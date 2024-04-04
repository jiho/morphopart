#
# Functions that define the steps of the pipeline
#
# All functions proceed the same way: they check if the output file exist,
# if it does, they read it and return the result; if not they produce it.
#

import os
import pickle as pkl
import numpy as np
import pandas as pd
import cuml
import rmm


def get_features(directory, params, log):
    """Extract features from original images

    Extract features.

    Args:
        directory: directory of the instruments that contain the original images
        params (DataFrame): a one row DataFrame with named elements containing:
            instrument (str): name of the instrument that took the images.
            features (str): name of the feature extractor.
                both of the arguments above determine the file to read.
        log : the logger.

    Returns:
        ndarray: an array of shape nb of objects x nb of features containing the features.
    """
    # TODO add DINO feature extraction and sub sample before feature extraction
    outfile = os.path.expanduser(
        f'~/datasets/morphopart/{params.instrument}/features_{params.features}.parquet'
    )

    if os.path.exists(outfile):
        log.info(' features already extracted')
        dataset_features=read_features(params,log)
    else :
        log.info(' extract features')
        if params.features=='uvplib':
            arr = os.listdir(directory+'/'+ params.instrument[0]+'/orig_imgs'); arr = [directory+'/'+ params.instrument[0]+'/orig_imgs/{x}' for x in arr]
            imagefilename=np.array(arr)
            # init features list
            features = list()
            # init filepath
            filepath = list()
            for i, path in enumerate(imagefilename):
                F =get_uvplib_features(path, params, log)  
                if len(F) > 0:  # test if feature extraction succeeded before appending to dataset
                    features.append(F)
                    filepath.append(path)
            dataset = pd.DataFrame(features)
            dataset['filename'] = filepath
            dataset_features = pd.DataFrame(features, index=dataset['objid'])
            dataset_features = dataset_features.rename(columns=str) # parquet need strings as column names
            
            dataset_features.to_parquet('~/datasets/morphopart/{params.instrument}/features_{params.features}.parquet')
        elif params.features=='mobilenet':
            log.info('	write them to disk')
            
            dataset_features=get_mobilenet_features(directory, params, log)
            
            dataset_features.to_parquet('~/datasets/morphopart/{params.instrument}/features_{params.features}.parquet')
        else:
            print("unknown features extraction")
    return(dataset_features)

def read_features(params, log):
    """Read features from a parquet file on disk

    Read all features. When there are many, subsample them to a manageable number.

    Args:
        params (DataFrame): a one row DataFrame with named elements containing:
            instrument (str): name of the instrument that took the images.
            features (str): name of the feature extractor.
                both of the arguments above determine the file to read.
            n_obj_max (int): maximum number of obejcts to consider. If more are available from the file, subsample it to reduce the number. This will be considered as "all" the objects for the purpose of the analysis.
        log : the logger.

    Returns:
        ndarray: an array of shape nb of objects x nb of features containing the features.
    """

    outfile = os.path.expanduser(
        f'~/datasets/morphopart/out/features_all__{params.instrument}_{params.features}_{params.n_obj_max}.pickle'
    )

    if os.path.exists(outfile):
        log.info('	load all features')
        with open(outfile, 'rb') as f:
            f_all = pkl.load(f)

    else :
        log.info('	read all features')
        # read all image features
        file = f'~/datasets/morphopart/{params.instrument}/features_{params.features}.parquet'
        features = pd.read_parquet(file)
        # NB: this is much faster than reading the .csv.gz file

        # when the amount of data is large, we will consider that "everything" is actually a subsample
        if params.n_obj_max < features.shape[0]:
            f_all = features.sample(n=params.n_obj_max, random_state=0)
            # NB: make it the same deterministic subsample across all replicates for replicability
            del features
        else:
            f_all = features
        #f_all = f_all.set_index('objid')
        # f_all.shape

        log.info('	write them to disk')
        with open(outfile, 'wb') as f:
            pkl.dump(f_all, f)

    return(f_all)


def subsample_features(f_all, params, log):
    """Subsample features for a smaller number of objects

    The goal will be to test the sensibility of the rest of the pipeline to the initial number of objects.

    Args:
        f_all (ndarray): the full array of features.
        params (DataFrame): a one row DataFrame with named elements containing all of the above and
            n_obj_sub (int): number of objects to subsample to test the robustness.
            replicate (int): an index of the replicate for the subsampling.
        log : the logger.

    Returns:
        ndarray: an array of shape nb of subsampled objects x nb of features containing the features.
    """
    outfile = os.path.expanduser(
        f'~/datasets/morphopart/out/features_subset__{params.instrument}_{params.features}_{params.n_obj_max}_{params.n_obj_sub}_{params.replicate}.pickle'
    )

    if os.path.exists(outfile):
        log.info('	load subsample of features')
        with open(outfile, 'rb') as f:
            f_sub = pkl.load(f)

    else :
        if params.n_obj_sub < f_all.shape[0]:
            log.info('	subsample features')
            # subsample rows
            f_sub = f_all.sample(n=params.n_obj_sub, random_state=params.replicate)
            # NB: the random state is defined for this to be reproducable
            #     it depends on the replicate number (just to make sure it changes between replicates)
            f_sub.shape
        else:
            log.info('	no need to subsample, copying all features')
            f_sub = f_all

        log.info('	write them to disk')
        with open(outfile, 'wb') as f:
            pkl.dump(f_sub, f)

    return(f_sub)


def reduce_dimension(f_sub, params, log):
    """Reduce the dimension of features

    Use PCA or UMAP to reduce the dimension of initial features to a more manageable number, for clustering.

    Args:
        f_sub (ndarray): an array of features.
        params (DataFrame): a one row DataFrame with named elements containing all of the above and
            dim_reducer (str): name of the dimensionality reduction method; PCA or UMAP are supported
        log : the logger.

    Returns:
        dict: containing
            - scaler: the feature scaler (mean=0 and variance=1) fitted to the data; has a .transform() method for new data
            - dim_reducer: the dimensional reduction method fitted to the data; also has a .transform() method for new data
            - features_reduced (ndarray): array of shape nb of objects in f_sub x nb of components retained
    """
    outfile = os.path.expanduser(
        f'~/datasets/morphopart/out/dimred__{params.instrument}_{params.features}_{params.n_obj_max}_{params.n_obj_sub}_{params.replicate}_{params.dim_reducer}.pickle'
    )

    if os.path.exists(outfile):
        log.info('	load dimension reduction info')
        with open(outfile, 'rb') as f:
            output = pkl.load(f)

    else:

        log.info('	scale data')
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(f_sub)
        f_sub_scaled = scaler.transform(f_sub)
        # f_sub_scaled.shape

        log.info('	impute missing values')
        # since we have scaled the data, we can simply replace missing values by 0
        f_sub_scaled = np.nan_to_num(f_sub_scaled, copy=False)

        log.info('	define dimensionality reducer')
        if params.dim_reducer == 'PCA':
            import cuml
            # define the number of components:
            #   set a maximum to 50
            #   keep only those bringing more than 1% more explained variance
            dim_reducer = cuml.PCA(n_components=50)
            dim_reducer.fit(f_sub_scaled)
            expl_var = dim_reducer.explained_variance_ratio_
            n_components = np.min(np.where(expl_var < 0.01))

            # then define the dimensionality reduction based on this number of components
            dim_reducer = cuml.PCA(n_components=n_components)

        elif params.dim_reducer == 'UMAP':
            # define n_neighbours as a Michalis-Menten type function from the number of points
            def umap_n_neighbours(x):
                n_min = 10
                n_max = 200
                n = np.round(n_min + n_max*x / (500000+x))
                return(n)

            import cuml
            dim_reducer = cuml.UMAP(
                n_neighbors=umap_n_neighbours(f_sub.shape[0]),
                n_components=4
            )

        else:
            print('Unknown dimensionality reducer; crashing')

        log.info('	fit dimensionality reducer')
        dim_reducer.fit(f_sub_scaled)

        log.info('	reduce dimension of features')
        # split in chunks to apply the transformation (avoid memory errors on the GPU)
        f_sub_scaled = np.vsplit(f_sub_scaled, 10)
        f_sub_reduced = [dim_reducer.transform(chunk) for chunk in f_sub_scaled]
        f_sub_reduced = np.vstack(f_sub_reduced)

        log.info('	write to disk')
        output = {'scaler': scaler, 'dim_reducer': dim_reducer, 'features_reduced': f_sub_reduced}
        with open(outfile, 'wb') as f:
            pkl.dump(output, f)

        # clean CUDA memory
        rmm.reinitialize()
    return(output)


def cluster(f_sub_reduced, params, log):
    """Cluster features

    Use kmeans to cluster objects into a smaller number of morphs.

    Args:
        f_sub_reduced (ndarray): an array of features (or reduced dimension, for clustering to work well).
        params (DataFrame): a one row DataFrame with named elements containing all of the above and
            n_clusters_tot (int): number of groups to cluster the data into
        log : the logger.

    Returns:
        dict: containing
            - clusterer: the clustering function, fitted to the data; has a .transform() method for new data
            - centroids (ndarray): array of shape n_clusters_tot x nb of ccolumsn in f_sub_reduced, the coordinates of the cluster centroids in the reduced space.
    """
    outfile = os.path.expanduser(
        f'~/datasets/morphopart/out/clust__{params.instrument}_{params.features}_{params.n_obj_max}_{params.n_obj_sub}_{params.replicate}_{params.dim_reducer}_{params.n_clusters_tot}.pickle'
    )

    if os.path.exists(outfile):
        log.info('	load cluster info')
        with open(outfile, 'rb') as f:
            output = pkl.load(f)

    else :
        log.info('	define clusterer')

        import cuml
        clust = cuml.KMeans(n_clusters=params.n_clusters_tot,
                       init='scalable-k-means++', n_init=10,
                       random_state=params.replicate
                      )

        clust.fit(f_sub_reduced)

        log.info('	define cluster centroids')
        centroids = clust.cluster_centers_

        log.info('	compute cluster membership')
        clusters = clust.predict(f_sub_reduced)

        log.info('	write to disk')
        output = {'clusterer': clust, 'centroids': centroids, 'clusters': clusters}
        with open(outfile, 'wb') as f:
            pkl.dump(output, f)

        # clean CUDA memory
        rmm.reinitialize()
    return(output)


def hierarchize(centroids, params, log):
    """Build a hierachical tree of centroids

    Use AgglomerativeClustering to build a hiearchical tree of centroids and compute the cluster values at all cutting levels.

    Args:
        centroids (ndarray): coordinates of the points to hierachize
        params (DataFrame): a one row DataFrame with named elements containing all of the above and
            linkage (str): linkage method in the agglomerative clustering (ward, complete, etc.)
        log : the logger.

    Returns:
        tree (DataFrame): with as many rows and columns as there are initial clusters; columns are numbered from 1 and each gives the cluster membership for the corresponding number of clusters. This means column 1 contains 1 cluster, so all 0; column 2 contains 2 clusters, so either 0 or 1; column 3...; and the last column contains all different numbers corresponding to the maximum level of clusters.
    """

    outfile = os.path.expanduser(
        f'~/datasets/morphopart/out/tree__{params.instrument}_{params.features}_{params.n_obj_max}_{params.n_obj_sub}_{params.replicate}_{params.dim_reducer}_{params.n_clusters_tot}_{params.linkage}.pickle'
    )

    if os.path.exists(outfile):
        log.info('	load tree')
        with open(outfile, 'rb') as f:
            tree = pkl.load(f)

    else :
        log.info('	define tree of centroids')

        from sklearn.cluster import AgglomerativeClustering

        # number of clusters
        n = centroids.shape[0]
        # NB: should be params.n_clusters_tot, but we may as well drop this dependency

        tree = np.zeros([n,n]).astype(int)
        for i in range(0,n):
            hclust = AgglomerativeClustering(n_clusters=i+1, linkage='ward')
            clusters = hclust.fit_predict(centroids)
            tree[:,i] = clusters
        tree = pd.DataFrame(tree)
        tree.columns = np.arange(1,n+1)

        log.info('	write to disk')
        with open(outfile, 'wb') as f:
            pkl.dump(tree, f)

    return(tree)


def transform_features(f_all, dimred, params, log):
    """Transform all features in the reduced space

    Args:
        f_all (ndarray): features of all objects .
        dimred (dict): dimensionality reduction information, output by function reduce_dimension().
        params (DataFrame): a one row DataFrame with named elements containing the nececarry parameters
        log : the logger.

    Returns:
        f_all_reduced (ndarray): features in f_all reduced through dimred['dim_reducer'].
    """

    outfile = os.path.expanduser(
        f'~/datasets/morphopart/out/features_all_reduced__{params.instrument}_{params.features}_{params.n_obj_max}_{params.n_obj_sub}_{params.replicate}_{params.dim_reducer}.pickle'
    )

    if os.path.exists(outfile):
        log.info('	load full data predicted based on current subsample')
        with open(outfile, 'rb') as f:
            f_all_reduced = pkl.load(f)

    else :
        # when n_obj_sub = n_obj_max, we would do the same thing twice
        if params['n_obj_sub'] == params['n_obj_max']:
            log.info('	read already reduced features')
            f_all_reduced = dimred['features_reduced']
        else:
            log.info('	reduce all features based on current subsample')

            f_all_scaled = dimred['scaler'].transform(f_all)
            f_all_scaled = np.nan_to_num(f_all_scaled, copy=False)

            # split in chunks to apply the transformation (avoid memory errors on the GPU)
            f_all_scaled = np.vsplit(f_all_scaled, 10)
            f_all_reduced = [dimred['dim_reducer'].transform(chunk) for chunk in f_all_scaled]
            f_all_reduced = np.vstack(f_all_reduced)

        log.info('	write to disk')
        with open(outfile, 'wb') as f:
            pkl.dump(f_all_reduced, f)

        # clean CUDA memory
        rmm.reinitialize()

    return(f_all_reduced)


def fast_merge(x, y, on, **kwargs):
    """Merge two DataFrames based on a column

    This is a faster implementation of .merge()

    Args:
        x,y (DataFrame): DataFrames to merge
        on (string, int): name/index of the column to merge on
        **kwargs: passed on .join()

    Returns:
        x (DataFrame): x with relevant rows of y appended by the join
    """
    x.set_index(on, inplace=True)
    y.set_index(on, inplace=True)
    x = x.join(y, **kwargs)
    x.reset_index(inplace=True)
    return(x)

def safe_sample(x, size, **kwargs):
    """Take a random sample of rows of a table with some conditions
    
    If size is larger than the number of rows of the table, just take all elements.
    If the table has only one element, return an empty array.
    
    Args:
        x (DataFrame): to sample rows from.
        size (int): the sample size.
        **kwargs: passed to pandas.sample() (useful for random_state for example)
    
    Returns:
        smp (ndarray): the rows sampled from the DataFrame
    """
    import numpy as np
    
    nrows = x.shape[0]
    if nrows == 1:
        # return empty set
        smp = x.iloc[np.arange(0, 0)]
    elif nrows <= size:
        # return all
        smp = x
    else:
        # take sample
        smp = x.sample(n=size, axis=0, ignore_index=False, replace=False, **kwargs)
    return(smp)

def sample_stratified_by_category(n, size, by, **kwargs):
    """Sample rows of a table stratified according to a categorical variable

    Args:
        n (int): number of rows of the table.
        size (int): number of elements to take.
        by (ndarray or list): of length n, values of the categories to stratifiy by.
        **kwargs: passed to pandas.sample()

    Returns:
        idx (ndarray): indexes of the rows sampled.
    """
    import pandas as pd
    import numpy as np
    
    # compute number of elements to sample in each stratum
    n_strata = len(np.unique(by))
    n_per_stratum = int(size / n_strata)
    # sample
    df = pd.DataFrame({'strat': by})
    smp = df.groupby('strat', group_keys=False).apply(safe_sample, size=n_per_stratum, **kwargs)
    # and get indexes
    idx = smp.index.values
    return(idx)

def sample_stratified_continuous(n, size, by, **kwargs):
    """Sample rows of a table stratified according to a continuous variable

    Args:
        n (int): number of rows of the table.
        size (int): number of elements to take.
        by (ndarray or DataFrame): the continous variable(s) to stratify by. 
        **kwargs: passed to pandas.sample()

    Returns:
        idx (ndarray of int): indexes of the rows of x sampled
    """
    import pandas as pd
    
    # cut the stratification columns in 5 pieces of ~ the same size
    bydf = pd.DataFrame(by)
    bydf = bydf.reset_index(drop=True)
    for i in bydf:        
        bydf[i] = pd.cut(bydf[i], bins=np.quantile(bydf[i], np.linspace(0, 1, 6)))

    # compute number of elements to sample per stratum
    # NB: ensure there are at least 2 per stratum
    n_per_stratum = np.max([int(size / 5**by.shape[1]), 2])
    
    # sample
    smp = bydf.groupby(bydf.columns.values.tolist(), group_keys=False).apply(safe_sample, size=n_per_stratum, **kwargs)
    # and get indexes
    idx = smp.index.values
    
    return(idx)

def evaluate(f_all_reduced, clust, tree, f_all_reduced_ref, clusters_ref, tree_ref, params, log):
    """Evaluate this pipeline thanks to the ARI, DBCV metrics

    Args:
        f_all_reduced (ndarray): features for all objects reduced based on the dimensionality reduction fitted on the *current subsample*; output of transform_predict().
        clust (dict): clusterer fitted on the *current subsample*; output of cluster().
        tree (DataFrame): hierarchical tree fitted on the *current subsample*; output of tree().
        f_all_reduced_ref (ndarray): features for all objects reduced based the *full dataset*.
        clusters_ref (ndarray): cluster numbers for all objects based on the clusterer fitted on the *full dataset*.
        tree_ref (DataFrame): hierarchical tree fitted on the *full dataset*
        params (DataFrame): a one row DataFrame with named elements containing all of the above and:
            n_clusters_eval (int): number of clusters at which to perform the evaluation.
            n_obj_eval (int): size of the subsamples with which to estimate DBCV (using all data is much too expensive computationnally).
        log : the logger.

    Returns:
        results (DataFrame): containing the quality metrics
    """
    outfile = os.path.expanduser(
        f'~/datasets/morphopart/out/eval__{params.instrument}_{params.features}_{params.n_obj_max}_{params.n_obj_sub}_{params.replicate}_{params.dim_reducer}_{params.n_clusters_tot}_{params.linkage}_{params.n_clusters_eval}_{params.n_obj_eval}.csv'
    )
    if os.path.exists(outfile):
        # log.info('    load evaluation results')
        # with open(outfile, 'rb') as f:
        #     results = pd.read_csv(f)
        log.info('	done')

    else :
        log.info('	predict cluster number of all objects in the reduced features space')
        # make a DataFrame with cluster level as column index
        # NB: internally, this is likely using a nearest neighbour classifier
        c_all = pd.DataFrame({params.n_clusters_tot: clust['clusterer'].predict(f_all_reduced)})

        if params.n_clusters_eval != params.n_clusters_tot:
            # reduce to the number of clusters requested for evaluation
            # = merge the level of the tree with the correct number of clusters
            log.info('	reduce to the target number of clusters')
            c_all = fast_merge(c_all, tree[[params.n_clusters_tot, params.n_clusters_eval]], on=params.n_clusters_tot)

        # compute metrics
        log.info('	compute ARI score')

        # define the reference clusters (at n_cluster_eval level)
        c_all_ref = pd.DataFrame({params.n_clusters_tot: clusters_ref})
        # NB: if we are evaluating at n_clusters_tot, we do not need to add a column
        if params.n_clusters_eval != params.n_clusters_tot:
            c_all_ref = fast_merge(c_all_ref, tree_ref[[params.n_clusters_tot, params.n_clusters_eval]], on=params.n_clusters_tot)

        from sklearn.metrics.cluster import adjusted_rand_score # = ARI score
        score_ARI = adjusted_rand_score(c_all_ref[params.n_clusters_eval].values, c_all[params.n_clusters_eval].values)

        # from cuml.metrics.cluster.adjusted_rand_index import adjusted_rand_score
        # score_ARI = adjusted_rand_score(c_all_ref[params.n_clusters_eval].values, c_all[params.n_clusters_eval].values)
        # NB: returns negative values sometimes!

        # subsample the data for DBCV and silouhette computation (too long otherwise)
        n_repetitions = 5
        # eval_subsamples = [sample_stratified_by_category(
        #                      n=f_all_reduced.shape[0],
        #                      size=params.n_obj_eval,
        #                      by=c_all[params.n_clusters_eval].values,
        #                      random_state=i)
        #                      for i in range(n_repetitions)]
        # NB: another possibility is to sample the reduced space, stratified by dimensions 1 and 2
        #     but it may result in all points being in the same subsample and that throws DBCV out
        eval_subsamples = [sample_stratified_continuous(
                             n=f_all_reduced.shape[0],
                             size=params.n_obj_eval,
                             by=f_all_reduced[:,[0,1]],
                             random_state=i)
                             for i in range(n_repetitions)]

        # log.info('    compute DBCV')
        # import ipdb; ipdb.set_trace()
        #
        # import hdbscan
        # # or https://github.com/FelSiq/DBCV but it is slower
        # DBCVs = [hdbscan.validity.validity_index(f_all_reduced[idx,:].astype('double'), labels=c_all[params.n_clusters_eval].values[idx], metric='euclidean') for idx in eval_subsamples]

        log.info('	compute Silhouette score')
        # import ipdb; ipdb.set_trace()
        def safe_silhouette_score(X, labels):
            from cuml.metrics.cluster.silhouette_score import cython_silhouette_score
            import numpy as np
            if len(np.unique(labels))==1:
                score = float('NaN')
            else:
                score = cython_silhouette_score(X, labels)
            return(score)
        SILs = [safe_silhouette_score(f_all_reduced[idx,:].astype('double'), labels=c_all[params.n_clusters_eval].values[idx]) for idx in eval_subsamples]

        log.info('	compute pairwise distances in reduced space')
        # PCA that produces the _ref one is fitted on the full dataset while the other is fitted on the subset. They may not have the same number of components.
        # The number of components is fixed to the lower number components between the full dataset and the subset.
        n_axis_ref=np.shape(f_all_reduced_ref)[1]
        n_axis=np.shape(f_all_reduced)[1]
        print(n_axis_ref, n_axis)
        if n_axis_ref != n_axis:
            fix_axis=min(n_axis_ref,n_axis)
            DISTs = np.linalg.norm(f_all_reduced_ref[:,0:fix_axis] - f_all_reduced[:,0:fix_axis], axis=0)
        else:
            DISTs = np.linalg.norm(f_all_reduced_ref - f_all_reduced, axis=0)


        # TODO compute purity of labels ?
        from sklearn.metrics.cluster import homogeneity_score
        homogeneity_score=homogeneity_score(c_all_ref[params.n_clusters_eval].values, c_all[params.n_clusters_eval].values)

        log.info('	write to disk')
        results = dict(params) | {
            'ARI': score_ARI,
            # record the actual number in the subsample used for the evaluation
            # (there can be fewer than params.n_obj_eval if there are small clusters)
            'n_obj_eval_actual': len(eval_subsamples[0]),
            # 'DBCV': np.mean(DBCVs), 'sdDBCV': np.std(DBCVs),
            'SIL': np.mean(SILs), 'sdSIL': np.std(SILs),
            # TODO count nans, exlude them and compute without them
            'DIST': np.mean(DISTs), 'sdDIST': np.std(DISTs)
        }

        results = pd.DataFrame(results, index=[0])
        results.to_csv(outfile, index=False)
    
    # clean CUDA memory
    rmm.reinitialize()

    return(results)


def get_uvplib_features(imagefilename, params, log):
    """  
    -        -
    Parameters
    ----------
    imagefilename : str
        Name of the image file containing the object for features extraction.
        
    params (DataFrame): a one row DataFrame with named elements containing:
        instrument (str): name of the instrument that took the images.
        features (str): name of the feature extractor.
            both of the arguments above determine the file to read.
        n_obj_max (int): maximum number of obejcts to consider. If more are available from the file, subsample it to reduce the number. This will be considered as "all" the objects for the purpose of the analysis.

    Returns
    -------
    features : OrderedDict
        Ordered Dictionary containing the features extracted from biggest
        connected region found in image.
        An empty OrderedDict is returned if no regions found.
    """
    #Package
    import imageio as iio
    import uvpec 
    from skimage import io, measure
    from numpy import argmax, histogram
    from math import sqrt, atan2
    from collections import OrderedDict
    from cython_uvp6 import py_get_features
    
    # load image file
    try :
        img = io.imread(imagefilename)    
    except :
        print("get_uvplib_features function : Failed to open file", imagefilename)
        return OrderedDict()
    
    # because images had 3 identical bands and uvpec was designed for grey-one-band images, we only keep one "layer"
    if params.instrument == 'uvp5hd':
        img = img[:,:,0] # TO INCLUDE FOR UVP5 IMAGES; exclude for uvp6
        threshold=4
        use_C=False
    elif params.instrument == 'uvp5sd':
        img = img[:,:,0] # TO INCLUDE FOR UVP5 IMAGES; exclude for uvp6
        threshold=8
        use_C=False
    elif params.instrument == 'uvp6':
        threshold=21
        use_C=True
        
    # crop the scale bar
    height = img.shape[0]
    img = img[0:(height-31),:]

    # invert the image (white over black)
    img_black = 255 - img #np.invert(img)

    # reassign black image to image
    img = img_black
    
    # apply thresholding 
    thresh_img = img > threshold
    if thresh_img.sum() < 1 : # there are no pixels above the threshold, return empty dict
        print("get_uvp6_features function : No objects found in", 
              imagefilename, "with threshold", threshold)
        return OrderedDict()
    
    # segmentation into connected regions 
    label_img = measure.label(thresh_img)
    
    # get region properties for connected regions found
    props = measure.regionprops(label_img, img)
    
    # get index of the region presenting the biggest area in square pixels
    Areas = list()
    for region in props:
        Areas.append(region.area)        
    max_area_idx=argmax(Areas)
    region = props[max_area_idx]
    
    if use_C is True:
        # execute C code
        cfeatures = py_get_features(img, region, threshold)
        
        # build an output ordered dict with the features vector
        # ATTENTION : feature insertion order is VERY important,
        # as it has to match exactly the feature order used on UVP6
        features = OrderedDict()
        features["area"] = cfeatures[0]
        features["width"] = cfeatures[1]
        features["height"] = cfeatures[2]
        features["mean"] = cfeatures[3]
        features["stddev"] = cfeatures[4]
        features["mode"] = cfeatures[5]
        features["min"] = cfeatures[6]
        features["max"] = cfeatures[7]
        features["x"] = cfeatures[8]
        features["y"] = cfeatures[9]
        features["xm"] = cfeatures[10]
        features["ym"] = cfeatures[11]
        features["major"] = cfeatures[12]
        features["minor"] = cfeatures[13]
        features["angle"] = cfeatures[14]
        features["eccentricity"] = cfeatures[15]
        features["intden"] = cfeatures[16]
        features["median"] = cfeatures[17]
        features["histcum1"] = cfeatures[18]
        features["histcum3"] = cfeatures[19]
        features["esd"] = cfeatures[20]
        features["range"] = cfeatures[21]
        features["meanpos"] = cfeatures[22]
        features["cv"] = cfeatures[23]
        features["sr"] = cfeatures[24]
        features["bbox_area"] = cfeatures[25]
        features["extent"] = cfeatures[26]

        features["central_moment-2-0"] = cfeatures[27]
        features["central_moment-1-1"] = cfeatures[28]
        features["central_moment-0-2"] = cfeatures[29]
        features["central_moment-3-0"] = cfeatures[30]
        features["central_moment-2-1"] = cfeatures[31]
        features["central_moment-1-2"] = cfeatures[32]
        features["central_moment-0-3"] = cfeatures[33]
    
        features["hu_moment-1"] = cfeatures[34]
        features["hu_moment-2"] = cfeatures[35]
        features["hu_moment-3"] = cfeatures[36]
        features["hu_moment-4"] = cfeatures[37]
        features["hu_moment-5"] = cfeatures[38]
        features["hu_moment-6"] = cfeatures[39]
        features["hu_moment-7"] = cfeatures[40]

        features["gray_central_moment-2-0"] = cfeatures[41]
        features["gray_central_moment-1-1"] = cfeatures[42]
        features["gray_central_moment-0-2"] = cfeatures[43]
        features["gray_central_moment-3-0"] = cfeatures[44]
        features["gray_central_moment-2-1"] = cfeatures[45]
        features["gray_central_moment-1-2"] = cfeatures[46]
        features["gray_central_moment-0-3"] = cfeatures[47]

        features["gray_hu_moment-1"] = cfeatures[48]
        features["gray_hu_moment-2"] = cfeatures[49]
        features["gray_hu_moment-3"] = cfeatures[50]
        features["gray_hu_moment-4"] = cfeatures[51]
        features["gray_hu_moment-5"] = cfeatures[52]
        features["gray_hu_moment-6"] = cfeatures[53]
        features["gray_hu_moment-7"] = cfeatures[54]
        
        return (features)

    else:
        # get gray values histogram for this region, and clear the 0 bin (background pixels)
        hist = histogram(region.intensity_image, bins=256, range=(0,256))[0]
        hist[0] = 0
    
        # calculate histogram related features
        mean = region.mean_intensity
        vmin = region.min_intensity
        vmax = region.max_intensity
        intden = region.weighted_moments_central[0][0] # this is the sum of all pixel values
        mode = argmax(hist)
        vrange = vmax - vmin
        meanpos = (mean - vmin)/vrange
    
        # get quartiles, and accumulate squared pixels values for stddev calculation
        nb_pixels = region.area
        first_quartile = 0.25 * nb_pixels
        second_quartile = 0.5 * nb_pixels
        third_quartile = 0.75 * nb_pixels
    
        square_gray_acc = 0; pix_acc = 0
        median = -1; histcum1 = -1; histcum3 = -1
    
        for gray_level, count in enumerate(hist) :
            if count != 0 : 
                square_gray_acc += count*gray_level*gray_level
                pix_acc += count
                if (histcum1 == -1) and (pix_acc > first_quartile) : histcum1 = gray_level
                if (median == -1) and (pix_acc > second_quartile) : median = gray_level
                if (histcum3 == -1) and (pix_acc > third_quartile) : histcum3 = gray_level            
   
        stddev = sqrt((square_gray_acc/nb_pixels) - (mean*mean))    
        cv = 100*(stddev/mean)
        sr = 100*(stddev/vrange)
    
        angle = 0.5 * atan2(2*region.moments_central[1][1], (region.moments_central[0][2] - region.moments_central[2][0]))
    
        # build an output ordered dict with the features vector
        # ATTENTION : feature insertion order is VERY important,
        # as it has to match exactly the feature order used on UVP6    
        features = OrderedDict()
        features["area"] = nb_pixels
        features["width"] = region.bbox[3] - region.bbox[1]
        features["height"] = region.bbox[2] - region.bbox[0]
        features["mean"] = mean
        features["stddev"] = stddev
        features["mode"] = mode
        features["min"] = vmin
        features["max"] = vmax
        features["x"] = region.local_centroid[1]
        features["y"] = region.local_centroid[0]
        features["xm"] = region.weighted_local_centroid[1]
        features["ym"] = region.weighted_local_centroid[0]
        features["major"] = region.major_axis_length
        features["minor"] = region.minor_axis_length    
        features["angle"] = angle
        features["eccentricity"] = region.eccentricity
        features["intden"] = intden
        features["median"] = median
        features["histcum1"] = histcum1
        features["histcum3"] = histcum3
        features["esd"] = region.equivalent_diameter
        features["range"] = vrange
        features["meanpos"] = meanpos
        
        features["cv"] = cv
        features["sr"] = sr
        features["bbox_area"] = region.bbox_area
        features["extent"] = region.extent
    
        features["central_moment-2-0"] = region.moments_central[0][2]
        features["central_moment-1-1"] = region.moments_central[1][1]
        features["central_moment-0-2"] = region.moments_central[2][0]
        features["central_moment-3-0"] = region.moments_central[0][3]
        features["central_moment-2-1"] = region.moments_central[1][2]
        features["central_moment-1-2"] = region.moments_central[2][1]
        features["central_moment-0-3"] = region.moments_central[3][0]
    
        """
        Current SciKit Hu Moments implementation is apparently wrong !
        (bad coordinate system convention rc <-> xy)
        It only has an impact on the sign of seventh Hu moment (mirroring)
        This is why we're inverting the sign here for hu_moment-7
        
        """
        features["hu_moment-1"] = region.moments_hu[0]
        features["hu_moment-2"] = region.moments_hu[1]
        features["hu_moment-3"] = region.moments_hu[2]
        features["hu_moment-4"] = region.moments_hu[3]
        features["hu_moment-5"] = region.moments_hu[4]
        features["hu_moment-6"] = region.moments_hu[5]
        features["hu_moment-7"] = - region.moments_hu[6] # see comment above
    
        features["gray_central_moment-2-0"] = region.weighted_moments_central[0][2]
        features["gray_central_moment-1-1"] = region.weighted_moments_central[1][1]
        features["gray_central_moment-0-2"] = region.weighted_moments_central[2][0]
        features["gray_central_moment-3-0"] = region.weighted_moments_central[0][3]
        features["gray_central_moment-2-1"] = region.weighted_moments_central[1][2]
        features["gray_central_moment-1-2"] = region.weighted_moments_central[2][1]
        features["gray_central_moment-0-3"] = region.weighted_moments_central[3][0]
    
        features["gray_hu_moment-1"] = region.weighted_moments_hu[0]
        features["gray_hu_moment-2"] = region.weighted_moments_hu[1]
        features["gray_hu_moment-3"] = region.weighted_moments_hu[2]
        features["gray_hu_moment-4"] = region.weighted_moments_hu[3]
        features["gray_hu_moment-5"] = region.weighted_moments_hu[4]
        features["gray_hu_moment-6"] = region.weighted_moments_hu[5]
        features["gray_hu_moment-7"] = - region.weighted_moments_hu[6] # see comment above
    
        return(features)

def training_model_mobilenet(directory, params, log):
    
       from deep import tensorflow_tricks  # settings for tensorflow to behave nicely

       import pandas as pd
       # pd.set_option('display.max_columns', None)
       import numpy as np
       import tensorflow as tf
       from sklearn import metrics

       from importlib import reload
       from deep import dataset            # custom data generator
       from deep import cnn                # custom functions for CNN generation
       dataset = reload(dataset)
       cnn = reload(cnn)
       
       exec(open('set_cnn_option.py').read()) 
       # prevent HDF file locking to be able to write on complex
       # needed to save checkpoints
       os.system("export HDF5_USE_FILE_LOCKING='FALSE'")

       print('Prepare datasets') ## ----
       # read DataFrame with image ids, paths and labels
       # NB: those would be in the database in EcoTaxa

       # read labels
       # TODO swap the comments in the next two lines for tests
       df = pd.read_csv(data_dir + '/taxa.csv.gz', usecols = ['objid','taxon'], nrows=10000)
       #df = pd.read_csv(data_dir + '/taxa.csv.gz', usecols = ['objid','taxon'])

       df = df.rename(columns={'taxon': 'label'})
       # compute path to images
       df['img_path'] = [data_dir + '/orig_imgs/' + str(objid) + '.' + img_format for objid in df['objid']]
       print('  training from ' + str(df.shape[0]) + ' objects')

       # extract a small validation set
       seed = 1
       # 95% in train
       df_train = df.groupby('label').sample(frac=0.85, random_state=seed)
       # the rest in val
       df_valid   = df.loc[list(set(df.index) - set(df_train.index))]

       # count nb of examples per class in the training set
       class_counts = df_train.groupby('label').size()
       # print(class_counts)

       # list classes
       classes = class_counts.index.to_list()

       # generate categories weights
       # i.e. a dict with format { class number : class weight }
       if use_class_weight:
           max_count = np.max(class_counts)
           class_weights = {}
           for idx,count in enumerate(class_counts.items()):
               class_weights.update({idx : (max_count / count[1])**weight_sensitivity})
       else:
           class_weights = None

       # define numnber of  classes to train on
       nb_of_classes = len(classes)

       # define data generators
       train_batches = dataset.EcoTaxaGenerator(
           images_paths=df_train['img_path'].values,
           input_shape=input_shape,
           labels=df_train['label'].values, classes=classes,
           batch_size=batch_size, augment=augment, shuffle=True,
           crop=[0,0,bottom_crop,0])

       valid_batches = dataset.EcoTaxaGenerator(
           images_paths=df_valid['img_path'].values,
           input_shape=input_shape,
           labels=df_valid['label'].values, classes=classes,
           batch_size=batch_size, augment=False, shuffle=False,
           crop=[0,0,bottom_crop,0])
       # NB: do not shuffle or augment data for validation, it is useless

       print('Prepare model') ## ----
       # try loading the model from a previous training checkpoint
       my_cnn,initial_epoch = cnn.Load(ckpt_dir)

       # if nothing is loaded this means the model was never trained
       # in this case, define it
       if (my_cnn is not None) :
           print('  restart from model trained until epoch ' + str(initial_epoch))
       else :
           print('  define model')
           # define CNN
           my_cnn = cnn.Create(
               # feature extractor
               fe_url=fe_url,
               input_shape=input_shape,
               fe_trainable=fe_trainable,
               # fully connected layer(s)
               fc_layers_sizes=fc_layers_sizes,
               fc_layers_dropout=fc_layers_dropout,
               # classification layer
               classif_layer_size=nb_of_classes,
               classif_layer_dropout=classif_layer_dropout
           )

           print('  compile model')
           # compile CNN
           my_cnn = cnn.Compile(
               my_cnn,
               initial_lr=initial_lr,
               lr_method=lr_method,
               decay_steps=len(train_batches),
               decay_rate=decay_rate,
               loss=loss
           )

       print('Train model') ## ----

       # train CNN
       history = cnn.Train(
           model=my_cnn,
           train_batches=train_batches,
           valid_batches=valid_batches,
           epochs=epochs,
           initial_epoch=initial_epoch,
           log_frequency=log_frequency,
           class_weight=class_weights,
           output_dir=ckpt_dir,
           workers=workers
       )

def mobilenet_feature_extractor(directory, params, log):
    import matplotlib.pyplot as plt # science packages
    import tensorflow as tf
    from importlib import reload
    from deep import dataset            # custom data generator
    from deep import cnn                # custom functions for CNN generation
    dataset = reload(dataset)
    cnn = reload(cnn)
    
    os.system("export HDF5_USE_FILE_LOCKING='FALSE'")
    
    exec(open('set_cnn_option.py').read()) 
    outfile = os.path.expanduser(ckpt_dir + '/training_log.tsv')
    
    if os.path.exists(outfile):
        print('Model and feature extractor already exist') ## ---- 
        df = pd.read_csv(ckpt_dir + '/training_log.tsv', sep='\t')
    else:
        training_model_mobilenet(directory, params, log)
        # Lis le log de l'entrainement et fais un plot. Tu veux que la val_loss et val_accuracy saturent
        df = pd.read_csv(ckpt_dir + '/training_log.tsv', sep='\t')
    df = df.drop(['batch', 'learning_rate'], axis='columns')

    df.plot(x='step', subplots=True)
    plt.show()

    df.plot(x='epoch', subplots=True)
    plt.show()
    # define best_epoch
    # Il faut choisir l'epoch de val_loss minimale et val_accuracy maximale.
    
    best_epoch = input("Enter the best epoch (use None to get the latest epoch): ")
    # Create Model and features extraction
    print('Create model and feature extractor') ## ----    
    # load model for best epoch
    my_cnn,epoch = cnn.Load(ckpt_dir, epoch=int(best_epoch))
    print(' at epoch {:d}'.format(epoch))
    # save model (just in case)
    my_cnn.save(cnn_dir + '/best_model', include_optimizer=False)
    # drop the last two layers to get a feature extractor + the middle MLP layer
    my_fe = tf.keras.models.Sequential([layer for layer in my_cnn.layers[0:-2] ])
    my_fe.summary()

    # save feature extractor (just in case)
    my_fe.save(cnn_dir + '/feature_extractor')
                                                                    
def get_mobilenet_features(directory, params, log):
    import tensorflow as tf
    from deep import progress # custom functions to track progress of training/prediction
    from deep import dataset            # custom data generator
    import pandas as pd
    
    exec(open('set_cnn_option.py').read()) 
    outfile = os.path.expanduser(cnn_dir + '/feature_extractor')
    
    if os.path.exists(outfile):
        print('Load data and extract features') ## ----
        my_fe = tf.keras.models.load_model(cnn_dir + '/feature_extractor', compile=False)
    else:
        my_fe = mobilenet_feature_extractor(directory, params, log) ######
        my_fe = tf.keras.models.load_model(cnn_dir + '/feature_extractor', compile=False)   
    # get model input shape
    input_shape = my_fe.layers[0].input_shape
    # remove the None element at the start (which is where the batch size goes)
    input_shape = tuple(x for x in input_shape if x is not None)

    # TODO swap the comments in the next two lines for tests
    df = pd.read_csv(data_dir + '/taxa.csv.gz', usecols = ['objid','taxon'], nrows=10)
    #df = pd.read_csv(data_dir + '/taxa.csv.gz', usecols = ['objid','taxon'])
    df = df.rename(columns={'taxon': 'label'})
    # compute path to images
    df['img_path'] = [data_dir + '/orig_imgs/' + str(objid) + '.' + img_format for objid in df['objid']]
    print('  found ' + str(df.shape[0]) + ' objects')

    batches = dataset.EcoTaxaGenerator(
        images_paths=df['img_path'].values,
        input_shape=input_shape,
        # NB: although the labels are in the file, we don't use them here
        labels=None, classes=None,
        batch_size=batch_size, augment=False, shuffle=False,
        crop=[0,0,bottom_crop,0])

    # extract features by going through the batches
    features = my_fe.predict(batches, callbacks=[progress.TQDMPredictCallback()],
                                  max_queue_size=max(10, workers*2), workers=workers)
    dataset_features = pd.DataFrame(features, index=df['objid'])
    dataset_features = dataset_features.rename(columns=str) # parquet need strings as column names
    
    return(dataset_features)
