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


def extract_features(params, log):
    # TODO add the code to extract features here
    # TODO the read_features function should therefore be included here, to read them when they have already been extracted
    return(None)

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
        f_all = f_all.set_index('objid')
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
        # TODO this could be a separate step (would save time of only the clustering method changes and not the feature extraction... but I am lazy for now)

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
        from cuml.metrics.cluster.silhouette_score import cython_silhouette_score
        SILs = [cython_silhouette_score(f_all_reduced[idx,:].astype('double'), labels=c_all[params.n_clusters_eval].values[idx]) for idx in eval_subsamples]

        log.info('	compute pairwise distances in reduced space')
        DISTs = np.linalg.norm(f_all_reduced_ref - f_all_reduced, axis=0)

        # TODO compute purity of labels ?

        log.info('	write to disk')
        results = dict(params) | {
            'ARI': score_ARI,
            # record the actual number in the subsample used for the evaluation
            # (there can be fewer than params.n_obj_eval if there are small clusters)
            'n_obj_eval_actual': len(eval_subsamples[0]),
            # 'DBCV': np.mean(DBCVs), 'sdDBCV': np.std(DBCVs),
            'SIL': np.mean(SILs), 'sdSIL': np.std(SILs),
            'DIST': np.mean(DISTs), 'sdDIST': np.std(DISTs)
        }

        results = pd.DataFrame(results, index=[0])
        results.to_csv(outfile, index=False)
    
    # clean CUDA memory
    rmm.reinitialize()

    return(results)
