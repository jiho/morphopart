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
 
