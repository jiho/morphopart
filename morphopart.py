#
# Functions that define the steps of the pipeline
#
# All functions proceed the same way: they check if the output file exist,
# if it does, they read it and return the result; if not they produce it.
#

# TODO add DINO feature extraction and sub sample before feature extraction

import os
import pickle as pkl
import numpy as np
import pandas as pd
import cuml
import rmm
import re

from sklearn.cluster import BisectingKMeans
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import row_norms
from sklearn.cluster._kmeans import _labels_inertia_threadpool_limit
from sklearn.base import BaseEstimator, TransformerMixin
#------------------------ Core pipeline function -------------------------------#

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
    # Create the full path to the output pickle file based on the current parameters
    outfile = os.path.expanduser(
        f'~/datasets/morphopart/out_test/features_all__{params.instrument}_{params.features}_{params.n_obj_max}.pickle'
    )

    if os.path.exists(outfile):                                                         # Check if the file already exists
        log.info(' features already extracted')                                         # Log that the features have already been extracted
        log.info(' load features')                                                      # Log that we are loading the existing features
        with open(outfile, 'rb') as f:                                                  # Open the pickle file in read-binary mode
            f_all = pkl.load(f)                                                         # Load the features into 'f_all'
    else :
        # Define the path to the parquet file for the features based on instrument and feature type
        outfile1 = os.path.expanduser(
            f'~/datasets/morphopart/{params.instrument}/features_{params.features}.parquet'
        )
        if os.path.exists(outfile1):                                                    # Check if the parquet file exists
            log.info(' All features already extracted')
            file = f'~/datasets/morphopart/{params.instrument}/features_{params.features}.parquet'
            features = pd.read_parquet(file)                                            # Load the parquet file into a DataFrame
            
            if params.n_obj_max < features.shape[0]:                                    # If the dataset has more rows than n_obj_max, subsample to reduce size
                log.info(' Original dataset of features is too large - subsampling to {params.n_obj_max} obj_max')
                f_all = features.sample(n=params.n_obj_max, random_state=0)             # Deterministic subsample for reproducibility
                del features                                                            # Free memory
            else:
                f_all = features                                                        # If dataset is already small enough, use it as is
            
            f_all = f_all.set_index('objid')                                            # Set 'objid' as the index
            f_all.shape

            log.info('	write them to disk')                                            # Log that the processed features will be saved to disk
            os.makedirs(os.path.dirname(outfile), exist_ok=True)                        # Ensure the output directory exists
            with open(outfile, 'wb') as f:
                pkl.dump(f_all, f)                                                      # Save the features DataFrame to a pickle file for faster future loading
        else :
            log.info(' extract features')                                               # Log that the extraction features will start
            image_dir='/home/irisson/datasets/morphopart/'+params.instrument+'/orig_imgs/' # Directory containing the raw images
            #image_dir='/home/jiho/datasets/morphopart/all/UVP5SD/'output'n_clusters
            
            arr = os.listdir(image_dir);                                        # List all files in the image directory
            # Determine the image file format based on the instrument
            if params.instrument=="uvp6":
                img_format = 'png'
            else:
                img_format='jpg'
            obj_id=[re.match(r'(.*)\.'+img_format, f).group(1) for f in arr]   # Extract the object IDs from the filenames by removing the file extension
        
            # Sub-sample raw data if the original dataset is too large
            if params.n_obj_max < len(obj_id):
                # Sub-sample
                df_sub_all=pd.DataFrame(arr)                                        # Convert the list of filenames to a DataFrame for easy sampling
                df_sub_all=df_sub_all.sample(n=params.n_obj_max, random_state=0)    # Randomly sample n_obj_max filenames deterministically (for reproducibility)
                df_sub_all = df_sub_all[0].values.tolist()                          # Convert the sampled DataFrame back to a list
            
                arr=df_sub_all                                                          # Update 'arr' to contain only the sampled filenames
                obj_id=[re.match(r'(.*)\.'+img_format, f).group(1) for f in df_sub_all] # Update 'obj_id' to match the sampled filenames by stripping the file extensions
            else:
                print(' extraction all raw data')
        
            # Extraction features following feature extractors
            if params.features=='uvplib':                                       # If feature extractors is uvplib
                imagefilename=np.array(arr)                                     # Convert the list of sampled filenames to a NumPy array for easier iteration
                features = list()                                               # Initialize an empty list to store extracted features
                filepath = list()                                               # Initialize an empty list to store corresponding file paths
                for i, path in enumerate(imagefilename):                        # Loop over each image file            
                    F =get_uvplib_features(image_dir+'/'+path, params, log)     # Extract features from the image using a custom function  
                    if len(F) > 0:                                              # test if feature extraction succeeded before appending to dataset
                        features.append(F)
                        filepath.append(path)
                dataset = pd.DataFrame(features)                                # Convert the list of features to a DataFrame
                dataset['filename'] = filepath
                f_all = pd.DataFrame(features, index=obj_id)                    # Add a column with the original filenames
                f_all = f_all.rename(columns=str)                               # parquet need strings as column names
        
                log.info('	write them to disk')                                # Log that the processed features will be saved to disk
                os.makedirs(os.path.dirname(outfile), exist_ok=True)            # Ensure the output directory exists
                with open(outfile, 'wb') as f:
                    pkl.dump(f_all, f)                                          # Save the features DataFrame to a pickle file for faster future loading
                        
            elif params.features=='mobilenet':                                  # if we want features from deep learning (here mobilenet)
                f_all=get_mobilenet_features(directory, params, obj_id, log)    # Extract features from the image using a custom function
                
                log.info('	write them to disk')                                # Log that the processed features will be saved to disk
                os.makedirs(os.path.dirname(outfile), exist_ok=True)            # Ensure the output directory exists
                with open(outfile, 'wb') as f:
                    pkl.dump(f_all, f)                                          # Save the features DataFrame to a pickle file for faster future loading
            else:
                print("unknown features extraction")                            # Display a message when the feature extractor is missing or not available
            
    return(f_all)                                                   # Return the features

# The "read_features" function below may no longer be needed / is deprecated
#def read_features(params, log):
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
        f'~/datasets/morphopart/out_test/features_all__{params.instrument}_{params.features}_{params.n_obj_max}.pickle'
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
    # Create the full path to the output pickle file based on the current parameters
    outfile = os.path.expanduser(
        f'~/datasets/morphopart/out_test/features_subset__{params.instrument}_{params.features}_{params.n_obj_max}_{params.n_obj_sub}_{params.replicate}.pickle'
    )

    if os.path.exists(outfile):                     # Check if the file already exists
        log.info('	load subsample of features')    # Log that the subsample of features have already been extracted
        with open(outfile, 'rb') as f:              # Open the pickle file in read-binary mode
            f_sub = pkl.load(f)                     # Load the features into 'f_sub'
    else :                                          # If the file doesn't exists
        if params.n_obj_sub < f_all.shape[0]:       # And if the number of objects in the subsample to use is smaller than the total number of features       
            log.info('	subsample features')        # Log that the subsample of features will be perform
            # subsample rows
            f_sub = f_all.sample(n=params.n_obj_sub, random_state=params.replicate) # NB: the random state is defined for this to be reproducable. It depends on the replicate                                                                      number (just to make sure it changes between replicates)
            f_sub.shape
        else:
            log.info('	no need to subsample, copying all features')                # Log that the subsample of features is already too small
            f_sub = f_all                                                           # So the total number of features is similar to the subsample

        log.info('	write them to disk')            # Log that the processed subsample features will be saved to disk
        with open(outfile, 'wb') as f:
            pkl.dump(f_sub, f)                      # Save the features DataFrame to a pickle file for faster future loading

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
    # Create the full path to the output pickle file based on the current parameters
    outfile = os.path.expanduser(
        f'~/datasets/morphopart/out_test/dimred__{params.instrument}_{params.features}_{params.n_obj_max}_{params.n_obj_sub}_{params.replicate}_{params.dim_reducer}.pickle'
    )

    if os.path.exists(outfile):                                                 # Check if the file already exists
        log.info('	load dimension reduction info')                             # Log that the dimensionality reduction has already been done
        with open(outfile, 'rb') as f:
            output = pkl.load(f)                                                # Load subsampled, reduced features
    else:
        log.info('	data preparation')                                                # Log that feature scaling on the subsampled dataset is starting
        
        if params.features =='uvplib':                                          # Clean dataset before transformer and scaler for dimensional reduction
            cols_to_drop = ["x:ym","angle",                                     # meaningless, related to location/orientation on image 
                            "bbox_area", "width", "height",                     # property of image, not particle
                            "esd",                                              # directly proportional to area
                            "min", "mode",                                      # unusable distributions
                            "range"]                                            # same as max                 

            cols_xy = [c for c in f_sub.columns if c.startswith(("x", "y"))]
            cols_to_remove = set(cols_to_drop).union(cols_xy)
            f_sub = f_sub.drop(columns=cols_to_remove, errors="ignore")
            
            masker_treshold_file = os.path.expanduser(f'~/datasets/morphopart/masker_quantile_threshold_uvplib.pickle')
            if os.path.exists(masker_treshold_file):
                with open(masker_treshold_file, "rb") as f:
                    masker = pkl.load(f)
                thresholds = masker.thresholds_
                f_sub = masker.transform_with_thresholds(f_sub, thresholds=thresholds)
                
            else: 
                masker = MaskExtremeByFeature(dict_para=dict_para,feature_names=feature_names)
                masker.fit_transform(f_sub.values)
                # sauvegarder l’objet entier
                with open(masker_treshold_file, "wb") as f:
                    pkl.dump(masker, f)
            
            f_sub, vars_kept, objs_kept = mask_nan(f_sub, max_var_na=10, max_obj_na=5)
                
        #################################################################
        log.info('	scale data')                                                # Log that feature scaling on the subsampled dataset is starting
        from sklearn.preprocessing import PowerTransformer, StandardScaler              
        yeo=PowerTransformer(method='yeo-johnson', standardize=False)
        yeo.fit(f_sub)
        f_yeo = yeo.transform(f_sub)
        
        scaler = StandardScaler()                                               # Initialize a StandardScaler to standardize features
        scaler.fit(f_yeo)                                                       # Fit the scaler to the Yeo-Johnson transformed data
        f_sub_scaled = scaler.transform(f_yeo)                                  # Transform the data to have zero mean and unit variance 
        # f_sub_scaled.shape
            
        log.info('	impute missing values')                                     # Log that missing values will be imputed
        f_sub_scaled = np.nan_to_num(f_sub_scaled, copy=False)                  # since we have scaled the data, we can simply replace missing values by 0

        log.info('	define dimensionality reducer')                             # Log that dimensionality reduction is starting
        if params.dim_reducer == 'PCA':                                         # If PCA is selected as the dimensionality reduction method
            import cuml                                                         # Import RAPIDS cuML for GPU-accelerated PCA
            n_components_max = min(f_sub.shape[1], 50)                          # Initialize PCA with a maximum of 50 components. Later we will keep only those bringing more than 1% more explained variance
            dim_reducer = cuml.PCA(n_components=n_components_max)               # Perform PCA
            dim_reducer.fit(f_sub_scaled)                                       # Fit PCA to the scaled data
            expl_var = dim_reducer.explained_variance_ratio_                    # Get the proportion of variance explained by each component
            n_components = np.min(np.where(expl_var < 0.01))                    # Determine the number of components that explain at least 1% variance each
            
            dim_reducer = cuml.PCA(n_components=n_components)                   # then define the dimensionality reduction based on this number of components

        elif params.dim_reducer == 'UMAP':                                      # If UMAP is selected as the dimensionality reduction method
            # define n_neighbours as a Michalis-Menten type function from the number of points
            def umap_n_neighbours(x):
                n_min = 10
                n_max = 200
                n = np.round(n_min + n_max*x / (500000+x))
                return(n)

            import cuml                                                         # Import RAPIDS cuML for GPU-accelerated PCA
            dim_reducer = cuml.UMAP(                                            # Initialize UMAP with a maximum of 4 components and number neighbours as Michalis-Menten type function
                n_neighbors=umap_n_neighbours(f_sub.shape[0]),
                n_components=4
            )

        else:                                                                   # Dimensionality reduction method not implemented
            print('Unknown dimensionality reducer; crashing')                   # Display a message when the dimensionality reducer is missing or not available

        log.info('	fit dimensionality reducer')                                # Log that the dimensionality reducer is being fitted                               
        dim_reducer.fit(f_sub_scaled)

        log.info('	reduce dimension of features')                              # Log that the feature matrix will now be reduced in dimensionality
        if params.features =='uvplib':
            f_sub = subsample_features(f_all, params[step_params], log)
            f_sub = f_sub[vars_kept]
            f_sub_scaled= scaler.transform(yeo.transform(f_sub))
            f_sub_scaled = np.nan_to_num(f_sub_scaled, copy=False)                  # since we have scaled the data, we can simply replace missing values by 0
 
        chunks = np.array_split(f_sub_scaled, 10)        
        f_sub_reduced = np.vstack([dim_reducer.transform(chunk) for chunk in chunks])
        
        #f_sub_scaled = np.vsplit(f_sub_scaled, 10)                              # Split in chunks to apply the transformation (avoid memory errors on the GPU)
        #f_sub_reduced = [dim_reducer.transform(chunk) for chunk in f_sub_scaled]# Apply the dimensionality reduction to each chunk
        #f_sub_reduced = np.vstack(f_sub_reduced)                                # Stack the reduced chunks back into a single array

        log.info('	write to disk')                                             # Log that the reduced features will be saved to disk
        output = {'transformer': yeo, 'scaler': scaler, 'dim_reducer': dim_reducer, 'features_reduced': f_sub_reduced, 'features_names': f_sub.columns}
        with open(outfile, 'wb') as f:
            pkl.dump(output, f)                                                 # Save the scaler, dimensionality reducer, and reduced features to a pickle file

        rmm.reinitialize()                                                      # Clean GPU memory (RAPIDS memory manager) to free resources
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
    # Create the full path to the output pickle file based on the current parameters
    outfile = os.path.expanduser(
        f'~/datasets/morphopart/out_test/clust__{params.instrument}_{params.features}_{params.n_obj_max}_{params.n_obj_sub}_{params.replicate}_{params.dim_reducer}_{params.clust_method}_{params.n_clusters_tot}.pickle'
    )

    if os.path.exists(outfile):                                             # Check if the file already exists
        log.info('	load cluster info')                                     # Log that the clustering approach has already been done
        with open(outfile, 'rb') as f:
            output = pkl.load(f)                                            # Load clusterer, centroïds & clusters

    else :
        output = {}
        if params.clust_method=='Kmean_seq':
            
            log.info('	Clusterer via the sequential K-means approach')    # Log that Kmean_seq is starting
            import cuml                                                                 # Import RAPIDS cuML for GPU-accelerated Kmean
            for n_clusters in range(2, params.n_clusters_tot + 1):                      # Loop over cluster numbers from 1 to n_clusters_tot
                clust = cuml.KMeans(n_clusters=n_clusters,                              # Initialize Kmean with a setting the total number of clusters
                               init='scalable-k-means++', n_init=10,                    # Use scalable KMeans++ initialization and the algorithm will be run 10 times with different centroid seeds
                               random_state=params.replicate)                           # Ensure reproducible clustering for each replicate

                clust.fit(f_sub_reduced)                                                # Fit clustering to the reduced data

                #log.info('	define cluster centroids')                                  # Log that cluster centroids are being extracted
                centroids = clust.cluster_centers_                                      # Retrieve the coordinates of the cluster centroids from the fitted KMeans model

                #log.info('	compute cluster membership')                                # Log that cluster assignments for each data point will be computed
                clusters = clust.predict(f_sub_reduced)                                 # Predict the cluster label for each reduced feature vector
                
                output[n_clusters] = {'n_clusters': n_clusters, 'clusterer': clust, 'centroids': centroids, 'clusters': clusters} # Store all outpouts
                    
        elif params.clust_method=='Kmean_hclust':
            
            log.info('	Clusterer via the K_mean + hierarchical approach')
            import cuml                                                         # Import RAPIDS cuML for GPU-accelerated Kmean
            clust = cuml.KMeans(n_clusters=params.n_clusters_tot,               # Initialize Kmean with a setting the total number of clusters
                           init='scalable-k-means++', n_init=10,                # Use scalable KMeans++ initialization and the algorithm will be run 10 times with different centroid seeds
                           random_state=params.replicate)                       # Ensure reproducible clustering for each replicate

            clust.fit(f_sub_reduced)                                            # Fit clustering to the reduced data

            #log.info('	define cluster centroids')                              # Log that cluster centroids are being extracted
            centroids = clust.cluster_centers_                                  # Retrieve the coordinates of the cluster centroids from the fitted KMeans model

            #log.info('	compute cluster membership')                            # Log that cluster assignments for each data point will be computed
            clusters = clust.predict(f_sub_reduced)                             # Predict the cluster label for each reduced feature vector
            
            output = {'clusterer': clust, 'centroids': centroids, 'clusters': clusters} # Store all outpouts
        
        elif params.clust_method=='Kmean_bisecting':
            
            log.info('	Clusterer via the the bissecting k-means approach')    # Log that Bisecting_Kmean is starting
            clust = BisectingKMeansTree(                                                    # Initialize the BisectingKMeansTree clusterer
                    n_clusters=params.n_clusters_tot,                                       # Current number of clusters
                    n_init=10,                                                              # Number of centroid seeds
                    random_state=params.replicate)                                          # Ensure reproducible clustering for each replicate
    
            clust.fit(dimred['features_reduced'])                                       # Fit clustering on the reduced data
    
            #log.info('    Computing cluster membership')                               # Log that cluster assignments for each data point will be computed
            clusters = clust.predict(dimred['features_reduced'])                        # Predict the cluster label for each reduced feature vector
            
            #log.info('    Extracting cluster centroids')                               # Log that cluster centroids are being extracted
            centroids = clust._centers_per_step                                          # Retrieve the coordinates of the cluster centroids from the fitted KMeans model
                
            output = {'clusterer': clust, 'centroids': centroids, 'clusters': clusters}  # Store all outpouts
                                                        
        else :
            print ("algo not included")
        
        # Save results to a file with the number of clusters in the filename
        with open(outfile, 'wb') as f:
            pkl.dump(output, f)                                             # Save the KMeans model, centroids, and cluster assignments to a pickle file
            
        rmm.reinitialize()                                                       # Clean GPU memory (RAPIDS memory manager) to free resources
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
    # Create the full path to the output pickle file based on the current parameters
    outfile = os.path.expanduser(
        f'~/datasets/morphopart/out_test/tree__{params.instrument}_{params.features}_{params.n_obj_max}_{params.n_obj_sub}_{params.replicate}_{params.dim_reducer}_{params.clust_method}_{params.n_clusters_tot}_{params.linkage}.pickle'
    )

    if os.path.exists(outfile):                                 # Check if the file already exists
        log.info('	load tree')                                 # Log that the hierarchical classification approach has already been done
        with open(outfile, 'rb') as f:
            tree = pkl.load(f)                                  # Load hierarchical classification 

    else :
        log.info('	define tree of centroids')                  # Log that the hierarchical tree of centroids is being computed

        from sklearn.cluster import AgglomerativeClustering
        n = centroids.shape[0]                                  # Number of centroids from the previous KMeans clustering. NB: should be params.n_clusters_tot, but we may as well drop this dependency
        tree = np.zeros([n,n]).astype(int)                      # Initialize an empty array to store hierarchical cluster labels for each centroid
        for i in range(0,n):                                    # Build a hierarchical clustering tree by iteratively clustering centroids
            hclust = AgglomerativeClustering(n_clusters=i+1, linkage=params.linkage)        # Perform agglomerative clustering with i+1 clusters
            clusters = hclust.fit_predict(centroids)            # Assign each centroid to a cluster
            tree[:,i] = clusters                                # Store cluster labels for a number of clusters
        tree = pd.DataFrame(tree)                               # Convert the tree to a pandas DataFrame for easier handling
        tree.columns = np.arange(1,n+1)                         # Columns correspond to the number of clusters
        
        log.info('	write to disk')                             # Log that the hierarchical tree will be saved
        with open(outfile, 'wb') as f:
            pkl.dump(tree, f)                                   # Save the hierarchical tree to a pickle file

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
    # Create the full path to the output pickle file based on the current parameters
    outfile = os.path.expanduser(
        f'~/datasets/morphopart/out_test/features_all_reduced__{params.instrument}_{params.features}_{params.n_obj_max}_{params.n_obj_sub}_{params.replicate}_{params.dim_reducer}.pickle'
    )

    if os.path.exists(outfile):                                             # Check if the file already exists
        log.info('	load full data predicted based on current subsample')   # Log that the dimensionality reduction has already been done on the full data based on the subsample
        with open(outfile, 'rb') as f:
            f_all_reduced = pkl.load(f)                                     # Load full data, reduced features based on the subsample
    else :
        if (params['n_obj_sub'] == params['n_obj_max'] and params['features'] != 'uvplib'):                      # Similarly, when n_obj_sub = n_obj_max, we would do the same thing twice
            log.info('	read already reduced features')                     # Log that the dimensionality reduction has already been done on the full data
            f_all_reduced = dimred['features_reduced']                      # Store the full data, reduced features based on the subsample in 'f_all_reduced'
        else:
            log.info('	reduce all features based on current subsample')
            
            if params.features =='uvplib':                                          # Clean dataset before transformer and scaler for dimensional reduction
                f_all=f_all[dimred['features_names']]
                
            from sklearn.preprocessing import PowerTransformer, StandardScaler
        
            f_yeo = dimred['transformer'].transform(f_all)                        # Perform Yeo-Johnson transformation to normalize            
            f_all_scaled = dimred['scaler'].transform(f_yeo)                        # Transform the data to have zero mean and unit variance 
            f_all_scaled = np.nan_to_num(f_all_scaled, copy=False)                  # since we have scaled the data, we can simply replace missing values by 0

            f_all_scaled = np.vsplit(f_all_scaled, 10)                              # Split in chunks to apply the transformation (avoid memory errors on the GPU)
            f_all_reduced = [dimred['dim_reducer'].transform(chunk) for chunk in f_all_scaled] # Apply the dimensionality reduction to each chunk
            f_all_reduced = np.vstack(f_all_reduced)                                # Stack the reduced chunks back into a single array

        log.info('	write to disk')                                         # Log that the reduced features will be saved to disk
        with open(outfile, 'wb') as f:
            pkl.dump(f_all_reduced, f)                                      # Save the full reduced features based on the subsample to a pickle file

        rmm.reinitialize()                                                  # Clean GPU memory (RAPIDS memory manager) to free resources

    return(f_all_reduced)

def evaluate(f_all, f_all_reduced, clust, tree, f_all_reduced_ref, clusters_ref, tree_ref, params, log):
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
    # Create the full path to the output pickle file based on the current parameters
    outfile = os.path.expanduser(
        f'~/datasets/morphopart/out_test/eval__{params.instrument}_{params.features}_{params.n_obj_max}_{params.n_obj_sub}_{params.replicate}_{params.dim_reducer}_{params.clust_method}_{params.n_clusters_tot}_{params.linkage}_{params.n_clusters_eval}_{params.n_obj_eval}.csv'
    )
    if os.path.exists(outfile):                                                                     # Check if the file already exists
         log.info('    load evaluation results')                                                    # Log that the evaluation has already been done
         with open(outfile, 'rb') as f:                                                             # Open the csv file in read-binary mode
             results = pd.read_csv(f)                                                               # Read the evaluation into 'results'
         log.info('	done')                                                                          # Log that the evaluation is now read
    else :                                                                                          # If the file doesn't exists
        log.info('	predict cluster number of all objects in the reduced features space')           # Log the initiation of cluster assignment for all objects
        # make a DataFrame with cluster level as column index
        # NB: internally, this is likely using a nearest neighbour classifier
        df = pd.read_csv( f'~/datasets/morphopart/{params.instrument}/taxa.csv.gz', usecols = ['objid','taxon']) # Load the taxonomy data (object IDs and taxon labels) from a compressed CSV file
        df= df[np.isin(df["objid"].values, f_all.index.values)]                                                  # Filter the taxonomy data to include only objects present in all features dataset
        df=df.set_index(df["objid"])                                                                             # Set 'objid' as the index to align with f_all
        df=df.reindex(f_all.index)                                                                               # Reindex to ensure the same order as f_all for consistent alignment    
        
        if params.clust_method == 'Kmean_hclust':
            c_all = pd.DataFrame({params.n_clusters_tot: clust['clusterer'].predict(f_all_reduced)})                 # Predict cluster assignments for all reduced feature vectors
            c_all["taxon"]=df["taxon"].values                                                                        # Add the corresponding taxon labels to the cluster assignment DataFrame
            
            if params.n_clusters_eval != params.n_clusters_tot:                                                      # If the evaluation cluster count differs from the total (applies to KMeans–hierarchical approach)
                # Reduce the clustering to the desired number of clusters for evaluation (i.e., merge the hierarchical clustering tree to reach the specified level)
                log.info('	reduce to the target number of clusters')
            
                # Map each object’s total-cluster assignment to its corresponding evaluated-cluster label using the hierarchical tree structure
                #c_all = fast_merge(c_all, tree[[params.n_clusters_tot, params.n_clusters_eval]], on=params.n_clusters_tot) # Note: commented out because using fast_merge here reorders clusters by their labels, not by the original object IDs. This breaks the correspondence with the original data order.
                c_all = c_all.merge(tree[[params.n_clusters_tot, params.n_clusters_eval]], left_on=params.n_clusters_tot, right_on=params.n_clusters_tot, how="left")
            
            log.info('	compute metrics score')                                                                         # Log that metrics (e.g. Adjusted Rand Index, SIL, DIST, DBCV) computation is starting
            c_all_ref = pd.DataFrame({params.n_clusters_tot: clusters_ref['clusters']})                                 # Define the reference cluster assignments at the total cluster level

            if params.n_clusters_eval != params.n_clusters_tot:                                                         # If evaluating at a different cluster level, map the reference clusters to the desired number of clusters using the hierarchical tree
                #c_all_ref = fast_merge(c_all_ref, tree_ref[[params.n_clusters_tot, params.n_clusters_eval]], on=params.n_clusters_tot) # Note: commented out because using fast_merge here reorders clusters by their labels, not by the original object IDs. This breaks the correspondence with the original data order.
                c_all_ref = c_all_ref.merge(tree_ref[[params.n_clusters_tot, params.n_clusters_eval]], left_on=params.n_clusters_tot, right_on=params.n_clusters_tot, how="left")
        
            del tree_ref                                                                                                # Clean up temporary DataFrames to free memory
        
        elif params.clust_method == 'Kmean_seq':
            c_all = pd.DataFrame({params.n_clusters_eval: clust[params.n_clusters_eval]['clusterer'].predict(f_all_reduced)})                 # Predict cluster assignments for all reduced feature vectors
            c_all["taxon"]=df["taxon"].values                                                                                          # Add the corresponding taxon labels to the cluster assignment DataFrame
            
            c_all_ref = pd.DataFrame({params.n_clusters_eval: clusters_ref[params.n_clusters_eval]['clusters']})                              # Define the reference cluster assignments at the total cluster level
        
        elif params.clust_method == 'Kmean_bisecting':
            c_all = pd.DataFrame({params.n_clusters_eval: clust['clusterer'].predict(f_all_reduced)[:,params.n_clusters_eval-1]})      # Predict cluster assignments for all reduced feature vectors
            c_all["taxon"]=df["taxon"].values                                                                                          # Add the corresponding taxon labels to the cluster assignment DataFrame
            
            c_all_ref = pd.DataFrame({params.n_clusters_eval: clusters_ref['clusters'][:,params.n_clusters_eval-1]})                   # Define the reference cluster assignments at the total cluster level
        
        else :
            print('Error all object in the reduced space cannot be predict')
        
        # --------------------------------------------------------------------
        # Compute ARI (Adjusted Rand Index)                                    # Range: -1 to 1 -> 1: Perfect agreement — the two clusterings are identical; 0: Random clustering — the agreement is what you’d expect by chance; Negative: Worse than random — clusters are anti-correlated with true labels.
        # --------------------------------------------------------------------
        log.info('	compute ARI score')                                                                                      # Log that Adjusted Rand Index computation is starting
        from sklearn.metrics.cluster import adjusted_rand_score                                                             # Import the ARI metric from scikit-learn --> CPU
        score_ARI = adjusted_rand_score(c_all_ref[params.n_clusters_eval].values, c_all[params.n_clusters_eval].values)     # Compute the Adjusted Rand Index between reference and predicted clusters
    
        # NOTE: The cuML ARI function is commented out here because on large datasets, it can produce incorrect or extremely large/small ARI values due to GPU float32 precision and overflow issues. Sometimes ARI_score reached -495 ...
        #from cuml.metrics.cluster.adjusted_rand_index import adjusted_rand_score                                             # Import the GPU-accelerated Adjusted Rand Index (ARI) function from cuML
        #score_ARI = adjusted_rand_score(c_all_ref[params.n_clusters_eval].values, c_all[params.n_clusters_eval].values)      # Compute the ARI score between reference and predicted clusters. Note: ARI can occasionally return negative values if the clustering is worse than random
        
        # --------------------------------------------------------------------
        # Compute NMI (Normalized Mutual Information)                                    # Range: 0 to 1 -> 1: one clustering fully predicts the other; 0: knowing a point’s cluster in A tells you nothing at all about its cluster in B.
        # --------------------------------------------------------------------
        log.info('	compute NMI score')                                                                                               # Log that Adjusted Rand Index computation is starting
        from sklearn.metrics.cluster import normalized_mutual_info_score                                                              # Import the ARI metric from scikit-learn --> CPU
        score_NMI = normalized_mutual_info_score(c_all_ref[params.n_clusters_eval].values, c_all[params.n_clusters_eval].values)      # Compute the Adjusted Rand Index between reference and predicted clusters
        
        del c_all_ref                                                                                                         # Clean up temporary DataFrames to free memory
        # --------------------------------------------------------------------
        # Subsample the data for DBCV and Silhouette score computation
        # (computing these metrics on the full dataset is too time-consuming)
        # --------------------------------------------------------------------
        n_repetitions = 5                                                                                                   # Number of subsampling repetitions for stability
        
        # Previous approach (commented out) sampled stratified by cluster category:
        # eval_subsamples = [sample_stratified_by_category(
        #                      n=f_all_reduced.shape[0],                                                                    # Total number of points in dataset
        #                      size=params.n_obj_eval,                                                                      # Number of points to include in each subsample
        #                      by=c_all[params.n_clusters_eval].values,                                                     # Stratify by cluster category
        #                      random_state=i)                                                                              # Random seed for reproducibility
        #                      for i in range(n_repetitions)]                                                               # Repeat for the specified number of subsamples
        
        # Note: Another approach is to stratify using the first two dimensions of the reduced space,
#       but naive sampling may put all points in one subsample, causing DBCV to fail.
        
        # Current approach: stratified sampling in continuous 2D space (first two components)
        eval_subsamples = [sample_stratified_continuous(
                             n=f_all_reduced.shape[0],                                                                      # Total number of points in dataset
                             size=params.n_obj_eval,                                                                        # Number of points to include in each subsample
                             by=f_all_reduced[:,[0,1]],                                                                     # Stratify by first two reduced dimensions of reduced features
                             random_state=i)                                                                                # Random seed for reproducibility
                             for i in range(n_repetitions)]                                                                 # Repeat for the specified number of subsamples

        
        # --------------------------------------------------------------------
        # Compute DBCV (Density-Based Cluster Validation) for each subsample   # Range: -1 to 1 -> 1: Perfect density separation — clusters are very dense and clearly separated; 0: Clusters are no better than random points (densities overlap); Negative: Poor clustering — points in the same cluster are not denser than points in other clusters.
        # --------------------------------------------------------------------
        # NOTE: The cuML ARI function is commented out here because we excluded HDBSCAN in the pipeline 
        #log.info('    compute DBCV')                                                                                        # Log that Density-Based Cluster Validation computation is starting
        # import ipdb; ipdb.set_trace()
        #import hdbscan                                                                                                      # HDBSCAN library provides the validity_index function. Alternative implementation exists (e.g., https://github.com/FelSiq/DBCV), but slower.
        #DBCVs = [hdbscan.validity.validity_index(f_all_reduced[idx,:].astype('double'), labels=c_all[params.n_clusters_eval].values[idx], metric='euclidean') for idx in eval_subsamples] # Compute DBCV score for each subsample using cluster labels

        # --------------------------------------------------------------------
        # Compute Silhouette score for each subsample                          # Range: -1 to 1 -> 1: clusters are well-separated and tight; 0: clusters overlap or are not clearly separated; Negative: clusters are poorly assigned; points are closer to other clusters
        # --------------------------------------------------------------------
        log.info('	compute Silhouette score')                                                                              # Log that Silhouette score computation is starting
        # import ipdb; ipdb.set_trace()
        
        def safe_silhouette_score(X, labels):                                                                               # Define a safe Silhouette scoring function to handle single-cluster edge cases
            """
            Compute Silhouette score safely: returns NaN if all points belong to the same cluster.
            Uses cuML's Cython-accelerated implementation for speed.
            """
            from cuml.metrics.cluster.silhouette_score import cython_silhouette_score                                       # Import cuML's GPU-accelerated Silhouette score function for fast clustering evaluation
            import numpy as np
            if len(np.unique(labels))==1:                                                                                   # Check if all points are in the same cluster
                score = float('NaN')                                                                                        # Single cluster: Silhouette undefined
            else:
                score = cython_silhouette_score(X, labels)                                                                  # Compute Silhouette score with cuML
            return(score)
        
        SILs = [safe_silhouette_score(f_all_reduced[idx,:].astype('double'), labels=c_all[params.n_clusters_eval].values[idx]) for idx in eval_subsamples] # Compute Silhouette score for each subsample using the safe function

        # --------------------------------------------------------------------
        # Compute Pairwise distances in reduced space
        # --------------------------------------------------------------------
        log.info('	compute pairwise distances in reduced space')                                                           # Log that pairwise distance computation in the reduced feature space is starting
        # PCA that produces the _ref one is fitted on the full dataset while the other is fitted on the subset. They may not have the same number of components.
        # The number of components is fixed to the lower number components between the full dataset and the subset.
        n_axis_ref=np.shape(f_all_reduced_ref)[1]                                                                           # Number of components in the reference reduced space
        n_axis=np.shape(f_all_reduced)[1]                                                                                   # Number of components in the current reduced space
        print(n_axis_ref, n_axis)                                                                                           # Print the number of components for debugging
        if n_axis_ref != n_axis:                                                                                            # If the number of components differs,
            fix_axis=min(n_axis_ref,n_axis)                                                                                 # Take the lower number of components
            DISTs = np.linalg.norm(f_all_reduced_ref[:,0:fix_axis] - f_all_reduced[:,0:fix_axis], axis=0)                   # Compute the Euclidean distance between corresponding components along each axis
        else:                                                                                                               # If the number of components is the same, 
            DISTs = np.linalg.norm(f_all_reduced_ref - f_all_reduced, axis=0)                                               # compute distance across all axes

        # --------------------------------------------------------------------
        # TODO Compute Purity of labels                                         # Range: 0 to 1 -> 1: each cluster contains points from only a single label (perfect match); 0: clusters are completely mixed 
        # --------------------------------------------------------------------
        log.info('	compute purity of labels')                                                                              # Log that computation of purity for taxonomic labels is starting
        def compute_purity(df, cluster_col='cluster', label_col='taxon'):                                                   # Define a function to compute the purity of clustering with respect to known labels
            total_points = len(df)                                                                                          # Total number of points in the dataset
            purity_sum = 0                                                                                                  # Initialize sum of majority counts across clusters

            for cluster_id in df[cluster_col].unique():                                                                     # Loop over each cluster
                cluster_points = df[df[cluster_col] == cluster_id]                                                          # Extract all points belonging to this cluster
                most_common_count = cluster_points[label_col].value_counts().max()                                          # Count occurrences of each label in the cluster and take the maximum (majority label count)
                purity_sum += most_common_count                                                                             # Add to the weighted sum

            purity = purity_sum / total_points                                                                              # Divide by total number of points to get overall purity
            return purity
        
        purity_score = compute_purity(c_all, cluster_col=params.n_clusters_eval, label_col='taxon')                         # Compute the purity score
        
        # --------------------------------------------------------------------
        # TODO Compute ecological index (Simspons/Shannon)                      Shannon index: Higher → more diverse; Simpson index: Higher → less dominance; 
        # --------------------------------------------------------------------
        log.info('	compute purity of taxonomie')                                                           # Log that purity of taxonomie computation is starting
        def diversity_index(list_input):                                                                    # Define a function to calculate diversity indices (Shannon and Simpson) for a list of items
            """
            Calculate Shannon and Simpson diversity indices for a given list.
            Shannon index: measures entropy (uncertainty) in the distribution
            Simpson index: measures probability that two randomly chosen items belong to the same category
            """
            import math
            unique_base = set(list_input)                                                                   # Get the set of unique categories (taxa)
            M   =  len(list_input)                                                                          # Total number of items
            entropy_list = []                                                                               # List to store individual Shannon contributions
            P_i_list = []                                                                                   # List to store squared probabilities for Simpson index
            for base in unique_base:
                n_i = list_input.count(base)                                                                # Count occurrences of this category
                P_i = n_i/float(M)                                                                          # Probability of this category
                entropy_i = P_i*(math.log(P_i))                                                             # Contribution to Shannon entropy
                entropy_list.append(entropy_i)
                P_i_list.append(P_i**2)                                                                     # Contribution to Simpson index
            sh_entropy = -(sum(entropy_list))                                                               # Shannon entropy (sum of contributions, negated)
            Si_index= 1/(sum(P_i_list))                                                                     # Simpson diversity index
            return (sh_entropy, Si_index)                                                                   # Return both diversity metrics
            
        clust_diversity_indices=[[cluster_id, diversity_index(list(c_all[c_all[params.n_clusters_eval]==cluster_id]["taxon"]))] for cluster_id in np.unique(c_all[params.n_clusters_eval].values)]  # Compute diversity indices for each cluster
        
        # --------------------------------------------------------------------
        # Compute Compacity index                                                # Range 0 to +∞ -> 0: clusters are tight and compact; high values: clusters are spread out / loose
        # --------------------------------------------------------------------
        log.info('	compacity of clusters')                                                                 # Log that compacity of clusters computation is starting
        def compute_compacity(features, clusters):                                                          # Define a function to compute the compacity (tightness) of clusters
            """
            Compute cluster compacity for a clustering.
    
            features: numpy array (n_samples x n_features)
            clusters: array-like of cluster assignments
            """
            clusters = np.array(clusters)                                                                   # Ensure cluster assignments are a NumPy array
            unique_clusters = np.unique(clusters)                                                           # Find all unique cluster IDs
            total_points = features.shape[0]                                                                # Total number of data points
    
            compacity_sum = 0.0                                                                             # Initialize weighted sum of cluster compacities

            for cluster_id in unique_clusters:                                                              # Loop over each cluster to compute its individual compacity
                idx = np.where(clusters == cluster_id)[0]                                                   # indices of points in the cluster
                cluster_points = features[idx, :]                                                           # features of cluster points
                centroid = np.mean(cluster_points, axis=0)                                                  # cluster centroid
                distances = np.linalg.norm(cluster_points - centroid, axis=1)                               # Euclidean distances
                cluster_compacity = np.mean(distances ** 2)                                                 # mean squared distance
                compacity_sum += len(idx) * cluster_compacity                                               # weighted sum

            overall_compacity = compacity_sum / total_points                                                # Overall compacity across all clusters
            return overall_compacity                                                                        # Return the final score

        compacity_score = compute_compacity(f_all_reduced, c_all[params.n_clusters_eval].values)            # Compute the overall compacity of clusters in the reduced feature space
        
        # --------------------------------------------------------------------
        # Save all metrics
        # --------------------------------------------------------------------
        log.info('	write to disk')                                                                         # Log that the metrics will be saved to disk
        if sum(np.isnan(SILs))>0:                                                                           # Check if any Silhouette scores are NaN
            results = dict(params) | {                                                                      # Start with parameters dictionary and merge evaluation metrics
                'ARI': score_ARI,                                                                           # Adjusted Rand Index
                'NMI': score_NMI,                                                                           # Normalized Mutual Information
                'n_obj_eval_actual': len(eval_subsamples[0]),                                               # Record the actual number of objects in the evaluation subsample. Note: Some subsamples may be smaller than params.n_obj_eval if clusters are small
                #'DBCV': np.mean(DBCVs), 'sdDBCV': np.std(DBCVs),                                            # DBCV metrics
                'SIL': np.mean([~np.isnan(SILs)]), 'sdSIL': np.std([~np.isnan(SILs)]), 'nanSIL': sum(np.isnan(SILs)), # Silhouette score metrics
                'DIST': np.mean(DISTs), 'sdDIST': np.std(DISTs),                                            # Pairwise distance metrics
                'Purity': purity_score,                                                                     # Purity of labels score
                'Compacity': compacity_score                                                                # Compacity index
            }
        else:
            results = dict(params) | {                                                                      # Start with parameters dictionary and merge evaluation metrics
                'ARI': score_ARI,                                                                           # Adjusted Rand Index
                'NMI': score_NMI,                                                                           # Normalized Mutual Information
                'n_obj_eval_actual': len(eval_subsamples[0]),                                               # Record the actual number of objects in the evaluation subsample. Note: Some subsamples may be smaller than params.n_obj_eval if clusters are small
                #'DBCV': np.mean(DBCVs), 'sdDBCV': np.std(DBCVs),
                'SIL': np.mean(SILs), 'sdSIL': np.std(SILs), 'nanSIL': sum(np.isnan(SILs)),                 # Silhouette score metrics
                'DIST': np.mean(DISTs), 'sdDIST': np.std(DISTs),                                            # Pairwise distance metrics
                'Purity': purity_score,                                                                     # Purity of labels score
                'Compacity': compacity_score                                                                # Compacity index 
            }

        results = pd.DataFrame(results, index=[0])                                                          # Convert results dictionary into a single-row pandas DataFrame
        results.to_csv(outfile, index=False)                                                                # Save results to CSV file
    
    rmm.reinitialize()                                                                                      # Clean GPU memory (RAPIDS memory manager) to free resources

    return(results)

#-------- Function for extracting features from raw images --------------------#
# Zooprocess features
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

# Deep features
# training
def training_model_mobilenet(directory, params, log):
    
       from deep import tensorflow_tricks                           # settings for tensorflow to behave nicely

       import pandas as pd
       # pd.set_option('display.max_columns', None)
       import numpy as np
       import tensorflow as tf
       from sklearn import metrics

       from importlib import reload
       from deep import dataset                                     # custom data generator
       from deep import cnn                                         # custom functions for CNN generation
       dataset = reload(dataset)
       cnn = reload(cnn)
       
       ########################## set_cnn_option ##########################################
       uvp_type = params.instrument
       data_dir=directory+'/'+params.instrument
       # image format
       if params.instrument=="uvp6":
           img_format = 'png'
       else:
           img_format='jpg'
       
       ## Data generator (see dataset.EcoTaxaGenerator)
       batch_size = 256  # increase until GPU memory is saturated
       augment = True
       upscale = True
       bottom_crop = 31
       fe_url = 'https://tfhub.dev/google/imagenet/mobilenet_v2_035_128/feature_vector/5'
       input_shape = (128, 128, 3)
       # input_shape = (224, 224, 3)
       fe_trainable = True
       fc_layers_sizes = [384]
       fc_layers_dropout = 0.4
       classif_layer_dropout = 0.2

       ## CNN training (see cnn.Train)
       use_class_weight = True
       weight_sensitivity = 0.5  # 0.5 = sqrt
       lr_method = 'decay'
       initial_lr = 0.0005
       decay_rate = 0.97
       loss = 'cce'
       epochs = 20 # or 20
       log_frequency = 2   # how many times to log per epoch
       workers = 10
       
       # directory to save training checkpoints
       cnn_dir = directory +'/cnn_mobilenet_v2_035_128_5_384/'+uvp_type
       ckpt_dir = cnn_dir + '/checkpoints'

       # create checkpoints dir if it does not exist
       os.makedirs('cnn_mobilenet_v2_035_128_5_384/'+uvp_type, exist_ok=True)
       os.makedirs('cnn_mobilenet_v2_035_128_5_384/'+uvp_type+'/checkpoints', exist_ok=True)
       ######################################################################################
       
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

# Define feature_extractor
def mobilenet_feature_extractor(directory, params, log):
    import matplotlib.pyplot as plt # science packages
    import tensorflow as tf
    from importlib import reload
    from deep import dataset            # custom data generator
    from deep import cnn                # custom functions for CNN generation
    import tf_keras as keras
    dataset = reload(dataset)
    cnn = reload(cnn)
    
    ########################## set_cnn_option ##########################################
    uvp_type = params.instrument
    data_dir=directory+'/'+params.instrument
    
    # directory to save training checkpoints
    cnn_dir = directory +'/cnn_mobilenet_v2_035_128_5_384/'+uvp_type
    ckpt_dir = cnn_dir + '/checkpoints'

    # create checkpoints dir if it does not exist
    os.makedirs('cnn_mobilenet_v2_035_128_5_384/'+uvp_type, exist_ok=True)
    os.makedirs('cnn_mobilenet_v2_035_128_5_384/'+uvp_type+'/checkpoints', exist_ok=True)
    ######################################################################################
    
    outfile = os.path.expanduser(ckpt_dir + '/training_log.tsv')
    if os.path.exists(outfile):
        print('Model and feature extractor already exist') ## ---- 
        df = pd.read_csv(ckpt_dir + '/training_log.tsv', sep='\t')
    else:
        training_model_mobilenet(directory, params, log)
        # Lis le log de l'entrainement et fais un plot. Tu veux que la val_loss et val_accuracy saturent
        df = pd.read_csv(ckpt_dir + '/training_log.tsv', sep='\t')
    df = df.drop(['batch', 'learning_rate'], axis='columns')
    
    output_path = os.path.join(os.path.expanduser(ckpt_dir), f'step.png')
    df.plot(x='step', subplots=True)
    plt.savefig(output_path)
    plt.close()
    
    output_path = os.path.join(os.path.expanduser(ckpt_dir), f'epoch.png')
    df.plot(x='epoch', subplots=True)
    plt.savefig(output_path)
    
    # define best_epoch
    # Il faut choisir l'epoch de val_loss minimale et val_accuracy maximale.
    print(df)
    
    best_epoch = input("Enter the best epoch (use None to get the latest epoch): ")
    # Create Model and features extraction
    print('Create model and feature extractor') ## ----    
    # load model for best epoch
    if best_epoch=='' or best_epoch=='None':
        my_cnn,epoch = cnn.Load('cnn_mobilenet_v2_035_128_5_384/'+params.instrument+'/checkpoints/')
    else:
        my_cnn,epoch = cnn.Load('cnn_mobilenet_v2_035_128_5_384/'+params.instrument+'/checkpoints/', epoch=int(best_epoch))
    print(' at epoch {:d}'.format(epoch))
    # save model (just in case)
    my_cnn.save('cnn_mobilenet_v2_035_128_5_384/'+params.instrument+'/best_model', include_optimizer=False)
    # drop the last two layers to get a feature extractor + the middle MLP layer
    my_fe = keras.models.Sequential([layer for layer in my_cnn.layers[0:-2] ])
    my_fe.summary()

    # save feature extractor (just in case)
    my_fe.save('cnn_mobilenet_v2_035_128_5_384/'+params.instrument+'/feature_extractor')

# extract deep features for raw images                                                                     
def get_mobilenet_features(directory, params, obj_id, log):
    import tensorflow as tf
    from deep import progress           # custom functions to track progress of training/prediction
    from deep import dataset            # custom data generator
    import pandas as pd
    import tf_keras as keras
    
    ########################## set_cnn_option ##########################################
    uvp_type = params.instrument
    data_dir=directory+'/'+params.instrument
    # image format
    if params.instrument=="uvp6":
        img_format = 'png'
    else:
        img_format='jpg'
    
    batch_size = 256  # increase until GPU memory is saturated
    bottom_crop = 31
    workers = 10
    
    # directory to save training checkpoints
    cnn_dir = directory +'/cnn_mobilenet_v2_035_128_5_384/'+uvp_type
    ckpt_dir = cnn_dir + '/checkpoints'

    # create checkpoints dir if it does not exist
    os.makedirs('cnn_mobilenet_v2_035_128_5_384/'+uvp_type, exist_ok=True)
    os.makedirs('cnn_mobilenet_v2_035_128_5_384/'+uvp_type+'/checkpoints', exist_ok=True)
    ######################################################################################
    outfile = os.path.expanduser('cnn_mobilenet_v2_035_128_5_384/'+params.instrument+'/feature_extractor')
    
    if os.path.exists(outfile):
        print('Load data and extract features') ## ----
        my_fe = keras.models.load_model('cnn_mobilenet_v2_035_128_5_384/'+params.instrument+'/feature_extractor', compile=False)
    else:
        my_fe = mobilenet_feature_extractor(directory, params, log) ######
        my_fe = keras.models.load_model('cnn_mobilenet_v2_035_128_5_384/'+params.instrument+'/feature_extractor', compile=False)   
    # get model input shape
    input_shape = my_fe.layers[0].input_shape
    # remove the None element at the start (which is where the batch size goes)
    input_shape = tuple(x for x in input_shape if x is not None)

    # TODO swap the comments in the next two lines for tests
    img_path = [data_dir + '/orig_imgs/' + str(objid) + '.' + img_format for objid in obj_id]
    #print('  found ' + str(df.shape[0]) + ' objects')

    batches = dataset.EcoTaxaGenerator(
        images_paths=img_path,
        input_shape=input_shape,
        # NB: although the labels are in the file, we don't use them here
        labels=None, classes=None,
        batch_size=batch_size, augment=False, shuffle=False,
        crop=[0,0,bottom_crop,0])

    # extract features by going through the batches
    features = my_fe.predict(batches, callbacks=[progress.TQDMPredictCallback()],
                                  max_queue_size=max(10, workers*2), workers=workers)
    f_all = pd.DataFrame(features, index=obj_id)
    f_all = f_all.rename(columns=str) # parquet need strings as column names
    
    return(f_all)


#------------------ Additional helper functions -------------------------------#
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


class BisectingKMeansTree(BisectingKMeans):                                                     # define a new class BisectingKMeansTree that inherits from BisectingKMeans
    import numpy as np

    from sklearn.cluster import BisectingKMeans
    from sklearn.utils.validation import check_is_fitted
    from sklearn.utils.extmath import row_norms
    from sklearn.cluster._kmeans import _labels_inertia_threadpool_limit
    
    # redefine the predict method to 
    # (1) allows an argument k = the maximum number of clusters to predict
    # (2) predict labels at all levels from 0 to k 
    def predict(self, X, k=None, sample_weight=None):
        # check arguments like the regular predict() method does
        check_is_fitted(self)
        X = self._check_test_data(X)
        X = X - self._X_mean
        x_squared_norms = row_norms(X, squared=True)
        sample_weight = np.ones_like(x_squared_norms)
        
        # if k is not specified, extract everything (the default)
        if k is None:
            k = self._n_features_out
      
        # initialise empty labels
        labels = np.zeros((X.shape[0], k), dtype=np.int32)
        
        # initialise the nodes dict (with the root node)
        # it is defined as {label: node}
        nodes = {0: self._bisecting_tree}
        
        self._centers_per_step = []   # list of centers at each step
        # centers: only root center available
        self._centers_per_step.append(np.asarray([nodes[0].center], dtype=float))
        # TODO record tree

        for step in range(k-1):
          # get the cluster to cut next
          # = the one of maximum inertia/number of element
          scores = {k:v.score for k,v in nodes.items()}
          labmax = max(scores, key=scores.get)
          
          # define which label it corresponds too
          mask = (labels[:,step] == labmax)
          
          # "split" this cluster the K-Means way and define the two new labels
          # NB: actually, this just assigns labels based on the distance from
          #     the centers
          centers = np.vstack((nodes[labmax].left.center,
                               nodes[labmax].right.center))
          labs = _labels_inertia_threadpool_limit(
              X[mask,:],
              sample_weight[mask],
              centers,
              return_inertia=False,
          )
          # copy labels from the previous clustering step
          labels[:,step+1] = labels[:,step]
          # replace the labels of the splitted cluster
          # NB: we number them starting from step * k to avoid cluster label
          #     collisions across the clustering steps. We will relabel them
          #     from 0 to step afterwards
          label_offset = step * k
          labels[mask,step+1] = labs + label_offset
          
          # update the list of nodes
          # NB: 0 is the left cluster, 1 is the right cluster, by definition
          #     of the centers
          new_nodes = {0 + label_offset : nodes[labmax].left,
                       1 + label_offset : nodes[labmax].right}
          drop = nodes.pop(labmax)
          nodes.update(new_nodes)
          
          # collect centers in order of node keys for reproducibility
          ordered_keys = sorted(nodes.keys())
          step_centers = np.vstack([nodes[k].center for k in ordered_keys])
          self._centers_per_step.append(step_centers)
        
        def relabel(x):
            """Relabel clusters
            
            Rank unique values of x in increasing order and relabel them from
            0 to the total number of unique values.
            """
            lab = np.unique(x)
            dict = {l:i for i,l in enumerate(lab)}
            return [dict[y] for y in x]
        
        # apply the relabelling to each column of the labels = to each step
        labels = np.apply_along_axis(relabel, axis=0, arr=labels)

        return labels


class MaskExtremeByFeature(BaseEstimator, TransformerMixin):
    def __init__(self, dict_para, feature_names, iqr_factor=1.5):
        """
        dict_para : dict {feature: [low%, high%] or None}
        feature_names : liste des colonnes dans l'ordre de X
        iqr_factor : multiplicateur IQR pour features None (default 1.5)
        """
        self.dict_para = dict_para
        self.feature_names = feature_names
        self.iqr_factor = iqr_factor
        self.lower_ = None
        self.upper_ = None
        self.thresholds_ = None
        self.masked_percent_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n_features = X.shape[1]

        self.lower_ = np.full(n_features, -np.inf)
        self.upper_ = np.full(n_features, np.inf)
        self.masked_percent_ = np.zeros(n_features)

        for j, feat in enumerate(self.feature_names):
            params = self.dict_para.get(feat, None)

            if params is None:
                # Masquage IQR pour les features None
                Q1 = np.nanpercentile(X[:, j], 25)
                Q3 = np.nanpercentile(X[:, j], 75)
                IQR = Q3 - Q1
                self.lower_[j] = Q1 - self.iqr_factor * IQR
                self.upper_[j] = Q3 + self.iqr_factor * IQR
            else:
                low_p, high_p = params
                low_q  = low_p / 100
                high_q = high_p / 100
                self.lower_[j] = np.nanquantile(X[:, j], low_q)
                self.upper_[j] = np.nanquantile(X[:, j], 1 - high_q)

        # sauvegarde seuils dans dict pour usage externe
        self.thresholds_ = {feat: (self.lower_[j], self.upper_[j])
                            for j, feat in enumerate(self.feature_names)}
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X_masked = X.copy()
        mask_total = np.zeros_like(X_masked, dtype=bool)

        for j in range(X_masked.shape[1]):
            mask = (X_masked[:, j] < self.lower_[j]) | (X_masked[:, j] > self.upper_[j])
            X_masked[mask] = np.nan
            mask_total[:, j] = mask

        # calcul du pourcentage de valeurs masquées par feature
        self.masked_percent_ = mask_total.sum(axis=0) / X.shape[0] * 100

        return X_masked

    def transform_with_thresholds(self, X, thresholds=None):
        if thresholds is None:
            thresholds = self.thresholds_

        # Assurer que X est un DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
    
        X_masked = X.copy()
        mask_total = pd.DataFrame(False, index=X.index, columns=X.columns)

        for feat in self.feature_names:
            lower, upper = thresholds.get(feat, (-np.inf, np.inf))
            mask = (X_masked[feat] < lower) | (X_masked[feat] > upper)
            X_masked.loc[mask, feat] = np.nan
            mask_total.loc[mask, feat] = True

        self.masked_percent_ = mask_total.sum() / len(X) * 100

        return X_masked

def mask_nan(df, max_var_na=10, max_obj_na=5):
    """
    Remove variables and objects with too many NaNs.
    
    Parameters
    ----------
    df : pd.DataFrame
    max_var_na : float
        Maximum allowed % of NaNs per variable.
    max_obj_na : int
        Maximum allowed number of NaNs per object (row).
    
    Returns
    -------
    df_masked : pd.DataFrame
        Filtered DataFrame (NaNs preserved).
    selected_vars : pd.Index
        Kept variables.
    selected_objs : pd.Index
        Kept objects.
    """

    # 1. Variables (columns)
    mask_vars = (df.isna().mean(axis=0) * 100) < max_var_na
    selected_vars = df.columns[mask_vars]

    # 2. Objects (rows)
    mask_objs = df[selected_vars].isna().sum(axis=1) <= max_obj_na
    selected_objs = df.index[mask_objs]

    # 3. Subset
    df_masked = df.loc[selected_objs, selected_vars]

    return df_masked, selected_vars, selected_objs
