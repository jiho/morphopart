conda activate morphopart                                   # activate conda environment

#########--------------------------------Vérification GPU--------------------------------------
# Add Conda environment libraries to search path
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Set CUDA paths to the Conda environment
export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX

ipython                                                                      # launch ipython

import os, ctypes

ctypes.CDLL(os.path.join(os.environ["CONDA_PREFIX"], "lib", "libcudart.so")) # Load CUDA runtime library from the Conda environment
ctypes.CDLL(os.path.join(os.environ["CONDA_PREFIX"], "lib", "libcublas.so")) # Load cuBLAS (CUDA BLAS) library
ctypes.CDLL(os.path.join(os.environ["CONDA_PREFIX"], "lib", "libcudnn.so"))  # Load cuDNN (CUDA Deep Neural Network) library


import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))                                # List and print all detected GPU devices available to TensorFlow

#########-----------------Define all combinations of parameters for the pipeline---------------
    
def expand_grid(data):
    import pandas as pd
    import itertools
    rows = itertools.product(*data.values())                                # Create Cartesian product of all values in the input dictionary
    return(pd.DataFrame.from_records(rows, columns=data.keys()))            # Convert the combinations into a pandas DataFrame

params_grid = expand_grid({
    'instrument': ['uvp5sd'],
    'features': ['mobilenet', 'uvplib', 'dino'],
    'n_obj_max': [4844991],
    'replicate': [1, 2, 3, 4, 5],
    'n_obj_sub': [4844991, 3000000, 2000000, 1000000, 500000, 250000, 100000, 50000, 10000, 50000, 10000, 5000, 2500, 1000, 500],
    'dim_reducer': ['UMAP', 'PCA'],
    'clust_method': ['Kmean_hclust', 'Kmean_seq', 'Kmean_bisecting'],
    'n_clusters_tot': [200],
    'linkage': ['ward','average'],
    'n_clusters_eval': [5, 15, 100, 200],
    'n_obj_eval': [100000]
})

params_grid.loc[params_grid['clust_method'] != 'Kmean_hclust', ['linkage']] = 'NaN'        # Disable irrelevant params for non-hclust methods
params_grid = params_grid.drop_duplicates().reset_index(drop=True)                                           # Remove duplicates created by disabling parameters

# Perform only one replicate in case n_obj_sub = n_obj_max.
params_grid = params_grid[
    (params_grid['n_obj_sub'] != params_grid['n_obj_max']) |
    ((params_grid['n_obj_sub'] == params_grid['n_obj_max']) & (params_grid['replicate'] == 1))
].reset_index(drop=True)


print(f'Defined {params.shape[0]} combinations of parameters')
params.to_csv('params_grid.csv', index=False)                   # Save the DataFrame 'params' to a CSV file named 'params_grid.csv'













######------------------------------------------------------##############

def expand_grid(data):
    import pandas as pd
    import itertools
    rows = itertools.product(*data.values())                                # Create Cartesian product of all values in the input dictionary
    return(pd.DataFrame.from_records(rows, columns=data.keys()))            # Convert the combinations into a pandas DataFrame

params_grid = expand_grid({
    'instrument': ['uvp5sd'],
    'features': ['mobilenet'],
    'n_obj_max': [2000000],
    'replicate': [1],
    'n_obj_sub': [2000000, 1900000, 1700000, 1500000, 1000000, 500000, 250000, 100000, 50000, 10000, 50000, 10000, 5000, 2500, 1000, 500],
    'dim_reducer': ['PCA'],
    'clust_method': ['Kmean_hclust', 'Kmean_seq', 'Kmean_bisecting'],
    'n_clusters_tot': [200],
    'linkage': ['ward','average'],
    'n_clusters_eval': [5, 15, 100, 200],
    'n_obj_eval': [100000]
})




########------------------------------------------------##########@
params_grid = expand_grid({
    'instrument': ['uvp5sd'],
    'features': ['mobilenet'],
    'n_obj_max': [2000000],
    'replicate': [1],
    'n_obj_sub': [2000000, 1900000, 1700000, 1500000, 1000000, 500000, 250000, 100000, 50000, 10000, 50000, 10000, 5000, 2500, 1000, 500],
    'dim_reducer': ['PCA'],
    'n_clusters_tot': [200],
    'linkage': ['ward','average'],
    'n_clusters_eval': [5, 15, 100, 200],
    'n_obj_eval': [100000]
})




params_grid = expand_grid({
    'instrument': ['uvp5sd'],
    'features': ['mobilenet'],
    'n_obj_max': [20000],
    'replicate': [1],
    'n_obj_sub': [20000, 10000, 1000, 500],
    'dim_reducer': ['PCA'],
    'clust_method': ['Kmean_seq','Kmean_hclust','Kmean_bisecting'],
    'n_clusters_tot': [200],
    'linkage': ['ward','average'],
    'n_clusters_eval': [5, 15, 200],
    'n_obj_eval': [100000]
})
params_grid.loc[params_grid['clust_method'] != 'Kmean_hclust', ['linkage']] = 'NaN'        # Disable irrelevant params for non-hclust methods
params_grid = params_grid.drop_duplicates().reset_index(drop=True)                                           # Remove duplicates created by disabling parameters

# Perform only one replicate in case n_obj_sub = n_obj_max.
params_grid = params_grid[
    (params_grid['n_obj_sub'] != params_grid['n_obj_max']) |
    ((params_grid['n_obj_sub'] == params_grid['n_obj_max']) & (params_grid['replicate'] == 1))
].reset_index(drop=True)