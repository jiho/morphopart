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
from itertools import product
import pandas as pd

grid_list = []
for method in ['Kmean_hclust', 'Kmean_seq', 'Kmean_bisecting']:
    if method == 'Kmean_hclust':
        combos = product(
            ['uvp5sd'],
            ['mobilenet', 'uvplib', 'dino'],
            [4844991],
            [1, 2, 3, 4, 5],
            [4844991, 3000000, 2000000, 1000000, 500000, 250000, 100000, 50000, 10000, 50000, 10000, 5000, 2500, 1000, 500],
            ['UMAP', 'PCA'],
            [method],
            [200],
            ['ward', 'average'],
            [5, 15, 100, 200],
            [100000]
        )
    else:
        combos = product(
            ['uvp5sd'],
            ['mobilenet', 'uvplib', 'dino'],
            [4844991],
            [1, 2, 3, 4, 5],
            [4844991, 3000000, 2000000, 1000000, 500000, 250000, 100000, 50000, 10000, 50000, 10000, 5000, 2500, 1000, 500],
            ['UMAP', 'PCA'],
            [method],
            [200],
            [np.nan],   # valeur par défaut
            [np.nan],        # valeur par défaut
            [np.nan]
        )
    for c in combos:
        grid_list.append(c)

# Convert the list of parameter combinations 'grid_list' into a pandas DataFrame and assign meaningful column names to each parameter
params_grid = pd.DataFrame(grid_list, columns=[
        'instrument', 'features', 'n_obj_max', 'replicate', 'n_obj_sub', 
        'dim_reducer', 'clust_method', 'n_clusters_tot', 'linkage', 
        'n_clusters_eval', 'n_obj_eval'
    ])

# Perform only one replicate in case n_obj_sub = n_obj_max.
params_grid = params_grid[
    (params_grid['n_obj_sub'] != params_grid['n_obj_max']) |
    ((params_grid['n_obj_sub'] == params_grid['n_obj_max']) & (params_grid['replicate'] == 1))
].reset_index(drop=True)

print(f'Defined {params.shape[0]} combinations of parameters')
params.to_csv('params_grid.csv', index=False)                   # Save the DataFrame 'params' to a CSV file named 'params_grid.csv'




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
    'n_clusters_tot': [200],
    'linkage': ['ward','average'],
    'n_clusters_eval': [5, 15, 100, 200],
    'n_obj_eval': [100000]
})