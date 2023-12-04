#!/usr/bin/env python
#
# Define all combinations of parameters for the pipeline
#

def expand_grid(data):
    import pandas as pd
    import itertools
    rows = itertools.product(*data.values())
    return(pd.DataFrame.from_records(rows, columns=data.keys()))

params = expand_grid({
    'instrument': ['uvp5sd','uvp6'],
    'features': ['mobilenet', 'uvplib', 'dino'],
    'n_obj_max': [2000000],
    'replicate': [1, 2, 3, 4, 5],
    'n_obj_sub': [10000, 50000, 100000, 250000, 500000, 1000000, 2000000],
    'dim_reducer': ['UMAP', 'PCA'],
    'n_clusters_tot': [200],
    'linkage': ['ward'],
    'n_clusters_eval': [5, 15, 100, 200],
    'n_obj_eval': [15000]
})

# # test
# params = expand_grid({
#     'instrument': ['uvp5sd'],
#     'features': ['mobilenet'],
#     'n_obj_max': [50000],
#     'replicate': [1],
#     'n_obj_sub': [10000],
#     'dim_reducer': ['UMAP'],
#     'n_clusters_tot': [200],
#     'linkage': ['ward'],
#     'n_clusters_eval': [5],
#     'n_obj_dbcv': [25000]
# })

# TODO possibly perform only one replicate in case n_obj_sub = n_obj_max? = remove the rows from the params_grid?

print(f'Defined {params.shape[0]} combinations of parameters')

params.to_csv('params_grid.csv', index=False)
