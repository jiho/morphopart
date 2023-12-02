#!//usr/bin/env python
#
# Explore possible parameters for UVP particles clustering
#

import os                       # general packages
import pickle as pkl
import logging

import matplotlib.pyplot as plt # science packages
import numpy as np
import pandas as pd

from morphopart import *        # local package

# import ipdb                     # debugging


## Prepare output ----

# create output directory
os.makedirs(os.path.expanduser('~/datasets/morphopart/out'), exist_ok=True)

# log to a file and to the console
log_format = logging.Formatter('%(asctime)s	%(message)s')
# console
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_format)
console_handler.setLevel(logging.INFO)
# file
log_file_path = os.path.join('log.tsv')
file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(log_format)
file_handler.setLevel(logging.INFO)
# start the logger(s)
log = logging.getLogger('root')
log.setLevel(logging.INFO)
log.addHandler(console_handler)
log.addHandler(file_handler)



## Define parameters ----

# defined as a grid of all relevant parameters
# one row is one set of parameters for which there will be performance metrics

# read parameters
params_grid = pd.read_csv('params_grid.csv')
# we will go through each row, within a loop

# initialise a Series which will hold the values of the previous execution loop
# this will allow to skip some steps if we have the data in memory already
previous_params = params_grid.iloc[0].copy()
# NB: the copy() is required to not get just a view of the original data
previous_params[:] = np.nan

for i in range(params_grid.shape[0]):
    # pick one row to process
    params = params_grid.iloc[i]
    log.info(f'start	Start parameters set {i} : {params.to_dict()}')


    
    log.info('step 0	Read and prepare data') # ----

    step_params = ['instrument', 'features', 'n_obj_max']
    if all(params[step_params] == previous_params[step_params]):
        log.info('	skip: data already read')
    else:
        f_all = read_features(params[step_params], log)

    step_params = step_params + ['n_obj_sub', 'replicate']
    if all(params[step_params] == previous_params[step_params]):
        log.info('	skip: data already subsampled')
    else:
        f_sub = subsample_features(f_all, params[step_params], log)



    log.info('step 1	Reduce dimension') # ----

    step_params = step_params + ['dim_reducer']
    if all(params[step_params] == previous_params[step_params]):
        log.info('	skip: data already reduced')
    else:
        dimred = reduce_dimension(f_sub, params[step_params], log)



    log.info('step 2	Cluster to define morphs') # ----

    step_params = step_params + ['n_clusters_tot']
    if all(params[step_params] == previous_params[step_params]):
        log.info('	skip: data already clustered')
    else:
        clust = cluster(dimred['features_reduced'], params[step_params], log)

    # plt.scatter(dimred['features_reduced'][:,0], dimred['features_reduced'][:,1], s=0.5)
    # plt.scatter(clust['centroids'][:,0], clust['centroids'][:,1], s=2, c='red')



    log.info('step 3	Hierarchize clusters') # ----

    step_params = step_params + ['linkage']
    if all(params[step_params] == previous_params[step_params]):
        log.info('	skip: clusters tree already built')
    else:
        tree = hierarchize(clust['centroids'], params[step_params], log)

    # set params for next turn of the loop
    previous_params = params

    log.info('end	End')
