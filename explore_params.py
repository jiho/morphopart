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

    # fit dimensionality reduction on the current subsample
    step_params = step_params + ['dim_reducer']
    if all(params[step_params] == previous_params[step_params]):
        log.info('	skip: dimensionality reduction already fitted')
    else:
        dimred = reduce_dimension(f_sub, params[step_params], log)

    # use this dimensionality reduction model to reduce the full data
    # NB: we do this as a separate step because it is long so we want to save it
    if all(params[step_params] == previous_params[step_params]):
        log.info('	skip: full data already reduced')
    else:
        f_all_reduced = transform_features(f_all, dimred, params[step_params], log)



    log.info('step 2	Cluster to define morphs') # ----

    step_params = step_params + ['n_clusters_tot']
    if all(params[step_params] == previous_params[step_params]):
        log.info('	skip: clusters already fitted')
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



    log.info('step 4	Evaluate clusters') # ----

    if all(params[step_params] == previous_params[step_params]):
        log.info('	skip: objects already assigned')
    else:
        pred_all = transform_features(f_all, dimred, params[step_params], log)


    # file_params = ['instrument', 'features', 'n_obj_max', 'n_obj_max', 'replicate', 'dim_reducer']
    # if all(params[file_params] == previous_params[file_params]):
    #     log.info('    skip: reference dimensionality reduction already loaded')
    # else:
    #     pred_all = transform_predict(, params[step_params], log)
    #
    # log.info('    load reference dimensionality reduction and clusters')
    # # = clustering based on the *full* data
    # # TODO test if those are not allready loaded
    # # TODO use only replicate 1? = use 1 instead of params.replicate
    # file_params = ['instrument', 'features', 'n_obj_max', 'n_obj_max', 'replicate', 'dim_reducer']
    # dimred_ref_file = os.path.expanduser(
    #     f'~/datasets/morphopart/out/dimred__{params.instrument}_{params.features}_{params.n_obj_max}_{params.n_obj_max}_{params.replicate}_{params.dim_reducer}.pickle'
    # )
    # with open(dimred_ref_file, 'rb') as f:
    #     dimred_ref = pkl.load(f)
    # # get the features, for easier access
    # f_all_reduced_ref = dimred_ref['features_reduced']
    # f_all_reduced_ref.shape
    #
    # cluster_ref_file = os.path.expanduser(
    #     f'~/datasets/morphopart/out/clust__{params.instrument}_{params.features}_{params.n_obj_max}_{params.n_obj_max}_{params.replicate}_{params.dim_reducer}_{params.n_clusters_tot}.pickle'
    # )
    # with open(cluster_ref_file, 'rb') as f:
    #     cluster_ref = pkl.load(f)
    #
    # tree_ref_file = os.path.expanduser(
    #     f'~/datasets/morphopart/out/tree__{params.instrument}_{params.features}_{params.n_obj_max}_{params.n_obj_max}_{params.replicate}_{params.dim_reducer}_{params.n_clusters_tot}_{params.linkage}.pickle'
    # )
    # with open(tree_ref_file, 'rb') as f:
    #     tree_ref = pkl.load(f)
    #
    #
    #
    # step_params = step_params + ['n_clusters_eval', 'n_obj_dbcv']
    # if all(params[step_params] == previous_params[step_params]):
    #     log.info('    skip: evaluation already performed')
    # else:
    #     results = evaluate(pred_all,params[step_params], log)
    
 

    # set params for next turn of the loop
    previous_params = params

    log.info('end	End')
