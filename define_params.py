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
    'instrument': ['uvp5sd'],
    'features': ['mobilenet', 'uvplib', 'dino'],
    'n_obj_max': [2000000],
    'replicate': [1, 2, 3, 4, 5],
    'n_obj_sub': [2000000, 1000000, 500000, 250000, 100000, 50000, 10000, 50000, 10000, 5000, 2500, 1000, 500],
    'dim_reducer': ['UMAP', 'PCA'],
    'n_clusters_tot': [200],
    'linkage': ['ward', 'average', 'complete'],
    'n_clusters_eval': [5, 15, 100, 200],
    'n_obj_eval': [100000]
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



params = expand_grid({
    'instrument': ['uvp5sd'],
    'features': ['dino'],
    'n_obj_max': [2000000],
    'replicate': [1],
    'n_obj_sub': [2000000],
    'dim_reducer': ['PCA'],
    'n_clusters_tot': [200],
    'linkage': ['ward'],
    'n_clusters_eval': [5, 10, 20, 50, 70, 100, 150, 200],
    'n_obj_eval': [100000]
})



###### test sur le Kmean direct

import os                       # general packages
import pickle as pkl
import logging

import matplotlib.pyplot as plt # science packages
import numpy as np
import pandas as pd

#from morphopart import *        # local package
# import ipdb                     # debugging


## Prepare output ----

# create output directory
os.makedirs(os.path.expanduser('~/datasets/morphopart/out_yeo'), exist_ok=True)

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

previous_params = params_grid.iloc[0].copy()
# NB: the copy() is required to not get just a view of the original data
previous_params[:] = np.nan

directory= '~/datasets/morphopart'

df_Alluvial = pd.DataFrame()  # ou pd.DataFrame(columns=c_all_ref.columns)
for i in range(params_grid.shape[0]):
    # pick one row to process
    params = params_grid.iloc[i]
    log.info(f'start	Start parameters set {i} : {params.to_dict()}')
    
    #----------------------------------------------------------------------------------------------------------------------------------------------#
    log.info('step 0	Extract, read and prepare data') # ----

    step_params = ['instrument', 'features', 'n_obj_max']
    if all(params[step_params] == previous_params[step_params]):
        log.info('	skip: data already read')
    else:
        f_all = get_features(directory, params[step_params], log)
    #----------------------------------------------------------------------------------------------------------------------------------------------#
    ref_params = ['instrument', 'features', 'n_obj_max', 'replicate', 'dim_reducer']
    # NB: we use replicate 1 all the time here
    # TODO this change from replicate 1 to others means the ARI score is not always 1; check if sample is done with or without replacement
    if all(params[ref_params] == previous_params[ref_params]):
        log.info('	skip: reference dimensionality reduction already loaded')
    else:
        log.info('	load reference dimensionality reduction')
        dimred_ref_file = os.path.expanduser(
            '~/datasets/morphopart/out_yeo/dimred__'
            f'{params.instrument}_{params.features}_{params.n_obj_max}_{params.n_obj_max}_{params.replicate}_{params.dim_reducer}'
            '.pickle'
        )
        with open(dimred_ref_file, 'rb') as f:
            dimred_ref = pkl.load(f)

    import cuml
    import rmm
    
    f_all_reduced_ref = dimred_ref['features_reduced']
    
    clust_ref = cuml.KMeans(n_clusters=params.n_clusters_eval,
                   init='scalable-k-means++', n_init=10,
                   random_state=params.replicate
                  )

    clust_ref.fit(f_all_reduced_ref)

    log.info('	define cluster centroids')
    centroids = clust_ref.cluster_centers_

    log.info('	compute cluster membership')
    clusters = clust_ref.predict(f_all_reduced_ref)

    cluster_ref = {'clusterer': clust_ref, 'centroids': centroids, 'clusters': clusters}

    # clean CUDA memory
    rmm.reinitialize()

    #----------------------------------------------------------------------------------------------------------------------------------------------#

    df = pd.read_csv( f'~/datasets/morphopart/{params.instrument}/taxa.csv.gz', usecols = ['objid','taxon'])
    df= df[np.isin(df["objid"].values, f_all.index.values)]
    df=df.set_index(df["objid"])
    df=df.reindex(f_all.index)    
    
    # compute metrics
    # define the reference clusters (at n_cluster_eval level)
    clusters_ref=cluster_ref['clusters']
    c_all_ref = pd.DataFrame({params.n_clusters_eval: cluster_ref['clusters']})

    df_Alluvial = pd.concat([df_Alluvial, c_all_ref], axis=1, ignore_index=True)
        #----------------------------------------------------------------------------------------------------------------------------------------------#

    log.info('end	End')
    
    
import pandas as pd
import plotly.graph_objects as go

# Étape 1 : Renommer les colonnes
clusters = [5, 10, 20, 50, 70, 100, 150, 200]
df_Alluvial.columns = [f"{i} clusters" for i in clusters]
cols = df_Alluvial.columns.tolist()
cols = cols[::-1]  # ["10 clusters", "9 clusters", ..., "2 clusters"]


import pandas as pd

all_transitions = []

for i in range(len(cols) - 1):
    src_col = cols[i]     # Ex: "10 clusters"
    tgt_col = cols[i + 1] # Ex: "9 clusters"
    
    # Grouper les individus par cluster source → target
    transitions = df_Alluvial.groupby([src_col, tgt_col]).size().reset_index(name='count')
    
    # Total pour chaque cluster source
    total_per_source = transitions.groupby(src_col)['count'].transform('sum')
    transitions['percentage'] = transitions['count'] / total_per_source * 100
    
    # Formater les labels pour Sankey
    transitions['source'] = transitions[src_col].astype(str) + f' ({src_col})'
    transitions['target'] = transitions[tgt_col].astype(str) + f' ({tgt_col})'
    
    # Conserver uniquement les colonnes nécessaires
    all_transitions.append(transitions[['source', 'target', 'percentage']])

df_transitions_pct = pd.concat(all_transitions, ignore_index=True)

import plotly.graph_objects as go

# Créer les nœuds uniques
labels = list(pd.unique(df_transitions_pct[['source', 'target']].values.ravel()))
label_to_index = {label: idx for idx, label in enumerate(labels)}

# Mapper les sources / targets vers leurs indices
source_ids = df_transitions_pct['source'].map(label_to_index)
target_ids = df_transitions_pct['target'].map(label_to_index)
values = df_transitions_pct['percentage']

# Créer le diagramme
fig = go.Figure(data=[go.Sankey(
    arrangement="snap",
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels,
        color="lightblue"
    ),
    link=dict(
        source=source_ids,
        target=target_ids,
        value=values,
        color="rgba(100,100,200,0.4)"
    )
)])

fig.update_layout(
    title_text="Alluvial Plot – Réduction progressive de 200 à 5 clusters (en %)",
    font_size=10
)



import pandas as pd
import plotly.graph_objects as go

# Étape 1 : Renommer les colonnes
clusters = [5, 10, 20, 50, 70, 100, 150, 200]
df_Alluvial.columns = [f"{i} clusters" for i in clusters]

# Conserver uniquement les colonnes 10 → 5
cols = ["200 clusters", "5 clusters"]

all_transitions = []

for i in range(len(cols) - 1):
    src_col = cols[i]     # "10 clusters"
    tgt_col = cols[i + 1] # "5 clusters"
    
    transitions = df_Alluvial.groupby([src_col, tgt_col]).size().reset_index(name='count')
    total_per_source = transitions.groupby(src_col)['count'].transform('sum')
    transitions['percentage'] = transitions['count'] / total_per_source * 100

    transitions['source'] = transitions[src_col].astype(str) + f' ({src_col})'
    transitions['target'] = transitions[tgt_col].astype(str) + f' ({tgt_col})'
    
    all_transitions.append(transitions[['source', 'target', 'percentage']])

df_transitions_pct = pd.concat(all_transitions, ignore_index=True)

# Créer les labels pour le Sankey
labels = list(pd.unique(df_transitions_pct[['source', 'target']].values.ravel()))
label_to_index = {label: idx for idx, label in enumerate(labels)}

source_ids = df_transitions_pct['source'].map(label_to_index)
target_ids = df_transitions_pct['target'].map(label_to_index)
values = df_transitions_pct['percentage']

# Sankey Plot
fig = go.Figure(data=[go.Sankey(
    arrangement="snap",
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels,
        color="lightblue"
    ),
    link=dict(
        source=source_ids,
        target=target_ids,
        value=values,
        color="rgba(100,100,200,0.4)"
    )
)])

fig.update_layout(
    title_text="Alluvial Plot – Réduction de 10 à 5 clusters (en %)",
    font_size=10
)

fig.write_html("sankey_plot_100_10.html")



    
    