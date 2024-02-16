# morphopart

## Principle

This code explores the objective categorisation of UVP particles based on their morphology.

The training process is

1. extract descriptive features
2. reduce their dimension
3. cluster particles in a large number of types (hundreds)
4. hierarchise the particle clusters to be able to regroup them reproducibly in several groups
5. evaluate the quality of the clustering based on a reference clustering (ARI score) and the density of clusters in the reduced space (Silhouette score, DBCV score, etc.)

At each step, several choices are possible:

1. features - which features: from UVPlib, from deep learning approaches, etc. from which instrument: UVP6, UVP5SD, UVP5HD.
2. dimension reduction - which method: PCA, UMAP, etc.
3. clustering - which method: kmeans, dbscan, etc.
4. hierarchisation - which method (agglemerative, etc.) with which linkage (Ward, single, complete, etc.)

and a final question is how many particles are needed to get consistent results (i.e. explore "all" possibilities).

So we perform each step with the various choices and several subsamples of a reference (large) dataset and plot the evaluation results.

Once the "best" choices have been found, the evaluation process for new images, to be included in EcoPart will be:

1. extract the features
2. apply the dimensionality reduction
3. assign each particle to the nearest cluster centroid
4. follow the hierarchical tree to regroup them


## Code

### Workhorse

All functions are in `morphopart.py`, which can be imported as a module. They mirror the steps above

- `extract_features`
- `reduce_dimension` for training and `transform_features` for evaluation
- `cluster`
- `hierarchize`
- `evaluate` which assigns to clusters and compute the evaluation metrics

All functions work the same way:
- they take a dict of parameters as input and log their process
- they store their result in a file, check its existence when they run and skip all computation if it does; this allows to re-run several combinations of parameters without having to recompute everything every time.

In addition `morphopart.py` contains utility functions

- `read_features`, because feature extraction is not yet included in this code
- `subsample_features` to extract the subsamples of various sizes
- `fast_merge`: faster version of pd.merge()
- `safe_sample`, `sample_stratified_by_category`, `sample_stratified_continuous` for subsampling a DataFrame for evaluation

### Pipeline evaluation

`define_params.py` defines a grid of parameters to explore for each step of the pipeline

`explore_params.py` performs all steps of the training process with the chosen parameters grid. This is the main function to run things. It stores all result and intermediate files in a path separate from the code since the files get quite big.

`inspect_results.R` and `inspect_timing.R` parse the evaluation results and the log file to make some plots.


### Technical aspects

Many functions use [cuML](https://docs.rapids.ai/api/cuml/stable/) from rapids.ai. The installation is easier with conda and an `environment.yml` file to create the appropriate conda environment is included. The commands are

```
conda activate base
conda env create -f environment.yml
conda env list
conda activate morphopart
```

An alternative, that may work if you are on an Ubuntu 20.04 system is

```
conda activate base
conda create --name morphopart --file environment_explicit.txt
conda env list
conda activate morphopart
```



The code is version controlled with git.

