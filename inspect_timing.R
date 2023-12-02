#
# Read and inspect the processing time, from the log
#

library("tidyverse")

## Parse and prepare log ----

l <- read_tsv("log.tsv", col_names=c("time", "step", "action"), show_col_types=FALSE) |> 
  mutate(time=time |> 
           str_replace(",", ".") |> 
           parse_datetime(format="%Y-%m-%d %H:%M:%OS")
  ) |> 
  fill(step)

# parse parameters
s <- filter(l, step=="start")
s_desc <- map_dfr(s$action, function(x) {
    x <- reticulate::py_run_string(str_c("dict=", x |> str_extract(pattern="\\{.*?\\}")))
    x$dict |> as_tibble()
  }) |> 
  # fix inconsistency
  replace_na(list(linkage="ward"))
s <- bind_cols(s, s_desc)

l <- left_join(l, s) |> 
  fill(instrument:linkage)

# detect full runs, identify them, and remove the others
l <- l |> 
  mutate(block = (step=="start") |> cumsum()) |> 
  group_by(block) |> 
  group_modify(.f=function(.x, .y) {
    if (sum(.x$step == "end") != 1) {
      .x$run <- NA
    } else {
      .x$run <- .y$block
    }
    return(.x)
  }) |> 
  ungroup() |> 
  select(-block) |> 
  filter(!is.na(run))

# compute durations
l <- l |> 
  group_by(run) |> 
  mutate(dur=c(diff(time), 0) |> as.numeric()) |> 
  ungroup()

## Compute statistics ----

# list all actions
l |> 
  filter(str_detect(step, "step")) |> 
  count(action)
# list all parameters
names(l)


l |> 
  filter(action=="read all features") |> 
  group_by(instrument, features, n_obj_max) |> summarise(mean(dur), sd(dur)) |> ungroup()
# -> directly linked to the number fo elements to subsample x nb of columns
#    for the 384 features, in the 2 min range

l |> 
  filter(action=="subsample features") |> 
  group_by(instrument, features, n_obj_max, n_obj_sub) |> summarise(mean(dur), sd(dur)) |> ungroup()
# -> directly linked to the number fo elements to subsample x nb of columns
#    always quite fast

l |> 
  filter(action=="fit dimensionality reducer") |> 
  group_by(instrument, features, dim_reducer, n_obj_sub) |> summarise(mean(dur), sd(dur), n=n()) |> ungroup()
# -> PCA considerably faster to fit than UMAP (<1 min vs. up to 1h for 1M rows)
#    for UMAP: dino >> mobilenet ~ uvplib, so the time is *not* directly linked with the dimension
#    for PCA : dino > mobilenet >> uvplib, so the time is roughly linked with the dimension
l |> 
  filter(action=="fit dimensionality reducer") |> 
  group_by(dim_reducer) |> summarise(tot=sum(dur)) |> ungroup() |> 
  mutate(tot=tot/3600)
# -> 17h doing only UMAPs...

l |> 
  filter(action=="define clusterer") |> 
  group_by(instrument, features, dim_reducer, n_obj_sub) |> summarise(mean(dur), sd(dur), n=n()) |> ungroup()
# -> kmeans is easier with UMAP than PCA (3.3 vs 6 min, for 1M rows)

l |> 
  filter(action=="define tree of centroids") |> 
  group_by(instrument) |> summarise(mean(dur), sd(dur)) |> ungroup()
# -> always very fast

l |> 
  filter(str_detect(action, "^load")) |> 
  summarise(tot=sum(dur))
# -> 1h just loading stuff into memory (the new skipping feature should shave some of that)