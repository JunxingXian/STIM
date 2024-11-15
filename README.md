# STIM 

STIM (Synchronized Topological Individual Mapper) is a framework designed for investigating individual differences in low-dimensional brain dynamics under naturalistic stimuli. Based on the Mapper algorithm, STIM modifies its approach according to the naturalistic stimulus conditions. It maps time series data into a lower-dimensional space, generating group-level dynamics and measuring both global and local topological similarities between individuals.

For a detailed description of the STIM framework and its application across multiple datasets, please refer to the article:  
[**Mapping Individual Differences in the Topological Landscape of Naturalistic Brain Dynamics**](https://www.biorxiv.org/content/10.1101/2024.06.20.599966v1)

For step-by-step instructions on how to use STIM, please refer to the tutorial section below.

## Tutorial

In this tutorial, we provide an overview of how to use STIM for preliminary feature extraction. For more detailed information, refer to the notebook located in `STIM_Tutorial.ipynb`. STIM is fully implemented in Python, and the necessary packages are listed in the code availability section of the referenced article.

### Data Import

The input data for STIM consists of time-aligned neural data (same number of time points). In this example, we use preprocessed HCP data (for detailed preprocessing steps, please refer to the Method section of the article).

```python
# Import data
hcp_18clips_ts = np.load(data_path)

# Remove clips with less than two minutes of data
hcp_13clips_ts = []
movie_ind = [0, 1, 2, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16]
for i in movie_ind:
    hcp_13clips_ts.append(hcp_18clips_ts[i])
```

### Building Low-Dimensional Embeddings with STIM

We begin by using standard parameters to compute low-dimensional embeddings. At this point, data from all movies are combined, and the resulting individual low-dimensional dynamics time series represent the aggregated time points across all movie clips. The group-average low-dimensional dynamics is then converted into a shape graph using the traditional Mapper approach (here, we convert to a NetworkX graph).

```python
# Select a filter (here, we use UMAP)
umap_model = UMAP(n_components=3, n_neighbors=25, random_state=42, min_dist=0.01, metric='euclidean')

# Apply STIM to the data
hcp_13clips_umap = STIM_LowDimSpace(hcp_13clips_ts, umap_model)

# Compute group consensus
group_umap = hcp_13clips_umap.mean(axis=0)
graph_group = g_mapper_graph(low_d_embeds=scale(group_umap, axis=0), cover_n=12, cover_overlap=0.5, eps=0.7)

# visualize shape graph
dG = DyNeuGraph(G=graph_group)
dG.visualize('your_path')
```

### Building Individual and Group TCMs with STIM

After data normalization, individual and group low-dimensional embeddings are merged to compute the Time Connectivity Matrix (TCM), a time × time matrix. For a detailed calculation procedure, please refer to the Method section of the article.

```python
# Calculate the TCM for each individual relative to the group (optional: specify group_embeds)
# If group_embeds is not provided, the group embedding is set as the mean of all individual embeddings.
hcp_170subs_tcm = STIM_individual_mapper(sub_embeds=hcp_13clips_umap, group_embeds=None)
```

### Calculating Global Topology and Local Geometry

Global topology refers to the diagonal of each individual's TCM. Local geometry involves two steps: 1) Event segmentation of the template TCM, and 2) Scoring the individual TCM within each event.

In this tutorial, we use the first movie clip as an example:

```python
# The first movie clip has 243 time points
hcp_tcm_mov1 = np.array(hcp_170subs_tcm)[:,:243,:243]

# Use the average TCM as the template
hcp_local_geo, clip_events = STIM_local_geometry(indi_TCM=hcp_tcm_mov1,template_TCM=hcp_tcm_mov1.mean(axis=0))
```
