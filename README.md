# STIM 

STIM (Synchronized Topological Individual Mapper) is a framework designed for investigating individual differences in low-dimensional brain dynamics under naturalistic stimuli. By leveraging Topological Data Analysis (TDA), STIM aligns high-dimensional whole-brain activities across individuals during movie fMRI into a shared low-dimensional state space, enabling quantification of both global and local topological similarities between individuals.

For a detailed description of the STIM framework and its application across multiple datasets, please refer to the pre-print:  
[**Probing Individual Differences in the Topological Landscape of Naturalistic Brain Dynamics**](https://www.biorxiv.org/content/10.1101/2024.06.20.599966v2)

For step-by-step instructions on how to use STIM, please refer to the quick start section below.

## Dependencies

STIM aims to quantify both shared and individual-specific dynamic brain activity patterns in low-dimensional space. A crucial dependency is Topological Data Analysis (TDA), specifically the Mapper algorithm - a method that combines dimensionality reduction, clustering, and graph network techniques to understand high-dimensional data structure. For the Mapper implementation, we utilize **DyNeuSR**, a Python toolbox described in the following papers:

- *Generating Dynamical Neuroimaging Spatiotemporal Representations (DyNeuSR) using Topological Data Analysis* (Geniesse et al., 2019)
- *Towards a New Approach to Reveal Dynamical Organization of the Brain using Topological Data Analysis* (Saggar et al., 2018)

For filter function, we utilize UMAP (Uniform Manifold Approximation and Projection) for dimensionality reduction, specifically the Python implementation umap-learn (McInnes et al., 2018). For graph operations and analysis, we use NetworkX, a Python package for the creation, manipulation, and study of complex networks.

## Quick Start

In this README, we provide a brief guide to using STIM for feature extraction. For detailed examples and usage, please refer to `STIM_Tutorial.ipynb`. STIM is implemented in Python, with all required dependencies listed in the code availability section of our paper.


### Data Input
Here we provide an example input data for STIM. Suppose we have fMRI data from 10 subjects, with each subject having 271 ROIs and 600 TRs. We concatenate the data along the ROI dimension:

```python
# generate random time series: shape 10 subjects * 271 ROIs * 600 TRs
time_series = np.random.rand(10,271,600)
```

### Building Low-Dimensional Embeddings with STIM


We begin by computing low-dimensional embeddings using standard parameters. The data from all movies are combined, with the resulting individual low-dimensional dynamics representing the aggregated time points across all movie clips. The group-average low-dimensional dynamics is then transformed into a shape graph using the Mapper algorithm and converted to a NetworkX graph representation.

```python
# Select a filter (here, we use UMAP)
umap_model = UMAP(n_components=3, n_neighbors=25, random_state=42, min_dist=0.01, metric='euclidean')

# Apply STIM to the data
umap_embeddings = STIM_LowDimSpace(time_series, umap_model)

# Compute group consensus
group_umap = umap_embeddings.mean(axis=0)
graph_group = g_mapper_graph(low_d_embeds=scale(group_umap, axis=0), cover_n=12, cover_overlap=0.5, eps=0.7)

# visualize shape graph
dG = DyNeuGraph(G=graph_group)
dG.visualize('your_path')
```

### Building Individual and Group TCMs with STIM

After data normalization, individual and group low-dimensional embeddings are integrated to compute the Time Connectivity Matrix (TCM), a time × time matrix. For detailed calculation procedures, please refer to the Method section.

```python
# Calculate the TCM for each individual relative to the group (optional: specify group_embeds)
# If group_embeds is not provided, the group embedding is set as the mean of all individual embeddings.
tcm = STIM_individual_mapper(sub_embeds=umap_embeddings, group_embeds=None)
```

### Calculating Global Topology and Local Geometry

Global topology is quantified through the diagonal of each individual's TCM. Local geometry analysis consists of two steps: 1) Event segmentation of the template TCM, and 2) Scoring the individual TCM within each event.


```python
# extract global topology
glob_topo = STIM_global_topology(tcm)

# extract local geometry
local_geo, clip_events = STIM_local_geometry(indi_TCM=tcm,template_TCM=tcm.mean(axis=0))
