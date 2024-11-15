# Import necessary libraries and modules
from dyneusr.tools import process_graph  # Import the process_graph function from dyneusr.tools
from dyneusr import DyNeuGraph  # Import the DyNeuGraph class from dyneusr
from kmapper import KeplerMapper, Cover, adapter  # Import KeplerMapper, Cover, and adapter from kmapper
import networkx as nx  # Import NetworkX for graph operations
import multiprocessing  # Import multiprocessing for parallel processing
import numpy as np
from sklearn.cluster import DBSCAN

# Function to shape a graph into a Temporal Coherence Matrix (TCM)
def g_TCM(graph, nTR):
    # Extract unique data IDs from the graph's nodes
    data_ids = np.unique([_ for n, d in graph['nodes'].items() for _ in d])
    
    # Process the graph using DyNeuGraph tools
    G = process_graph(graph)
    
    # Extract adjacency matrix (A), node degree matrix (M), and TCM (T)
    A, M, TCM = extract_matrices_(G=G, nTR=nTR, index=data_ids)
    
    return TCM  # Return the Temporal Coherence Matrix

# Function to extract matrices from a processed graph
def extract_matrices_(G, nTR, index=None, **kwargs):
    # If no index is provided, create a unique list of node members
    if index is None:
        index = np.unique([__ for n, _ in G.nodes(data='members') for __ in _])
    
    # Create the adjacency matrix (A) from the graph
    A = nx.to_numpy_matrix(G).A  # Convert the graph to a numpy adjacency matrix
    
    # Initialize the node degree matrix (M) and the Temporal Coherence Matrix (T)
    M = np.zeros((nTR, A.shape[0]))  # TR x node matrix
    T = np.zeros((nTR, nTR))  # TR x TR matrix

    # If the graph is empty, return empty matrices
    if not len(G):
        return A, M, T

    # Map nodes to indices and get their members
    node_to_index = {n: i for i, n in enumerate(G)}
    node_to_members = dict(nx.get_node_attributes(G, 'members'))
    node_members = np.array(list(node_to_members.values()))

    # Loop over time points (TRs) to fill in the matrices
    for TR in range(nTR):
        # Find nodes that contain the current TR
        TR_nodes = [n for n, _ in G.nodes(data='members') if TR in _]

        # Skip this TR if no nodes are found
        if not len(TR_nodes):
            continue
        
        # Get indices of the found nodes and update the node degree matrix (M)
        node_index = [node_to_index[_] for _ in TR_nodes]
        M[TR, node_index] += 1.0

        # Find similar nodes (nodes and their neighbors)
        similar_nodes = np.nonzero(A[node_index, :])[-1]
        similar_nodes = np.r_[node_index, similar_nodes]
        similar_nodes = list(np.unique(similar_nodes))

        # Count the members found in similar nodes
        similar_TRs = np.hstack(node_members[similar_nodes])
        similar_TRs, counts = np.unique(similar_TRs, return_counts=True)

        # Update the Temporal Coherence Matrix (T) with the degree of connectivity between TRs
        T[TR, similar_TRs] += counts / max(counts)

    # Symmetrize the matrix T
    T = (T + T.T) / 2.0

    return A, M, T  # Return the adjacency matrix (A), node degree matrix (M), and Temporal Coherence Matrix (T)

# Function to create a point cloud mapper and generate a Temporal Coherence Matrix (TCM)
def points_cloud_mapper(embeds, nTR, cover_n, cover_overlap, lens=None, eps=0.7):
    # Initialize KeplerMapper
    mapper = KeplerMapper(verbose=0)
    
    # If no lens is provided, create a lens using the first three dimensions of the embeddings
    if lens is None:
        lens = mapper.fit_transform(embeds, projection=[0, 1, 2])
    
    # Create a graph from the lens and embeddings using the specified cover and clustering parameters
    graph = mapper.map(lens, X=embeds, cover=Cover(cover_n, cover_overlap, limits=np.array([[0, 1], [0, 1], [0, 1]])), clusterer=DBSCAN(eps=eps))
    
    # Generate the Temporal Coherence Matrix (TCM) from the graph
    tcm = g_TCM(graph=graph, nTR=nTR)
    
    return tcm  # Return the Temporal Coherence Matrix

# Function to process UMAP embeddings in parallel and generate a sub-matrix of the TCM
def process_sub_embeds(args):
    # Unpack the arguments
    embeds_i, g_size, cover_n, cover_overlap, lens_i, eps = args
    
    # Generate the Temporal Coherence Matrix (TCM) for the current embedding
    tcm = points_cloud_mapper(embeds=embeds_i, nTR=int(g_size * 2), cover_n=cover_n, cover_overlap=cover_overlap, lens=lens_i, eps=eps)
    
    # Extract the sub-matrix of the TCM that corresponds to the current size
    sub_tcm = tcm[:g_size, g_size:]
    
    return sub_tcm  # Return the sub-matrix of the Temporal Coherence Matrix
