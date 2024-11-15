from sklearn.preprocessing import scale  # Import the scale function for normalizing data
from sklearn.decomposition import PCA  # Import PCA for dimensionality reduction
from umap import UMAP  # Import UMAP for non-linear dimensionality reduction
import numpy as np

def align_projection(ts_arr, model, template=None, scale_axis=1, if_norm=False):
    '''
    Align time-series data using UMAP with optional PCA initialization.
    
    Parameters:
    ts_arr (numpy.ndarray): Time-series data, shape (n_subj, n_ROIs, n_TR).
    model: A filter model, with fit and fit_transform. 
    template (numpy.ndarray or None): Optional template for fit. Default is None.
    scale_axis (int): Axis along which to scale the data. Default is 1 (features).
    if_norm (bool): Whether to normalize each time-series vector. Default is False.
    
    Returns:
    sub_umap (numpy.ndarray): UMAP embeddings for each subject, shape (n_subj, n_TR, n_components).
    '''
    n_sub = ts_arr.shape[0]  # Number of subjects
    nTR = ts_arr.shape[2]  # Number of time points (TR)
    
    # Normalize the time-series data along the specified axis
    ts_norm = np.array([scale(ts_arr[i], axis=scale_axis) for i in range(n_sub)])
    
    # Stack the normalized time-series data from all subjects
    ts_stack = np.vstack([ts_norm[i].T for i in range(n_sub)])
    
    if if_norm:
        # Normalize each time-series vector by its L2 norm if specified
        ts_stack = np.array([i / np.linalg.norm(i) for i in ts_stack])
    
    # Initialize PCA for dimensionality reduction
    pca = PCA(n_components=model.n_components, random_state=42)
    
    if template is not None:
        # If a template is provided, fit PCA to the template and initialize UMAP with PCA results
        pca_res = pca.fit_transform(template)
        model.init = pca_res
        model.fit(template)  # Fit UMAP to the template
        embeds = model.transform(ts_stack)  # Transform the test data using the fitted UMAP
    else:
        # If no template is provided, fit PCA to the test data and initialize UMAP with PCA results
        pca_res = pca.fit_transform(ts_stack)
        model.init = pca_res
        embeds = model.fit_transform(ts_stack)  # Fit and transform the test data using UMAP
    
    # Reshape the UMAP embeddings back into the original subject structure
    sub_embeds = np.array([embeds[i*nTR:(i+1)*nTR] for i in range(n_sub)])
    
    return sub_embeds  # Return the UMAP embeddings for each subject
