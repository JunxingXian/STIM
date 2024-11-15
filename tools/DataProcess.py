import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar display
import pandas as pd  # Import pandas for data manipulation
import os  # Import os for interacting with the file system

# Generate a list of subject IDs that have data for all specified movies
def g_id_list(path):
    """
    Function to generate a list of subject IDs that have complete movie data.
    
    Parameters:
    path (str): The directory path where the subject data is stored.
    
    Returns:
    list: A list of subject IDs that have data for all four movies.
    """
    # List of all subject IDs in the given path
    id_list = os.listdir(path)
    id_intersec = []  # List to store subject IDs that have data for all movies
    movie_name_ls = ['movie1_AP', 'movie2_PA', 'movie3_PA', 'movie4_AP']  # List of movie names to check
    
    for sub in id_list:
        flag = []  # Initialize a list to check the existence of movie data for the current subject
        for mov_i in movie_name_ls:
            # Check if the movie data file exists for the current subject
            flag.append(os.path.exists(os.path.join(path, sub, sub + '_' + mov_i + '_with_gs.npy')))
        if np.sum(flag) == 4:  # If all four movies are present, add the subject ID to the intersection list
            id_intersec.append(sub)
    return id_intersec  # Return the list of valid subject IDs

# Generate a list of time series data for all subjects and movies
def g_ts_ls(path):
    """
    Function to generate a list of time series data for all valid subjects and movies.
    
    Parameters:
    path (str): The directory path where the subject data is stored.
    
    Returns:
    list: A list of numpy arrays containing time series data for each movie.
    """
    
    id_list = os.listdir(path)  # List of all subject IDs in the given path
    id_intersec = []  # List to store subject IDs that have data for all movies
    movie_name_ls = ['movie1_AP', 'movie2_PA', 'movie3_PA', 'movie4_AP']  # List of movie names to check
    
    for sub in id_list:
        flag = []  # Initialize a list to check the existence of movie data for the current subject
        for mov_i in movie_name_ls:
            # Check if the movie data file exists for the current subject
            flag.append(os.path.exists(os.path.join(path, sub, sub + '_' + mov_i + '_with_gs.npy')))
        if np.sum(flag) == 4:  # If all four movies are present, add the subject ID to the intersection list
            id_intersec.append(sub)

    sub_ts_ls = []  # List to store time series data for each movie
    for movie_name_i in movie_name_ls:
        movie_i_ts_ls = []  # List to store time series data for the current movie
        for sub in id_intersec:
            # Load the time series data for the current subject and movie
            movie_i_ts_ls.append(np.load(os.path.join(path, sub, sub + '_' + movie_name_i + '_with_gs.npy')))
        movie_i_ts_ls = np.array(movie_i_ts_ls)  # Convert the list to a numpy array
        print(movie_i_ts_ls.shape)  # Print the shape of the loaded time series data
        sub_ts_ls.append(movie_i_ts_ls)  # Add the time series data to the main list

    return sub_ts_ls  # Return the list of time series data for all movies

from nilearn import signal  # Import the signal module from nilearn for signal processing

# Apply a bandpass filter to time series data
def g_ts_filter(ts_ls, low_pass=None, high_pass=None, t_r=1):
    """
    Function to apply a bandpass filter to the time series data.
    
    Parameters:
    ts_ls (numpy.ndarray): The time series data, shape (n_subjs, ROIs, nTR).
    low_pass (float or None): The low-pass frequency for the filter. Default is None.
    high_pass (float or None): The high-pass frequency for the filter. Default is None.
    t_r (float): The repetition time (TR) of the time series. Default is 1.
    
    Returns:
    numpy.ndarray: The filtered time series data.
    """
    
    filter_ts = []  # List to store filtered time series data
    for ts_i in tqdm(ts_ls):  # Loop over each movie's time series data with a progress bar
        sub_filter_i = []  # List to store filtered data for each subject
        for sig_i in ts_i:  # Loop over each subject's time series data
            # Apply bandpass filter to the time series data
            sig_i_filter = signal.clean(sig_i, detrend=False, standardize=False,
                                        confounds=None, low_pass=low_pass, high_pass=high_pass, t_r=t_r,
                                        ensure_finite=False)
            sub_filter_i.append(sig_i_filter)  # Add the filtered data to the list
        filter_ts.append(np.array(sub_filter_i))  # Convert the list to a numpy array and add to the main list
    return np.array(filter_ts)  # Return the filtered time series data as a numpy array
    
def load_npz(path):
    """
    Function to load a .npz file and return its contents as a list.
    
    Parameters:
    path (str): The file path to the .npz file.
    
    Returns:
    list: A list containing the arrays stored in the .npz file.
    """
    file = np.load(path)  # Load the .npz file
    file_ls = [file[key] for key in file.files]  # Extract each array from the .npz file and store in a list
    return file_ls  # Return the list of arrays

def triu_mtx(mtx, k=1):
    """
    Function to extract the upper triangular part of a matrix.
    
    Parameters:
    mtx (numpy.ndarray): The input matrix.
    k (int): Diagonal offset (default is 1, which excludes the main diagonal).
    
    Returns:
    numpy.ndarray: The upper triangular part of the matrix as a 1D array.
    """
    return mtx[np.triu_indices_from(mtx, k=k)]  # Return the upper triangular part of the matrix

# Compute the correlation coefficient between two matrices
def corr2_coeff(A, B):
    """
    Function to compute the Pearson correlation coefficient between two matrices.
    
    Parameters:
    A (numpy.ndarray): The first input matrix, shape (m, n).
    B (numpy.ndarray): The second input matrix, shape (m, n).
    
    Returns:
    numpy.ndarray: A correlation matrix representing the pairwise Pearson correlation coefficients
                   between the rows of A and B.
    """
    A = A.T  # Transpose matrix A to align rows for correlation computation
    B = B.T  # Transpose matrix B to align rows for correlation computation
    
    # Subtract the row-wise mean from each row of A and B
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Compute the sum of squares for each row of A and B
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Compute the Pearson correlation coefficient matrix
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))
