import numpy as np
from tqdm import tqdm  # Import the tqdm module for progress bar display
from sklearn.preprocessing import scale  # Import the scale function for normalizing data
import sys
import os

# Get the directory of the current script and add it to the system path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
import PELT  # Import the PELT module

def PELT_segment(dts_mtx, penalty=10, min_size=10):
    '''
    Function to segment time points using the PELT algorithm.
    
    Parameters:
    dts_mtx (numpy.ndarray): A symmetric matrix representing distances (tr * tr).
    penalty (int): Penalty parameter for the PELT algorithm. Default is 10.
    min_size (int): Minimum segment size for PELT. Default is 10.
    
    Returns:
    scene_tr (list): List of tuples representing the start and end points of segments.
    '''
    nTR = dts_mtx.shape[0]  # Number of time points (TR)
    
    tr_jump = [0]  # Initialize the list of change points
    det = PELT.Pelt(cost="l2", min_size=min_size, jump=1)  # Initialize PELT detector with L2 cost
    tr_jump += list(det.find_changepoints(dts_mtx, penalty=penalty))  # Find change points
    tr_jump.append(nTR-1)  # Add the last time point as a change point
    
    scene_tr = []
    for i in range(len(tr_jump) - 1):
        a = tr_jump[i] + 1
        b = tr_jump[i + 1]
        scene_length = b - a
        if scene_length > 5:  # Filter out very short segments
            scene_tr.append((a, b))  # Store the segment
    return scene_tr

def check_tr_in_clique(tr_pair, clique_tr):
    '''
    Check if time points in a pair are within a specified segment.
    
    Parameters:
    tr_pair (tuple): Pair of time points to check.
    clique_tr (tuple): Segment boundaries (start, end).
    
    Returns:
    int: 1 if either time point is outside the segment, 0 otherwise.
    '''
    start_t = clique_tr[0]
    end_t = clique_tr[1]
    for tr_i in tr_pair:
        if (tr_i < start_t) or (tr_i >= end_t):
            return 1  # Return 1 if any time point is outside the segment
    return 0  # Return 0 if both time points are within the segment

def check_tr_out_mov(tr_pair, nTR):
    '''
    Check if time points in a pair are out of the valid range.
    
    Parameters:
    tr_pair (tuple): Pair of time points to check.
    nTR (int): Number of total time points (TR).
    
    Returns:
    bool: True if any time point is out of bounds, False otherwise.
    '''
    test1 = tr_pair[0] < 0
    test2 = tr_pair[1] < 0
    test3 = tr_pair[0] >= nTR
    test4 = tr_pair[1] >= nTR
    return (True in [test1, test2, test3, test4])  # Return True if any time point is out of bounds

def clique_performance(dts_mtx, scene_tr):
    '''
    Calculate within-segment and between-segment performance based on the distance matrix.
    
    Parameters:
    dts_mtx (numpy.ndarray): Distance matrix (tr * tr).
    scene_tr (list): List of segment boundaries.
    
    Returns:
    tuple: Dictionary and results containing within-segment and between-segment pairs and their corresponding distances.
    '''
    nTR = dts_mtx.shape[0]  # Number of time points
    within_bwt_dict = {'with_in': [], 'bwt': []}  # Initialize dictionary for within and between segment pairs
    within_bwt_res = {'with_in': [], 'bwt': []}  # Initialize dictionary for within and between segment distances
    keys = list(within_bwt_dict.keys())
    
    for clique_tr_i in scene_tr:
        start_t = clique_tr_i[0]
        end_t = clique_tr_i[1]
        duration = end_t - start_t
        pair = []
        for i in range(start_t, end_t):
            pair += [[(i+j, i), (i-j, i)] for j in range(1, duration)]  # Generate pairs of time points within the segment
        for pair_i in pair:
            tuple_i = pair_i[0]
            tuple_j = pair_i[1]
            if check_tr_out_mov(tuple_i, nTR) or check_tr_out_mov(tuple_j, nTR):
                continue  # Skip if any time point is out of bounds
            ind_i = check_tr_in_clique(tuple_i, clique_tr_i)
            ind_j = check_tr_in_clique(tuple_j, clique_tr_i)
            if ind_i + ind_j == 1:  # Classify pairs as within or between segments
                within_bwt_dict[keys[ind_i]].append(tuple_i)
                within_bwt_dict[keys[ind_j]].append(tuple_j)
    within_bwt_res['with_in'] = [dts_mtx[_] for _ in within_bwt_dict['with_in']]  # Get distances for within-segment pairs
    within_bwt_res['bwt'] = [dts_mtx[_] for _ in within_bwt_dict['bwt']]  # Get distances for between-segment pairs
    
    return within_bwt_dict, within_bwt_res

def clique_score(dts_mtx, scene_tr):
    '''
    Calculate the ratio of within-segment to between-segment distances.
    
    Parameters:
    dts_mtx (numpy.ndarray): Distance matrix (tr * tr).
    scene_tr (list): List of segment boundaries.
    
    Returns:
    float: Ratio of mean within-segment to mean between-segment distances.
    '''
    inner_score_ls = []
    for i in range(len(scene_tr)):
        ci_t1 = scene_tr[i][0]
        ci_t2 = scene_tr[i][1]
        inner_score_ls.append(np.mean(dts_mtx[ci_t1:ci_t2, ci_t1:ci_t2]))  # Calculate mean within-segment distance
    
    bwt_score_ls = []
    for i in range(len(scene_tr) - 1):
        ci_t1 = scene_tr[i][0]
        ci_t2 = scene_tr[i][1]
        cj_t1 = scene_tr[i+1][0]
        cj_t2 = scene_tr[i+1][1]
        bwt_score_ls.append(np.mean(dts_mtx[ci_t1:ci_t2, cj_t1:cj_t2]))  # Calculate mean between-segment distance
    return np.mean(inner_score_ls) / np.mean(bwt_score_ls)  # Return the ratio of within-segment to between-segment distance

def search_best_param(dts_mtx, method='tr_level', penalty=range(1, 50)):
    '''
    Search for the best penalty parameter by evaluating segment performance.
    
    Parameters:
    dts_mtx (numpy.ndarray): Distance matrix (tr * tr).
    method (str): Evaluation method ('tr_level' or 'clique_level'). Default is 'tr_level'.
    penalty (range): Range of penalty values to search. Default is range(1, 50).
    
    Returns:
    tuple: Best segment boundaries and performance dictionary.
    '''
    performance = {}
    for p_i in tqdm(penalty):  # Iterate over penalty values with a progress bar
        scene_tr = PELT_segment(dts_mtx, penalty=p_i, min_size=5)  # Segment the time points with current penalty
        if method == 'tr_level':
            within_bwt_dict, within_bwt_res = clique_performance(dts_mtx=dts_mtx, scene_tr=scene_tr)
            score = np.nanmean(within_bwt_res['with_in']) / np.nanmean(within_bwt_res['bwt'])  # Calculate score
            if np.isnan(score):
                score = np.inf  # Handle NaN scores
            performance[p_i] = score
        elif method == 'clique_level':
            score = clique_score(dts_mtx, scene_tr)
            if np.isnan(score):
                score = np.inf  # Handle NaN scores
            performance[p_i] = score
    best_p = list(performance.keys())[np.argsort(list(performance.values()))[-1]]  # Get the best penalty value
    best_scene_tr = PELT_segment(dts_mtx, penalty=best_p, min_size=5)  # Segment with the best penalty value
    return best_scene_tr, performance

def weighted_clique_score(clique, weight=None):
    '''
    Calculate a weighted score for cliques (segments).
    
    Parameters:
    clique (list): List of cliques (segments).
    weight (numpy.ndarray or None): Weights for the cliques. Default is None, meaning equal weights.
    
    Returns:
    numpy.ndarray: Weighted score for each clique.
    '''
    if weight is None:
        weight = np.mean(clique, axis=0)  # Default weight is the mean of the clique
    weight_triu = weight[np.triu_indices_from(weight, k=1)]  # Get upper triangular part of weight matrix
    
    clique_triu = np.array([(_)[np.triu_indices_from(_, k=1)] for _ in clique])  # Get upper triangular part of each clique
    cliuqe_diag = np.array([(_).diagonal() for _ in clique])  # Get diagonal of each clique
    
    sub_diag_score = scale(np.sum(cliuqe_diag, axis=1))  # Scale the sum of diagonals
    sub_triu_score = clique_triu * (weight_triu)  # Calculate weighted upper triangular scores
    sub_triu_score_final = scale(np.sum(sub_triu_score, axis=1))  # Scale the sum of weighted upper triangular scores
    dts_score = sub_diag_score + sub_triu_score_final  # Combine diagonal and upper triangular scores
    
    return dts_score