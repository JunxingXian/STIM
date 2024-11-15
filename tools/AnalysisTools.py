import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.api as sm
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import itertools
from scipy.optimize import linear_sum_assignment
import os
import re

def pc1_comp(mtx):
    """
    Perform PCA on the input matrix and return the first principal component and its coefficients.

    Parameters:
    mtx (numpy.ndarray): The input matrix.

    Returns:
    tuple: The first principal component (as a 1D array) and its corresponding coefficients.
    """
    pca = PCA(n_components=1)
    pca_res = pca.fit_transform(mtx)[:, 0]  # Extract the first principal component
    components = pca.components_[0, :]  # Get the coefficients for the first component

    # Ensure that the sum of components is positive (for consistency)
    if components.sum() < 0:
        components = -components
        pca_res = -pca_res
    
    return pca_res, components

def g_residuals(x, y):
    """
    Compute the residuals of y after regressing out x.

    Parameters:
    x (numpy.ndarray): The independent variable (e.g., covariates like age, gender).
    y (numpy.ndarray): The dependent variable (e.g., feature scores).

    Returns:
    numpy.ndarray: The residuals of y after removing the influence of x.
    """
    ols = sm.OLS(y, x)  # Ordinary least squares regression
    models = ols.fit()  # Fit the model
    y_p = models.predict(x)  # Predict y based on x
    return y - y_p  # Return the residuals

def df_pcorr(df, label_x, label_y, covar_ls=['Age', 'Sex']):
    """
    Compute partial correlation between two variables, controlling for covariates.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    label_x (str): The name of the independent variable.
    label_y (str): The name of the dependent variable.
    covar_ls (list): List of covariate names to control for.

    Returns:
    pandas.DataFrame: The result of the partial correlation analysis.
    """
    n_subs = df.shape[0]  # Number of subjects
    dict_orig = {}
    for var_i in [label_x, label_y] + covar_ls:
        dict_orig[var_i] = np.array(df[var_i].values).astype(float)  # Extract and convert variables to float
    
    nan_ind = np.zeros(n_subs)
    for var_i in dict_orig.keys():
        nan_ind += (np.isnan(dict_orig[var_i])).astype(int)  # Identify rows with missing values
    
    valid_ind = ~(nan_ind.astype(bool))  # Create a boolean mask for valid (non-NaN) rows
    
    dict_new = {}
    for var_i in dict_orig.keys():
        dict_new[var_i] = dict_orig[var_i][valid_ind]  # Filter out rows with NaN values

    df_partial = pd.DataFrame(dict_new)  # Create a DataFrame with valid data
    res = pg.partial_corr(data=df_partial, x=label_x, y=label_y, covar=covar_ls, method='spearman')  # Perform partial correlation
    return res

def compute_r_all_behav(df_all_agesort, follow_score, behav_ls=None, covar_ls=['Sex', 'Age']):
    """
    Compute correlations between follow-up scores and various behavioral measures.

    Parameters:
    df_all_agesort (pandas.DataFrame): DataFrame containing all subject data sorted by age.
    follow_score (numpy.ndarray): The follow-up scores for the subjects.
    behav_ls (list or None): List of behavioral measures to consider. If None, all available measures are used.
    covar_ls (list): List of covariate names to control for in the correlation analysis.

    Returns:
    pandas.DataFrame: DataFrame containing the correlation results for each behavioral measure.
    """
    df = pd.DataFrame({'EID': list(df_all_agesort['EID']), 'follow_score': follow_score}).merge(df_all_agesort, on='EID')  # Merge follow scores with subject data
    path = '/share/user_data/xianjunxing/HCP_7T_movie/mapper/graph/hbn_behavior/behavior_score/'  # Path to behavioral data
    
    if behav_ls is None:
        behav_ls = os.listdir(path)
        behav_ls.remove('.ipynb_checkpoints')  # Remove unwanted files
        behav_ls_new = [i.split('.')[0] for i in behav_ls]  # Clean up file names
    else:
        behav_ls_new = behav_ls
    
    ban_ls = ['Site', 'Year', 'Baseline']  # Banned keywords to exclude certain measures
    tail_ls = ['Valid', 'Complete']  # Exclude measures with these suffixes
    
    behav_corr_res = {'behav': [], 'r': [], 'p-val': [], 'n': []}  # Dictionary to store results
    for behav in behav_ls_new:
        df_behav = pd.read_csv(path + behav + '.csv')
        if (behav + ',EID') not in df_behav.columns.to_list():
            continue
        df_behav['EID'] = df_behav[behav + ',EID']
        df_cog = df.merge(df_behav, on='EID')  # Merge cognitive data with follow scores
        label_ls = df_cog.columns.tolist()
        for label in label_ls:
            if len(re.findall('\d', label)) != 0:  # Skip labels containing digits
                continue
            if np.sum([_ in label for _ in ban_ls]) > 0:  # Skip labels containing banned keywords
                continue
            if label.split('_')[-1] in tail_ls:  # Skip labels ending with excluded suffixes
                continue
            try:
                behav_values = df_cog[label].values.astype(float)  # Convert to float
                if np.isnan(behav_values).any():  # Handle missing values
                    behav_values[np.isnan(behav_values)] = np.nanmean(behav_values)
                    df_cog[label] = behav_values
                    
                res = df_pcorr(df_cog, 'follow_score', label, covar_ls=covar_ls)  # Compute partial correlation
                behav_corr_res['behav'].append(label)  # Store the behavioral measure
                behav_corr_res['r'].append(res['r'][0])  # Store the correlation coefficient
                behav_corr_res['p-val'].append(res['p-val'][0])  # Store the p-value
                behav_corr_res['n'].append(res['n'][0])  # Store the sample size
            except:
                None
    gloabl_res_df = pd.DataFrame(behav_corr_res)  # Convert results to a DataFrame
    
    return gloabl_res_df  # Return the correlation results

def fast_min_match(data1, data2):
    """
    Perform a fast minimum-cost matching between two datasets.

    Parameters:
    data1 (list or numpy.ndarray): The first dataset.
    data2 (list or numpy.ndarray): The second dataset.

    Returns:
    tuple: The average difference between matched pairs and the list of matched pairs.
    """
    # Determine the smaller and larger datasets
    smaller_data = [data1, data2][np.argsort([len(data1), len(data2)])[0]]
    larger_data = [data1, data2][np.argsort([len(data1), len(data2)])[1]]
    
    # Create a cost matrix where each element represents the absolute difference between data points
    cost_matrix = np.zeros((len(larger_data), len(smaller_data)))
    for i in range(len(larger_data)):
        for j in range(len(smaller_data)):
            cost_matrix[i][j] = abs(larger_data[i] - smaller_data[j])

    # Use linear sum assignment to find the optimal matching with minimum cost
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Output the best matching pairs
    best_match = [(larger_data[i], smaller_data[j]) for i, j in zip(row_ind, col_ind)]
    diff = np.abs(np.array(best_match)[:, 0] - np.array(best_match)[:, 1]).mean()  # Compute the average difference between matched pairs
    return diff, best_match  # Return the average difference and the best match pairs

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

def p_rank_star(pval_arr): 
    threshold = [1,0.05,0.01,0.001,0.0001,-1]
    rank = ['','*','**','***','****']
    p_rank = np.zeros_like(pval_arr)
    for i in range(len(threshold)-1):
        a,b = threshold[i],threshold[i+1]
        p_rank=np.where(((pval_arr<a)&(pval_arr>=b)),rank[i],p_rank)
    return p_rank