import numpy as np
from scipy.sparse import coo_matrix, block_diag
import torch
import torch.nn.functional as F
from .pygcn_utils import *

def get_graph_inputs(celltype_ad_list):
    """
    Generate input data for graph neural network (GNN) from a list of AnnData objects.
    
    Parameters
    ----------
    celltype_ad_list : list
        List of AnnData objects, each containing expression data and spatial information
        for a different tissue section or dataset.
        
    Returns
    -------
    X_raw : numpy.ndarray
        Combined gene expression matrix of shape (n_total_spots, n_genes).
    A : scipy.sparse.coo_matrix
        Block diagonal adjacency matrix combining all spatial graphs.
    nb_mask : numpy.ndarray
        Indices of non-zero elements in the adjacency matrix, representing connected spots.
    slice_class_onehot : torch.Tensor
        One-hot encoded tensor indicating which dataset/slice each spot belongs to.
        
    Notes
    -----
    This function:
    1. Extracts gene expression data (X) from each AnnData object
    2. Extracts spatial connectivity information (A) from each AnnData's obsp['spatial_distances']
    3. Combines individual adjacency matrices into a block diagonal matrix
    4. Creates a one-hot encoding to track which dataset/slice each spot belongs to
    
    The resulting data can be directly used as input for graph neural network models
    that operate on spatial transcriptomics data.
    """
    print('Generating GNN inputs...')
    A_list = []
    X_list = []
    for celltype_ad in celltype_ad_list:
        X_tmp = np.matrix(celltype_ad.X,dtype='float32')
        X_list.append(X_tmp)
        A_list.append(coo_matrix(celltype_ad.obsp['spatial_distances'],dtype='float32'))

    X_raw = np.concatenate(X_list)
    class_index = 0
    slice_class = []
    for A_tmp in A_list:
        slice_class = slice_class + [class_index]*A_tmp.shape[0]
        class_index += 1
    A = block_diag(A_list)
    nb_mask = np.argwhere(A > 0).T
    slice_class_onehot = F.one_hot(torch.tensor(slice_class)).float()
    return X_raw,A,nb_mask,slice_class_onehot
from scipy.sparse.linalg import eigsh
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

import numpy as np
import torch
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import eigsh

import numpy as np
import torch

def get_graph_kernel(features, adj, k=2):
     """
    Generate graph convolutional kernels using Chebyshev polynomials.
    
    This function creates the necessary components for spectral graph convolution
    by computing Chebyshev polynomial approximations of the graph Laplacian.
    
    Parameters
    ----------
    features : numpy.ndarray
        Feature matrix of shape (n_nodes, n_features).
    adj : scipy.sparse.coo_matrix
        Adjacency matrix representing the graph structure.
    k : int, optional
        Order of the Chebyshev polynomial. Higher values capture larger
        graph neighborhoods but increase computational complexity. Default: 2.
        
    Returns
    -------
    features_scaled : torch.Tensor
        Normalized feature matrix with mean 0 and standard deviation 1.
    graph : list
        List containing the scaled features and Chebyshev polynomial transforms
        of the graph Laplacian, where graph[0] is the scaled features and
        graph[1:] are the sparse tensor representations of the polynomials.
    support : int
        Number of polynomial terms (k+1), representing the filter support size.
        
    Notes
    -----
    This implementation follows the approach described in the paper:
    "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering"
    by Defferrard et al.
    
    The function:
    1. Normalizes features to have zero mean and unit variance
    2. Computes the normalized graph Laplacian
    3. Rescales the Laplacian to the range [-1, 1]
    4. Computes Chebyshev polynomials up to order k
    5. Converts the polynomials to PyTorch sparse tensors
    
    These components are used in graph convolutional networks to perform
    localized spectral filtering.
    """
    features_scaled = (features - features.mean(0)) / features.std(0)
    features_scaled = torch.tensor(features_scaled)
    
    SYM_NORM = False  
    L = normalized_laplacian(adj, SYM_NORM) 
    L_scaled = rescale_laplacian(L) 
    T_k = chebyshev_polynomial(L_scaled, k)  
    support = k + 1  

    
    graph = [features_scaled] + T_k
    for _i in range(1, len(graph)):
        graph[_i] = sparse_mx_to_torch_sparse_tensor(graph[_i])
    
    return features_scaled, graph, support




def split_train_test_idx(X,train_prop):
    """
    Split data indices into training and testing sets.
    
    Parameters
    ----------
    X : numpy.ndarray
        Data matrix of shape (n_samples, n_features).
    train_prop : float
        Proportion of data to use for training, between 0 and 1.
        
    Returns
    -------
    train_idx : numpy.ndarray
        Indices for training data.
    test_idx : numpy.ndarray
        Indices for testing data.
        
    Notes
    -----
    This function randomly permutes the indices of the data and then splits them
    according to the specified proportion. The randomization ensures that there
    is no selection bias in the train/test split.
    
    Example
    -------
    >>> X = np.random.rand(100, 10)
    >>> train_idx, test_idx = split_train_test_idx(X, 0.8)
    >>> X_train, X_test = X[train_idx], X[test_idx]
    """
    rand_idx = np.random.permutation(X.shape[0])
    train_idx = rand_idx[:int(len(rand_idx)*train_prop)]
    test_idx = rand_idx[int(len(rand_idx)*train_prop):]
    return train_idx, test_idx