import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
import torch

def normalize_adj(adj, symmetric=True):
    """
    Normalize the adjacency matrix of a graph.
    
    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        Adjacency matrix of the graph.
    symmetric : bool, optional
        If True, performs symmetric normalization: D^(-1/2) * A * D^(-1/2)
        If False, performs row-wise normalization: D^(-1) * A
        Default: True.
        
    Returns
    -------
    scipy.sparse.csr_matrix
        Normalized adjacency matrix.
        
    Notes
    -----
    The symmetric normalization is typically used for undirected graphs, while
    the row-wise normalization is more suitable for directed graphs.
    
    For symmetric normalization, each element A_ij is normalized by the square root
    of the product of the degrees of nodes i and j: A_ij / sqrt(deg(i) * deg(j))
    
    For row-wise normalization, each element A_ij is normalized by the degree
    of node i: A_ij / deg(i)
    """
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def normalized_laplacian(adj, symmetric=True):
    """
    Compute the normalized Laplacian matrix of a graph.
    
    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        Adjacency matrix of the graph.
    symmetric : bool, optional
        Whether to use symmetric normalization for the adjacency matrix.
        Default: True.
        
    Returns
    -------
    scipy.sparse.csr_matrix
        Normalized Laplacian matrix L = I - D^(-1/2) * A * D^(-1/2) (symmetric)
        or L = I - D^(-1) * A (non-symmetric).
        
    Notes
    -----
    The normalized Laplacian matrix encodes the graph structure and is used in
    spectral graph theory, graph signal processing, and graph neural networks.
    
    Its eigenvalues provide information about the graph's connectivity and
    community structure, and its eigenvectors are used for spectral clustering
    and dimensionality reduction.
    """
    adj_normalized = normalize_adj(adj, symmetric)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian


def rescale_laplacian(laplacian):
    """
    Rescale the Laplacian matrix to the range [-1, 1].
    
    Parameters
    ----------
    laplacian : scipy.sparse.csr_matrix
        Normalized Laplacian matrix of a graph.
        
    Returns
    -------
    scipy.sparse.csr_matrix
        Rescaled Laplacian matrix with eigenvalues in the range [-1, 1].
        
    Notes
    -----
    This function rescales the Laplacian to ensure its eigenvalues lie in
    the range [-1, 1], which is required for the stable computation of
    Chebyshev polynomials.
    
    The rescaling is done by computing the largest eigenvalue λ_max of the
    Laplacian, then rescaling as: (2/λ_max) * L - I
    
    If eigenvalue computation fails to converge, λ_max=2 is used as a fallback
    since the eigenvalues of a normalized Laplacian are bounded by 2.
    """
    try:
        print('Calculating largest eigenvalue of normalized graph Laplacian...')
        largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    except ArpackNoConvergence:
        print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2

    scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian


def chebyshev_polynomial(X, k):
    """
    Calculate Chebyshev polynomials of the rescaled Laplacian up to order k.
    
    Parameters
    ----------
    X : scipy.sparse.csr_matrix
        Rescaled Laplacian matrix of a graph.
    k : int
        Maximum order of Chebyshev polynomials to compute.
        
    Returns
    -------
    list of scipy.sparse.csr_matrix
        List of sparse matrices representing Chebyshev polynomials of the
        rescaled Laplacian from order 0 to k.
        
    Notes
    -----
    Chebyshev polynomials are defined recursively as:
    T_0(x) = 1
    T_1(x) = x
    T_n(x) = 2x T_{n-1}(x) - T_{n-2}(x)
    
    These polynomials form a basis for graph convolution operations, allowing
    for efficient k-localized spectral filtering of graph signals. This is
    particularly useful in graph convolutional networks (GCNs) as they provide
    a way to perform convolutions on irregular graph structures.
    
    The parameter k determines the size of the receptive field in the graph
    (i.e., how many hops away information can flow). A larger k allows the
    model to capture more global structures but increases computational cost.
    """
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        """
        Recurrence relation for computing Chebyshev polynomials.
        
        T_k(x) = 2x * T_{k-1}(x) - T_{k-2}(x)
        """
        X_ = sp.csr_matrix(X, copy=True)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k+1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    return T_k


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    
    Parameters
    ----------
    sparse_mx : scipy.sparse.csr_matrix
        Scipy sparse matrix to convert.
        
    Returns
    -------
    torch.sparse.FloatTensor
        PyTorch sparse tensor representation of the input matrix.
        
    Notes
    -----
    This function is essential for using scipy sparse matrices (commonly used
    for graph operations) in PyTorch-based graph neural networks.
    
    The conversion process involves:
    1. Converting to COO format (coordinate format) for easy extraction of indices
    2. Extracting row and column indices to form the indices tensor
    3. Extracting data values
    4. Creating a PyTorch sparse tensor with these components
    
    The resulting tensor can be used in PyTorch operations that support sparse tensors,
    such as the sparse matrix multiplication operations in graph neural networks.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)