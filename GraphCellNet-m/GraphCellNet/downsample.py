import numba
import numpy as np
import multiprocessing as mp
from functools import partial
import random


@numba.jit(nopython=True, parallel=True)
def get_bin_edges(a, bins):
    """
    Compute evenly spaced bin edges for a given array.
    
    Parameters
    ----------
    a : numpy.ndarray
        Input array to compute bin edges for.
    bins : int
        Number of bins to create.
        
    Returns
    -------
    numpy.ndarray
        Array of bin edges with shape (bins+1,).
    
    Notes
    -----
    This function computes bin edges that span the range from the minimum to the
    maximum value in the input array. The bins are equally spaced.
    """
    bin_edges = np.zeros((bins+1,), dtype=np.float32)
    a_min = a.min()
    a_max = a.max()
    delta = (a_max - a_min) / bins
    for i in numba.prange(bin_edges.shape[0]):
        bin_edges[i] = a_min + i * delta

    bin_edges[-1] = a_max  
    return bin_edges

@numba.jit(nopython=True, parallel=False)
def compute_bin(x, bin_edges):
    """
    Determine which bin a value falls into based on bin edges.
    
    Parameters
    ----------
    x : float
        Value to find bin for.
    bin_edges : numpy.ndarray
        Array of bin edges.
        
    Returns
    -------
    int or None
        Index of the bin that x falls into, or None if x is outside all bins.
    
    Notes
    -----
    The maximum value of the array is always assigned to the last bin.
    """
    n = bin_edges.shape[0] - 1
    a_max = bin_edges[-1]
    if x == a_max:
        return n - 1 # a_max always in last bin
    bin = np.searchsorted(bin_edges, x)-1
    if bin < 0 or bin >= n:
        return None
    else:
        return bin

@numba.jit(nopython=True, parallel=False)
def numba_histogram(a, bin_edges):
    """
    Compute histogram of a dataset using specified bin edges.
    
    This is a Numba-optimized implementation of numpy's histogram function.
    
    Parameters
    ----------
    a : numpy.ndarray
        Input data to compute the histogram for.
    bin_edges : numpy.ndarray
        Array of bin edges, including the rightmost edge.
        
    Returns
    -------
    hist : numpy.ndarray
        The histogram counts.
    bin_edges : numpy.ndarray
        The array of bin edges.
    
    Notes
    -----
    This function counts how many values from input array 'a' fall into each bin
    defined by 'bin_edges'. Values outside the range of bin_edges are ignored.
    """
    hist = np.zeros((bin_edges.shape[0] - 1,), dtype=np.intp)
    for x in a.flat:
        bin = compute_bin(x, bin_edges)
        if bin is not None:
            hist[int(bin)] += 1
    return hist, bin_edges

@numba.jit(nopython=True, parallel=True)
def downsample_cell(cell_counts,fraction):
    """
    Downsample a cell's expression counts by randomly selecting a fraction of reads.
    
    Parameters
    ----------
    cell_counts : numpy.ndarray
        Array of gene expression counts for a cell.
    fraction : float
        Fraction of total counts to retain, between 0 and 1.
        
    Returns
    -------
    numpy.ndarray
        Downsampled counts array of the same shape as the input.
    
    Notes
    -----
    This function:
    1. Calculates the total number of reads to keep based on the fraction
    2. Randomly selects that number of reads from the total counts
    3. Redistributes the selected reads across genes according to the original distribution
    
    The downsampling is performed without replacement.
    """
    n = np.floor(np.sum(cell_counts) * fraction)
    readsGet = np.sort(np.random.choice(np.arange(np.sum(cell_counts)), np.intp(n), replace=False))
    cumCounts = np.concatenate((np.array([0]),np.cumsum(cell_counts)))
    counts_new = numba_histogram(readsGet,cumCounts)[0]
    counts_new = counts_new.astype(np.float32)
    return counts_new

def downsample_cell_python(cell_counts,fraction):
    """
    Python implementation of downsample_cell without Numba optimization.
    
    Parameters
    ----------
    cell_counts : numpy.ndarray
        Array of gene expression counts for a cell.
    fraction : float
        Fraction of total counts to retain, between 0 and 1.
        
    Returns
    -------
    numpy.ndarray
        Downsampled counts array of the same shape as the input.
    
    Notes
    -----
    This function provides the same functionality as downsample_cell but uses
    Python's random.sample instead of numpy's random.choice, which may be preferred
    in certain contexts where Numba optimization is not available.
    """
    n = np.floor(np.sum(cell_counts) * fraction)
    readsGet = np.sort(random.sample(range(np.intp(np.sum(cell_counts))), np.intp(n)))
    cumCounts = np.concatenate((np.array([0]),np.cumsum(cell_counts)))
    counts_new = numba_histogram(readsGet,cumCounts)[0]
    counts_new = counts_new.astype(np.float32)
    return counts_new

@numba.jit(nopython=True, parallel=True)
def downsample_per_cell(cell_counts,new_cell_counts):
    """
    Downsample a cell's counts to a specific number of total counts.
    
    Parameters
    ----------
    cell_counts : numpy.ndarray
        Array of gene expression counts for a cell.
    new_cell_counts : int
        Target number of total counts after downsampling.
        
    Returns
    -------
    numpy.ndarray
        Downsampled counts array with approximately new_cell_counts total counts.
    
    Notes
    -----
    If the target count is greater than or equal to the current total count,
    the original counts are returned unchanged. Otherwise, random downsampling
    is performed to reach the target count.
    """
    n = new_cell_counts
    if n < np.sum(cell_counts):
        readsGet = np.sort(np.random.choice(np.arange(np.sum(cell_counts)), np.intp(n), replace=False))
        cumCounts = np.concatenate((np.array([0]),np.cumsum(cell_counts)))
        counts_new = numba_histogram(readsGet,cumCounts)[0]
        counts_new = counts_new.astype(np.float32)
        return counts_new
    else:
        return cell_counts.astype(np.float32)

def downsample_per_cell_python(param):
    """
    Python implementation of downsample_per_cell designed for multiprocessing compatibility.
    
    Parameters
    ----------
    param : tuple
        Tuple containing (cell_counts, new_cell_counts) where:
        - cell_counts: array of gene expression counts for a cell
        - new_cell_counts: target number of total counts after downsampling
        
    Returns
    -------
    numpy.ndarray
        Downsampled counts array with approximately new_cell_counts total counts.
    
    Notes
    -----
    This function is specifically formatted to work with multiprocessing.Pool.map(),
    which requires a single parameter that is then unpacked inside the function.
    """
    cell_counts,new_cell_counts = param[0],param[1]
    n = new_cell_counts
    if n < np.sum(cell_counts):
        readsGet = np.sort(random.sample(range(np.intp(np.sum(cell_counts))), np.intp(n)))
        cumCounts = np.concatenate((np.array([0]),np.cumsum(cell_counts)))
        counts_new = numba_histogram(readsGet,cumCounts)[0]
        counts_new = counts_new.astype(np.float32)
        return counts_new
    else:
        return cell_counts.astype(np.float32)

def downsample_matrix_by_cell(matrix,per_cell_counts,n_cpus=None,numba_end=True):
     """
    Downsample each cell in a matrix to a specified number of counts.
    
    Parameters
    ----------
    matrix : numpy.ndarray
        2D array of gene expression counts with shape (n_cells, n_genes).
    per_cell_counts : numpy.ndarray
        1D array of target counts for each cell with length n_cells.
    n_cpus : int, optional
        Number of CPU cores to use for parallel processing. If None, no multiprocessing
        is used. Default: None.
    numba_end : bool, optional
        Whether to use the Numba-optimized version (True) or Python version (False)
        of the downsampling function. Default: True.
        
    Returns
    -------
    numpy.ndarray
        Downsampled matrix with the same shape as the input.
    
    Notes
    -----
    This function downsamples each cell (row) in the matrix to have approximately
    the corresponding number of counts specified in per_cell_counts. It can leverage
    multiprocessing for faster execution on large matrices.
    """
    if numba_end:
        downsample_func = downsample_per_cell
    else:
        downsample_func = downsample_per_cell_python
    if n_cpus is not None:
        with mp.Pool(n_cpus) as p:
            matrix_ds = p.map(downsample_func, zip(matrix,per_cell_counts))
    else:
        matrix_ds = [downsample_func(c,per_cell_counts[i]) for i,c in enumerate(matrix)]
    return np.array(matrix_ds)

def downsample_matrix_total(matrix,fraction):
    matrix_flat = matrix.reshape(-1)
    matrix_flat_ds = downsample_cell(matrix_flat,fraction)
    matrix_ds = matrix_flat_ds.reshape(matrix.shape)
    return matrix_ds

