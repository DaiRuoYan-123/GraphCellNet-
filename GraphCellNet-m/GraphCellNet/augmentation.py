import numba
import numpy as np

@numba.njit
def random_dropout(cell_expr,max_rate):
    """
    Randomly drop out non-zero gene expression values in a cell.
    
    Parameters
    ----------
    cell_expr : numpy.ndarray
        1D array of gene expression values for a single cell.
    max_rate : float
        Maximum dropout rate between 0 and 1. The actual dropout rate will be
        randomly selected between 0 and this value.
        
    Returns
    -------
    numpy.ndarray
        Modified cell expression array with randomly dropped out values.
    
    Notes
    -----
    This function only drops out non-zero expression values. The dropout rate is
    randomly selected between 0 and max_rate for each call.
    """
    non_zero_mask = np.where(cell_expr!=0)[0]
    zero_mask = np.random.choice(non_zero_mask,int(len(non_zero_mask)*np.float32(np.random.uniform(0,max_rate))))
    cell_expr[zero_mask] = 0
    return cell_expr

@numba.njit
def random_scale(cell_expr,max_val):
     """
    Randomly scale gene expression values in a cell by a factor.
    
    Parameters
    ----------
    cell_expr : numpy.ndarray
        1D array of gene expression values for a single cell.
    max_val : float
        Maximum scaling deviation. The actual scaling factor will be randomly 
        selected between (1-max_val) and (1+max_val).
        
    Returns
    -------
    numpy.ndarray
        Modified cell expression array with values scaled by a random factor.
    
    Notes
    -----
    This function applies the same scaling factor to all expression values in the cell.
    A max_val of 0.8 means the expression values could be scaled between 0.2x and 1.8x.
    """
    scale_factor = np.float32(1+np.random.uniform(-max_val,max_val))
    cell_expr = cell_expr*scale_factor
    return cell_expr

@numba.njit
def random_shift(cell_expr,kth):
    """
    Randomly shift non-zero gene expression values in a cell by adding/subtracting a value.
    
    Parameters
    ----------
    cell_expr : numpy.ndarray
        1D array of gene expression values for a single cell.
    kth : float
        A value between 0 and 1 that determines the maximum percentile of unique 
        expression values to consider for the shift magnitude.
        
    Returns
    -------
    numpy.ndarray
        Modified cell expression array with non-zero values shifted by a random amount.
    
    Notes
    -----
    This function:
    1. Randomly selects a direction (add, subtract, or no change)
    2. Randomly selects a shift value from the unique expression values
    3. Adds this shift to all non-zero expression values
    4. Clips negative values to zero
    """
    shift_value = np.random.choice(np.array([1,0,-1]),1)[0]*np.unique(cell_expr)[int(np.random.uniform(0,kth)*len(np.unique(cell_expr)))]
    cell_expr[cell_expr != 0] = cell_expr[cell_expr != 0]+shift_value
    cell_expr[cell_expr < 0] = 0
    return cell_expr

@numba.njit(parallel=True)
def random_augment(mtx,max_rate=0.8,max_val=0.8,kth=0.2):
    """
    Apply random augmentation to a batch of cells in parallel.
    
    This function applies dropout, scaling, and shifting augmentations to each cell
    in the provided matrix.
    
    Parameters
    ----------
    mtx : numpy.ndarray
        2D array of gene expression values with shape (n_cells, n_genes).
    max_rate : float, optional
        Maximum dropout rate. Default: 0.8.
    max_val : float, optional
        Maximum scaling deviation. Default: 0.8.
    kth : float, optional
        Maximum percentile for shift value selection. Default: 0.2.
        
    Returns
    -------
    numpy.ndarray
        Augmented expression matrix with the same shape as the input.
    
    Notes
    -----
    This function leverages Numba's parallel processing to efficiently augment
    large expression matrices. Each cell is processed independently.
    """
    for i in numba.prange(mtx.shape[0]):
        random_dropout(mtx[i,:],max_rate=max_rate)
        random_scale(mtx[i,:],max_val=max_val)
        random_shift(mtx[i,:],kth=kth)
    return mtx

@numba.njit
def random_augmentation_cell(cell_expr,max_rate=0.8,max_val=0.8,kth=0.2):
     """
    Apply a sequence of random augmentations to a single cell's expression vector.
    
    This function combines dropout, scaling, and shifting augmentations in sequence
    for a single cell.
    
    Parameters
    ----------
    cell_expr : numpy.ndarray
        1D array of gene expression values for a single cell.
    max_rate : float, optional
        Maximum dropout rate between 0 and 1. Default: 0.8.
    max_val : float, optional
        Maximum scaling deviation. Default: 0.8.
    kth : float, optional
        Maximum percentile for shift value selection. Default: 0.2.
        
    Returns
    -------
    numpy.ndarray
        Augmented cell expression array.
    
    Notes
    -----
    This function applies the augmentations in sequence:
    1. Random dropout
    2. Random scaling
    3. Random shifting
    
    The order matters as each transformation affects the input to the next.
    """
    cell_expr = random_dropout(cell_expr,max_rate=max_rate)
    cell_expr = random_scale(cell_expr,max_val=max_val)
    cell_expr = random_shift(cell_expr,kth=kth)
    return cell_expr