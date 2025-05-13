import numpy as np
import pandas as pd
import numba as nb
from numba import jit

import random
from .downsample import downsample_cell,downsample_matrix_by_cell
from .augmentation import random_augmentation_cell


def count_cell_counts(cell_counts):
    """
    Summarize the distribution of cell counts per spot.
    
    Parameters
    ----------
    cell_counts : pandas.Series or dict
        Cell counts per spot.
        
    Returns
    -------
    pandas.DataFrame
        Filtered distribution of cell counts, including only counts that 
        represent 99% of the total observations.
        
    Notes
    -----
    This function creates a histogram of cell counts, then filters out
    extreme values to focus on the most common counts.
    """
    cell_counts = np.array(cell_counts.values,dtype=int).reshape(-1)
    counts_list = np.array(np.histogram(cell_counts,range=[0,np.max(cell_counts)+1],bins=np.max(cell_counts)+1)[0],dtype=int)
    counts_index = np.array((np.histogram(cell_counts,range=[0,np.max(cell_counts)+1],bins=np.max(cell_counts)+1)[1][:-1]),dtype=int)
    counts_df = pd.DataFrame(counts_list,index=counts_index,columns=['count'],dtype=np.int32)
    counts_df = counts_df[(counts_df['count'] != 0) & (counts_df.index != 0)]
    count_sum = 0
    for i in np.arange(len(counts_df)):
        count_sum += counts_df.iloc[i].values
        if count_sum > counts_df.values.sum()*0.99:
            counts_df_filtered = counts_df.iloc[:i+1,:]
            break
    return counts_df_filtered


@nb.njit
def numba_set_seed(seed):
    """
    Set random seeds for both NumPy and Python's random module in Numba context.
    
    Parameters
    ----------
    seed : int
        Seed value for random number generators.
    """
    np.random.seed(seed)
    random.seed(seed)

@nb.njit
def np_apply_along_axis(func1d, axis, arr):
    """
    Numba-optimized version of numpy's apply_along_axis function.
    
    Parameters
    ----------
    func1d : function
        Function to apply to each 1D slice.
    axis : int
        Axis along which to apply the function (0 or 1).
    arr : numpy.ndarray
        2D input array.
        
    Returns
    -------
    numpy.ndarray
        Result of applying func1d along the specified axis.
        
    Notes
    -----
    This function only works with 2D arrays and axis values of 0 or 1.
    """
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1], dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0], dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


@nb.njit
def np_mean(array, axis):
    """
    Numba-optimized version of numpy's mean function for 2D arrays.
    
    Parameters
    ----------
    array : numpy.ndarray
        2D input array.
    axis : int
        Axis along which to compute the mean (0 or 1).
        
    Returns
    -------
    numpy.ndarray
        Mean values along the specified axis.
    """
    return np_apply_along_axis(np.mean, axis, array)

@nb.njit
def np_sum(array, axis):
    """
    Numba-optimized version of numpy's sum function for 2D arrays.
    
    Parameters
    ----------
    array : numpy.ndarray
        2D input array.
    axis : int
        Axis along which to compute the sum (0 or 1).
        
    Returns
    -------
    numpy.ndarray
        Sum values along the specified axis.
    """
    return np_apply_along_axis(np.sum, axis, array)

@jit(nopython=True,parallel=True)
def sample_cell(param_list,cluster_p,clusters,cluster_id,sample_exp,sample_cluster,cell_p_balanced,downsample_fraction=None,data_augmentation=True,max_rate=0.8,max_val=0.8,kth=0.2):
    """
    Generate simulated spots by sampling cells from single-cell data.
    
    Parameters
    ----------
    param_list : numpy.ndarray
        Array of parameters for each spot, where each row contains:
        [number_of_cells, number_of_clusters].
    cluster_p : numpy.ndarray
        Probability of selecting each cluster.
    clusters : numpy.ndarray
        Array of cluster identifiers.
    cluster_id : numpy.ndarray
        Cluster ID for each cell in the sample.
    sample_exp : numpy.ndarray
        Expression matrix of shape (n_cells, n_genes).
    sample_cluster : numpy.ndarray
        One-hot encoded cluster assignments of shape (n_clusters, n_clusters).
    cell_p_balanced : numpy.ndarray
        Cell sampling probability after balancing.
    downsample_fraction : float, optional
        Fraction of counts to retain after downsampling. Default: None (no downsampling).
    data_augmentation : bool, optional
        Whether to apply data augmentation. Default: True.
    max_rate : float, optional
        Maximum dropout rate for data augmentation. Default: 0.8.
    max_val : float, optional
        Maximum scaling factor for data augmentation. Default: 0.8.
    kth : float, optional
        Maximum percentile for shift value in data augmentation. Default: 0.2.
        
    Returns
    -------
    exp : numpy.ndarray
        Simulated expression data of shape (n_spots, n_genes).
    density : numpy.ndarray
        True cell type proportions of shape (n_spots, n_clusters).
        
    Notes
    -----
    This function implements a simulation approach where:
    1. For each spot, a number of clusters are randomly selected
    2. Cells from these clusters are randomly sampled
    3. Their expression profiles are combined (summed)
    4. Optional data augmentation and downsampling are applied
    
    The function is parallelized with Numba for performance.
    """
    exp = np.empty((len(param_list), sample_exp.shape[1]),dtype=np.float32)
    density = np.empty((len(param_list), sample_cluster.shape[1]),dtype=np.float32)

    for i in nb.prange(len(param_list)):
        params = param_list[i]
        num_cell = params[0]
        num_cluster = params[1]
        used_clusters = clusters[np.searchsorted(np.cumsum(cluster_p), np.random.rand(num_cluster), side="right")]
        cluster_mask = np.array([False]*len(cluster_id))
        for c in used_clusters:
            cluster_mask = (cluster_id==c)|(cluster_mask)
        # print('cluster_mask',cluster_mask)
        # print('used_clusters',used_clusters)
        used_cell_ind = np.where(cluster_mask)[0]
        used_cell_p = cell_p_balanced[cluster_mask]
        used_cell_p = used_cell_p/used_cell_p.sum()
        sampled_cells = used_cell_ind[np.searchsorted(np.cumsum(used_cell_p), np.random.rand(num_cell), side="right")]
        combined_exp = np_sum(sample_exp[sampled_cells,:],axis=0).astype(np.float32)
        if data_augmentation:
            combined_exp = random_augmentation_cell(combined_exp,max_rate=max_rate,max_val=max_val,kth=kth)
        if downsample_fraction is not None:
            combined_exp = downsample_cell(combined_exp, downsample_fraction)
        combined_clusters = np_sum(sample_cluster[cluster_id[sampled_cells]],axis=0).astype(np.float32)
        exp[i,:] = combined_exp
        density[i,:] = combined_clusters
    return exp,density

@jit(nopython=True,parallel=True)
def sample_cell_from_clusters(cluster_sample_list,ncell_sample_list,cluster_id,sample_exp,sample_cluster,cell_p_balanced,downsample_fraction=None,data_augmentation=True,max_rate=0.8,max_val=0.8,kth=0.2):
    """
    Generate simulated spots by sampling cells from specific clusters.
    
    Parameters
    ----------
    cluster_sample_list : numpy.ndarray
        Binary array indicating which clusters to sample from for each spot.
        Shape is (n_spots, n_clusters).
    ncell_sample_list : numpy.ndarray
        Number of cells to sample for each spot.
    cluster_id : numpy.ndarray
        Cluster ID for each cell in the sample.
    sample_exp : numpy.ndarray
        Expression matrix of shape (n_cells, n_genes).
    sample_cluster : numpy.ndarray
        One-hot encoded cluster assignments of shape (n_clusters, n_clusters).
    cell_p_balanced : numpy.ndarray
        Cell sampling probability after balancing.
    downsample_fraction : float, optional
        Fraction of counts to retain after downsampling. Default: None (no downsampling).
    data_augmentation : bool, optional
        Whether to apply data augmentation. Default: True.
    max_rate : float, optional
        Maximum dropout rate for data augmentation. Default: 0.8.
    max_val : float, optional
        Maximum scaling factor for data augmentation. Default: 0.8.
    kth : float, optional
        Maximum percentile for shift value in data augmentation. Default: 0.2.
        
    Returns
    -------
    exp : numpy.ndarray
        Simulated expression data of shape (n_spots, n_genes).
    density : numpy.ndarray
        True cell type proportions of shape (n_spots, n_clusters).
        
    Notes
    -----
    Unlike sample_cell, this function uses predefined cluster combinations for each spot.
    It's useful for creating specific mixtures or testing particular cell type combinations.
    """
    exp = np.empty((len(cluster_sample_list), sample_exp.shape[1]),dtype=np.float32)
    density = np.empty((len(cluster_sample_list), sample_cluster.shape[1]),dtype=np.float32)
    for i in nb.prange(len(cluster_sample_list)):
        used_clusters = np.where(cluster_sample_list[i] == 1)[0]
        num_cell = ncell_sample_list[i]
        cluster_mask = np.array([False]*len(cluster_id))
        for c in used_clusters:
            cluster_mask = (cluster_id==c)|(cluster_mask)
        used_cell_ind = np.where(cluster_mask)[0]
        used_cell_p = cell_p_balanced[cluster_mask]
        used_cell_p = used_cell_p/used_cell_p.sum()
        sampled_cells = used_cell_ind[np.searchsorted(np.cumsum(used_cell_p), np.random.rand(num_cell), side="right")]
        combined_exp = np_sum(sample_exp[sampled_cells,:],axis=0).astype(np.float32)
        if data_augmentation:
            combined_exp = random_augmentation_cell(combined_exp,max_rate=max_rate,max_val=max_val,kth=kth)
        if downsample_fraction is not None:
            combined_exp = downsample_cell(combined_exp, downsample_fraction)
        combined_clusters = np_sum(sample_cluster[cluster_id[sampled_cells]],axis=0).astype(np.float32)
        exp[i,:] = combined_exp
        density[i,:] = combined_clusters
    return exp,density

def init_sample_prob(sc_ad,celltype_key):
    """
    Initialize cell and cluster sampling probabilities.
    
    Parameters
    ----------
    sc_ad : AnnData
        Single-cell AnnData object.
    celltype_key : str
        Key in sc_ad.obs containing cell type annotations.
        
    Returns
    -------
    sc_ad : AnnData
        Updated AnnData object with sampling probabilities in obs and uns.
        
    Notes
    -----
    This function computes three different cluster sampling probabilities:
    1. unbalance: proportional to the original cell type distribution
    2. balance: uniform across all cell types
    3. sqrt: square root of the original distribution (a compromise between the two)
    
    These sampling strategies are useful for creating synthetic data with different
    cell type composition biases.
    """
    print('### Initializing sample probability')
    sc_ad.uns['celltype2num'] = pd.DataFrame(
        np.arange(len(sc_ad.obs[celltype_key].value_counts())).T,
        index=sc_ad.obs[celltype_key].value_counts().index.values,
        columns=['celltype_num']
    )
    sc_ad.obs['celltype_num'] = [sc_ad.uns['celltype2num'].loc[c,'celltype_num'] for c in sc_ad.obs[celltype_key]]
    cluster_p_unbalance = sc_ad.obs['celltype_num'].value_counts()/sc_ad.obs['celltype_num'].value_counts().sum()
    cluster_p_sqrt = np.sqrt(sc_ad.obs['celltype_num'].value_counts())/np.sqrt(sc_ad.obs['celltype_num'].value_counts()).sum()
    cluster_p_balance = pd.Series(
        np.ones(len(sc_ad.obs['celltype_num'].value_counts()))/len(sc_ad.obs['celltype_num'].value_counts()), 
        index=sc_ad.obs['celltype_num'].value_counts().index
    )

    cell_p_balanced = [1/cluster_p_unbalance[c] for c in sc_ad.obs['celltype_num']]
    cell_p_balanced = np.array(cell_p_balanced)/np.array(cell_p_balanced).sum()
    sc_ad.obs['cell_p_balanced'] = cell_p_balanced
    sc_ad.uns['cluster_p_balance'] = cluster_p_balance
    sc_ad.uns['cluster_p_sqrt'] = cluster_p_sqrt
    sc_ad.uns['cluster_p_unbalance'] = cluster_p_unbalance
    return sc_ad

def generate_sample_array(sc_ad, used_genes):
    """
    Convert an AnnData expression matrix to a NumPy array with selected genes.
    
    Parameters
    ----------
    sc_ad : AnnData
        AnnData object containing expression data.
    used_genes : list or None
        List of genes to include. If None, all genes are used.
        
    Returns
    -------
    numpy.ndarray
        Expression array with shape (n_cells, n_genes).
    """
    if used_genes is not None:
        sc_df = sc_ad.to_df().loc[:,used_genes]
    else:
        sc_df = sc_ad.to_df()
    return sc_df.values

def get_param_from_uniform(num_sample,cells_min=None,cells_max=None,clusters_min=None,clusters_max=None):
 """
    Generate spot parameters from uniform distributions.
    
    Parameters
    ----------
    num_sample : int
        Number of spots to simulate.
    cells_min : int, optional
        Minimum number of cells per spot.
    cells_max : int, optional
        Maximum number of cells per spot.
    clusters_min : int, optional
        Minimum number of clusters per spot.
    clusters_max : int, optional
        Maximum number of clusters per spot.
        
    Returns
    -------
    cell_count : numpy.ndarray
        Number of cells for each simulated spot.
    cluster_count : numpy.ndarray
        Number of clusters for each simulated spot.
    
    Notes
    -----
    The number of clusters is constrained to be at most the number of cells.
    """
    cell_count = np.asarray(np.ceil(np.random.uniform(int(cells_min),int(cells_max),size=num_sample)),dtype=int)
    cluster_count = np.asarray(np.ceil(np.clip(np.random.uniform(clusters_min,clusters_max,size=num_sample),1,cell_count)),dtype=int)
    return cell_count, cluster_count

def get_param_from_gaussian(num_sample,cells_min=None,cells_max=None,cells_mean=None,cells_std=None,clusters_mean=None,clusters_std=None):
"""
    Generate spot parameters from Gaussian distributions.
    
    Parameters
    ----------
    num_sample : int
        Number of spots to simulate.
    cells_min : int, optional
        Minimum number of cells per spot.
    cells_max : int, optional
        Maximum number of cells per spot.
    cells_mean : float, optional
        Mean number of cells per spot.
    cells_std : float, optional
        Standard deviation of cells per spot.
    clusters_mean : float, optional
        Mean number of clusters per spot.
    clusters_std : float, optional
        Standard deviation of clusters per spot.
        
    Returns
    -------
    cell_count : numpy.ndarray
        Number of cells for each simulated spot.
    cluster_count : numpy.ndarray
        Number of clusters for each simulated spot.
    
    Notes
    -----
    The number of clusters is constrained to be at most the number of cells and at least 1.
    """
    cell_count = np.asarray(np.ceil(np.clip(np.random.normal(cells_mean,cells_std,size=num_sample),int(cells_min),int(cells_max))),dtype=int)
    cluster_count = np.asarray(np.ceil(np.clip(np.random.normal(clusters_mean,clusters_std,size=num_sample),1,cell_count)),dtype=int)
    return cell_count,cluster_count


def get_param_from_cell_counts(
    num_sample,
    cell_counts,
    cluster_sample_mode='gaussian',
    cells_min=None,cells_max=None,
    cells_mean=None,cells_std=None,
    clusters_mean=None,clusters_std=None,
    clusters_min=None,clusters_max=None
):
 """
    Generate spot parameters based on observed cell counts.
    
    Parameters
    ----------
    num_sample : int
        Number of spots to simulate.
    cell_counts : numpy.ndarray
        Observed cell counts from spatial data.
    cluster_sample_mode : str, optional
        Method for sampling cluster counts ('gaussian' or 'uniform'). Default: 'gaussian'.
    cells_min : int, optional
        Minimum number of cells per spot.
    cells_max : int, optional
        Maximum number of cells per spot.
    cells_mean : float, optional
        Mean number of cells per spot.
    cells_std : float, optional
        Standard deviation of cells per spot.
    clusters_mean : float, optional
        Mean number of clusters per spot.
    clusters_std : float, optional
        Standard deviation of clusters per spot.
    clusters_min : int, optional
        Minimum number of clusters per spot.
    clusters_max : int, optional
        Maximum number of clusters per spot.
        
    Returns
    -------
    cell_count : numpy.ndarray
        Number of cells for each simulated spot.
    cluster_count : numpy.ndarray
        Number of clusters for each simulated spot.
        
    Notes
    -----
    This function samples cell counts from a Gaussian distribution with parameters
    derived from observed data, while cluster counts can be sampled from either
    Gaussian or uniform distributions.
    """
    cell_count = np.asarray(np.ceil(np.clip(np.random.normal(cells_mean,cells_std,size=num_sample),int(cells_min),int(cells_max))),dtype=int)
    if cluster_sample_mode == 'gaussian':
        cluster_count = np.asarray(np.ceil(np.clip(np.random.normal(clusters_mean,clusters_std,size=num_sample),1,cell_count)),dtype=int)
    elif cluster_sample_mode == 'uniform':
        cluster_count = np.asarray(np.ceil(np.clip(np.random.uniform(clusters_min,clusters_max,size=num_sample),1,cell_count)),dtype=int)
    else:
        raise TypeError('Not correct sample method.')
    return cell_count,cluster_count

def get_cluster_sample_prob(sc_ad,mode):
    """
    Get cluster sampling probabilities based on the specified mode.
    
    Parameters
    ----------
    sc_ad : AnnData
        Single-cell AnnData object with sampling probabilities in uns.
    mode : str
        Sampling mode: 'unbalance', 'balance', or 'sqrt'.
        
    Returns
    -------
    numpy.ndarray
        Cluster sampling probabilities.
        
    Notes
    -----
    Different modes produce different biases in cell type composition:
    - 'unbalance': preserves the original cell type distribution
    - 'balance': creates equal representation of all cell types
    - 'sqrt': creates a compromise between the two by using square root transformation
    """
    if mode == 'unbalance':
        cluster_p = sc_ad.uns['cluster_p_unbalance'].values
    elif mode == 'balance':
        cluster_p = sc_ad.uns['cluster_p_balance'].values
    elif mode == 'sqrt':
        cluster_p = sc_ad.uns['cluster_p_sqrt'].values
    else:
        raise TypeError('Balance argument must be one of [ None, banlance, sqrt ].')
    return cluster_p

def cal_downsample_fraction(sc_ad,st_ad,celltype_key=None):
    """
    Calculate appropriate downsampling fraction to match simulated data to spatial data.
    
    Parameters
    ----------
    sc_ad : AnnData
        Single-cell AnnData object.
    st_ad : AnnData
        Spatial transcriptomics AnnData object.
    celltype_key : str, optional
        Key in sc_ad.obs containing cell type annotations.
        
    Returns
    -------
    float or None
        Downsample fraction if spatial data has lower counts than simulated data,
        otherwise None.
        
    Notes
    -----
    This function compares the median UMI counts between spatial data and simulated data.
    If spatial data has fewer counts, it calculates a downsampling fraction to make
    the simulated data match the spatial data's count distribution.
    """
    st_counts_median = np.median(st_ad.X.sum(axis=1))
    simulated_st_data, simulated_st_labels = generate_simulation_data(sc_ad,num_sample=10000,celltype_key=celltype_key,balance_mode=['unbalance'])
    simulated_st_counts_median = np.median(simulated_st_data.sum(axis=1))
    if st_counts_median < simulated_st_counts_median:
        fraction = st_counts_median / simulated_st_counts_median
        print(f'### Simulated data downsample fraction: {fraction}')
        return fraction
    else:
        return None

def generate_simulation_data(
    sc_ad,
    celltype_key,
    num_sample: int, 
    used_genes=None,
    balance_mode=['unbalance','sqrt','balance'],
    cell_sample_method='gaussian',
    cluster_sample_method='gaussian',
    cell_counts=None,
    downsample_fraction=None,
    data_augmentation=True,
    max_rate=0.8,max_val=0.8,kth=0.2,
    cells_min=1,cells_max=20,
    cells_mean=10,cells_std=5,
    clusters_mean=None,clusters_std=None,
    clusters_min=None,clusters_max=None,
    cell_sample_counts=None,cluster_sample_counts=None,
    ncell_sample_list=None,
    cluster_sample_list=None,
    n_cpus=None
):
"""
    Generate simulated spatial transcriptomics data from single-cell data.
    
    Parameters
    ----------
    sc_ad : AnnData
        Single-cell AnnData object.
    celltype_key : str
        Key in sc_ad.obs containing cell type annotations.
    num_sample : int
        Number of spots to simulate.
    used_genes : list, optional
        List of genes to include. If None, all genes are used.
    balance_mode : list, optional
        List of sampling modes to use ('unbalance', 'balance', 'sqrt'). Default: ['unbalance','sqrt','balance'].
    cell_sample_method : str, optional
        Method for sampling cell counts ('gaussian' or 'uniform'). Default: 'gaussian'.
    cluster_sample_method : str, optional
        Method for sampling cluster counts ('gaussian' or 'uniform'). Default: 'gaussian'.
    cell_counts : numpy.ndarray, optional
        Observed cell counts from spatial data for realistic simulation. Default: None.
    downsample_fraction : float, optional
        Fraction of counts to retain after downsampling. Default: None (no downsampling).
    data_augmentation : bool, optional
        Whether to apply data augmentation. Default: True.
    max_rate : float, optional
        Maximum dropout rate for data augmentation. Default: 0.8.
    max_val : float, optional
        Maximum scaling factor for data augmentation. Default: 0.8.
    kth : float, optional
        Maximum percentile for shift value in data augmentation. Default: 0.2.
    cells_min : int, optional
        Minimum number of cells per spot. Default: 1.
    cells_max : int, optional
        Maximum number of cells per spot. Default: 20.
    cells_mean : float, optional
        Mean number of cells per spot. Default: 10.
    cells_std : float, optional
        Standard deviation of cells per spot. Default: 5.
    clusters_mean : float, optional
        Mean number of clusters per spot. Default: None (uses cells_mean/2).
    clusters_std : float, optional
        Standard deviation of clusters per spot. Default: None (uses cells_std/2).
    clusters_min : int, optional
        Minimum number of clusters per spot. Default: None (uses cells_min).
    clusters_max : int, optional
        Maximum number of clusters per spot. Default: None (uses min(cells_max/2, num_clusters)).
    cell_sample_counts : numpy.ndarray, optional
        Predefined number of cells for each spot. Default: None.
    cluster_sample_counts : numpy.ndarray, optional
        Predefined number of clusters for each spot. Default: None.
    ncell_sample_list : numpy.ndarray, optional
        Alternative predefined number of cells for each spot. Default: None.
    cluster_sample_list : numpy.ndarray, optional
        Predefined cluster combinations for each spot. Default: None.
    n_cpus : int, optional
        Number of CPUs to use for parallel processing. Default: None.
        
    Returns
    -------
    sample_data : numpy.ndarray
        Simulated expression data of shape (n_spots, n_genes).
    sample_labels : numpy.ndarray
        True cell type proportions of shape (n_spots, n_clusters).
        
    Notes
    -----
    This function provides a flexible framework for simulating spatial transcriptomics data
    by mixing single-cell profiles. It supports various sampling strategies for cell counts
    and cluster counts, as well as data augmentation and downsampling to mimic technical
    characteristics of spatial technologies.
    
    The function generates a mixture of spots with different cell type compositions based
    on the specified balance modes. This is useful for creating realistic training data
    for deconvolution algorithms.
    """

    if not 'cluster_p_unbalance' in sc_ad.uns:
        sc_ad = init_sample_prob(sc_ad,celltype_key)
    num_sample_per_mode = num_sample//len(balance_mode)
    cluster_ordered = np.array(sc_ad.obs['celltype_num'].value_counts().index)
    cluster_num = len(cluster_ordered)
    cluster_id = sc_ad.obs['celltype_num'].values
    cluster_mask = np.eye(cluster_num)
    if (cell_sample_counts is None) or (cluster_sample_counts is None):
        if cell_counts is not None:
            cells_mean = np.mean(np.sort(cell_counts)[int(len(cell_counts)*0.05):int(len(cell_counts)*0.95)])
            cells_std = np.std(np.sort(cell_counts)[int(len(cell_counts)*0.05):int(len(cell_counts)*0.95)])
            cells_min = int(np.min(np.sort(cell_counts)[int(len(cell_counts)*0.05):int(len(cell_counts)*0.95)]))
            cells_max = int(np.max(np.sort(cell_counts)[int(len(cell_counts)*0.05):int(len(cell_counts)*0.95)]))
        if clusters_mean is None:
            clusters_mean = cells_mean/2
        if clusters_std is None:
            clusters_std = cells_std/2
        if clusters_min is None:
            clusters_min = cells_min
        if clusters_max is None:
            clusters_max = np.min((cells_max//2,cluster_num))

        if cell_counts is not None:
            cell_sample_counts, cluster_sample_counts = get_param_from_cell_counts(num_sample_per_mode,cell_counts,cluster_sample_method,cells_mean=cells_mean,cells_std=cells_std,cells_max=cells_max,cells_min=cells_min,clusters_mean=clusters_mean,clusters_std=clusters_std,clusters_min=clusters_min,clusters_max=clusters_max)
        elif cell_sample_method == 'gaussian':
            cell_sample_counts, cluster_sample_counts = get_param_from_gaussian(num_sample_per_mode,cells_mean=cells_mean,cells_std=cells_std,cells_max=cells_max,cells_min=cells_min,clusters_mean=clusters_mean,clusters_std=clusters_std)
        elif cell_sample_method == 'uniform':
            cell_sample_counts, cluster_sample_counts = get_param_from_uniform(num_sample_per_mode,cells_max=cells_max,cells_min=cells_min,clusters_min=clusters_min,clusters_max=clusters_max)
        else:
            raise TypeError('Not correct sample method.')
    if cluster_sample_list is None or ncell_sample_list is None:
        params = np.array(list(zip(cell_sample_counts, cluster_sample_counts)))

        sample_data_list = []
        sample_labels_list = []
        for b in balance_mode:
            print(f'### Genetating simulated spatial data using scRNA data with mode: {b}')
            cluster_p = get_cluster_sample_prob(sc_ad,b)
            if downsample_fraction is not None:
                if downsample_fraction > 0.035:
                    sample_data,sample_labels = sample_cell(
                        param_list=params,
                        cluster_p=cluster_p,
                        clusters=cluster_ordered,
                        cluster_id=cluster_id,
                        sample_exp=generate_sample_array(sc_ad,used_genes),
                        sample_cluster=cluster_mask,
                        cell_p_balanced=sc_ad.obs['cell_p_balanced'].values,
                        downsample_fraction=downsample_fraction,
                        data_augmentation=data_augmentation,max_rate=max_rate,max_val=max_val,kth=kth,
                    )
                else:
                    sample_data,sample_labels = sample_cell(
                        param_list=params,
                        cluster_p=cluster_p,
                        clusters=cluster_ordered,
                        cluster_id=cluster_id,
                        sample_exp=generate_sample_array(sc_ad,used_genes),
                        sample_cluster=cluster_mask,
                        cell_p_balanced=sc_ad.obs['cell_p_balanced'].values,
                        data_augmentation=data_augmentation,max_rate=max_rate,max_val=max_val,kth=kth,
                    )
                    # logging.warning('### Downsample data with python backend')
                    sample_data = downsample_matrix_by_cell(sample_data, downsample_fraction, n_cpus=n_cpus, numba_end=False)
            else:
                sample_data,sample_labels = sample_cell(
                    param_list=params,
                    cluster_p=cluster_p,
                    clusters=cluster_ordered,
                    cluster_id=cluster_id,
                    sample_exp=generate_sample_array(sc_ad,used_genes),
                    sample_cluster=cluster_mask,
                    cell_p_balanced=sc_ad.obs['cell_p_balanced'].values,
                    data_augmentation=data_augmentation,max_rate=max_rate,max_val=max_val,kth=kth,
                )

            sample_data_list.append(sample_data)
            sample_labels_list.append(sample_labels)
    else:
        sample_data_list = []
        sample_labels_list = []
        for b in balance_mode:
            print(f'### Genetating simulated spatial data using scRNA data with mode: {b}')
            cluster_p = get_cluster_sample_prob(sc_ad,b)
            if downsample_fraction is not None:
                if downsample_fraction > 0.035:
                    sample_data,sample_labels = sample_cell_from_clusters(
                        cluster_sample_list=cluster_sample_list,
                        ncell_sample_list=ncell_sample_list,
                        cluster_id=cluster_id,
                        sample_exp=generate_sample_array(sc_ad,used_genes),
                        sample_cluster=cluster_mask,
                        cell_p_balanced=sc_ad.obs['cell_p_balanced'].values,
                        downsample_fraction=downsample_fraction,
                        data_augmentation=data_augmentation,max_rate=max_rate,max_val=max_val,kth=kth,
                    )
                else:
                    sample_data,sample_labels = sample_cell_from_clusters(
                        cluster_sample_list=cluster_sample_list,
                        ncell_sample_list=ncell_sample_list,
                        cluster_id=cluster_id,
                        sample_exp=generate_sample_array(sc_ad,used_genes),
                        sample_cluster=cluster_mask,
                        cell_p_balanced=sc_ad.obs['cell_p_balanced'].values,
                        data_augmentation=data_augmentation,max_rate=max_rate,max_val=max_val,kth=kth,
                    )
                    
                    sample_data = downsample_matrix_by_cell(sample_data, downsample_fraction, n_cpus=n_cpus, numba_end=False)
            else:
                sample_data,sample_labels = sample_cell_from_clusters(
                    cluster_sample_list=cluster_sample_list,
                    ncell_sample_list=ncell_sample_list,
                    cluster_id=cluster_id,
                    sample_exp=generate_sample_array(sc_ad,used_genes),
                    sample_cluster=cluster_mask,
                    cell_p_balanced=sc_ad.obs['cell_p_balanced'].values,
                    data_augmentation=data_augmentation,max_rate=max_rate,max_val=max_val,kth=kth,
                )
            sample_data_list.append(sample_data)
            sample_labels_list.append(sample_labels)
    return np.concatenate(sample_data_list), np.concatenate(sample_labels_list)

@jit(nopython=True,parallel=True)
def sample_cell_exp(cell_counts,sample_exp,cell_p,downsample_fraction=None,data_augmentation=True,max_rate=0.8,max_val=0.8,kth=0.2):
    exp = np.empty((len(cell_counts), sample_exp.shape[1]),dtype=np.float32)
    ind = np.zeros((len(cell_counts), np.max(cell_counts)),dtype=np.int32)
    cell_ind = np.arange(sample_exp.shape[0])
    for i in nb.prange(len(cell_counts)):
        num_cell = cell_counts[i]
        sampled_cells=cell_ind[np.searchsorted(np.cumsum(cell_p), np.random.rand(num_cell), side="right")]
        combined_exp=np_sum(sample_exp[sampled_cells,:],axis=0).astype(np.float64)
        if downsample_fraction is not None:
            combined_exp = downsample_cell(combined_exp, downsample_fraction)
        if data_augmentation:
            combined_exp = random_augmentation_cell(combined_exp,max_rate=max_rate,max_val=max_val,kth=kth)
        exp[i,:] = combined_exp
        ind[i,:cell_counts[i]] = sampled_cells + 1
    return exp,ind

def generate_simulation_st_data(
    st_ad,
    num_sample: int, 
    used_genes=None,
    balance_mode=['unbalance'],
    cell_sample_method='gaussian',
    cell_counts=None,
    downsample_fraction=None,
    data_augmentation=True,
    max_rate=0.8,max_val=0.8,kth=0.2,
    cells_min=1,cells_max=10,
    cells_mean=5,cells_std=3,
):
    print('### Genetating simulated data using spatial data')
    cell_p = np.ones(len(st_ad))/len(st_ad)
    if cell_counts is not None:
        cells_mean = np.mean(np.sort(cell_counts)[int(len(cell_counts)*0.05):int(len(cell_counts)*0.95)])
        cells_std = np.std(np.sort(cell_counts)[int(len(cell_counts)*0.05):int(len(cell_counts)*0.95)])
        cells_min = int(np.min(np.sort(cell_counts)[int(len(cell_counts)*0.05):int(len(cell_counts)*0.95)]))
        cells_max = int(np.max(np.sort(cell_counts)[int(len(cell_counts)*0.05):int(len(cell_counts)*0.95)]))
    elif cell_sample_method == 'gaussian':
        cell_counts = np.asarray(np.ceil(np.clip(np.random.normal(cells_mean,cells_std,size=num_sample),int(cells_min),int(cells_max))),dtype=int)
    elif cell_sample_method == 'uniform':
        cell_counts = np.asarray(np.ceil(np.random.uniform(int(cells_min),int(cells_max),size=num_sample)),dtype=int)
    else:
        raise TypeError('Not correct sample method.')

    sample_data,sample_ind = sample_cell_exp(
        cell_counts=cell_counts,
        sample_exp=generate_sample_array(st_ad,used_genes),
        cell_p=cell_p,
        downsample_fraction=downsample_fraction,
        data_augmentation=data_augmentation,max_rate=max_rate,max_val=max_val,kth=kth,
    )

    return sample_data,sample_ind
