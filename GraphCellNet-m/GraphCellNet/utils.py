import pandas as pd
import numpy as np
import scanpy as sc
import anndata
from scipy.sparse import issparse,csr_matrix
from sklearn.preprocessing import normalize
from . import downsample
from . import augmentation
from . import spatial_simulation
import logging


from rpy2.robjects.packages import importr
import rpy2.robjects as robjects

def normalize_adata(ad,target_sum=None):
     """
    Normalize gene expression data in an AnnData object.
    
    Parameters
    ----------
    ad : AnnData
        AnnData object containing gene expression data.
    target_sum : float, optional
        Target sum for normalization. If None, defaults to 1e4. Default: None.
        
    Returns
    -------
    AnnData
        AnnData object with normalized data in the 'norm' layer.
        
    Notes
    -----
    This function performs the following normalization steps:
    1. Total-count normalize to target_sum (10,000 by default)
    2. Log-transform the data (log1p)
    3. Stores the normalized data in the 'norm' layer
    """
    ad_norm = sc.pp.normalize_total(ad,inplace=False,target_sum=1e4)
    ad_norm  = sc.pp.log1p(ad_norm['X'])
    # ad_norm  = sc.pp.scale(ad_norm)
    # ad_norm = normalize(ad_norm,axis=1)
    ad.layers['norm'] = ad_norm
    return ad

def normalize_mtx(mtx,target_sum):
    """
    Normalize a gene expression matrix.
    
    Parameters
    ----------
    mtx : numpy.ndarray
        Gene expression matrix of shape (n_cells, n_genes).
    target_sum : float
        Target sum for normalization.
        
    Returns
    -------
    numpy.ndarray
        Normalized matrix of shape (n_cells, n_genes).
        
    Notes
    -----
    This function performs the following normalization steps:
    1. Removes cells with zero total counts
    2. Total-count normalizes to target_sum
    3. Log-transforms the data (log1p)
    4. Scales values to [0,1] range for each cell
    5. L2-normalizes each cell (row)
    """
    mtx = mtx[mtx.sum(1)!=0,:]
    mtx = np.nan_to_num(np.log1p((mtx.T*target_sum/mtx.sum(axis=1)).T))
    mtx = (mtx-mtx.min(axis=1,keepdims=True))/(mtx.max(axis=1,keepdims=True)-mtx.min(axis=1,keepdims=True))
    mtx = normalize(mtx,axis=1)
    return mtx
    

def find_sc_markers(sc_ad, celltype_key, layer='norm', deg_method=None, log2fc_min=0.5, pval_cutoff=0.01, n_top_markers=200, pct_diff=None, pct_min=0.1):
    """
    Find marker genes for each cell type in single-cell data.
    
    Parameters
    ----------
    sc_ad : AnnData
        Single-cell AnnData object.
    celltype_key : str
        Key in sc_ad.obs containing cell type annotations.
    layer : str, optional
        Layer in AnnData to use for differential expression. Default: 'norm'.
    deg_method : str, optional
        Method for differential expression analysis. Options: 'wilcoxon', 't-test', 'logreg'.
        If None, uses scanpy's default method. Default: None.
    log2fc_min : float, optional
        Minimum log2 fold change threshold for selecting marker genes. Default: 0.5.
    pval_cutoff : float, optional
        Maximum p-value threshold for selecting marker genes. Default: 0.01.
    n_top_markers : int, optional
        Number of top marker genes to select per cell type. Default: 200.
    pct_diff : float, optional
        Minimum percentage difference in cells expressing the gene between the group
        and the rest. Default: None.
    pct_min : float, optional
        Minimum percentage of cells that must express the gene in at least one group.
        Default: 0.1.
        
    Returns
    -------
    numpy.ndarray
        Array of marker gene names.
        
    Notes
    -----
    This function filters out cell types with only one sample, then identifies marker
    genes for each cell type using scanpy's rank_genes_groups. It applies multiple
    filtering criteria to select high-quality marker genes.
    """
    print('### Finding marker genes...')
    
    # filter celltype contain only one sample.
    filtered_celltypes = list(sc_ad.obs[celltype_key].value_counts()[(sc_ad.obs[celltype_key].value_counts() == 1).values].index)
    if len(filtered_celltypes) > 0:
        sc_ad = sc_ad[sc_ad.obs[~(sc_ad.obs[celltype_key].isin(filtered_celltypes))].index,:].copy()
        print(f'### Filter cluster contain only one sample: {filtered_celltypes}')

    sc.tl.rank_genes_groups(sc_ad, groupby=celltype_key, pts=True, layer=layer, use_raw=False, method=deg_method)
    marker_genes_dfs = []
    for c in np.unique(sc_ad.obs[celltype_key]):
        tmp_marker_gene_df = sc.get.rank_genes_groups_df(sc_ad, group=c, pval_cutoff=pval_cutoff, log2fc_min=log2fc_min)
        if (tmp_marker_gene_df.empty is not True):
            tmp_marker_gene_df.index = tmp_marker_gene_df.names
            tmp_marker_gene_df.loc[:,celltype_key] = c
            if pct_diff is not None:
                pct_diff_genes = sc_ad.var_names[np.where((sc_ad.uns['rank_genes_groups']['pts'][c]-sc_ad.uns['rank_genes_groups']['pts_rest'][c]) > pct_diff)]
                tmp_marker_gene_df = tmp_marker_gene_df.loc[np.intersect1d(pct_diff_genes, tmp_marker_gene_df.index),:]
            if pct_min is not None:
                # pct_min_genes = sc_ad.var_names[np.where((sc_ad.uns['rank_genes_groups']['pts'][c]) > pct_min)]
                tmp_marker_gene_df = tmp_marker_gene_df[tmp_marker_gene_df['pct_nz_group'] > pct_min]
            if n_top_markers is not None:
                tmp_marker_gene_df = tmp_marker_gene_df.sort_values('logfoldchanges',ascending=False)
                tmp_marker_gene_df = tmp_marker_gene_df.iloc[:n_top_markers,:]
            marker_genes_dfs.append(tmp_marker_gene_df)
    marker_gene_df = pd.concat(marker_genes_dfs,axis=0)
    print(marker_gene_df[celltype_key].value_counts())
    all_marker_genes = np.unique(marker_gene_df.names)
    return all_marker_genes

def find_st_hvg(st_ad,n_top_hvg=None):
    """
    Find highly variable genes in spatial transcriptomics data.
    
    Parameters
    ----------
    st_ad : AnnData
        Spatial transcriptomics AnnData object.
    n_top_hvg : int, optional
        Number of top highly variable genes to select. Default: None.
        
    Returns
    -------
    numpy.ndarray
        Array of highly variable gene names.
        
    Notes
    -----
    This function uses scanpy's highly_variable_genes function with the
    'seurat_v3' flavor to identify genes with high variance across spots.
    """
    print('### Finding HVG in spatial...')
    sc.pp.highly_variable_genes(st_ad,n_top_genes=n_top_hvg,flavor='seurat_v3')
    return st_ad.var_names[st_ad.var['highly_variable'] == True]

# 
def filter_model_genes(
    sc_ad, 
    st_ad, 
    celltype_key=None, 
    used_genes=None,
    sc_genes=None,
    st_genes=None,
    layer='norm',
    deg_method=None,
    log2fc_min=0.5, 
    pval_cutoff=0.01, 
    n_top_markers:int=200, 
    n_top_hvg=None,
    pct_diff=None, 
    pct_min=0.1,
):
 """
    Filter genes for deconvolution model based on marker genes and highly variable genes.
    
    Parameters
    ----------
    sc_ad : AnnData
        Single-cell AnnData object.
    st_ad : AnnData
        Spatial transcriptomics AnnData object.
    celltype_key : str, optional
        Key in sc_ad.obs containing cell type annotations.
    used_genes : list, optional
        Predefined list of genes to use. If provided, other gene selection methods
        are skipped. Default: None.
    sc_genes : list, optional
        Predefined list of genes from single-cell data. Default: None.
    st_genes : list, optional
        Predefined list of genes from spatial data. Default: None.
    layer : str, optional
        Layer in AnnData to use for differential expression. Default: 'norm'.
    deg_method : str, optional
        Method for differential expression analysis. Default: None.
    log2fc_min : float, optional
        Minimum log2 fold change threshold for marker genes. Default: 0.5.
    pval_cutoff : float, optional
        Maximum p-value threshold for marker genes. Default: 0.01.
    n_top_markers : int, optional
        Number of top marker genes per cell type. Default: 200.
    n_top_hvg : int, optional
        Number of top highly variable genes in spatial data. Default: None.
    pct_diff : float, optional
        Minimum percentage difference for marker genes. Default: None.
    pct_min : float, optional
        Minimum percentage of cells expressing marker genes. Default: 0.1.
        
    Returns
    -------
    sc_ad : AnnData
        Filtered single-cell AnnData object.
    st_ad : AnnData
        Filtered spatial transcriptomics AnnData object.
        
    Notes
    -----
    This function performs the following steps:
    1. Finds genes common to both single-cell and spatial data
    2. If used_genes is not provided:
       a. Identifies highly variable genes in spatial data (if n_top_hvg is set)
       b. Finds marker genes in single-cell data for each cell type
       c. Uses the intersection of marker genes and HVGs
    3. Filters both AnnData objects to include only the selected genes
    4. Filters out cells with no detected genes
    """
    overlaped_genes = np.intersect1d(sc_ad.var_names,st_ad.var_names)
    sc_ad = sc_ad[:,overlaped_genes].copy()
    st_ad = st_ad[:,overlaped_genes].copy()
    if used_genes is None:
        if st_genes is None:
            if n_top_hvg is None:
                st_genes = st_ad.var_names
            else:
                st_genes = find_st_hvg(st_ad, n_top_hvg)
        if sc_genes is None:
            sc_ad = sc_ad[:, st_genes].copy()
            sc_genes = find_sc_markers(sc_ad, celltype_key, layer, deg_method, log2fc_min, pval_cutoff, n_top_markers, pct_diff, pct_min)
        used_genes = np.intersect1d(sc_genes,st_genes)
    sc_ad = sc_ad[:,used_genes].copy()
    st_ad = st_ad[:,used_genes].copy()
    sc.pp.filter_cells(sc_ad,min_genes=1)
    sc.pp.filter_cells(st_ad,min_genes=1)
    print(f'### Used gene numbers: {len(used_genes)}')
    return sc_ad, st_ad

def check_data_type(ad):
    """
    Ensure AnnData object has dense array in float32 format.
    
    Parameters
    ----------
    ad : AnnData
        AnnData object to check and convert.
        
    Returns
    -------
    AnnData
        AnnData object with dense array in float32 format.
        
    Notes
    -----
    This function converts sparse matrices to dense arrays and
    ensures the data type is float32 for compatibility with PyTorch.
    """
    if issparse(ad.X):
        ad.X = ad.X.toarray()
    if ad.X.dtype != np.float32:
        ad.X =ad.X.astype(np.float32)
    return ad


def generate_sm_adata(sc_ad,num_sample,celltype_key,n_threads,cell_counts,clusters_mean,cells_mean,cells_min,cells_max,cell_sample_counts,cluster_sample_counts,ncell_sample_list,cluster_sample_list):
    """
    Generate simulated mixture AnnData object from single-cell data.
    
    Parameters
    ----------
    sc_ad : AnnData
        Single-cell AnnData object.
    num_sample : int
        Number of simulated spots to generate.
    celltype_key : str
        Key in sc_ad.obs containing cell type annotations.
    n_threads : int
        Number of CPU threads to use for parallel processing.
    cell_counts : dict, optional
        Dictionary mapping cell types to their counts for simulation.
    clusters_mean : float, optional
        Mean number of clusters per simulated spot.
    cells_mean : int, optional
        Mean number of cells per simulated spot.
    cells_min : int, optional
        Minimum number of cells per simulated spot.
    cells_max : int, optional
        Maximum number of cells per simulated spot.
    cell_sample_counts : dict, optional
        Dictionary mapping cell types to their sampling counts.
    cluster_sample_counts : dict, optional
        Dictionary mapping clusters to their sampling counts.
    ncell_sample_list : list, optional
        List of specific cell counts to sample for simulation.
    cluster_sample_list : list, optional
        List of specific clusters to sample for simulation.
        
    Returns
    -------
    AnnData
        Simulated mixture AnnData object with true proportions in .obsm['label'].
        
    Notes
    -----
    This function generates simulated spatial transcriptomics spots by sampling
    and combining cells from the single-cell data according to various parameters.
    The true cell type proportions for each spot are stored in the returned
    AnnData object's .obsm['label'] as a DataFrame.
    """
    sm_data,sm_labels = spatial_simulation.generate_simulation_data(sc_ad,num_sample=num_sample,celltype_key=celltype_key,downsample_fraction=None,data_augmentation=False,n_cpus=n_threads,cell_counts=cell_counts,clusters_mean=clusters_mean,cells_mean=cells_mean,cells_min=cells_min,cells_max=cells_max,cell_sample_counts=cell_sample_counts,cluster_sample_counts=cluster_sample_counts,ncell_sample_list=ncell_sample_list,cluster_sample_list=cluster_sample_list)
    sm_data_mtx = csr_matrix(sm_data)
    sm_ad = anndata.AnnData(sm_data_mtx)
    sm_ad.var.index = sc_ad.var_names
    sm_labels = (sm_labels.T/sm_labels.sum(axis=1)).T
    sm_ad.obsm['label'] = pd.DataFrame(sm_labels,columns=np.array(sc_ad.obs[celltype_key].value_counts().index.values),index=sm_ad.obs_names)
    return sm_ad

def downsample_sm_spot_counts(sm_ad,st_ad,n_threads):
    """
    Downsample simulated mixture data to match the library size distribution of spatial data.
    
    Parameters
    ----------
    sm_ad : AnnData
        Simulated mixture AnnData object.
    st_ad : AnnData
        Spatial transcriptomics AnnData object.
    n_threads : int
        Number of CPU threads to use for parallel processing.
        
    Notes
    -----
    This function:
    1. Fits a log-normal distribution to the library sizes (total UMI counts) in spatial data
    2. Samples library sizes from this distribution for each simulated spot
    3. Downsamples each simulated spot to match the sampled library size
    
    This helps make the simulated data more realistic by mimicking the technical 
    characteristics of the spatial technology.
    """
    fitdistrplus = importr('fitdistrplus')
    lib_sizes = robjects.FloatVector(np.array(st_ad.X.sum(1)).reshape(-1))
    res = fitdistrplus.fitdist(lib_sizes,'lnorm')
    loc = res[0][0]
    scale = res[0][1]

    sm_mtx_count = sm_ad.X.toarray()
    sample_cell_counts = np.random.lognormal(loc,scale,sm_ad.shape[0])
    sm_mtx_count_lb = downsample.downsample_matrix_by_cell(sm_mtx_count,sample_cell_counts.astype(np.int64), n_cpus=n_threads, numba_end=False)
    sm_ad.X = csr_matrix(sm_mtx_count_lb)
    
def split_shuffle_data(X,Y,shuffle=True,proportion=0.8):
    """
    Downsample simulated mixture data to match the library size distribution of spatial data.
    
    Parameters
    ----------
    sm_ad : AnnData
        Simulated mixture AnnData object.
    st_ad : AnnData
        Spatial transcriptomics AnnData object.
    n_threads : int
        Number of CPU threads to use for parallel processing.
        
    Notes
    -----
    This function:
    1. Fits a log-normal distribution to the library sizes (total UMI counts) in spatial data
    2. Samples library sizes from this distribution for each simulated spot
    3. Downsamples each simulated spot to match the sampled library size
    
    This helps make the simulated data more realistic by mimicking the technical 
    characteristics of the spatial technology.
    """
    if shuffle:
        reind = np.random.permutation(len(X))
        X = X[reind]
        Y = Y[reind]
    X_train = X[:int(len(X)*proportion)]
    Y_train = Y[:int(len(Y)*proportion)]
    X_test = X[int(len(X)*proportion):]
    Y_test = Y[int(len(Y)*proportion):]
    return X_train,Y_train,X_test,Y_test
