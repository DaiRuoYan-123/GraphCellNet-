from . import utils
from . import deconv

from . import spatial_simulation
import numpy as np
import torch
from scipy.sparse import csr_matrix
import numba
import random


def init_model(
    sc_ad,
    st_ad,
    celltype_key,
    sc_genes=None,
    st_genes=None,
    used_genes=None,
    deg_method:str='wilcoxon',
    n_top_markers:int=200,
    n_top_hvg:int=None,
    log2fc_min=0.5,
    pval_cutoff=0.01,
    pct_diff=None, 
    pct_min=0.1,
    use_rep='scvi',
    st_batch_key=None,
    sm_size:int=500000,
    cell_counts=None,
    clusters_mean=None,
    cells_mean=10,
    cells_min=1,
    cells_max=20,
    cell_sample_counts=None,
    cluster_sample_counts=None,
    ncell_sample_list=None,
    cluster_sample_list=None,
    scvi_layers=2,
    scvi_latent=128,
    scvi_gene_likelihood='zinb',
    scvi_dispersion='gene-batch',
    latent_dims=128, 
    hidden_dims=512,
    infer_losses=['kl','cos'],
    n_threads=4,
    seed=42,
    use_gpu=True
  
):
 """
    Initialize a SpatialNet model for cell type deconvolution in spatial transcriptomics data.
    
    This function prepares and initializes a SpatialNet model by:
    1. Setting global random seeds for reproducibility
    2. Normalizing single-cell and spatial transcriptomics data
    3. Filtering genes based on differential expression analysis
    4. Generating simulated mixture data from single-cell profiles
    5. Constructing and returning a configured SpatialNet model
    
    Parameters
    ----------
    sc_ad : AnnData
        Single-cell RNA-seq AnnData object containing gene expression data.
    st_ad : AnnData
        Spatial transcriptomics AnnData object containing gene expression data.
    celltype_key : str
        Key in sc_ad.obs that contains cell type annotations.
    sc_genes : list, optional
        List of genes to be used from single-cell data. If None, all genes in sc_ad will be considered. Default: None.
    st_genes : list, optional
        List of genes to be used from spatial data. If None, all genes in st_ad will be considered. Default: None.
    used_genes : list, optional
        List of specific genes to be used for analysis. If provided, this overrides gene selection via markers. Default: None.
    deg_method : str, optional
        Method for differential expression analysis to select marker genes. Options: 'wilcoxon', 't-test', 'logreg'. Default: 'wilcoxon'.
    n_top_markers : int, optional
        Number of top marker genes to select per cell type. Default: 200.
    n_top_hvg : int, optional
        Number of highly variable genes to select. If None, no HVG selection is performed. Default: None.
    log2fc_min : float, optional
        Minimum log2 fold change threshold for selecting differentially expressed genes. Default: 0.5.
    pval_cutoff : float, optional
        Maximum p-value threshold for selecting differentially expressed genes. Default: 0.01.
    pct_diff : float, optional
        Minimum percentage difference in cells expressing the gene between groups. Default: None.
    pct_min : float, optional
        Minimum percentage of cells that must express the gene in at least one group. Default: 0.1.
    use_rep : str, optional
        Which representation to use for model input. Options: 'scvi', 'X', 'pca'. Default: 'scvi'.
    st_batch_key : str, optional
        Key in st_ad.obs for batch information, if spatial data contains multiple batches. Default: None.
    sm_size : int, optional
        Number of simulated mixture samples to generate. Default: 500000.
    cell_counts : dict, optional
        Dictionary mapping cell types to their counts for simulation. Default: None.
    clusters_mean : float, optional
        Mean number of clusters per simulated spot. Default: None.
    cells_mean : int, optional
        Mean number of cells per simulated spot. Default: 10.
    cells_min : int, optional
        Minimum number of cells per simulated spot. Default: 1.
    cells_max : int, optional
        Maximum number of cells per simulated spot. Default: 20.
    cell_sample_counts : dict, optional
        Dictionary mapping cell types to their sampling counts. Default: None.
    cluster_sample_counts : dict, optional
        Dictionary mapping clusters to their sampling counts. Default: None.
    ncell_sample_list : list, optional
        List of specific cell counts to sample for simulation. Default: None.
    cluster_sample_list : list, optional
        List of specific clusters to sample for simulation. Default: None.
    scvi_layers : int, optional
        Number of layers in scVI model. Default: 2.
    scvi_latent : int, optional
        Latent dimension in scVI model. Default: 128.
    scvi_gene_likelihood : str, optional
        Gene likelihood model for scVI. Options: 'zinb', 'nb', 'poisson'. Default: 'zinb'.
    scvi_dispersion : str, optional
        Dispersion model for scVI. Options: 'gene', 'gene-batch', 'gene-label', 'gene-cell'. Default: 'gene-batch'.
    latent_dims : int, optional
        Latent dimensions in the SpatialNet model. Default: 128.
    hidden_dims : int, optional
        Hidden dimensions in the SpatialNet model. Default: 512.
    infer_losses : list, optional
        List of loss functions for inference. Options: 'kl', 'cos', 'rmse'. Default: ['kl', 'cos'].
    n_threads : int, optional
        Number of threads to use for parallel processing. Default: 4.
    seed : int, optional
        Random seed for reproducibility. Default: 42.
    use_gpu : bool, optional
        Whether to use GPU for model training if available. Default: True.
    
    Returns
    -------
    deconv.SpatialNet
        Initialized SpatialNet model ready for training.
    
    Notes
    -----
    This function requires a GPU for optimal performance. The preprocessing steps include
    data normalization, gene filtering, and simulation of pseudo-spots using single-cell data.
    """
    print('Setting global seed:', seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    spatial_simulation.numba_set_seed(seed)
    numba.set_num_threads(n_threads)

    sc_ad = utils.normalize_adata(sc_ad,target_sum=1e4)
    st_ad = utils.normalize_adata(st_ad,target_sum=1e4)
    sc_ad, st_ad = utils.filter_model_genes(
        sc_ad,
        st_ad,
        celltype_key=celltype_key,
        deg_method=deg_method,
        n_top_markers=n_top_markers,
        n_top_hvg=n_top_hvg,
        used_genes=used_genes,
        sc_genes=sc_genes,
        st_genes=st_genes,
        log2fc_min=log2fc_min, 
        pval_cutoff=pval_cutoff, 
        pct_diff=pct_diff, 
        pct_min=pct_min
    )
    sm_ad =utils.generate_sm_adata(sc_ad,num_sample=sm_size,celltype_key=celltype_key,n_threads=n_threads,cell_counts=cell_counts,clusters_mean=clusters_mean,cells_mean=cells_mean,cells_min=cells_min,cells_max=cells_max,cell_sample_counts=cell_sample_counts,cluster_sample_counts=cluster_sample_counts,ncell_sample_list=ncell_sample_list,cluster_sample_list=cluster_sample_list)
    utils.downsample_sm_spot_counts(sm_ad,st_ad,n_threads=n_threads)

    model = deconv.SpatialNet(
        st_ad,
        sm_ad,
        clusters = np.array(sm_ad.obsm['label'].columns),
        spot_names = np.array(st_ad.obs_names),
        used_genes = np.array(st_ad.var_names),
        use_rep=use_rep,
        st_batch_key=st_batch_key,
        scvi_layers=scvi_layers,
        scvi_latent=scvi_latent,
        scvi_gene_likelihood=scvi_gene_likelihood,
        scvi_dispersion=scvi_dispersion,
        latent_dims=latent_dims, 
        hidden_dims=hidden_dims,
        infer_losses=infer_losses,
        use_gpu=use_gpu,
        seed=seed
    )
    return model