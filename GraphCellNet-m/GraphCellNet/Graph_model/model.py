import numpy as np
import pandas as pd
import random
import torch
import squidpy as sq
from .graph import get_graph_inputs,get_graph_kernel,split_train_test_idx
from .utils import generate_celltype_ad_list,cal_celltype_weight
from .Graph_model import GraphTrainer

def init_model(
    expr_ad_list:list,
    n_clusters:int,
    k:int=2,
    use_weight=True,
    train_prop:float=0.5,
    n_neighbors=6,
    min_prop=0.01,
    lr:float=3e-3,
    l1:float=0.01,
    l2:float=0.01,
    latent_dim:int=16,
    hidden_dims:int=64,
    gnn_dropout:float=0.8,
    simi_neighbor=1,
    use_gpu=None,
    seed=42
)->GraphTrainer:
 """
    Initialize a graph-based model for spatial domain identification across multiple tissue sections.
    
    This function prepares and initializes all components required for the graph-based
    spatial domain identification model, including:
    1. Setting random seeds for reproducibility
    2. Computing spatial neighbors if not already present
    3. Generating cell type-specific AnnData objects
    4. Computing gene weights based on spatial autocorrelation
    5. Creating graph inputs and kernels
    6. Initializing the GraphTrainer
    
    Parameters
    ----------
    expr_ad_list : list
        List of AnnData objects containing gene expression data for each tissue section.
    n_clusters : int
        Number of spatial domains to identify.
    k : int, optional
        Order of the Chebyshev polynomial for graph convolution. Higher values
        capture larger graph neighborhoods but increase computational complexity. Default: 2.
    use_weight : bool, optional
        Whether to use gene-specific weights based on spatial autocorrelation. Default: True.
    train_prop : float, optional
        Proportion of data to use for training, between 0 and 1. Default: 0.5.
    n_neighbors : int, optional
        Number of spatial neighbors to consider for each spot when constructing
        the spatial graph. Default: 6.
    min_prop : float, optional
        Minimum proportion threshold for filtering genes. Default: 0.01.
    lr : float, optional
        Learning rate for the optimizers. Default: 3e-3.
    l1 : float, optional
        L1 regularization parameter. Default: 0.01.
    l2 : float, optional
        L2 regularization parameter. Default: 0.01.
    latent_dim : int, optional
        Number of dimensions in the latent space. Default: 16.
    hidden_dims : int, optional
        Number of hidden dimensions in the neural networks. Default: 64.
    gnn_dropout : float, optional
        Dropout rate for the graph neural network. Default: 0.8.
    simi_neighbor : int or None, optional
        Type of neighbors to use for similarity loss. If 1, uses direct neighbors;
        if None, uses all connected components. Default: 1.
    use_gpu : bool or None, optional
        Whether to use GPU for computation. If None, automatically detected. Default: None.
    seed : int, optional
        Random seed for reproducibility. Default: 42.
        
    Returns
    -------
    GraphTrainer
        Initialized GraphTrainer object ready for training.
        
    Notes
    -----
    This function performs several preprocessing steps:
    
    1. Sets random seeds for all components to ensure reproducibility
    2. Computes spatial neighborhood graphs for each tissue section if not already present
    3. Generates cell type-specific AnnData objects by filtering genes based on minimum proportion
    4. Calculates gene weights based on spatial autocorrelation (Moran's I)
    5. Prepares graph inputs including feature matrix, adjacency matrix, and polynomial kernels
    6. Splits data into training and testing sets
    7. Configures GPU usage if available
    
    The resulting GraphTrainer can be used to identify spatial domains that are consistent
    across multiple tissue sections, by training with adversarial learning to encourage
    domain representations that are section-invariant.
    
    Requires GPU for optimal performance, especially with large datasets or complex graphs.
    """
    print('Setting global seed:', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if use_gpu is None:
        if torch.cuda.is_available():
            use_gpu = True
        else:
            use_gpu = False
    
    for expr_ad in expr_ad_list:
        if 'spatial_connectivities' not in expr_ad.obsp.keys():
            sq.gr.spatial_neighbors(expr_ad,coord_type='grid',n_neighs=n_neighbors)
    celltype_ad_list = generate_celltype_ad_list(expr_ad_list,min_prop=min_prop)
    celltype_weights,morans_mean = cal_celltype_weight(celltype_ad_list)
    kept_ind = celltype_weights > 0
    if not use_weight:
        celltype_weights = np.ones(len(celltype_weights))/len(celltype_weights)
    X,A,nb_mask,slice_class_onehot = get_graph_inputs(celltype_ad_list)
    X_filtered, graph, support = get_graph_kernel(X[:,kept_ind],A,k=k)
    celltype_weights = celltype_weights[kept_ind]
    morans_mean = morans_mean[kept_ind]
    if simi_neighbor == 1:
        nb_mask = nb_mask
    elif simi_neighbor == None:
        nb_mask = np.array(np.where(graph[-1].to_dense())!=0)
        nb_mask = nb_mask[:,nb_mask[0] != nb_mask[1]]
    else:
        raise ValueError('simi_neighbor must be 1 or None.')
    train_idx,test_idx = split_train_test_idx(X,train_prop=0.5)
    if use_gpu:
        for i in range(len(graph)):
            graph[i] = graph[i].cuda()
        slice_class_onehot = slice_class_onehot.cuda()
    return GraphTrainer(
        expr_ad_list,
        n_clusters,
        X_filtered,
        graph,
        support,
        slice_class_onehot,
        nb_mask,
        train_idx,
        test_idx,
        celltype_weights,
        morans_mean,
        lr,
        l1,
        l2,
        latent_dim,
        hidden_dims,
        gnn_dropout,
        use_gpu
    )
