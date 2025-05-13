import random
import anndata
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics import davies_bouldin_score

def one_hot_encode(labels, unique_labels=None):
    """
    Convert categorical labels to one-hot encoded format.
    
    Parameters
    ----------
    labels : array-like
        Array of categorical labels to encode.
    unique_labels : array-like, optional
        Predefined list of all possible unique labels. If None, unique labels
        are extracted from the provided labels. Default: None.
        
    Returns
    -------
    encoded : numpy.ndarray
        One-hot encoded array of shape (n_samples, n_classes).
    unique_labels : numpy.ndarray
        Array of unique label values used for encoding.
        
    Notes
    -----
    This function creates a binary matrix where each row corresponds to a sample
    and each column corresponds to a unique label. A value of 1 indicates the
    presence of the corresponding label for that sample.
    
    Using a predefined set of unique_labels ensures consistent encoding across
    multiple calls, which is useful when encoding test data with the same
    categories as training data.
    """
    if unique_labels is None:
        unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    encoded = np.eye(num_classes)[np.array([label_map[label] for label in labels])]
    return encoded, unique_labels


def add_cell_type_composition(ad, prop_df=None, celltype_anno=None, all_celltypes=None):
    """
    Add cell type composition information to an AnnData object.
    
    Parameters
    ----------
    ad : AnnData
        AnnData object to which cell type composition will be added.
    prop_df : pandas.DataFrame, optional
        DataFrame containing cell type proportions with spots as rows and
        cell types as columns. Default: None.
    celltype_anno : array-like, optional
        Array of cell type annotations for each spot. Default: None.
    all_celltypes : array-like, optional
        Complete list of cell types to consider. Useful for ensuring consistent
        cell type sets across multiple datasets. Default: None.
        
    Returns
    -------
    None
        The function modifies the AnnData object in-place.
        
    Notes
    -----
    This function adds cell type composition information to the AnnData object
    in two ways:
    1. If prop_df is provided, it adds the proportions directly to ad.obs
    2. If celltype_anno is provided, it converts the annotations to one-hot encoding
       and adds them to ad.obs
       
    The cell type names are also stored in ad.uns['celltypes'] for reference.
    
    Raises
    ------
    ValueError
        If both prop_df and celltype_anno are None.
    """
    if prop_df is not None:
        if all_celltypes is not None:
            prop_df.loc[:,np.setdiff1d(all_celltypes, prop_df.columns)] = 0
        ad.obs[prop_df.columns] = prop_df.values
        ad.uns['celltypes'] = prop_df.columns
    elif celltype_anno is not None:
        encoded, unique_celltypes = one_hot_encode(celltype_anno, all_celltypes)
        ad.obs[unique_celltypes] = encoded
        ad.uns['celltypes'] = unique_celltypes
    else:
        raise ValueError("prop_df and celltype_anno can not both be None.")


def _morans_i_mtx(
    g_data: np.ndarray,
    g_indices: np.ndarray,
    g_indptr: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """
    Calculate Moran's I statistic for multiple features in a spatial graph.
    
    Parameters
    ----------
    g_data : numpy.ndarray
        Data array of the spatial connectivity matrix in CSR format.
    g_indices : numpy.ndarray
        Indices array of the spatial connectivity matrix in CSR format.
    g_indptr : numpy.ndarray
        Index pointer array of the spatial connectivity matrix in CSR format.
    X : numpy.ndarray
        Feature matrix of shape (n_features, n_spots) to calculate Moran's I for.
        
    Returns
    -------
    numpy.ndarray
        Array of Moran's I values for each feature, of shape (n_features,).
        
    Notes
    -----
    Moran's I is a measure of spatial autocorrelation, ranging from -1 (perfect
    dispersion) to 1 (perfect correlation), with 0 indicating random spatial distribution.
    
    This function efficiently calculates Moran's I for multiple features using the
    CSR format of the spatial connectivity matrix, which represents the spatial
    relationships between spots.
    """
    M, N = X.shape
    assert N == len(g_indptr) - 1
    W = g_data.sum()
    out = np.zeros(M, dtype=np.float_)
    for k in range(M):
        x = X[k, :]
        out[k] = _morans_i_vec_W(g_data, g_indices, g_indptr, x, W)
    return out


def _morans_i_vec_W(
    g_data: np.ndarray,
    g_indices: np.ndarray,
    g_indptr: np.ndarray,
    x: np.ndarray,
    W: np.float_,
) -> float:
    """
    Calculate Moran's I statistic for a single feature in a spatial graph.
    
    Parameters
    ----------
    g_data : numpy.ndarray
        Data array of the spatial connectivity matrix in CSR format.
    g_indices : numpy.ndarray
        Indices array of the spatial connectivity matrix in CSR format.
    g_indptr : numpy.ndarray
        Index pointer array of the spatial connectivity matrix in CSR format.
    x : numpy.ndarray
        Feature vector of shape (n_spots,) to calculate Moran's I for.
    W : float
        Sum of all weights in the spatial connectivity matrix.
        
    Returns
    -------
    float
        Moran's I value for the feature.
        
    Notes
    -----
    This function implements the Moran's I calculation for a single feature:
    
    I = (N/W) * (∑ᵢⱼ wᵢⱼ(xᵢ-x̄)(xⱼ-x̄)) / (∑ᵢ(xᵢ-x̄)²)
    
    where:
    - N is the number of spatial units (spots)
    - wᵢⱼ is the spatial weight between spots i and j
    - xᵢ is the feature value at spot i
    - x̄ is the mean of the feature
    - W is the sum of all wᵢⱼ
    """
    z = x - x.mean()
    z2ss = (z * z).sum()
    N = len(x)
    inum = 0.0

    for i in range(N):
        s = slice(g_indptr[i], g_indptr[i + 1])
        i_indices = g_indices[s]
        i_data = g_data[s]
        inum += (i_data * z[i_indices]).sum() * z[i]

    return len(x) / W * inum / z2ss


def fill_low_prop(ad, min_prop):
    """
    Filter out low cell type proportions from an AnnData object.
    
    Parameters
    ----------
    ad : AnnData
        AnnData object containing cell type proportions in the X matrix.
    min_prop : float
        Minimum proportion threshold. Values below this threshold will be set to 0.
        
    Returns
    -------
    AnnData
        Modified AnnData object with low proportions removed.
        
    Notes
    -----
    This function is useful for filtering out noise or spurious cell type assignments
    in spatial deconvolution results, where very low proportions may not be reliable
    or biologically meaningful.
    """
    mtx = ad.X
    mtx[mtx < min_prop] = 0
    ad.X = mtx
    return ad


def cal_celltype_moran(ad):
    """
    Calculate Moran's I statistic for cell type proportions in an AnnData object.
    
    Parameters
    ----------
    ad : AnnData
        AnnData object containing cell type proportions in the X matrix and
        spatial connectivity information in obsp['spatial_connectivities'].
        
    Returns
    -------
    None
        The function stores Moran's I values in ad.uns['moran_vals'].
        
    Notes
    -----
    This function calculates the spatial autocorrelation (Moran's I) for each
    cell type proportion across the spatial tissue. High Moran's I values indicate
    that cell types are spatially clustered, while low values indicate random
    or dispersed distributions.
    
    The spatial connectivity matrix must be precomputed and stored in the AnnData
    object as obsp['spatial_connectivities'] in CSR format.
    """
    moran_vals = _morans_i_mtx(
        ad.obsp['spatial_connectivities'].data,
        ad.obsp['spatial_connectivities'].indices,
        ad.obsp['spatial_connectivities'].indptr,
        ad.X.T
    )
    ad.uns['moran_vals'] = np.nan_to_num(moran_vals)
    

def cal_celltype_weight(ad_list):
    """
    Calculate cell type weights based on spatial autocorrelation across multiple datasets.
    
    Parameters
    ----------
    ad_list : list
        List of AnnData objects containing cell type proportions and spatial information.
        
    Returns
    -------
    celltype_weights : numpy.ndarray
        Normalized weights for each cell type based on their average Moran's I values.
    morans_mean : numpy.ndarray
        Average Moran's I values for each cell type across all datasets.
        
    Notes
    -----
    This function:
    1. Calculates Moran's I for each cell type in each dataset
    2. Averages these values across datasets
    3. Normalizes the average values to create weights
    
    Cell types with higher spatial autocorrelation receive higher weights, which
    can be used to prioritize spatially structured cell types in downstream analyses.
    """
    print('Calculating cell type weights...')
    for ad in ad_list:
        cal_celltype_moran(ad)
    moran_min=-1
    morans = ad_list[0].uns['moran_vals'].copy()
    for i, ad in enumerate(ad_list[1:]):
        morans += ad.uns['moran_vals'].copy()
    morans_mean = morans/len(ad_list)
    celltype_weights = morans_mean/morans_mean.sum()
    return celltype_weights, morans_mean


def generate_celltype_ad_list(expr_ad_list, min_prop):
    """
    Generate a list of AnnData objects containing only cell type proportion information.
    
    Parameters
    ----------
    expr_ad_list : list
        List of AnnData objects containing expression data and cell type proportions.
    min_prop : float
        Minimum proportion threshold. Values below this threshold will be set to 0.
        
    Returns
    -------
    list
        List of new AnnData objects containing only cell type proportions.
        
    Notes
    -----
    This function extracts cell type proportion information from each AnnData object
    in the input list and creates new AnnData objects with:
    - Cell type proportions as the X matrix
    - Original observation metadata (obs)
    - Original observation-level embeddings (obsm)
    - Original observation-level pairwise relationships (obsp)
    - Low proportions filtered out based on min_prop threshold
    
    This is useful for focusing downstream analyses on cell type composition
    rather than gene expression.
    """
    celltype_ad_list = []
    for expr_ad in expr_ad_list:
        celltype_ad = anndata.AnnData(expr_ad.obs[[c for c in expr_ad.uns['celltypes']]])
        celltype_ad.obs = expr_ad.obs
        celltype_ad.obsm = expr_ad.obsm
        celltype_ad.obsp = expr_ad.obsp
        celltype_ad = fill_low_prop(celltype_ad, min_prop)
        celltype_ad_list.append(celltype_ad)
    return celltype_ad_list


def clustering(Cluster, feature):
    """
    Perform clustering and evaluate quality using Davies-Bouldin score.
    
    Parameters
    ----------
    Cluster : sklearn.cluster estimator
        Clustering algorithm with fit_predict method (e.g., KMeans).
    feature : numpy.ndarray
        Feature matrix of shape (n_samples, n_features) to cluster.
        
    Returns
    -------
    float
        Davies-Bouldin score of the clustering result (lower is better).
        
    Notes
    -----
    This function:
    1. Applies the clustering algorithm to the feature matrix
    2. Prints the predicted cluster labels for inspection
    3. Calculates the Davies-Bouldin score to evaluate clustering quality
    
    The Davies-Bouldin score is a metric for evaluating clustering algorithms,
    where a lower score indicates better clustering (i.e., clusters are more
    compact and better separated).
    
    Raises
    ------
    ValueError
        If the clustering result contains fewer than 2 clusters, as Davies-Bouldin
        score requires at least 2 clusters.
    """
    predict_labels = Cluster.fit_predict(feature)
    print("Predicted labels:", predict_labels)  # Print predicted labels
    if len(set(predict_labels)) <= 1:
        raise ValueError("The clustering result contains fewer than 2 clusters.")
    db = davies_bouldin_score(feature, predict_labels)
    return db

    
def split_ad(ad, by):
    """
    Split an AnnData object into multiple objects based on a categorical variable.
    
    Parameters
    ----------
    ad : AnnData
        AnnData object to split.
    by : str
        Key in ad.obs for the categorical variable to split by.
        
    Returns
    -------
    list
        List of AnnData objects, one for each unique category in the specified variable.
        
    Notes
    -----
    This function is useful for dividing a dataset into subsets based on metadata
    categories such as batch, condition, or tissue region. Each returned AnnData
    object is a deep copy of the subset of the original data.
    
    Example
    -------
    >>> # Split an AnnData object by sample ID
    >>> ad_list = split_ad(ad, 'sample_id')
    >>> # Result is a list with one AnnData per sample
    """
    ad_list = []
    for s in np.unique(ad.obs[by]):
        ad_split = ad[ad.obs[by] == s].copy()
        ad_list.append(ad_split)
    return ad_list