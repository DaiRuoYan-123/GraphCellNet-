import numpy as np
import pandas as pd
import os 
import matplotlib
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import seaborn as sns
import squidpy as sq
from .utils import clustering
import math
import numpy as np
from tqdm import tqdm
from time import strftime, localtime
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvLayer(Module):
    """
    Graph Convolutional Layer for graph neural networks.
    
    This layer implements a spectral graph convolution using Chebyshev polynomials
    to approximate the graph Laplacian.
    
    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    support : int
        Size of the filter support, i.e., number of Chebyshev polynomials to use.
    bias : bool, optional
        Whether to include a bias term. Default: True.
        
    Attributes
    ----------
    weight : torch.nn.Parameter
        Learnable weight matrix of shape (in_features * support, out_features).
    bias : torch.nn.Parameter
        Learnable bias vector of shape (out_features).
    
    Notes
    -----
    This implementation follows the approach described in the paper 
    "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering"
    by Defferrard et al.
    """
    def __init__(self, in_features, out_features, support, bias=True):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.support = support
        self.weight = Parameter(torch.FloatTensor(in_features * support, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize weights and bias parameters.
        
        Initializes weights using uniform distribution and
        bias using uniform distribution if present.
        """
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, features, basis):
        """
        Forward pass of the graph convolutional layer.
        
        Parameters
        ----------
        features : torch.Tensor
            Input node features of shape (n_nodes, in_features).
        basis : list of torch.Tensor
            List of Chebyshev polynomial basis functions of the graph Laplacian.
            Each element should be a sparse tensor of shape (n_nodes, n_nodes).
            
        Returns
        -------
        torch.Tensor
            Output node features of shape (n_nodes, out_features).
        """
        supports = list()
        for i in range(self.support):
            supports.append(basis[i].matmul(features))
        supports = torch.cat(supports, dim=1)
        output = torch.spmm(supports, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        """
        String representation of the layer.
        
        Returns
        -------
        str
            Layer description string.
        """
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
            

class GCNED(nn.Module):
    """
    Graph Convolutional Network Encoder-Decoder for spatial domain identification.
    
    This neural network uses graph convolutional layers to encode spatial transcriptomics
    data into a low-dimensional latent space and then decode it back to the original space.
    
    Parameters
    ----------
    feature_dims : int
        Number of input features (genes).
    support : int
        Size of the filter support for graph convolution.
    latent_dims : int, optional
        Number of dimensions in the latent space. Default: 8.
    hidden_dims : int, optional
        Number of hidden dimensions. Default: 64.
    dropout : float, optional
        Dropout rate for regularization. Default: 0.8.
        
    Attributes
    ----------
    encode_gc1 : GraphConvLayer
        First graph convolutional layer of the encoder.
    encode_gc2 : GraphConvLayer
        Second graph convolutional layer of the encoder.
    decode_gc1 : GraphConvLayer
        First graph convolutional layer of the decoder.
    decode_gc2 : GraphConvLayer
        Second graph convolutional layer of the decoder.
    """
    def __init__(self, feature_dims, support, latent_dims=8, hidden_dims=64, dropout=0.8):
        super(GCNED, self).__init__()
        self.feature_dims = feature_dims
        self.support = support
        self.latent_dims = latent_dims
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.encode_gc1 = GraphConvLayer(feature_dims, hidden_dims, support)
        self.encode_gc2 = GraphConvLayer(hidden_dims, latent_dims, support)
        self.decode_gc1 = GraphConvLayer(latent_dims, hidden_dims, support)
        self.decode_gc2 = GraphConvLayer(hidden_dims, feature_dims, support)
        
        nn.init.kaiming_normal_(self.encode_gc1.weight)
        nn.init.xavier_uniform_(self.encode_gc2.weight)
        nn.init.kaiming_normal_(self.decode_gc1.weight)
        nn.init.xavier_uniform_(self.decode_gc2.weight)
        
    @staticmethod
    def l2_activate(x, dim):
        """
        Apply min-max scaling followed by L2 normalization.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        dim : int
            Dimension along which to normalize.
            
        Returns
        -------
        torch.Tensor
            Normalized tensor.
        """
        def scale(z):
            zmax = z.max(1, keepdims=True).values
            zmin = z.min(1, keepdims=True).values
            z_std = torch.nan_to_num(torch.div(z - zmin,(zmax - zmin)),0)
            return z_std
        
        x = scale(x)
        x = F.normalize(x, p=2, dim=1)
        return x
        
    def encode(self, x, adj):
        """
        Encode input features into latent space.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (n_nodes, feature_dims).
        adj : list of torch.Tensor
            List of Chebyshev polynomial basis functions.
            
        Returns
        -------
        torch.Tensor
            Encoded latent representation of shape (n_nodes, latent_dims).
        """
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.leaky_relu(self.encode_gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.encode_gc2(x, adj)
        return self.l2_activate(x, dim=1)
    
    def decode(self, x, adj):
        """
        Decode latent representation back to feature space.
        
        Parameters
        ----------
        x : torch.Tensor
            Latent representation of shape (n_nodes, latent_dims).
        adj : list of torch.Tensor
            List of Chebyshev polynomial basis functions.
            
        Returns
        -------
        torch.Tensor
            Reconstructed features of shape (n_nodes, feature_dims).
        """
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.leaky_relu(self.decode_gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.decode_gc2(x, adj)
        return x

    def forward(self, x, adj):
        """
        Forward pass through the encoder-decoder network.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (n_nodes, feature_dims).
        adj : list of torch.Tensor
            List of Chebyshev polynomial basis functions.
            
        Returns
        -------
        z : torch.Tensor
            Encoded latent representation of shape (n_nodes, latent_dims).
        x_ : torch.Tensor
            Reconstructed features of shape (n_nodes, feature_dims).
        """
        z = self.encode(x, adj)
        x_ = self.decode(z, adj)
        return z, x_
    

class DiscNet(nn.Module):
    """
    Discriminator network for domain adaptation between different tissue sections.
    
    This network is used to classify which dataset/slice each spot belongs to,
    enabling adversarial training for domain adaptation.
    
    Parameters
    ----------
    label : torch.Tensor
        One-hot encoded tensor indicating which dataset/slice each spot belongs to.
    latent_dims : int, optional
        Number of dimensions in the latent space. Default: 8.
    hidden_dims : int, optional
        Number of hidden dimensions. Default: 64.
    dropout : float, optional
        Dropout rate for regularization. Default: 0.5.
        
    Attributes
    ----------
    disc : nn.Sequential
        Sequential model for classification.
    class_num : int
        Number of classes (datasets/slices).
    """
    def __init__(self, label, latent_dims=8, hidden_dims=64, dropout=0.5):
        super(DiscNet, self).__init__()
        self.latent_dims = latent_dims
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.class_num = label.shape[1]
        self.disc = nn.Sequential(
            nn.Linear(latent_dims, hidden_dims),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dims),
            nn.Linear(hidden_dims, hidden_dims),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dims),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, self.class_num)
        )

    def forward(self, x):
        """
        Forward pass through the discriminator network.
        
        Parameters
        ----------
        x : torch.Tensor
            Input latent representation of shape (n_nodes, latent_dims).
            
        Returns
        -------
        torch.Tensor
            Predicted probabilities for each class of shape (n_nodes, class_num).
        """
        x = self.disc(x)
        y = F.softmax(x, dim=1)
        return y


class GraphTrainer():
    """
    Trainer for graph-based spatial domain identification.
    
    This class implements adversarial training of a graph convolutional encoder-decoder
    and a discriminator to identify spatial domains across multiple tissue sections.
    
    Parameters
    ----------
    expr_ad_list : list
        List of AnnData objects containing expression data for each tissue section.
    n_clusters : int
        Number of spatial domains to identify.
    X : torch.Tensor
        Feature matrix of shape (n_total_spots, n_genes).
    graph : list
        List containing the scaled features and Chebyshev polynomial transforms.
    support : int
        Number of polynomial terms, representing the filter support size.
    slice_class_onehot : torch.Tensor
        One-hot encoded tensor indicating which dataset/slice each spot belongs to.
    nb_mask : numpy.ndarray
        Indices of spatial neighbors in the adjacency matrix.
    train_idx : numpy.ndarray
        Indices for training data.
    test_idx : numpy.ndarray
        Indices for testing data.
    celltype_weights : numpy.ndarray
        Weights for different cell types/genes in the loss function.
    morans_mean : numpy.ndarray
        Average Moran's I values for spatial autocorrelation.
    lr : float
        Learning rate for optimizers.
    l1 : float
        L1 regularization parameter.
    l2 : float
        L2 regularization parameter.
    latent_dim : int
        Number of dimensions in the latent space.
    hidden_dims : int
        Number of hidden dimensions.
    gnn_dropout : float
        Dropout rate for the graph neural network.
    use_gpu : bool
        Whether to use GPU for computation.
        
    Attributes
    ----------
    model_g : GCNED
        Graph convolutional encoder-decoder model.
    model_d : DiscNet
        Discriminator model.
    optimizer_g : torch.optim.Optimizer
        Optimizer for the encoder-decoder.
    optimizer_d : torch.optim.Optimizer
        Optimizer for the discriminator.
    Cluster : sklearn.cluster.KMeans
        KMeans clustering model for domain identification.
    best_path : str
        Path to the best model weights.
    """
    def __init__(
        self,
        expr_ad_list,
        n_clusters,
        X,
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
    ):
        self.expr_ad_list = expr_ad_list
        self.model_g = GCNED(X.shape[1],support,latent_dims=latent_dim,hidden_dims=hidden_dims,dropout=gnn_dropout)
        self.model_d = DiscNet(slice_class_onehot,latent_dims=latent_dim,hidden_dims=hidden_dims)
        self.graph = graph
        self.slice_class_onehot = slice_class_onehot
        self.nb_mask = nb_mask
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.celltype_weights = torch.tensor(celltype_weights)
        self.morans_mean = morans_mean
        self.best_path = None
        self.cos_loss_obj = F.cosine_similarity
        self.d_loss_obj = F.cross_entropy
        self.n_clusters = n_clusters
        self.Cluster = KMeans(n_clusters=self.n_clusters,n_init=10,tol=1e-3,algorithm='elkan',max_iter=1000,random_state=42)
        self.optimizer_g = optim.RMSprop(self.model_g.parameters(),lr=lr)
        self.optimizer_d = optim.RMSprop(self.model_d.parameters(),lr=lr)
        self.l1 = l1
        self.l2 = l2
        if use_gpu:
            self.model_g = self.model_g.cuda()
            self.model_d = self.model_d.cuda()
            self.celltype_weights = self.celltype_weights.cuda()
    
    @staticmethod
    def kl_divergence(y_true, y_pred, dim=0):
        """
        Compute KL divergence between two distributions.
        
        Parameters
        ----------
        y_true : torch.Tensor
            True distribution.
        y_pred : torch.Tensor
            Predicted distribution.
        dim : int, optional
            Dimension along which to compute KL divergence. Default: 0.
            
        Returns
        -------
        torch.Tensor
            KL divergence value.
        """
        y_pred = torch.clip(y_pred, torch.finfo(torch.float32).eps)
        y_true = y_true.to(y_pred.dtype)
        y_true = torch.nan_to_num(torch.div(y_true, y_true.sum(dim, keepdims=True)),0)
        y_pred = torch.nan_to_num(torch.div(y_pred, y_pred.sum(dim, keepdims=True)),0)
        y_true = torch.clip(y_true, torch.finfo(torch.float32).eps, 1)
        y_pred = torch.clip(y_pred, torch.finfo(torch.float32).eps, 1)
        return torch.mul(y_true, torch.log(torch.nan_to_num(torch.div(y_true, y_pred)))).mean(dim)
        
    def train_model_g(self, d_l, simi_l):
        """
        Train the graph convolutional encoder-decoder model.
        
        Parameters
        ----------
        d_l : float
            Weight for the discriminator loss.
        simi_l : float
            Weight for the similarity loss.
            
        Returns
        -------
        torch.Tensor
            Total loss value.
        """
        self.model_g.train()
        self.optimizer_g.zero_grad()
        encoded, decoded = self.model_g(self.graph[0],self.graph[1:])
        y_disc = self.model_d(encoded)
        d_loss = F.cross_entropy(self.slice_class_onehot, y_disc)
        decoded_mask = decoded[self.train_idx]
        x_mask = self.graph[0][self.train_idx]
        simi_loss = -torch.mean(torch.sum(encoded[self.nb_mask[0]] * encoded[self.nb_mask[1]], dim=1)) + torch.mean(torch.abs(encoded[self.nb_mask[0]]-encoded[self.nb_mask[1]]))
        g_loss = -torch.sum(self.celltype_weights*F.cosine_similarity(x_mask, decoded_mask,dim=0))+torch.sum(self.celltype_weights*self.kl_divergence(x_mask, decoded_mask, dim=0)) + simi_l*simi_loss

        total_loss = g_loss - d_l*d_loss
        total_loss.backward()
        self.optimizer_g.step()
        return total_loss
    
    def train_model_d(self,):
        """
        Train the discriminator model.
        
        Returns
        -------
        torch.Tensor
            Discriminator loss value.
        """
        self.model_d.train()
        self.optimizer_d.zero_grad()
        encoded, decoded = self.model_g(self.graph[0],self.graph[1:])
        y_disc = self.model_d(encoded)
        d_loss = F.cross_entropy(self.slice_class_onehot, y_disc)
        d_loss.backward()
        self.optimizer_d.step()
        return d_loss
    
    def test_model(self, d_l, simi_l):
        """
        Evaluate both models on test data.
        
        Parameters
        ----------
        d_l : float
            Weight for the discriminator loss.
        simi_l : float
            Weight for the similarity loss.
            
        Returns
        -------
        total_loss : torch.Tensor
            Combined loss value.
        g_loss : torch.Tensor
            Generator (encoder-decoder) loss value.
        d_loss : torch.Tensor
            Discriminator loss value.
        accuarcy : torch.Tensor
            Discriminator accuracy.
        simi_loss : torch.Tensor
            Similarity loss value.
        db_loss : float
            Davies-Bouldin clustering quality score.
        encoded : torch.Tensor
            Encoded latent representations.
        decoded : torch.Tensor
            Reconstructed features.
        """
        self.model_g.eval()
        self.model_d.eval()
        encoded, decoded = self.model_g(self.graph[0],self.graph[1:])
        y_disc = self.model_d(encoded)
        d_loss = F.cross_entropy(self.slice_class_onehot, y_disc)
        decoded_mask = decoded[self.test_idx]
        x_mask = self.graph[0][self.test_idx]
        ll = torch.eq(torch.argmax(self.slice_class_onehot, -1), torch.argmax(y_disc, -1))
        accuarcy = ll.to(torch.float32).mean()
        simi_loss = -torch.mean(torch.sum(encoded[self.nb_mask[0]] * encoded[self.nb_mask[1]], dim=1)) + torch.mean(torch.abs(encoded[self.nb_mask[0]]-encoded[self.nb_mask[1]]))
        g_loss = -torch.sum(self.celltype_weights*F.cosine_similarity(x_mask, decoded_mask,dim=0))+torch.sum(self.celltype_weights*self.kl_divergence(x_mask, decoded_mask, dim=0)) + simi_l*simi_loss
        total_loss = g_loss - d_l*d_loss
        db_loss = clustering(self.Cluster, encoded.cpu().detach().numpy())
        return total_loss, g_loss, d_loss, accuarcy, simi_loss, db_loss, encoded, decoded
    
    def train(
        self,
        max_epochs=300,
        convergence=0.0001,
        db_convergence=0,
        early_stop_epochs=10,
        d_l=0.5,
        simi_l=None,
        g_step=1,
        d_step=1,
        plot_step=5,
        save_path=None,
        prefix=None
    ):
        """
        Train the model with adversarial learning.
        
        Parameters
        ----------
        max_epochs : int, optional
            Maximum number of training epochs. Default: 300.
        convergence : float, optional
            Convergence threshold for generator loss. Default: 0.0001.
        db_convergence : float, optional
            Convergence threshold for Davies-Bouldin score. Default: 0.
        early_stop_epochs : int, optional
            Number of epochs without improvement before early stopping. Default: 10.
        d_l : float, optional
            Weight for the discriminator loss. Default: 0.5.
        simi_l : float, optional
            Weight for the similarity loss. If None, calculated from Moran's I. Default: None.
        g_step : int, optional
            Number of generator training steps per epoch. Default: 1.
        d_step : int, optional
            Number of discriminator training steps per epoch. Default: 1.
        plot_step : int, optional
            Number of epochs between evaluations. Default: 5.
        save_path : str, optional
            Directory to save model weights. Default: None (uses temporary directory).
        prefix : str, optional
            Prefix for saved model files. Default: None.
        """
        best_loss = np.inf
        best_db_loss = np.inf
        best_simi_loss = np.inf
        if simi_l is None:
            simi_l = 1/np.mean(self.morans_mean)
            print(f'Setting the weight of similarity loss to {simi_l:.3f}')
        
        if save_path is None:
            save_path = os.path.join(tempfile.gettempdir() ,'Graph_models_'+strftime("%Y%m%d%H%M%S",localtime()))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        early_stop_count = 0
        pbar = tqdm(range(max_epochs))
        for epoch in pbar:
            for _ in range(g_step):
                train_total_loss = self.train_model_g(d_l=d_l, simi_l=simi_l)
            for _ in range(d_step):
                self.train_model_d()

            if epoch % plot_step == 0:
                test_total_loss, test_g_loss, test_d_loss, test_acc, simi_loss, db_loss, encoded, decoded = self.test_model(d_l=d_l, simi_l=simi_l)
                current_loss = test_g_loss.cpu().detach().numpy()
                current_db_loss = db_loss
                if (best_loss - current_loss > convergence) & (best_db_loss - current_db_loss > db_convergence):
                    if best_loss > current_loss:
                        best_loss = current_loss
                    if best_db_loss > current_db_loss:
                        best_db_loss = current_db_loss
                    pbar.set_description("The best epoch {0} total loss={1:.3f} g loss={2:.3f} d loss={3:.3f} d acc={4:.3f} simi loss={5:.3f} db loss={6:.3f}".format(epoch, test_total_loss, test_g_loss, test_d_loss, test_acc, simi_loss, db_loss),refresh=True)
                    old_best_path = self.best_path
                    early_stop_count = 0
                    if prefix is not None:
                        self.best_path = os.path.join(save_path,prefix+'_'+f'Graph_weights_epoch{epoch}.h5')
                    else:
                        self.best_path = os.path.join(save_path,f'Graph_weights_epoch{epoch}.h5')
                    if old_best_path is not None:
                        if os.path.exists(old_best_path):
                            os.remove(old_best_path)
                    torch.save(self.model_g.state_dict(), self.best_path)
                else:
                    early_stop_count += 1
                
                if early_stop_count > early_stop_epochs:
                    print('Stop trainning because of loss convergence')
                    break
    
    def identify_spatial_domain(self, key=None, colors=None):
        """
        Identify spatial domains in all tissue sections.
        
        Parameters
        ----------
        key : str, optional
            Key to store spatial domain annotations in AnnData objects. Default: 'spatial_domain'.
        colors : list, optional
            List of colors for spatial domains. Default: None (uses tab10/tab20 palette).
            
        Notes
        -----
        This method:
        1. Loads the best model weights
        2. Computes latent representations for all spots
        3. Clusters them using KMeans
        4. Assigns the resulting cluster IDs to each tissue section
        5. Updates the AnnData objects with domain annotations and colors
        """
        if colors is None:
            if self.n_clusters > 10:
                colors = [matplotlib.colors.to_hex(c) for c in sns.color_palette('tab20',n_colors=self.n_clusters)]
            else:
                colors = [matplotlib.colors.to_hex(c) for c in sns.color_palette('tab10',n_colors=self.n_clusters)]
            color_map = pd.DataFrame(colors,index=np.arange(self.n_clusters),columns=['color'])
        if key is None:
            key = 'spatial_domain'
        self.model_g.load_state_dict(torch.load(self.best_path))
        self.model_g.eval()
        encoded, decoded = self.model_g(self.graph[0], self.graph[1:])
        clusters = self.Cluster.fit_predict(encoded.cpu().detach().numpy())
        loc_index = 0
        for i in range(len(self.expr_ad_list)):
            if key in self.expr_ad_list[i].obs.columns:
                self.expr_ad_list[i].obs = self.expr_ad_list[i].obs.drop(columns=key)
            self.expr_ad_list[i].obs[key] = clusters[loc_index:loc_index+self.expr_ad_list[i].shape[0]]
            self.expr_ad_list[i].obs[key] = pd.Categorical(self.expr_ad_list[i].obs[key])
            self.expr_ad_list[i].uns[f'{key}_colors'] = [color_map.loc[c,'color'] for c in self.expr_ad_list[i].obs[key].cat.categories]
            loc_index += self.expr_ad_list[i].shape[0]