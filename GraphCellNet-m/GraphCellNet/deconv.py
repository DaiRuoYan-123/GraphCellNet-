from . import model
from . import utils
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import anndata
from . import metrics
import matplotlib.pyplot as plt
import os
import tempfile
import itertools
from functools import partial
from tqdm import tqdm
from time import strftime, localtime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, RandomSampler, BatchSampler

from . import kan
#from . import fast_kan
import psutil
from time import time

def compute_kernel(x, y):
    """
    Compute RBF kernel between two batches of vectors.
    
    Parameters
    ----------
    x : torch.Tensor
        First batch of vectors with shape (x_size, dim).
    y : torch.Tensor
        Second batch of vectors with shape (y_size, dim).
        
    Returns
    -------
    torch.Tensor
        Kernel matrix with shape (x_size, y_size) where each element is the RBF kernel
        evaluation between corresponding vectors from x and y.
    """
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)

    tiled_x = x.unsqueeze(1).expand(x_size, y_size, dim)
    tiled_y = y.unsqueeze(0).expand(x_size, y_size, dim)

    kernel = torch.exp(-torch.square(tiled_x - tiled_y).mean(dim=2) / dim)
    return kernel


def compute_mmd(x, y):
    """
    Compute Maximum Mean Discrepancy (MMD) between two batches of vectors using RBF kernel.
    
    MMD is a measure of distance between two distributions, useful for domain adaptation tasks.
    
    Parameters
    ----------
    x : torch.Tensor
        First batch of vectors.
    y : torch.Tensor
        Second batch of vectors.
        
    Returns
    -------
    torch.Tensor
        Scalar value representing the MMD distance between the distributions of x and y.
    """
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)

    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
    return mmd


class SelfAttention(nn.Module):
    def __init__(self, hidden_dims):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dims, hidden_dims)
        self.key = nn.Linear(hidden_dims, hidden_dims)
        self.value = nn.Linear(hidden_dims, hidden_dims)
        self.scale = hidden_dims ** -0.5

    def forward(self, x):
        # Ensure input x is 3D (batch_size, seq_len, hidden_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add a sequence length dimension

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_weights = F.softmax(torch.bmm(Q, K.transpose(1, 2)) * self.scale, dim=-1)
        out = torch.bmm(attn_weights, V)
        
        # Remove the sequence length dimension if it was added
        if out.size(1) == 1:
            out = out.squeeze(1)
        
        return out

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dims, dropout):
        super(ResidualBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dims),
            nn.Dropout(dropout)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dims),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        identity = x
        out = self.layer1(x)
        out = self.layer2(out)
        out += identity
        return out


class CellTypePredictionModel(nn.Module):
    def __init__(
        self,
        input_dims,
        latent_dims,
        hidden_dims,
        celltype_dims,
        dropout
    ):
        super(CellTypePredictionModel, self).__init__()
        print("all")
        self.encoder = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dims),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dims, dropout),
            nn.Linear(hidden_dims, latent_dims)
        )
        self.decoder = nn.Sequential(
            nn.Linear(celltype_dims, hidden_dims),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dims),
            SelfAttention(hidden_dims),
            ResidualBlock(hidden_dims, dropout),
            nn.Linear(hidden_dims, hidden_dims),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dims),
            nn.Linear(hidden_dims, hidden_dims),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dims),
            nn.Linear(hidden_dims, input_dims)
        )
        self.pred = nn.Sequential(
            kan.KANLinear(latent_dims, hidden_dims).cuda(),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dims),
            nn.Dropout(dropout),
            kan.KANLinear(hidden_dims, celltype_dims).cuda(),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        z = self.encoder(x)
        pred = self.pred(z)
        decoded = self.decoder(pred)
        return z, pred, decoded



class SpatialNet():
    """
    SpatialNet is a deep learning model for spatial transcriptomics cell type deconvolution.
    
    This model leverages single-cell RNA-seq data to deconvolve cell types in spatial transcriptomics data.
    It uses a neural network architecture with KAN (Kolmogorov-Arnold Network) layers for improved
    representation learning.
    
    Parameters
    ----------
    st_ad : AnnData
        Spatial transcriptomics AnnData object.
    sm_ad : AnnData
        Single-cell RNA-seq AnnData object.
    clusters : list
        List of cell type names to deconvolve.
    used_genes : list
        List of genes used for analysis.
    spot_names : list
        List of spot names in spatial data.
    use_rep : str
        Which representation to use. Options: 'scvi', 'X', 'pca'. Default: 'scvi'.
    st_batch_key : str, optional
        Batch key for spatial data if batches exist.
    scvi_layers : int, optional
        Number of layers in scVI model. Default: 2.
    scvi_latent : int, optional
        Latent dimension in scVI model. Default: 64.
    scvi_gene_likelihood : str, optional
        Gene likelihood model in scVI. Options: 'zinb', 'nb', 'poisson'. Default: 'zinb'.
    scvi_dispersion : str, optional
        Dispersion model in scVI. Options: 'gene', 'gene-batch', 'gene-label', 'gene-cell'. Default: 'gene-batch'.
    latent_dims : int, optional
        Latent dimensions in SpatialNet model. Default: 32.
    hidden_dims : int, optional
        Hidden dimensions in SpatialNet model. Default: 512.
    infer_losses : list, optional
        List of loss functions for inference. Options: 'kl', 'cos', 'rmse'. Default: ['kl', 'cos'].
    l1 : float, optional
        L1 regularization strength. Default: 0.01.
    l2 : float, optional
        L2 regularization strength. Default: 0.01.
    sm_lr : float, optional
        Learning rate for single-cell model optimization. Default: 3e-4.
    st_lr : float, optional
        Learning rate for spatial model optimization. Default: 3e-5.
    use_gpu : bool, optional
        Whether to use GPU if available. Default: True.
    seed : int, optional
        Random seed for reproducibility. Default: 42.
    
    Notes
    -----
    This model requires a GPU for optimal performance.
    """
    def __init__(
        self,
        st_ad,
        sm_ad,
        clusters,
        used_genes,
        spot_names,
        use_rep,
        st_batch_key=None,
        scvi_layers=2,
        scvi_latent=64,
        scvi_gene_likelihood='zinb',
        scvi_dispersion='gene-batch',
        latent_dims=32,
        hidden_dims=512,
        infer_losses=['kl', 'cos'],
        l1=0.01,
        l2=0.01,
        sm_lr=3e-4,
        st_lr=3e-5,
        use_gpu=True,
        seed=42,

    ):
        if ((use_gpu is None) or (use_gpu is True)) and (torch.cuda.is_available()):
            self.device = 'cuda',

        else:
            self.device = 'cpu'
        self.use_gpu = use_gpu

        self.st_ad = st_ad
        self.sm_ad = sm_ad
        self.scvi_dims = 64
        self.spot_names = spot_names
        self.used_genes = used_genes
        self.clusters = clusters
        self.st_batch_key = st_batch_key
        self.scvi_layers = scvi_layers
        self.scvi_latent = scvi_latent
        self.scvi_gene_likelihood = scvi_gene_likelihood
        self.scvi_dispersion = scvi_dispersion
        self.kl_infer_loss_func = partial(self.kl_divergence, dim=1)
        self.kl_rec_loss_func = partial(self.kl_divergence, dim=1)
        self.cosine_infer_loss_func = partial(F.cosine_similarity, dim=1)
        self.cosine_rec_loss_func = partial(F.cosine_similarity, dim=1)
        self.rmse_loss_func = self.rmse
        self.infer_losses = infer_losses
        self.mmd_loss = compute_mmd
        self.l1 = l1
        self.l2 = l2
        self.use_rep = use_rep
        if use_rep == 'scvi':
            self.feature_dims = scvi_latent
        elif use_rep == 'X':
            self.feature_dims = st_ad.shape[1]
        elif use_rep == 'pca':
            self.feature_dims = 50
        else:
            raise ValueError('use_rep must be one of scvi, pca and X.')
        self.latent_dims = latent_dims
        self.hidden_dims = hidden_dims
        self.sm_lr = sm_lr
        self.st_lr = st_lr
        self.init_model()
        self.st_data = None
        self.sm_data = None
        self.sm_labels = None
        self.best_path = None
        self.history = pd.DataFrame(columns=['sm_train_rec_loss', 'sm_train_infer_loss', 'sm_test_rec_loss', 'sm_test_infer_loss',
                                    'st_train_rec_loss', 'st_test_rec_loss', 'st_train_mmd_loss', 'st_test_mmd_loss', 'is_best'])
        self.batch_size = None
        self.seed = seed

    @staticmethod
    def rmse(y_true, y_pred):
        mse = F.mse_loss(y_pred, y_true)
        rmse = torch.sqrt(mse)
        return rmse

    @staticmethod
    def kl_divergence(y_true, y_pred, dim=0):
        y_pred = torch.clip(y_pred, torch.finfo(torch.float32).eps)
        y_true = y_true.to(y_pred.dtype)
        y_true = torch.nan_to_num(
            torch.div(y_true, y_true.sum(dim, keepdims=True)), 0)
        y_pred = torch.nan_to_num(
            torch.div(y_pred, y_pred.sum(dim, keepdims=True)), 0)
        y_true = torch.clip(y_true, torch.finfo(torch.float32).eps, 1)
        y_pred = torch.clip(y_pred, torch.finfo(torch.float32).eps, 1)
        return torch.mul(y_true, torch.log(torch.nan_to_num(torch.div(y_true, y_pred)))).mean(dim)

    def init_model(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = CellTypePredictionModel(self.feature_dims, self.latent_dims, self.hidden_dims, len(
            self.clusters), 0.8).to(self.device)
        self.sm_optimizer = optim.Adam(list(self.model.encoder.parameters(
        ))+list(self.model.pred.parameters()), lr=self.sm_lr,weight_decay=self.l2)
        self.st_optimizer = optim.Adam(list(self.model.encoder.parameters(
        ))+list(self.model.decoder.parameters()), lr=self.st_lr)

    def get_scvi_latent(
        self,
        n_layers=None,
        n_latent=None,
        gene_likelihood=None,
        dispersion=None,
        max_epochs=100,
        early_stopping=True,
        batch_size=4096,
    ):
        if self.st_batch_key is not None:
            if 'simulated' in self.st_ad.obs[self.st_batch_key]:
                raise ValueError(
                    f'obs[{self.st_batch_key}] cannot include "real".')
            self.st_ad.obs["batch"] = self.st_ad.obs[self.st_batch_key].astype(
                str)
            self.sm_ad.obs["batch"] = 'simulated'
        else:
            self.st_ad.obs["batch"] = 'real'
            self.sm_ad.obs["batch"] = 'simulated'

        adata = sc.concat([self.st_ad, self.sm_ad])
        adata.layers["counts"] = adata.X.copy()

        scvi.model.SCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch"
        )
        if n_layers is None:
            n_layers = self.scvi_layers
        if n_latent is None:
            n_latent = self.scvi_latent
        if gene_likelihood is None:
            gene_likelihood = self.scvi_gene_likelihood
        if dispersion is None:
            dispersion = self.scvi_dispersion
        vae = scvi.model.SCVI(adata, n_layers=n_layers, n_latent=n_latent,
                              gene_likelihood=gene_likelihood, dispersion=dispersion)
        vae.train(max_epochs=max_epochs, early_stopping=early_stopping,
                  batch_size=batch_size, use_gpu=self.use_gpu)
        adata.obsm["X_scVI"] = vae.get_latent_representation()

        st_scvi_ad = anndata.AnnData(
            adata[adata.obs['batch'] != 'simulated'].obsm["X_scVI"])
        sm_scvi_ad = anndata.AnnData(
            adata[adata.obs['batch'] == 'simulated'].obsm["X_scVI"])

        st_scvi_ad.obs = self.st_ad.obs
        st_scvi_ad.obsm = self.st_ad.obsm

        sm_scvi_ad.obs = self.sm_ad.obs
        sm_scvi_ad.obsm = self.sm_ad.obsm

        sm_scvi_ad = utils.check_data_type(sm_scvi_ad)
        st_scvi_ad = utils.check_data_type(st_scvi_ad)

        self.sm_data = sm_scvi_ad.X
        self.sm_labels = sm_scvi_ad.obsm['label'].values
        self.st_data = st_scvi_ad.X

        return sm_scvi_ad, st_scvi_ad

    def build_dataset(self, batch_size, device=None):
        if device is None:
            device = self.device
        x_train, y_train, x_test, y_test = utils.split_shuffle_data(np.array(
            self.sm_data, dtype=np.float32), np.array(self.sm_labels, dtype=np.float32))

        x_train = torch.tensor(x_train).to(device)
        y_train = torch.tensor(y_train).to(device)
        x_test = torch.tensor(x_test).to(device)
        y_test = torch.tensor(y_test).to(device)
        st_data = torch.tensor(self.st_data).to(device)

        self.sm_train_ds = TensorDataset(x_train, y_train)
        self.sm_test_ds = TensorDataset(x_test, y_test)
        self.st_ds = TensorDataset(st_data)

        self.sm_train_batch_size = min(len(self.sm_train_ds), batch_size)
        self.sm_test_batch_size = min(len(self.sm_test_ds), batch_size)
        self.st_batch_size = min(len(self.st_ds), batch_size)

        g = torch.Generator()
        g.manual_seed(self.seed)
        self.sm_train_sampler = BatchSampler(RandomSampler(
            self.sm_train_ds, generator=g), batch_size=self.sm_train_batch_size, drop_last=True)
        self.sm_test_sampler = BatchSampler(RandomSampler(
            self.sm_test_ds, generator=g), batch_size=self.sm_test_batch_size, drop_last=True)
        self.st_sampler = BatchSampler(RandomSampler(
            self.st_ds, generator=g), batch_size=self.st_batch_size, drop_last=True)

    def train_st(self, sm_data, st_data, rec_w=1, m_w=1):
        self.model.train()
        self.st_optimizer.zero_grad()
        sm_latent, sm_predictions, sm_rec_data = self.model(sm_data)
        st_latent, _, st_rec_data = self.model(st_data)
        sm_rec_loss = self.kl_rec_loss_func(sm_data, sm_rec_data).mean(
        ) - self.cosine_rec_loss_func(sm_data, sm_rec_data).mean()
        st_rec_loss = self.kl_rec_loss_func(st_data, st_rec_data).mean(
        ) - self.cosine_rec_loss_func(st_data, st_rec_data).mean()
        mmd_loss = self.mmd_loss(sm_latent, st_latent)
        loss = rec_w*sm_rec_loss + rec_w*st_rec_loss + m_w*mmd_loss
        loss.backward()
        self.st_optimizer.step()
        return loss, sm_rec_loss, st_rec_loss, mmd_loss

    def train_sm(self, sm_data, sm_labels, infer_w=1):
        self.model.train()
        self.sm_optimizer.zero_grad()
        sm_latent, sm_predictions, sm_rec_data = self.model(sm_data)
        infer_loss = 0
        for loss in self.infer_losses:
            if loss == 'kl':
                infer_loss += self.kl_infer_loss_func(
                    sm_labels, sm_predictions).mean()
            elif loss == 'cos':
                infer_loss -= self.cosine_infer_loss_func(
                    sm_labels, sm_predictions).mean()
            elif loss == 'rmse':
                infer_loss += self.rmse_loss_func(sm_labels, sm_predictions)
        loss = infer_w*infer_loss
        loss.backward()
        self.sm_optimizer.step()
        return loss, infer_loss

    def test_st(self, sm_data, st_data, rec_w=1, m_w=1):
        self.model.eval()
        sm_latent, sm_predictions, sm_rec_data = self.model(sm_data)
        st_latent, _, st_rec_data = self.model(st_data)
        sm_rec_loss = self.kl_rec_loss_func(sm_data, sm_rec_data).mean(
        ) - self.cosine_rec_loss_func(sm_data, sm_rec_data).mean()
        st_rec_loss = self.kl_rec_loss_func(st_data, st_rec_data).mean(
        ) - self.cosine_rec_loss_func(st_data, st_rec_data).mean()
        mmd_loss = self.mmd_loss(sm_latent, st_latent)
        loss = rec_w*sm_rec_loss + rec_w*st_rec_loss + m_w*mmd_loss
        return loss, sm_rec_loss, st_rec_loss, mmd_loss

    def test_sm(self, sm_data, sm_labels, infer_w=1):
        self.model.eval()
        sm_latent, sm_predictions, sm_rec_data = self.model(sm_data)
        infer_loss = 0
        for loss in self.infer_losses:
            if loss == 'kl':
                infer_loss += self.kl_infer_loss_func(
                    sm_labels, sm_predictions).mean()
            elif loss == 'cos':
                infer_loss -= self.cosine_infer_loss_func(
                    sm_labels, sm_predictions).mean()
            elif loss == 'rmse':
                infer_loss += self.rmse_loss_func(sm_labels, sm_predictions)
        loss = infer_w*infer_loss
        return loss, infer_loss

    def train_model_by_step(
        self,
        max_steps=5000,
        save_mode='all',
        save_path=None,
        prefix=None,
        sm_step=10,
        st_step=10,
        test_step_gap=1,
        convergence=0.001,
        early_stop=True,
        early_stop_max=2000,
        sm_lr=None,
        st_lr=None,
        rec_w=1,
        infer_w=1,
        m_w=1,
    ):
        process = psutil.Process() 
        start_time = time()  
      
        if len(self.history) > 0:
            best_ind = np.where(self.history['is_best'] == 'True')[0][-1]
            best_loss = self.history['sm_test_infer_loss'][best_ind]
            best_rec_loss = self.history['st_test_rec_loss'][best_ind]
        else:
            best_loss = np.inf
            best_rec_loss = np.inf
        early_stop_count = 0
        if sm_lr is not None:
            for g in self.sm_optimizer.param_groups:
                g['lr'] = sm_lr
        if st_lr is not None:
            for g in self.st_optimizer.param_groups:
                g['lr'] = st_lr

        pbar = tqdm(range(max_steps))
        sm_trainr_iter = itertools.cycle(self.sm_train_sampler)
        sm_test_iter = itertools.cycle(self.sm_test_sampler)
        st_iter = itertools.cycle(self.st_sampler)
        sm_train_shuffle_step = max(
            int(len(self.sm_train_ds)/(self.sm_train_batch_size*sm_step)), 1)
        sm_test_shuffle_step = max(
            int(len(self.sm_test_ds)/(self.sm_test_batch_size*sm_step)), 1)
        st_shuffle_step = max(
            int(len(self.st_ds)/(self.st_batch_size*st_step)), 1)
        for step in pbar:
            if step % sm_train_shuffle_step == 0:
                sm_train_iter = itertools.cycle(self.sm_train_sampler)
            if step % sm_test_shuffle_step == 0:
                sm_test_iter = itertools.cycle(self.sm_test_sampler)
            if step % st_shuffle_step == 0:
                st_iter = itertools.cycle(self.st_sampler)

            st_exp = self.st_ds[next(st_iter)][0]
            sm_exp, sm_proportion = self.sm_train_ds[next(sm_train_iter)]
            for i in range(st_step):
                st_train_total_loss, sm_train_rec_loss, st_train_rec_loss, st_train_mmd_loss = self.train_st(
                    sm_exp, st_exp, rec_w=rec_w, m_w=m_w)
            for i in range(sm_step):
                sm_train_total_loss, sm_train_infer_loss = self.train_sm(
                    sm_exp, sm_proportion, infer_w=infer_w)

            if step % test_step_gap == 0:
                sm_test_exp, sm_test_proportion = self.sm_test_ds[next(
                    sm_test_iter)]
                st_test_total_loss, sm_test_rec_loss, st_test_rec_loss, st_test_mmd_loss = self.test_st(
                    sm_test_exp, st_exp, rec_w=rec_w, m_w=m_w)
                sm_test_total_loss, sm_test_infer_loss = self.test_sm(
                    sm_test_exp, sm_test_proportion, infer_w=infer_w)

                current_infer_loss = sm_test_infer_loss.item()

                best_flag = 'False'
                if best_loss - current_infer_loss > convergence:
                    if best_loss > current_infer_loss:
                        best_loss = current_infer_loss
                    best_flag = 'True'
                    # print('### Update best model')
                    early_stop_count = 0
                    old_best_path = self.best_path
                    if prefix is not None:
                        self.best_path = os.path.join(
                            save_path, prefix+'_'+f'celleagle_weights_step{step}.h5')
                    else:
                        self.best_path = os.path.join(
                            save_path, f'celleagle_weights_step{step}.h5')
                    if save_mode == 'best':
                        if old_best_path is not None:
                            if os.path.exists(old_best_path):
                                os.remove(old_best_path)
                        torch.save(self.model.state_dict(), self.best_path)
                else:
                    early_stop_count += 1

                if save_mode == 'all':
                    if prefix is not None:
                        self.best_path = os.path.join(
                            save_path, prefix+'_'+f'celleagle_weights_step{step}.h5')
                    else:
                        self.best_path = os.path.join(
                            save_path, f'celleagle_weights_step{step}.h5')
                    torch.save(self.model.state_dict(), self.best_path)

                self.history = pd.concat([
                    self.history,
                    pd.DataFrame({
                        'sm_train_infer_loss': sm_train_infer_loss.item(),
                        'sm_train_rec_loss': sm_train_infer_loss.item(),
                        'sm_test_rec_loss': sm_test_rec_loss.item(),
                        'sm_test_infer_loss': sm_test_infer_loss.item(),
                        'st_train_rec_loss': st_train_rec_loss.item(),
                        'st_test_rec_loss': st_test_rec_loss.item(),
                        'st_train_mmd_loss': st_train_rec_loss.item(),
                        'st_test_mmd_loss': st_test_rec_loss.item(),
                        'is_best': best_flag
                    }, index=[0])
                ]).reset_index(drop=True)

                pbar.set_description(
                    f"Step {step + 1}: Test inference loss={sm_test_infer_loss.item():.3f}", refresh=True)

                if (early_stop_count > early_stop_max) and early_stop:
                    print('Stop trainning because of loss convergence')
                    break
 # 训练结束后记录资源消耗
        end_time = time()
        elapsed_time = end_time - start_time
        memory_usage = process.memory_info().rss / (1024 * 1024)  # MB

        print(f"Training completed in {elapsed_time:.2f} seconds with memory usage {memory_usage:.2f} MB.")
    def train_model(
        self,
        max_steps=5000,
        save_mode='all',
        save_path=None,
        prefix=None,
        sm_step=10,
        st_step=10,
        test_step_gap=1,
        convergence=0.001,
        early_stop=False,
        early_stop_max=2000,
        sm_lr=None,
        st_lr=None,
        batch_size=1024,
        rec_w=1,
        infer_w=1,
        m_w=1,
    ):
        
        self.init_model()
        self.train_model_by_step(
            max_steps=max_steps,
            save_mode=save_mode,
            save_path=save_path,
            prefix=prefix,
            sm_step=sm_step,
            st_step=st_step,
            test_step_gap=test_step_gap,
            convergence=convergence,
            early_stop=early_stop,
            early_stop_max=early_stop_max,
            sm_lr=sm_lr,
            st_lr=st_lr,
            rec_w=rec_w,
            infer_w=infer_w,
            m_w=m_w
        )

    def train(
        # self,
        # max_steps=5000,
        # save_mode='best',
        # save_path=None,
        # prefix=None,
        # sm_step=5,
        # st_step=5,
        # test_step_gap=1,
        # convergence=0.001,
        # early_stop=True,
        # early_stop_max=1000,
        # sm_lr=0.001,
        # st_lr=0.001,
        # batch_size=128,
        # rec_w=1,
        # infer_w=1,
        # m_w=1,
        # scvi_max_epochs=50,
        # scvi_early_stopping=True,
        # scvi_batch_size=128,
        #2025.4.15
        self,
        max_steps=5000,
        save_mode='best',
        save_path=None,
        prefix=None,
        sm_step=10,
        st_step=10,
        test_step_gap=1,
        convergence=0.001,
        early_stop=False,
        early_stop_max=2000,
        sm_lr=0.001,
        st_lr=0.002,
        # sm_lr=None,
        # st_lr=None,
        batch_size=2048,
        rec_w=1,
        infer_w=1,
        m_w=1,
        scvi_max_epochs=100,
        scvi_early_stopping=True,
        scvi_batch_size=4096,
        # self,
        # max_steps=4000,
        # save_mode='best',
        # save_path=None,
        # prefix=None,
        # sm_step=5,
        # st_step=5,
        # test_step_gap=10,
        # convergence=0.01,
        # early_stop=True,
        # early_stop_max=1000,
        # sm_lr=0.001,
        # st_lr=0.001,
        # batch_size=2048,
        # rec_w=1,
        # infer_w=1,
        # m_w=1,
        # scvi_max_epochs=100,
        # scvi_early_stopping=True,
        # scvi_batch_size=4096,
    ):
        """
    SpatialNet is a deep learning model for spatial transcriptomics cell type deconvolution.
    
    This model leverages single-cell RNA-seq data to deconvolve cell types in spatial transcriptomics data.
    It uses a neural network architecture with KAN (Kolmogorov-Arnold Network) layers for improved
    representation learning.
    
    Parameters
    ----------
    st_ad : AnnData
        Spatial transcriptomics AnnData object.
    sm_ad : AnnData
        Single-cell RNA-seq AnnData object.
    clusters : list
        List of cell type names to deconvolve.
    used_genes : list
        List of genes used for analysis.
    spot_names : list
        List of spot names in spatial data.
    use_rep : str
        Which representation to use. Options: 'scvi', 'X', 'pca'. Default: 'scvi'.
    st_batch_key : str, optional
        Batch key for spatial data if batches exist.
    scvi_layers : int, optional
        Number of layers in scVI model. Default: 2.
    scvi_latent : int, optional
        Latent dimension in scVI model. Default: 64.
    scvi_gene_likelihood : str, optional
        Gene likelihood model in scVI. Options: 'zinb', 'nb', 'poisson'. Default: 'zinb'.
    scvi_dispersion : str, optional
        Dispersion model in scVI. Options: 'gene', 'gene-batch', 'gene-label', 'gene-cell'. Default: 'gene-batch'.
    latent_dims : int, optional
        Latent dimensions in SpatialNet model. Default: 32.
    hidden_dims : int, optional
        Hidden dimensions in SpatialNet model. Default: 512.
    infer_losses : list, optional
        List of loss functions for inference. Options: 'kl', 'cos', 'rmse'. Default: ['kl', 'cos'].
    l1 : float, optional
        L1 regularization strength. Default: 0.01.
    l2 : float, optional
        L2 regularization strength. Default: 0.01.
    sm_lr : float, optional
        Learning rate for single-cell model optimization. Default: 3e-4.
    st_lr : float, optional
        Learning rate for spatial model optimization. Default: 3e-5.
    use_gpu : bool, optional
        Whether to use GPU if available. Default: True.
    seed : int, optional
        Random seed for reproducibility. Default: 42.
    
    Notes
    -----
    This model requires a GPU for optimal performance.
    """
        if save_path is None:
            save_path = os.path.join(tempfile.gettempdir(
            ), 'Spoint_models_'+strftime("%Y%m%d%H%M%S", localtime()))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.get_scvi_latent(max_epochs=scvi_max_epochs,
                             early_stopping=scvi_early_stopping, batch_size=scvi_batch_size)
        self.build_dataset(batch_size)
        self.train_model(
            max_steps=max_steps,
            save_mode=save_mode,
            save_path=save_path,
            prefix=prefix,
            sm_step=sm_step,
            st_step=st_step,
            test_step_gap=test_step_gap,
            convergence=convergence,
            early_stop=early_stop,
            early_stop_max=early_stop_max,
            sm_lr=sm_lr,
            st_lr=st_lr,
            rec_w=rec_w,
            infer_w=infer_w,
            m_w=m_w
        )

    def eval_model(self, model_path=None, use_best_model=True, batch_size=4096, metric='pcc'):
        if metric == 'pcc':
            metric_name = 'PCC'
            func = metrics.pcc
        if metric == 'spcc':
            metric_name = 'SPCC'
            func = metrics.spcc
        if metric == 'mae':
            metric_name = 'MAE'
            func = metrics.mae
        if metric == 'js':
            metric_name = 'JS'
            func = metrics.js
        if metric == 'rmse':
            metric_name = 'RMSE'
            func = metrics.rmse
        if metric == 'ssim':
            metric_name = 'SSIM'
            func = metrics.ssim

        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
        elif use_best_model:
            self.model.load_state_dict(torch.load(self.best_path))
        model.eval()
        pre = []
        prop = []
        for exp_batch, prop_batch in self.sm_test_dataloader:
            latent_tmp, pre_tmp, _ = self.model(exp_batch)
            pre.extend(pre_tmp.cpu().detach().numpy())
            prop.extend(prop_batch.cpu().detach().numpy())
        pre = np.array(pre)
        prop = np.array(prop)
        metric_list = []
        for i, c in enumerate(self.clusters):
            metric_list.append(func(pre[:, i], prop[:, i]))
        print('### Evaluate model with simulation data')
        for i in range(len(metric_list)):
            print(f'{metric_name} of {self.clusters[i]}, {metric_list[i]}')

   
    def plot_training_history(self, save=None, return_fig=False, show=True, dpi=300):
        if len(self.history) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.plot(np.arange(len(self.history)),
                        self.history['sm_test_infer_loss'], label='Simulated Test Inference Loss', linewidth=2)
            plt.plot(np.arange(len(self.history)),
                        self.history['st_test_rec_loss'], label='Spatial Test Reconstruction Loss', linewidth=2)
            plt.xlabel('Epochs', fontsize=14)
            plt.ylabel('Losses', fontsize=14)
            plt.title('Training History', fontsize=16)
            plt.legend(fontsize=12)
            if save is not None:
                plt.savefig(save, bbox_inches='tight', dpi=dpi)
            if show:
                plt.show()
            plt.close()
            if return_fig:
                return fig
        else:
            print('History is empty, train the model first')

    def deconv_spatial(self, st_data=None, min_prop=0.01, model_path=None, use_best_model=True, add_obs=True, add_uns=True):
       """
    Perform cell type deconvolution on spatial transcriptomics data.
    
    This method applies the trained model to predict cell type proportions in spatial data.
    
    Parameters
    ----------
    st_data : array-like, optional
        Spatial transcriptomics data to deconvolve. If None, uses the data from initialization. Default: None.
    min_prop : float, optional
        Minimum proportion threshold. Predictions below this value will be set to 0. Default: 0.01.
    model_path : str, optional
        Path to saved model weights. If None and use_best_model is True, uses the best model
        from training. Default: None.
    use_best_model : bool, optional
        Whether to use the best model from training. Default: True.
    add_obs : bool, optional
        Whether to add predictions to st_ad.obs. Default: True.
    add_uns : bool, optional
        Whether to add cell type list to st_ad.uns. Default: True.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing predicted cell type proportions for each spot.
    """
        if st_data is None:
            st_data = self.st_data
        st_data = torch.tensor(st_data).to(self.device)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
        elif use_best_model:
            self.model.load_state_dict(torch.load(self.best_path))
        self.model.to(self.device)
        self.model.eval()
        latent, pre, _ = self.model(st_data)
        pre = pre.cpu().detach().numpy()
        pre[pre < min_prop] = 0
        pre = pd.DataFrame(pre, columns=self.clusters,
                           index=self.st_ad.obs_names)
        self.st_ad.obs[pre.columns] = pre.values
        self.st_ad.uns['celltypes'] = list(pre.columns)
        return pre
