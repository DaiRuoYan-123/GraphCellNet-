# GraphCellNet: A Deep Learning Method for Integrated Single-Cell and Spatial Transcriptomic Analysis with Applications in Development and Disease
![image](picture/model.jpg)



## Getting Started
* [Requirements](#Requirements)

* Tutorials
    * [Deconvolution ](Visium_human_DLPFC_deconv.ipynb)
    * [Spatial domain recognition](Graph_model/Visium_human_DLPFC_Graph.ipynb)



## Requirements

To install `GraphCellNet`, you must first install [PyTorch](https://pytorch.org) with GPU support. If GPU acceleration is not required, you can skip the installation of `cudnn` and `cudatoolkit`.

python == 3.9  
torch == 1.13.0  
scanpy == 1.9.2  
anndata == 0.8.0  
numpy == 1.22.3  
rpy2 == 3.5.16  
matplotlib == 3.7.2  
tqdm == 4.64.1  
scikit-learn == 1.4.1.post1  
pip3 install torch==1.12.1.0+cu113 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 （GPU） 

pip3 install torch==1.12.1 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu  （CPU）    
