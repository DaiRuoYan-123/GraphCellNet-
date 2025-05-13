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
torch == 1.12.1 
scanpy == 1.9.8  
anndata == 0.10.5.post1 
numpy == 1.26.4  
rpy2 == 3.5.15  
matplotlib == 3.8.3  
tqdm == 4.67.1  
scikit-learn == 1.4.1.post1  
# CUDA 11.3
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113 （GPU） 

pip3 install torch==1.12.1 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu  （CPU）    
## install
conda create -n GraphCellNet python=3.9
conda activate GraphCellNet
## step1 Installing PyTorch’s CUDA support or CPU support on Linux
pip3 install torch==1.12.1.0+cu113 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 （GPU） 

# CPU only
pip install torch==1.12.0+cpu torchvision==0.13.0+cpu torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cpu  （CPU）
## step2 Download dependencies
pip install -r requirements.txt
pip install GraphCellNet==1.0.0
or
git clone https://github.com/DaiRuoYan-123/GraphCellNet-.git
cd GraphCellNet
python setup.py build
python setup.py install --user


