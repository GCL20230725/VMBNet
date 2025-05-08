# VMBNet

## Abstract
Designing computationally efficient network architectures persists as an ongoing necessity in computer vision. In this paper, we transplant Mamba, a state-space language model, into VMamba, a vision backbone that works in linear time complexity. At the core of VMamba lies a stack of Visual State-Space (VSS) blocks with the 2D Selective Scan (SS2D) module. By traversing along four scanning routes, SS2D helps bridge the gap between the ordered nature of 1D selective scan and the non-sequential structure of 2D vision data, which facilitates the gathering of contextual information from various sources and perspectives. Based on the VSS blocks, we develop a family of VMamba architectures and accelerate them through a succession of architectural and implementation enhancements. Extensive experiments showcase VMamba’s promising performance across diverse visual perception tasks, highlighting its advantages in input scaling efficiency compared to existing benchmark models.

## Create Environment

conda create -n vmbnet python=3.10.3

conda activate vmbnet 

pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117 

pip install packaging pip install timm==0.4.12 

pip install pytest chardet yacs termcolor

pip install submitit tensorboardX 

pip install triton==2.0.0

pip install causal_conv1d==1.0.0 # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl

pip install mamba_ssm==1.0.1 # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl

pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs

## Train
python train.py

## Visualization Result
![Visualization](https://github.com/user-attachments/assets/cc2bc017-6fc3-45b5-92a3-e700745ccb46)
![12](https://github.com/user-attachments/assets/519293ba-1612-445d-9cb4-a87351cc529c)


