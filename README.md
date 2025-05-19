# VMBNet

## Abstract
Existing crowd counting methods are primarily based on convolutional neural networks (CNNs) and Transformer. Although Transformers rely on self-attention mechanisms to achieve global modeling, their quadratic computational complexity limits inference efficiency. Recently, visual state space models have emerged as a novel modeling paradigm, demonstrating efficient long-range dependency capture. Motivated by this, we propose VMBNet, which introduces a Dynamic Global-Local Perception Block (DGLP Block) based on the selective scanning strategy, reducing the quadratic complexity of traditional global modeling to linear complexity and thus effectively improving inference efficiency. To further enhance feature representation, we design a Multiscale Atrous Spatial Pyramid Module (MASPM), which integrates atrous convolutions to enrich both local and global features. In addition, a Channel-Aware Mamba Decoder (CAM) is proposed to further strengthen channel-wise modeling capability. Extensive experiments conducted on five public datasets demonstrate that VMBNet outperforms most CNN-based and Transformer-based methods in terms of counting accuracy and model size, while reducing FLOPs by 79.8% compared to state-of-the-art approaches, significantly lowering computational complexity.

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
``` python train.py ```

## Visualization Result
![Visualization](https://github.com/user-attachments/assets/cc2bc017-6fc3-45b5-92a3-e700745ccb46)
![12](https://github.com/user-attachments/assets/519293ba-1612-445d-9cb4-a87351cc529c)


