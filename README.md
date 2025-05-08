# VMBNet

## Abstract
Designing computationally efficient network architectures persists as an ongoing necessity in computer vision. In this paper, we transplant Mamba, a state-space language model, into VMamba, a vision backbone that works in linear time complexity. At the core of VMamba lies a stack of Visual State-Space (VSS) blocks with the 2D Selective Scan (SS2D) module. By traversing along four scanning routes, SS2D helps bridge the gap between the ordered nature of 1D selective scan and the non-sequential structure of 2D vision data, which facilitates the gathering of contextual information from various sources and perspectives. Based on the VSS blocks, we develop a family of VMamba architectures and accelerate them through a succession of architectural and implementation enhancements. Extensive experiments showcase VMamba’s promising performance across diverse visual perception tasks, highlighting its advantages in input scaling efficiency compared to existing benchmark models.

## Environment
Please set up the environment according to the requirements.txt file.
## Train
python train.py
## Visualization Result
![Visualization](https://github.com/user-attachments/assets/cc2bc017-6fc3-45b5-92a3-e700745ccb46)
![12](https://github.com/user-attachments/assets/519293ba-1612-445d-9cb4-a87351cc529c)

## Getting Started

### Installation

**Step 1: Clone the VMamba repository:**

To get started, first clone the VMamba repository and navigate to the project directory:

```bash
git clone https://github.com/MzeroMiko/VMamba.git
cd VMamba
```

**Step 2: Environment Setup:**

VMamba recommends setting up a conda environment and installing dependencies via pip. Use the following commands to set up your environment:
Also, We recommend using the pytorch>=2.0, cuda>=11.8. But lower version of pytorch and CUDA are also supported.

***Create and activate a new conda environment***

```bash
conda create -n vmamba
conda activate vmamba
```

