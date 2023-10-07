# Stick

Stick is an educational deep learning framework. The skeleton of this repository is from the [CMU 10-414/714 DLSys course](https://dlsyscourse.org). An alternative of this course is the [Cornell MiniTorch course](https://minitorch.github.io/).

## Features
- Common operators (e.g. matmul, conv), optimizers (e.g. SGD, Adam), and models (e.g. ResNet, LSTM)
- Automatic differentiation
- CPU (C++) and GPU (CUDA) NDArray backends
- Checkpointing (rematerialization)
- Dynamic tensor rematerialization (DTR)

The features checkpointing and DTR are from [Tianqi Chen's paper](https://arxiv.org/abs/1604.06174) and [Marisa Kirisame's paper](https://openreview.net/pdf?id=Vfs_2RnOD0H), respectively. They serve as examples of implementing research ideas on a DIY deep learning framework. Other features are NEcessary Elements for Deep LEarning (NEEDLE).

## How To Run

To run it as simply as possible, you may use Google Colab, which provides a unified environment and free GPU resources. Please read [Colab.ipynb](./Colab.ipynb) to run this repo in Colab.

Following is the process to run it on my Mac M1.

**Download**: ``git clone https://github.com/Sunt-ing/stick``

Note that this repository includes several datasets (~205MB), so it may be slow to download.

**Build Conda environment**
```shell
conda create --name dlsys python=3.9
conda activate dlsys
pip3 install numpy typing pybind11
# for tests
pip3 install numdifftools pytest torch
```

**Compile NDArray backends**: ``make``

**Run tests**

In terminal: ``pytest``

Or in Jupyter Notebook: ``!python3 -m pytest``

