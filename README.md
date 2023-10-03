# Stick

Stick is an educational deep learning framework. You can DIY such a framework by folllowing the [CMU 10-414/714 DLSys](https://dlsyscourse.org) or [Cornell MiniTorch course](https://minitorch.github.io/). The skeleton of this repository is from the CMU 10-414/714 DLSys course.

## Features
- Common operators (e.g. matmul, conv), optimizers (e.g. SGD, Adam), and models (e.g. ResNet, LSTM)
- Automatic Differentiation
- CPU (C++) and GPU (CUDA) NDArray backends
- Checkpointing (rematerialization)

## How To Run

Note that this repository includes several datasets (~205MB), so it may be slow to download.

To run it as simply as possible, you may use Google Colab, which provides a unified environment and GPU resources. Please read [Colab.ipynb](./Colab.ipynb) to run this repo in Colab.

Following is the process to run it on my Mac M1.

**Build Conda environment**
```shell
conda create --name dlsys python=3.9
conda activate dlsys
pip3 install numpy typing pybind11
# for tests
pip3 install numdifftools pytest torch
```

**Compile**

``
make
``

**Run tests:**

In CLI:
``
pytest
``

Or in ipynb:
``
!python3 -m pytest
``

