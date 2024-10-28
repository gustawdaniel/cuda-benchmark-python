---
title: How to benchmark CUDA in Python?
publishDate: 2024-10-26
---

Create conda environment

```bash
conda create -n cuda-benchmark python=3.10
conda activate cuda-benchmark
```

Install pytorch using link: https://pytorch.org/get-started/locally/

First check your cuda version by `nvidia-smi`. For example.:

```bash
nvidia-smi
```

In my case it is

```
CUDA Version: 12.6 
```

You can check by `nvcc --version`:

```bash
nvcc --version
```

For example:

```
Cuda compilation tools, release 12.6, V12.6.77
```

Unfortunately there is no pytorch for cuda 12.6. So I will use 12.4

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```
