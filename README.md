# Perceptual Image Processing ALgorithms Dataset and Codebase

## PIPAL: a Large-Scale Image Quality Assessment Dataset for Perceptual Image Restoration
<a href="超链接地址" target="_blank">Jinjin Gu</a>, 
<a href="超链接地址" target="_blank">Haoming Cai</a>, 
<a href="超链接地址" target="_blank">Haoyu Chen</a>, 
<a href="超链接地址" target="_blank">Xiaoxing ye</a>, 
<a href="超链接地址" target="_blank">Jimmy S.Ren</a>, 
<a href="超链接地址" target="_blank">Chao Dong</a>. In ECCV, 2020.

<p align="center">
<img src="figures/comparison.png" >
</p>

## Navigation
- [ECCV 2020 Paper]() | [Project Web]() | [NTIRE 2021 Challenge]() |  
- Any questions, please contact with [haomingcai@link.cuhk.edu.cn](haomingcai@link.cuhk.edu.cn)

## Dependencies and Installation
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install numpy opencv-python lmdb pyyaml`
- TensorBoard:
  - PyTorch >= 1.1: `pip install tb-nightly future`
  - PyTorch == 1.0: `pip install tensorboardX`


## How to Train
- **Your IQA**
	1. Prepare IQA dataset PIPAL public training [NTIRE 2021] or BAPPS. More details are in [`codes/data`](codes/data).
    1. Modify the dataset format based on your need in [`codes/data/PairedTrain_dataset.py`](codes/data/PairedTrain_dataset.py) and [`ValidorTest_dataset.py`](ValidorTest_dataset.py)
	1. Modify the configuration file [`codes/options/train_test_yml/train_our_IQA.yml`](codes/options/train_test_yml/train_our_IQA.yml)
	1. Run command:
	```c++
	python train.py -opt options/train_test_yml/train_our_IQA.yml
	```


## How to Test
- **Prepare the test dataset**
	1. Prepare IQA dataset PIPAL public validation [NTIRE 2021]. More details are in [`codes/data`](codes/data).
	1. Modify the dataset format based on your need in [`ValidorTest_dataset.py`](ValidorTest_dataset.py)
	1. Modify the configuration file [`codes/options/train_test_yml/test_IQA.yml`](codes/options/train_test_yml/test_IQA.yml)


## Ackowledgement
- This code is based on [mmsr](https://github.com/open-mmlab/mmsr).