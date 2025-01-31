<h2 align="center"> <a href="https://arxiv.org/abs/2501.18452">ReSA: positive-feedback self-supervised learning</a><h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2501.18452-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2501.18452)[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/winci-ai/resa/blob/main/LICENSE)


<p align="center">
    <img src="https://github.com/winci-ai/resa/releases/download/figure/postive-feedback.jpg" width="350">
</p>


## Method
ReSA is designed within our positive-feedback SSL framework, which directly leverages the stable and semantically meaningful clustering properties of the encoder’s outputs. Through an online self-clustering mechanism, ReSA refines its own
objective during training, outperforming state of-the-art SSL baselines and achieving higher efficiency.

<p align="center">
    <img src="https://github.com/winci-ai/resa/releases/download/figure/resa.jpg" width="600">
</p>

### ReSA Excels at both Fine-grained and Coarse-grained Learning

<p align="center">
    <img src="https://github.com/winci-ai/resa/releases/download/figure/tsne.jpg" width="800">
</p>


## Installation and Requirements

1. Clone this repository and navigate to the folder
```bash
git clone https://github.com/winci-ai/resa_cifar.git
cd resa
```

2. Create a conda environment, activate it and install packages (newer versions of python are supported)
```Shell
conda create -n resa python=3.8.18
conda activate resa
pip install -r requirements.txt
```

## Pretraining on CIFAR-10/100

If you would like to pretrain ReSA on ImageNet, please refer to the main [repository](https://github.com/winci-ai/resa).

### ResNet-18 with 1-node (1-GPU) training, a batch size of 256 

```
torchrun --nproc_per_node=1 main.py \
--dataset cifar10 \
--dump_path /path/to/saving_dir \
```

The command for pretraining on CIFAR-100 is identical; you simply need to set `--dataset cifar100`.

## License
See the [LICENSE](./LICENSE) file for details about the license under which this code is made available.

## Citation
If you find this repository useful in your research, please consider giving a star :star: and a citation
```
@article{weng2025resa,
  title={Clustering Properties of Self-Supervised Learning},
  author={Weng, Xi and An, Jianing and Ma, Xudong and Qi, Binhang and Luo, Jie and Yang, Xi and Dong, Jin Song and Huang, Lei},
  journal={arXiv preprint arXiv:2501.18452},
  year={2025}
}
