# GSPO: Group Sequence Policy Optimization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

A toy implementation of **Group Sequence Policy Optimization (GSPO)** for training large language models on mathematical reasoning tasks using the GSM8K dataset.

## Overview

GSPO combines the stability of PPO with sequence-level optimizations that are particularly well-suited for tasks requiring coherent, multi-step reasoning rather than just token-by-token generation quality.

## Quick Start




**Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```



### Training

```bash
./run.sh
```



## Reference


```bibtex
@article{zheng2025group,
  title={Group Sequence Policy Optimization},
  author={Zheng, Chujie and Liu, Shixuan and Li, Mingze and Chen, Xiong-Hui and Yu, Bowen and Gao, Chang and Dang, Kai and Liu, Yuqiong and Men, Rui and Yang, An and others},
  journal={arXiv preprint arXiv:2507.18071},
  year={2025}
}
```
