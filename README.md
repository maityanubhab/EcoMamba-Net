# EcoMamba-Net: Parameter-Efficient Architecture for Medical Image Segmentation

This repository contains the official implementation for the paper:

**"EcoMamba-Net: Parameter-Efficient Architecture for Medical Image Segmentation"**

📄 *Submitted to [PREMI]*

## Code Availability

The code is currently under preparation and will be released upon paper acceptance or camera-ready submission.

Stay tuned! 

## Project Description

EcoMamba introduces a lightweight and efficient Mamba-inspired state-space model tailored for 2D medical image segmentation. The design focuses on minimizing computational cost (GFLOPs) while maintaining high segmentation performance on benchmark datasets like ISIC 2018 and BTCV.

Key contributions:
- EcoMamba blocks with depthwise convolution
- Efficient attention block

## 📊 Datasets

We used the following datasets:
- [ISIC 2018](https://challenge2018.isic-archive.com/)
- [BTCV](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)

Instructions to download and preprocess datasets will be included with the final release.

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@article{eco2025,
  title={EcoMamba: Economical State-Space Blocks for Scalable Medical Image Segmentation},
  author={Anubhab Maity and Sushmita Mitra},
  journal={Under Review},
  year={2025}
}
