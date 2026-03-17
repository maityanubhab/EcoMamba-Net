# 🧠 EcoMamba-Net  
### Parameter-Efficient Architecture for Medical Image Segmentation

📄 **Published at PREMI (Pattern Recognition and Machine Intelligence)**  
🔗 DOI: *To be updated*

---

## 🚀 Overview

**EcoMamba-Net** is a lightweight and efficient deep learning architecture designed for **2D medical image segmentation**.

It integrates **state-space inspired modeling (Mamba-like blocks)** with a **U-Net style encoder–decoder framework**, achieving a strong balance between:

- ⚡ Computational efficiency  
- 🧠 Expressive feature modeling  
- 📉 Reduced parameter count  
- 🏥 High segmentation performance  

---

## 🏗️ Architecture Design

EcoMamba-Net follows a **hierarchical encoder–decoder structure** with carefully designed components for efficiency and performance.

### 🔹 Core Building Blocks

#### 1. EcoMamba Block
- State-space inspired module for feature modeling  
- Depthwise convolution for efficient spatial interaction  
- Lightweight design with reduced parameters  
- Enables long-range dependency modeling  

#### 2. Efficient Attention Module
- Hybrid **channel + spatial attention**
- Enhances feature refinement with minimal overhead  
- Adaptive feature recalibration  

#### 3. Encoder–Decoder Framework
- Multi-scale feature extraction  
- Skip connections for spatial information preservation  
- Progressive feature aggregation  

---

## ✨ Key Highlights

- ✔️ **Parameter-Efficient Design**  
  Optimized for low memory and compute usage  

- ✔️ **Mamba-Inspired Modeling**  
  Efficient alternative to heavy attention mechanisms  

- ✔️ **Hybrid Attention Integration**  
  Improves feature quality without significant overhead  

- ✔️ **Scalable Architecture**  
  Easily adaptable to different medical segmentation tasks  

---

## 📊 Benchmark Datasets

EcoMamba-Net has been evaluated on standard medical imaging datasets:

- 🧪 **ISIC 2018** — Skin lesion segmentation  
  https://challenge2018.isic-archive.com/

- 🧪 **Kvasir-SEG** — Polyp segmentation  
  https://www.synapse.org/#!Synapse:syn3193805/wiki/217789  

---

## 🧪 Model

This repository provides the **complete architecture implementation** of EcoMamba-Net, including:

- EcoMamba Blocks  
- Efficient Attention Modules  
- Encoder–Decoder Network Design  

The model is designed to be **modular, extensible, and easy to integrate** into different pipelines.

---

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@article{eco2025,
  title={EcoMamba-Net: Parameter-Efficient Architecture for Medical Image Segmentation},
  author={Anubhab Maity and Pallabi Dutta and Sushmita Mitra},
  journal={PREMI},
  year={2025},
  note={DOI will be updated}
}
