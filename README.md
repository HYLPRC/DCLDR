# DCLDR: Dual Channel Graph Contrastive Learning for Drug Repositioning

This repository contains the official implementation of the paper:

> **DCLDR: A Dual Channel Graph Contrastive Learning Framework for Drug Repositioning**  
> 🧬 Predicting potential drug–disease associations via integrating heterogeneous biological networks and similarity features.

## 🔬 Introduction

Drug repositioning aims to find new therapeutic uses for existing drugs. In this work, we propose **DCLDR**, a novel contrastive learning framework that jointly leverages **heterogeneous biological networks** and **multi-source semantic similarity**.


## 📁 Directory Structure

```bash
DCLDR/
├── data/                    # Raw and processed datasets
├── model/                   # Model architecture
├── utils/                   # Utilities: metrics, losses, etc.
├── main.py                  # Training and evaluation pipeline
├── config.py                # Hyperparameters and settings
└── README.md                # You are here


🧱 Environment Setup
This project was developed and tested under the following environment:

🔧 System & Runtime
Python: 3.8.20

CUDA: 12.4

PyTorch: 1.12.1

Torch Geometric: 2.6.1

DGL (Deep Graph Library): 0.9.1 (with CUDA 11.3 support)

⚠️ Note: While the system has CUDA 12.4 installed, dgl-cu113 is built for CUDA 11.3. Ensure CUDA compatibility when changing versions.

📦 Key Python Dependencies
Package	Version	Description
torch	1.12.1	Core deep learning framework
torch-geometric	2.6.1	Graph neural network extension for PyTorch
dgl-cu113	0.9.1	Deep Graph Library (built with CUDA 11.3)
scikit-learn	0.24.2	Machine learning utilities
pandas	2.0.3	Data processing and manipulation
numpy	1.23.3	Numerical computing
matplotlib	3.7.5	Data visualization
networkx	3.1	Graph operations (used in preprocessing)
pyvis	0.3.2	Visualization of graphs in notebooks/browsers
tensorboard	2.14.0	Training visualization
gpustat	1.1.1	Lightweight GPU monitoring
