# 🎵 VAE for Hybrid Language Music Clustering

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Course:** Neural Networks  
> **Project:** Unsupervised Learning with Variational Autoencoders  
> **Dataset:** 4MuLA Tiny (1,988 songs, 93 artists, 27 genres, 3 languages)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Dataset Setup](#-dataset-setup)
- [Usage](#-usage)
- [Methods Implemented](#-methods-implemented)
- [Results](#-results)
- [Evaluation Metrics](#-evaluation-metrics)
- [Visualizations](#-visualizations)
- [References](#-references)

---

## 🎯 Overview

This project implements an **unsupervised learning pipeline** using Variational Autoencoders (VAE) for clustering multilingual music tracks. We extract latent representations from audio features (melspectrograms) and lyrics, then perform clustering analysis to discover meaningful groupings in music data.

### Key Features

- **Multiple VAE Architectures**: Basic VAE, Convolutional VAE, Beta-VAE, Conditional VAE (CVAE)
- **Multi-modal Learning**: Combines audio features, lyrics embeddings, and genre information
- **Comprehensive Clustering**: K-Means, Agglomerative Clustering, DBSCAN
- **Extensive Evaluation**: Silhouette Score, NMI, ARI, Cluster Purity, Davies-Bouldin Index
- **Rich Visualizations**: t-SNE, UMAP, latent space plots, reconstruction examples

### Tasks Completed

| Task | Description | Status |
|------|-------------|--------|
| **Easy** | Basic VAE + K-Means + t-SNE/UMAP + PCA baseline | ✅ |
| **Medium** | Conv-VAE + Lyrics embeddings + Multiple clustering algorithms | ✅ |
| **Hard** | Beta-VAE/CVAE + Multi-modal clustering + Extended metrics | ✅ |

---

## 📁 Project Structure

```
VAE-Music-Clustering/
│
├── 📓 notebooks/
│   └── VAE_Music_Clustering_Complete.ipynb    # Main notebook with all tasks
│
├── 📂 data/
│   ├── 4mula_tiny.parquet                     # Dataset (download separately)
│   └── DATASET_DOWNLOAD.txt                   # Download instructions
│
├── 📂 results/
│   ├── easy_task_results.csv                  # Easy task metrics
│   ├── medium_task_results.csv                # Medium task metrics
│   ├── hard_task_results.csv                  # Hard task metrics
│   └── visualizations/
│       ├── training_curves.png
│       ├── cluster_selection.png
│       ├── tsne_visualization.png
│       ├── umap_visualization.png
│       ├── beta_vae_latent_spaces.png
│       ├── cluster_distributions.png
│       ├── reconstruction_examples.png
│       ├── latent_interpolation.png
│       └── comprehensive_results.png
│
├── 📂 models/
│   ├── basic_vae.pth                          # Trained Basic VAE
│   ├── beta_vae_4.pth                         # Trained Beta-VAE (β=4)
│   ├── cvae.pth                               # Trained Conditional VAE
│   ├── conv_vae.pth                           # Trained Convolutional VAE
│   └── autoencoder.pth                        # Trained Autoencoder
│
├── 📄 requirements.txt                         # Python dependencies
├── 📄 README.md                                # This file
└── 📄 LICENSE                                  # MIT License
```

---

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster training)

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/VAE-Music-Clustering.git
cd VAE-Music-Clustering
```

### Step 2: Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Dataset Setup

### Download the Dataset

The 4MuLA Tiny dataset is too large for GitHub. Download it from the link provided in `data/DATASET_DOWNLOAD.txt`.

**Direct Download Link:** [See data/DATASET_DOWNLOAD.txt]

### Place the Dataset

After downloading, place `4mula_tiny.parquet` in the `data/` directory:

```
VAE-Music-Clustering/
└── data/
    └── 4mula_tiny.parquet    # Place the downloaded file here
```

### Dataset Information

| Property | Value |
|----------|-------|
| **Total Songs** | 1,988 |
| **Total Artists** | 93 |
| **Total Genres** | 27 |
| **Languages** | English, Portuguese, Spanish |
| **Features** | Melspectrograms (128 mel bands), Lyrics, Metadata |
| **File Format** | Parquet |

---

## 🚀 Usage

### Option 1: Run in Google Colab (Recommended)

1. Upload the notebook to Google Colab
2. Upload the dataset to your Google Drive at: `/MyDrive/Datasets/4mula_tiny.parquet`
3. Set `USE_COLAB = True` in the notebook
4. Run all cells

### Option 2: Run Locally

1. Ensure the dataset is in the `data/` directory
2. Set `USE_COLAB = False` in the notebook
3. Open the notebook:

```bash
jupyter notebook notebooks/VAE_Music_Clustering_Complete.ipynb
```

4. Run all cells

### Quick Start

```python
# The notebook automatically detects the environment
# Just run all cells after setting up the dataset!
```

---

## 🧠 Methods Implemented

### 1. VAE Architectures

| Model | Description | Latent Dim |
|-------|-------------|------------|
| **Basic VAE** | Standard VAE with MLP encoder/decoder | 32 |
| **Conv-VAE** | Convolutional VAE for 2D spectrograms | 64 |
| **Beta-VAE** | VAE with β > 1 for disentanglement | 32 |
| **CVAE** | Conditional VAE with genre conditioning | 32 |
| **Autoencoder** | Standard AE (baseline) | 32 |

### 2. Feature Representations

| Feature | Dimensions | Source |
|---------|------------|--------|
| **Audio (Statistical)** | 512 | Melspectrogram statistics |
| **Audio (Conv-VAE)** | 64 | 2D spectrogram encoding |
| **Lyrics** | 384 | Sentence-BERT embeddings |
| **Genre** | 27 | One-hot encoding |
| **Multi-modal** | 443+ | Combined features |

### 3. Clustering Algorithms

- **K-Means**: Optimal K selected via Elbow method and Silhouette analysis
- **Agglomerative Clustering**: Hierarchical clustering with Ward linkage
- **DBSCAN**: Density-based clustering with adaptive epsilon

---

## 📊 Results

### Clustering Performance Comparison

| Method | Silhouette ↑ | NMI ↑ | ARI ↑ | Purity ↑ |
|--------|-------------|-------|-------|----------|
| PCA (Baseline) | 0.32 | 0.15 | 0.02 | 0.18 |
| Basic VAE | 0.26 | 0.14 | 0.02 | 0.17 |
| Beta-VAE (β=4) | 0.28 | 0.16 | 0.03 | 0.19 |
| CVAE | 0.27 | 0.15 | 0.02 | 0.18 |
| Multi-Modal | 0.25 | 0.18 | 0.04 | 0.21 |

*Note: Actual results may vary slightly due to random initialization.*

### Key Findings

1. **VAE vs PCA**: PCA achieves competitive results, suggesting the data has significant linear structure
2. **Multi-modal Benefits**: Adding lyrics improves genre alignment (higher ARI)
3. **Beta-VAE**: Higher β values provide more disentangled representations
4. **CVAE**: Genre conditioning creates structured latent spaces

---

## 📐 Evaluation Metrics

| Metric | Description | Range | Optimal |
|--------|-------------|-------|---------|
| **Silhouette Score** | Cluster cohesion vs separation | [-1, 1] | Higher |
| **Calinski-Harabasz** | Between/within cluster variance | [0, ∞) | Higher |
| **Davies-Bouldin** | Cluster similarity measure | [0, ∞) | Lower |
| **ARI** | Agreement with ground truth | [-1, 1] | Higher |
| **NMI** | Mutual information with labels | [0, 1] | Higher |
| **Cluster Purity** | Dominant class fraction | [0, 1] | Higher |

---

## 📈 Visualizations

### Latent Space (t-SNE & UMAP)
![Latent Space](results/visualizations/umap_visualization.png)

### Beta-VAE Comparison
![Beta VAE](results/visualizations/beta_vae_latent_spaces.png)

### Cluster Distributions
![Distributions](results/visualizations/cluster_distributions.png)

### Reconstruction Examples
![Reconstructions](results/visualizations/reconstruction_examples.png)

---

## 🛠️ Technical Details

### Hyperparameters

```python
# VAE Architecture
INPUT_DIM = 512          # Melspectrogram statistics
HIDDEN_DIM = 256         # Hidden layer size
LATENT_DIM = 32          # Latent space dimension

# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 100
BETA_VALUES = [1.0, 2.0, 4.0, 8.0]  # For Beta-VAE

# Clustering
OPTIMAL_K = 10           # Number of clusters (determined by analysis)
```

### Loss Function

$$\mathcal{L} = \mathcal{L}_{recon} + \beta \cdot D_{KL}(q(z|x) \| p(z))$$

Where:
- $\mathcal{L}_{recon}$: MSE reconstruction loss
- $D_{KL}$: KL divergence regularization
- $\beta$: KL weight (1.0 for VAE, >1 for Beta-VAE)

---

## 📚 References

1. Kingma, D. P., & Welling, M. (2013). *Auto-Encoding Variational Bayes*. arXiv:1312.6114

2. Higgins, I., et al. (2017). *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*. ICLR 2017

3. Sohn, K., et al. (2015). *Learning Structured Output Representation using Deep Conditional Generative Models*. NeurIPS 2015

4. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP 2019

5. McInnes, L., et al. (2018). *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*. arXiv:1802.03426

---

## 👤 Author

**Nadim Mahmud Dipu**
- Course: Neural Networks
- Institution: Brac University
- Email: nadim.mahmud.dipu@g.bracu.ac.bd

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Course Instructor: Moin Mostakim
- Dataset: 4MuLA (Four Multi-lingual Audio) Dataset
- Libraries: PyTorch, Scikit-learn, Sentence-Transformers, UMAP

---

<p align="center">
  <b>⭐ If you found this project helpful, please consider giving it a star! ⭐</b>
</p>
