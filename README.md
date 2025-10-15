# Driving Maneuver Classification with Domain Knowledge & Transfer Learning

Reproducing methods from:
- Conference (Supervised only)  
- IEEE Access (Unsupervised + Supervised)  

> This repo contains Jupyter notebooks implementing two core methods:
> 1. **Supervised LSTM**  
> 2. **Unsupervised Autoencoder + Supervised LSTM transfer**

---

## 🔍 Methods

### 1. Supervised LSTM  
- **Notebook**: `k_fold_supervised_LSTM.ipynb`  
- **What it does**: Trains a 2-layer LSTM using labeled sensor windows  
- **Why**: LSTMs model temporal dynamics; combined with domain-specific features, they yield high classification accuracy (≈97.2 %) in the conference work  
- **Illustration**:  
  ![Supervised LSTM Architecture](figures/supervised_lstm_arch.png)

---

### 2. Unsupervised + Supervised (Transfer Learning)  
- **Notebooks**:  
  - `k_fold_unsupervised_(300)_(mendaly_data).ipynb` — trains LSTM autoencoder  
  - `k_fold_semi_supervised_(300, 1000)_(mendaly_data)_final.ipynb` — transfers latent features to LSTM classifier  
- **What it does**: Learns latent features from unlabeled data, then fine-tunes a classifier  
- **Why**: Leverages unlabeled data to reduce dependence on labelled samples; improves generalization and flexibility  
- **Illustration**:  
  ![Semi-supervised Flow](figures/semi_supervised_flow.png)

---

## 📁 Repository Structure



#### If you are interested to explore rule-based data labeling and classification you can follow the following papers:
#### [3] A data change rule based empirical framework for labeling unlabeled time series driving data
##### Link: https://ieeexplore.ieee.org/abstract/document/9333495
#### [4] Driving Maneuver Classification From Time Series Data: A Rule Based Machine Learning Approach 
##### Link: Will be availble soon
