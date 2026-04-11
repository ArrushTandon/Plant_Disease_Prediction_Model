# 🌿 Plant Disease Detection using CNN, LBP & Metaheuristic Optimization

This project presents a hybrid deep learning + machine learning pipeline for plant leaf disease detection using grayscale images. It combines CNN-based feature extraction with handcrafted texture features (LBP), followed by dimensionality reduction (PCA) and advanced metaheuristic feature selection techniques including:

- Elephant Herding Optimization (EHO)
- Dung Beetle Optimization (DBO)
- Slime Mould Algorithm (SMA)

The system is designed to deliver high accuracy, robustness, and computational efficiency, and supports GPU acceleration using TensorFlow.

---

## 🚀 Key Features

- 🧠 **CNN Feature Extraction** using TensorFlow/Keras
- 🎯 **Local Binary Pattern (LBP)** for texture-based feature extraction
- 📉 **PCA** for dimensionality reduction
- ⚙️ **Metaheuristic Feature Selection:**
  - EHO (Elephant Herding Optimization)
  - DBO (Dung Beetle Optimization)
  - SMA (Slime Mould Algorithm)
- ⚖️ **SMOTE** for class imbalance handling
- 📊 **Comprehensive Evaluation Metrics:**
  - Accuracy, Precision, Recall
  - Confusion Matrix
  - ROC-AUC & Precision-Recall curves
- ⚡ **GPU Acceleration** + Mixed Precision Training

---

## 🏗️ System Pipeline

```
Input Image
     ↓
Preprocessing (Resize + Normalize)
     ↓
LBP Feature Extraction
     ↓
CNN Feature Extraction
     ↓
Feature Fusion (CNN + LBP)
     ↓
PCA (Dimensionality Reduction)
     ↓
Metaheuristic Optimization (EHO / DBO / SMA)
     ↓
SMOTE (Balancing)
     ↓
MLP Classifier
     ↓
Evaluation & Threshold Optimization
```

---

## 📁 Folder Structure

```
Plant_Disease_Prediction/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── baselines/
│   ├── train.py                        # LBP + MLP baseline
│   └── train_cnn.py                    # Basic CNN classifier
│
├── data/
│   └── processed/
│       ├── healthy/                    # Healthy leaf images
│       └── diseased/                   # Diseased leaf images
│
├── scripts/
│   └── data_preparation.py             # Data preprocessing & splitting
│
├── src/
│   ├── models/
│   │   └── cnn_model.py                # CNN architecture
│   ├── optimization/
│   │   ├── eho_optimization.py
│   │   ├── dbo_optimization.py
│   │   └── sma_optimization.py
│   ├── utils/
│   │   ├── visualize.py
│   │   └── lbp_feature_extraction.py
│   ├── cnn_eho_gpu_updated.py          # EHO pipeline
│   ├── cnn_dbo_gpu_updated.py          # DBO pipeline
│   └── cnn_sma_gpu_updated.py          # SMA pipeline
│
├── experiments/
│   ├── convergence_curves.py
│   └── cnn_eho_gpu.py                  # Earlier prototype
```

---

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/ArrushTandon/Plant_Disease_Prediction_Model.git
cd Plant_Disease_Prediction_Model

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🖥️ Hardware Requirements

### Minimum
| Component | Requirement        |
|-----------|--------------------|
| CPU       | Intel i5 / Ryzen 5 |
| RAM       | 8 GB               |
| Storage   | 15-20 GB           |
| GPU       | Optional           |

### Recommended (for training)
| Component | Requirement               |
|-----------|---------------------------|
| CPU       | Intel i7 / Ryzen 7        |
| RAM       | 16 GB+                    |
| GPU       | NVIDIA GPU (CUDA support) |
| Storage   | SSD                       |

> ⚠️ GPU significantly speeds up CNN feature extraction, training time, and metaheuristic optimization.

---

## 📊 Dataset

**Dataset used:** [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)

**Steps:**
1. Download the dataset
2. Run the preprocessing script:

```bash
python scripts/data_preparation.py
```

This generates:

```
data/processed/
    ├── healthy/
    └── diseased/
```

---

## 🧠 How It Works

1. **Image Preprocessing** — Resize to 128×128, normalize grayscale images
2. **Feature Extraction** — CNN extracts deep features; LBP extracts texture features
3. **Feature Fusion** — Combine CNN + LBP features
4. **Dimensionality Reduction** — PCA reduces feature space
5. **Feature Selection** — EHO / DBO / SMA selects optimal subset
6. **Data Balancing** — SMOTE handles class imbalance
7. **Classification** — MLP neural network
8. **Evaluation** — ROC-AUC, Confusion Matrix, Precision-Recall, Threshold tuning

---

## 📈 Results

| Metric              | Value (example) |
|---------------------|-----------------|
| Accuracy            | 96.3%           |
| Precision (Healthy) | 95.2%           |
| Recall (Diseased)   | 97.8%           |
| ROC-AUC             | 0.98            |

**Visualizations include:**
- Confusion Matrix
- Precision-Recall curves
- ROC curves
- Optimization convergence tracking

---

## 🧪 Run the Project

### Main Pipelines

```bash
# EHO-based model
python src/cnn_eho_gpu_updated.py

# DBO-based model
python src/cnn_dbo_gpu_updated.py

# SMA-based model
python src/cnn_sma_gpu_updated.py
```

### Baselines

```bash
python baselines/train_cnn.py
python baselines/train.py
```

### Experiments

```bash
python experiments/convergence_curves.py
```

---

## ✍️ Authors

- Arrush Tandon
- Jiya Shrivastava
- Anshuman Semwal
- Irkan A. Saifi

---

## 📜 License

This project is licensed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch
3. Submit a pull request

Feel free to raise issues or suggest improvements 🚀