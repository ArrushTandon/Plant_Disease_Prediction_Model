# Plant Disease Detection using CNN, LBP & EHO

This project is a hybrid deep learning and machine learning pipeline for detecting plant leaf diseases from grayscale images. It combines handcrafted features (LBP) with CNN-based visual features, applies dimensionality reduction (PCA), and optimizes feature selection using Elephant Herding Optimization (EHO). The system evaluates model performance using multiple statistical metrics.

---

## 🚀 Key Features
- **CNN Feature Extraction** with Keras
- **Local Binary Pattern (LBP)** for handcrafted image features
- **PCA** for dimensionality reduction
- **Elephant Herding Optimization (EHO)** for intelligent feature selection
- **Robust Evaluation**: Precision-Recall, Confusion Matrix, ROC-AUC
- **Mixed Precision** training for GPU acceleration

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
│       ├── healthy/                    # Preprocessed healthy leaf images
│       └── diseased/                   # Preprocessed diseased leaf images
│
├── scripts/
│   └── data_preparation.py             # Preprocessing & dataset splitting
│
├── src/
│   ├── models/
│   │   └── cnn_model.py                # CNN Base model architecture
│   ├── optimization/
│   │   └── eho_optimization.py         # EHO logic
│   ├── utils/
│   │    ├── visualize.py               # LBP histogram visualization
│   │    └── lbp_feature_extraction.py  # LBP Feature Extraction
│   └── cnn_eho_gpu_updated             # Main File
│
├── experiments/
│   ├── convergence_curves.py           # Fitness & accuracy tracking
│   └── cnn_eho_gpu.py                  # Earlier prototype
```


---

## 📦 Installation
```bash
# Clone the repository
$ git clone https://github.com/ArrushTandon/Plant_Disease_Prediction_Model.git
$ cd Plant_Disease_Prediction_Model

# Create virtual environment (optional but recommended)
$ python -m venv venv
$ source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
$ pip install -r requirements.txt
```

---

## 📊 Dataset
Uses the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease).
- Prepare data using `scripts/data_preparation.py`
- It will create `processed/healthy/` and `processed/diseased/`

---

## 🧠 How It Works
1. **Data Preprocessing:** Resize, normalize, extract LBP
2. **Feature Extraction:** CNN + LBP
3. **Dimensionality Reduction:** PCA
4. **Feature Selection:** EHO
5. **Balancing:** SMOTE
6. **Classification:** MLP
7. **Evaluation:** Metrics + threshold tuning

---

## 📈 Results
| Metric              | Value (example) |
|---------------------|-----------------|
| Accuracy            | 96.3%           |
| Precision (Healthy) | 95.2%           |
| Recall (Diseased)   | 97.8%           |
| ROC-AUC             | 0.98            |

Includes:
- Confusion Matrix
- Precision-Recall Curves
- Convergence curves (training loss, accuracy, EHO fitness)

---

## 🧪 Run Main Pipeline
```bash
python src/cnn_eho_gpu_updated.py
```

You can also run:
```bash
python baselines/train_cnn.py        # Basic CNN
python baselines/train.py            # LBP + EHO
python experiments/convergence_curves.py  # Extended analysis
```

---

## ✍️ Authors
- Arrush Tandon
- Jiya Shrivastava
- Anshuman Semwal
- Irkan A. Saifi

---

## 📜 License
[MIT License](LICENSE)

---

## 🙋‍♂️ Want to Contribute?
Pull requests, issues, and forks are welcome! Feel free to create issues or suggest improvements.
