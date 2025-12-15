# 🗑️ Material Stream Identification System
Machine Learning Course – Fall 2025

Classical computer vision + machine learning pipeline for post-consumer waste classification using SVM and k-NN.

================================================================

## 🎯 Project Overview
This project implements an end-to-end feature-based vision system to classify waste materials into 6 known classes + 1 rejection class ("Unknown").

**Pipeline**:
Data Augmentation → Feature Extraction → Model Training → Evaluation → Real-time Deployment

**Target**:
≥ 85% validation accuracy on the 6 primary classes.

================================================================

### 🧾 Classes
------------

| ID | Class              |
|----|--------------------|
| 0  | Glass              |
| 1  | Paper              |
| 2  | Cardboard          |
| 3  | Plastic            |
| 4  | Metal              |
| 5  | Trash              |
| 6  | Unknown (Rejection)|

================================================================

## 📁 Repository Structure
----------------------
```
msi-project/
├── data/
│   ├── raw/                    # Original dataset (class folders 0–5) ⭐
│   ├── augmented/              # Augmented training images ⭐
│   └── splits/                 # train_paths.txt, val_paths.txt ⭐
│
├── features/                   # Extracted feature vectors (.npy) ⭐
│   ├── train_features.npy
│   ├── train_labels.npy
│   ├── val_features.npy
│   └── val_labels.npy
│
├── models/                     # Trained SVM & k-NN models
│   ├── svm/
│   └── knn/
│
├── deployment/                 # Real-time OpenCV camera app
│   └── realtime_app.py
│
├── src/                        # Core ML pipeline scripts
│   ├── data_pipeline.py
│   └── feature_extraction.py
│
├── notebooks/                  # EDA and experiments
├── tests/                      # End-to-end tests
├── report/                     # Technical report and figures
│   └── report.pdf
│
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
└── .gitignore                  # Ignore large data files
```

================================================================

## 🚀 Quick Start
--------------
### 1) Clone & Setup
----------------
```bash
git clone https://github.com/<your-username>/msi-project.git
cd msi-project
pip install -r requirements.txt
```
### 2) Place Dataset
----------------
Copy the raw dataset into:
data/raw/
```
Folder structure example:
data/raw/
├── glass/
├── paper/
├── cardboard/
├── plastic/
├── metal/
└── trash/
```
### 3) Run Pipeline (Step-by-Step)
------------------------------
#### Step 1: Data preparation + augmentation
python src/data_pipeline.py

Output:
- Augmented images
- train_paths.txt
- val_paths.txt

#### Step 2: Feature extraction
python features/feature_extraction.py

Output:
```
features/
├── train_features.npy   (3000, 98)
├── train_labels.npy
├── val_features.npy     (373, 98)
└── val_labels.npy
```
================================================================

📊 Current Progress
------------------

| Step                     | Status | Output                          |
|--------------------------|--------|---------------------------------|
| Data Prep + Augmentation | ✅     | 3000 train, 373 val            |
| Feature Extraction       | ✅     | 98-dim feature vectors          |
| SVM Training             | ⏳     | models/svm/best_svm_model.pkl   |
| k-NN Training            | ⏳     | models/knn/best_knn_model.pkl   |
| Model Comparison         | ⏳     | models/evaluation/results.csv   |
| Real-time Deployment     | ⏳     | deployment/realtime_app.py      |
| Technical Report         | ⏳     | report/report.pdf               |

================================================================

🛠️ Technical Stack
------------------
**Data:**
- PIL
- NumPy
- Text-based dataset splits

**Feature Extraction:**
- scikit-image
- RGB & HSV histograms
- Sobel gradient statistics

**Models:**
- scikit-learn
- Support Vector Machine (SVM)
- k-Nearest Neighbors (k-NN)

**Deployment:**
- OpenCV (real-time camera feed)

================================================================

📈 Feature Extraction Details
----------------------------
**Method:**
hist_grad

**Image Size:**
64 × 64

**Features:**
- RGB histograms: 3 × 16 bins
- HSV histograms: 3 × 16 bins
- Sobel gradients: mean + std

**Total Features:**
96 (histograms) + 2 (gradients) = 98-dimensional vector
