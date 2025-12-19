# 🗑️ Material Stream Identification System

Classical computer vision + machine learning pipeline for material classification using SVM and k-NN.

================================================================

## 🎯 Project Overview
This project implements an end-to-end feature-based vision system to classify materials into 6 known classes + 1 class ("Unknown").

**Pipeline**:
Data Augmentation → Feature Extraction → Model Training → Evaluation → Real-time Deployment

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
│   ├── augmented/
│   ├── raw/
│   │   ├── cardboard/
│   │   ├── glass/
│   │   ├── metal/
│   │   ├── paper/
│   │   ├── plastic/
│   │   └── trash/
│   └── splits/
│       ├── train_paths.csv
│       └── val_paths.csv
│
├── features/
│   ├── cnn_feature_extraction.py
│   ├── feature_extraction.py
│   ├── hand_crafted_feature_extraction.py
│   └── (resnet|effientnet_lbp|etc.)/(train|val)_features*.npy
│
├── models/
│   ├── svm/
│   │   ├── train_svm.py
	│   └── evaluate_svm.py
│   └── knn/
│       ├── train_knn.py
		 └── evaluate_knn.py
│
├── deployment/
│   ├── class_mapping.json
│   ├── realtime_app.py
│   └── inference.py
│
├── src/
│   ├── config.py
│   ├── data_pipeline.py
│   ├── feature_extraction.py
	└── trash_unkown_handler.py
│
├── notebooks/
│   └── (many notebooks for experiments)
│
├── tests/
│   └── test.py
│
├── report/
│   └── MSI_Technical_Report.pdf
│
├── README.md
├── requirements.txt
└── .gitignore
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
#### Run the whole project (module-style commands)
Run the full pipeline and individual steps using the module interface below.

- Prepare data + augmentations (creates augmented images and split CSVs):

```bash
python -m src.data_pipeline
```

- Feature extraction (set `method` inside `features/feature_extraction.py` if needed):
- Recommended feature-extraction methods to use (configure `method` in `features/feature_extraction.py`):

	- **SVM (best):** `efficientnet_lbp`
	- **k-NN (best):** `resnet_lbp`

```bash
python -m features.feature_extraction
```

Output examples (feature files will be saved to `features/`):

```
features/
├── train_features.npy
├── train_labels.npy
├── val_features.npy
└── val_labels.npy
```

- Train & evaluate models:

```bash
python -m models.svm.train_svm
python -m models.svm.evaluate_svm
python -m models.knn.train_knn
python -m models.knn.evaluate_knn
```

- Realtime application (OpenCV camera app):

```bash
python -m deployment.realtime_app
```

- Run tests (place test images in `tests/test_images` first):

```bash
python -m tests.test
```
