# 🧠 Learning-Driven Brain Tumor Diagnosis Using Magnetic Resonance Imaging

## 📖 Introduction
Brain tumor detection from MRI is a critical yet time-consuming process requiring expert radiologists.  
This project explores **machine learning (ML)** and **deep learning (DL)** models for **automated brain tumor classification**.

Each MRI slice is categorized into one of four classes:
- 🧩 `no_tumor`
- 🧩 `pituitary`
- 🧩 `meningioma`
- 🧩 `glioma`

We compare traditional and deep approaches:  
➡️ **SVM (RBF + PCA)** — classical ML baseline  
➡️ **SimpleCNN** — lightweight CNN  
➡️ **DeepCNN** — deeper convolutional architecture with best performance

---

## 🗂️ Table of Contents
- [Introduction](#-introduction)
- [Dataset](#-dataset)
- [Installation](#️-installation)
- [Usage](#️-usage)
- [Models](#-models)
- [Results](#-results)
- [Examples](#-examples)
- [Limitations & Future Work](#-limitations--future-work)
- [Contributors](#-contributors)
- [License](#-license)

---

## 📊 Dataset
**Dataset**: [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)  
**Author**: Masoud Nickparvar  
**Total Images**: 7,023 axial MRI slices  
**Classes**: `no_tumor`, `pituitary`, `meningioma`, `glioma`

### Processing
- Original *Training* and *Testing* folders merged  
- New **80/20 stratified split** created (to preserve class proportions)  
- Images resized to **200×200**, grayscale normalized to `[0,1]`  
- Pixel histograms inspected (strong right skew due to dark background)  

---

## ⚙️ Installation

### 1️⃣ Clone Repository
```bash
git clone https://github.com/yourusername/brain-tumor-diagnosis.git
cd brain-tumor-diagnosis
2️⃣ Install Dependencies
bash
Kodu kopyala
pip install -r requirements.txt
3️⃣ Download Dataset
python
Kodu kopyala
import kagglehub
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
print(path)
Or manually place dataset in ./data/brain_tumor_mri_dataset/.

▶️ Usage
🧩 Train Models
bash
Kodu kopyala
python train.py --model svm
python train.py --model simplecnn
python train.py --model deepcnn
🧾 Evaluate Performance
After training, metrics such as accuracy, F1-score, and confusion matrices will be generated automatically.

🧠 Models
🔹 SVM (RBF + PCA)
Flattened 200×200 images → PCA (retain 98% variance)

StandardScaler applied on training PCA features

RBF kernel SVM (C=10, gamma=scale)

Loss: hinge loss

Library: scikit-learn

🔹 SimpleCNN
3× Conv-ReLU + MaxPooling blocks

Global Average Pooling → Dense(128) → Dropout → Softmax

Regularization: BatchNorm, Dropout

Optimizer: Adam (lr=1e-3), Loss: Categorical Cross-Entropy

🔹 DeepCNN
5× Conv blocks (filters: 32 → 512)

Double Conv per block + BatchNorm + MaxPooling

GAP → Dense(512) → Dropout → Softmax

Scheduler: ReduceLROnPlateau

Early stopping on validation loss

Optimizer: Adam (lr=1e-3)

📈 Results
Model	Accuracy	Balanced Acc	Precision (Macro)	Recall (Macro)	F1 (Macro)
🧠 DeepCNN	0.9779	0.9770	0.9770	0.9770	0.9770
🧩 SimpleCNN	0.9025	0.8971	0.9005	0.8971	0.8964
⚙️ SVM (RBF + PCA98%)	0.8498	0.8463	0.8514	0.8463	0.8456

✅ Key Findings
DeepCNN achieved 97.8% accuracy, outperforming both baselines

Most confusion between meningioma and glioma classes

Deep learning clearly surpasses classical ML with end-to-end learned features

🖼️ Examples
Sample MRI slices: Shown per class (visual inspection)

Pixel intensity histograms: Show mid-intensity overlap between tumors

PCA plots: 2D separation reveals class clusters

Confusion matrices: DeepCNN has near-diagonal performance

(Figures available in /assets/ folder of the full repository)

⚠️ Limitations & Future Work
Slice-based analysis (no 3D context)

Merged dataset (not subject-wise split)

Single hold-out test → Potential optimistic bias

🔮 Future Work
Subject-level & multi-center validation

MRI-specific intensity normalization

2.5D / 3D CNN architectures

Calibrated probability outputs (Platt scaling)

ROC / PR curve analysis for clinical decision thresholds

Model compression for real-time inference

👨‍💻 Contributors
Furkan Yardımcı — Data Preparation, Model Development, Report Writing

📄 License
This project is released under the MIT License.
See LICENSE for more information.

🔗 References
SEER Cancer Stats (2025)

CBTRUS Report (2024)

GLOBOCAN (2022)

Khalighi et al., npj Precision Oncology (2024)

Al-Rahbi et al., EJNPNS (2025)

Masoud Nickparvar — Kaggle Dataset

Kingma & Ba (2015), Adam Optimizer
