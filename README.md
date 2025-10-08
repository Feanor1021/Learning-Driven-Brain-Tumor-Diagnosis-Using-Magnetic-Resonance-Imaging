# 🧠 Learning-Driven Brain Tumor Diagnosis Using Magnetic Resonance Imaging

## 📖 Introduction
Brain tumor detection from MRI is a critical, expert-driven task. This project applies machine learning and deep learning techniques to classify axial T1-weighted contrast-enhanced brain MRI slices into one of four categories:
- `no_tumor`
- `pituitary`
- `meningioma`
- `glioma`

Three models are compared:
- 📌 **SVM (RBF + PCA)**
- 📌 **SimpleCNN**
- 📌 **DeepCNN** (final model with best performance)

---

## 📊 Dataset

- **Source**: [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Total Images**: 7,023
- **Classes**: 4
- **Format**: Grayscale axial slices, resized to 200×200
- **Split**: Custom stratified 80/20 split from merged Training and Testing folders

---

## ⚙️ Installation

### 🔹 Clone Repository
```bash
git clone https://github.com/yourusername/brain-tumor-diagnosis.git
cd brain-tumor-diagnosis
```

### 🔹 Install Dependencies
```bash
pip install -r requirements.txt
```

### 🔹 Download Dataset
```python
import kagglehub
kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
```



---

## 🧠 Models

### ✅ SVM (RBF + PCA)
- Preprocessing: flatten → PCA (98% variance) → StandardScaler
- Model: SVC with RBF kernel
- Loss: Hinge loss (1-vs-1)

### ✅ SimpleCNN
- Conv → ReLU → MaxPool → Dropout (×3)
- GlobalAvgPool → Dense(128) → Dropout → Softmax

### ✅ DeepCNN
- 5 Conv blocks: (32 → 512 filters)
- BatchNorm + MaxPool + Dropout
- GlobalAvgPool → Dense(512) → Dropout → Softmax
- Optimizer: Adam with ReduceLROnPlateau and early stopping

---

## 📈 Results

| Model                 | Accuracy | Balanced Acc | F1 Macro |
|----------------------|----------|---------------|----------|
| 🧠 **DeepCNN**        | **97.8%** | **97.7%**     | **97.7%** |
| 📦 SimpleCNN          | 90.2%    | 89.7%         | 89.6%    |
| ⚙️ SVM (RBF + PCA98%) | 84.9%    | 84.6%         | 84.5%    |

- DeepCNN outperforms both SVM and SimpleCNN by a large margin.
- Most confusion occurs between `meningioma` and `glioma`.

---

## ⚠️ Limitations & Future Work

### Limitations:
- Slice-wise classification (no 3D volume context)
- No subject-wise or cross-center validation
- Single test split, not cross-validated

### Future directions:
- 2.5D / 3D CNNs
- Patient-level aggregation
- External datasets
- Probability calibration (Platt scaling)
- Deployment with real-time inference

---

## 📚 References

- SEER, CBTRUS, GLOBOCAN cancer statistics
- Khalighi et al. (2024), Al-Rahbi et al. (2025)
- Kingma & Ba — *Adam Optimizer*
- Dataset by Masoud Nickparvar (Kaggle)
