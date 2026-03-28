# Hybrid-CNN-Morphological-features-for-advance-Breast-Cancer-Classification-
# 🎗️ Hybrid CNN + Morphological Features for Advanced Breast Cancer Classification

A deep learning pipeline that fuses **Convolutional Neural Network (CNN)** extracted features with hand-crafted **morphological descriptors** for robust, multi-class breast cancer classification from histopathological images.

---

## 📌 Overview

Manual interpretation of breast histopathology slides is time-consuming and prone to inter-pathologist variability. This project addresses that gap by combining the automatic spatial feature learning of CNNs with morphological texture features (shape, boundary, texture) to improve classification accuracy and interpretability.

---

## 🧠 Approach

The pipeline operates in three stages:

1. **Preprocessing** — Image normalization, color correction (H&E stain standardization), and data augmentation (rotation, flipping, shearing).
2. **Feature Extraction** — Dual-branch extraction:
   - *CNN branch*: Convolutional layers learn hierarchical spatial representations.
   - *Morphological branch*: Hand-crafted descriptors such as GLCM, area, perimeter, and eccentricity capture structural tissue patterns.
3. **Hybrid Classification** — Fused feature vectors are passed through fully connected layers for final class prediction.

---

## 📂 Dataset

The model is evaluated on publicly available breast histopathology datasets. Recommended options:

- [BreaKHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) — Microscopic biopsy images at multiple magnifications
- [BreakHis / IDC on Kaggle](https://www.kaggle.com/) — Invasive Ductal Carcinoma patches

> Place your dataset under `data/` and update paths in `config.py`.

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.9+ |
| Deep Learning | TensorFlow / Keras or PyTorch |
| Image Processing | OpenCV, scikit-image |
| Feature Engineering | NumPy, SciPy |
| Visualization | Matplotlib, Seaborn |

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/Themahattva/Hybrid-CNN-Morphological-features-for-advance-Breast-Cancer-Classification-.git
cd Hybrid-CNN-Morphological-features-for-advance-Breast-Cancer-Classification-

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

# Evaluate
python evaluate.py
```

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| CNN only | — | — | — | — |
| Morphological only | — | — | — | — |
| **Hybrid (Ours)** | **—** | **—** | **—** | **—** |

> Results will be updated upon final training runs.

---

## 📁 Project Structure

```
├── data/                  # Dataset directory
├── models/                # Saved model weights
├── notebooks/             # Exploratory analysis
├── src/
│   ├── preprocessing.py   # Image preprocessing
│   ├── cnn_features.py    # CNN feature extractor
│   ├── morph_features.py  # Morphological descriptor extraction
│   ├── fusion.py          # Feature fusion module
│   └── classifier.py      # Final classification head
├── train.py
├── evaluate.py
├── config.py
└── requirements.txt
```

---

## 👤 Author

**Mahattva** — [@Themahattva](https://github.com/Themahattva)  
BIT Durg | 
---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
