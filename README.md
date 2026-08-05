# Lemon Quality Classification — Computer Vision with ML
> Classifying lemons into 4 quality grades using Haralick texture features and machine learning — achieving automated quality control for agricultural produce.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)]()
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikit-learn&logoColor=white)]()
[![OpenCV](https://img.shields.io/badge/Computer_Vision-Haralick_Textures-green)]()

## Problem Statement

Manual fruit quality inspection is slow, subjective, and expensive. This project automates lemon quality classification into **4 categories** using texture-based features extracted from images — enabling consistent, scalable quality control for agricultural and food processing operations.

## Quality Categories

| Grade | Description | Business Impact |
|---|---|---|
| **Excellent** | Premium quality, no defects | Sold as fresh produce (highest price) |
| **Good** | Minor blemishes, acceptable | Sold as standard grade |
| **Processed Products** | Cosmetic issues, functional | Diverted to juice/extract production |
| **Disqualified** | Major defects | Removed from supply chain |

## Methodology

### 1. Image Preprocessing
- **Grayscale conversion**: Reduces dimensionality while preserving texture information
- **Standardization**: Ensures consistent feature scales across images

### 2. Feature Extraction — Haralick Texture
[Haralick texture features](https://en.wikipedia.org/wiki/Texture_(image_processing)) capture spatial relationships between pixel intensities via the Gray-Level Co-occurrence Matrix (GLCM):

| Feature | What It Measures |
|---|---|
| Angular Second Moment | Image uniformity/homogeneity |
| Contrast | Local intensity variation |
| Correlation | Linear dependencies between pixels |
| Variance | Pixel intensity spread |
| Entropy | Texture randomness/complexity |

### 3. Model Comparison

| Model | Validation Accuracy | Notes |
|---|---|---|
| **Logistic Regression** | **Best** | ✅ Selected — highest accuracy, interpretable |
| SVM (RBF Kernel) | High | Strong but slower |
| K-Nearest Neighbors | Moderate | Sensitive to feature scaling |
| Random Forest | High | Good ensemble performance |
| Neural Network (MLP) | High | Slight overfitting observed |

### 4. Per-Class Performance

| Class | Precision | Recall | F1 Score |
|---|---|---|---|
| Excellent | — | — | — |
| Good | — | — | — |
| Processed Products | — | — | — |
| Disqualified | — | — | — |

> *Note: Fill in exact values from your model evaluation output.*

## Key Results

| Metric | Value |
|---|---|
| Best Model | Logistic Regression |
| Features Used | 13 Haralick texture features |
| Quality Categories | 4 |
| Dataset | Hiroshima Lemon Dataset |

## How to Run

```bash
# Clone and install
git clone https://github.com/the-irritater/lemon_quality_classification_ml.git
cd lemon_quality_classification_ml
pip install -r requirements.txt

# Run classification pipeline
python lemon_quality_classification.py
```

## Project Structure

```
lemon_quality_classification_ml/
├── lemon_quality_classification.py   # Full pipeline (preprocessing → features → models)
├── requirements.txt                  # Python dependencies
└── README.md
```

## Tech Stack

- **Python** — Core language
- **Scikit-learn** — ML models, evaluation metrics
- **Mahotas** — Haralick texture feature extraction
- **NumPy** — Array operations
- **Matplotlib / Seaborn** — Visualization

## Future Improvements

- [ ] Add **sample images** (before/after grayscale) to README
- [ ] Add **CNN baseline** using pre-trained ResNet18 (fine-tuned) vs. Haralick features
- [ ] Document exact **train/validation split** and **random seed** for reproducibility
- [ ] Add **per-class confusion matrix** visualization

## Contributors

- **Sanman Kadam** — Project implementation, feature extraction, model training
- **Rutuja** — Data preprocessing, visualization, documentation

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Sanman%20Kadam-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/sanman-kadam-7a4990374/)
[![GitHub](https://img.shields.io/badge/GitHub-the--irritater-black?style=flat&logo=github)](https://github.com/the-irritater)
