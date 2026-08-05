# Lemon Quality Classification: Computer Vision and Machine Learning

Automated classification of lemons into four quality categories utilizing Haralick texture feature extraction and PyTorch ResNet18 Convolutional Neural Network architectures.

## Problem Statement

Manual agricultural produce inspection is subjective and labor-intensive. Automated computer vision inspection standardizes quality control across four commercial categories:
- **Excellent**: Premium grade suitable for fresh produce markets.
- **Good**: Standard commercial quality with minor cosmetic variations.
- **Processed Products**: Blemished produce suitable for juice or extract processing.
- **Disqualified**: Sub-standard produce rejected from commercial distribution.

## Feature Extraction & Model Benchmarking

### Haralick Texture Features (GLCM)
Extracted Gray-Level Co-occurrence Matrix (GLCM) statistics capturing spatial pixel intensity relationships:
- Angular Second Moment (Uniformity)
- Contrast (Local intensity variations)
- Correlation (Linear spatial dependencies)
- Variance (Intensity dispersion)
- Entropy (Texture complexity)

### Model Performance Comparison

| Model Architecture | Feature Representation | Validation Accuracy | Notes |
|---|---|---|---|
| Logistic Regression | Haralick Texture (GLCM) | 86.5% | Optimal classical baseline |
| Support Vector Machine (RBF) | Haralick Texture (GLCM) | 84.2% | High precision on Disqualified class |
| K-Nearest Neighbors | Haralick Texture (GLCM) | 78.0% | Sensitive to feature normalization |
| Random Forest | Haralick Texture (GLCM) | 83.5% | Robust ensemble baseline |
| ResNet18 CNN (PyTorch) | Learned Deep Feature Maps | 89.2% | Transfer learning fine-tuning |

## Project Structure

```
lemon_quality_classification_ml/
├── outputs/
│   └── model_comparison.csv
├── lemon_quality_classification.py
├── resnet_baseline.py
├── requirements.txt
└── README.md
```

## How to Run

### Classical ML Pipeline (Haralick Features)
```bash
pip install -r requirements.txt
python lemon_quality_classification.py
```

### PyTorch ResNet18 CNN Baseline
```bash
python resnet_baseline.py
```

## Authors

- Sanman Kadam (MSc Statistics | Data Analyst)
- Rutuja (Data Preprocessing and Analytics)
