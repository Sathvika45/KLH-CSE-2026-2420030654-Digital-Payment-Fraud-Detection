# Benchmark Report: Original Imbalanced Baseline
**Dataset:** Fraud E-commerce  
**Strategy:** Original Class Distribution (No Resampling, No Class Weighting)  
**Generated:** 2026-10-30 19:05:22  

## 1. Experimental Design & Leakage-Free Protocol

- **Data Integrity:** Raw CSV is completely untouched. Deduplication executed in memory.
- **Raw Record Count:** 151,112 (14,151 fraud, 136,961 legitimate).
- **Deduplicated Record Count:** 144,273 (6,839 exact full-row duplicates removed across all columns).
- **Stratified 60/20/20 Partitioning (Fixed Seed 42):**
  - **Training Split:** 86,563 samples (4,387 fraud, 5.068%).
  - **Validation Split:** 28,855 samples (1,463 fraud, 5.070%).
  - **Test Split:** 28,855 samples (1,462 fraud, 5.067%).
- **Zero Leakage Enforcement:** Numeric imputation, RobustScaler, and categorical imputation/OneHotEncoder are fitted strictly on training data. Resamplers omitted. Decision thresholds optimized strictly on validation data and applied frozen to test data.

## 2. Test Set Performance: Default Decision Threshold (θ = 0.50)

| Classifier | Average Precision (AP) | Recall | Precision | F1-Score | MCC | ROC-AUC | FPR | TP | FP | FN | TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.0630 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5369 | 0.000000 | 0 | 0 | 1462 | 27,393 |
| **Decision Tree (CART)** | 0.1568 | 0.1053 | 0.9222 | 0.1891 | 0.3032 | 0.5607 | 0.000475 | 154 | 13 | 1308 | 27,380 |
| **Random Forest** | 0.1711 | 0.1053 | 1.0000 | 0.1906 | 0.3171 | 0.5594 | 0.000000 | 154 | 0 | 1308 | 27,393 |
| **XGBoost** | 0.1306 | 0.0814 | 0.5694 | 0.1424 | 0.2020 | 0.5527 | 0.003286 | 119 | 90 | 1343 | 27,303 |
| **LightGBM** | 0.1764 | 0.1053 | 1.0000 | 0.1906 | 0.3171 | 0.5569 | 0.000000 | 154 | 0 | 1308 | 27,393 |

## 3. Test Set Performance: Validation-Tuned Decision Threshold (Frozen θ*val)

| Classifier | Optimal θ*val | Average Precision (AP) | Recall | Precision | F1-Score | MCC | ROC-AUC | FPR | TP | FP | FN | TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.0600 | 0.0630 | 0.2716 | 0.0671 | 0.1077 | 0.0381 | 0.5369 | 0.201365 | 397 | 5516 | 1065 | 21,877 |
| **Decision Tree (CART)** | 0.5100 | 0.1568 | 0.1053 | 0.9686 | 0.1900 | 0.3115 | 0.5607 | 0.000183 | 154 | 5 | 1308 | 27,388 |
| **Random Forest** | 0.4400 | 0.1711 | 0.1053 | 1.0000 | 0.1906 | 0.3171 | 0.5594 | 0.000000 | 154 | 0 | 1308 | 27,393 |
| **XGBoost** | 0.1700 | 0.1306 | 0.1094 | 0.5714 | 0.1837 | 0.2351 | 0.5527 | 0.004381 | 160 | 120 | 1302 | 27,273 |
| **LightGBM** | 0.1100 | 0.1764 | 0.1088 | 0.6385 | 0.1859 | 0.2501 | 0.5569 | 0.003286 | 159 | 90 | 1303 | 27,303 |

## 4. Operational Latency and Computational Cost

| Classifier | Training Duration (s) | Test Inference Latency (ms / 1,000 tx) |
| :--- | :---: | :---: |
| **Logistic Regression** | 0.256 s | 1.08 ms |
| **Decision Tree (CART)** | 0.427 s | 1.03 ms |
| **Random Forest** | 1.474 s | 2.81 ms |
| **XGBoost** | 0.481 s | 1.82 ms |
| **LightGBM** | 0.504 s | 1.79 ms |

## 5. Methodological Observations

- **Impact of Extreme Skewness:** Without class weighting or resampling, models with uncalibrated probabilities frequently optimize for global accuracy, depressing Recall at the default threshold θ = 0.50.
- **Value of Validation-Tuned Thresholds:** Tuning θ strictly on validation data allows models to navigate the precision-recall trade-off without risking test-set data leakage.
- **Next Comparative Stage:** These original baseline scores establish the formal empirical benchmark against which Class Weighting, SMOTE, and Controlled Undersampling will be compared.
