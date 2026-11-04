# Benchmark Report: Original Imbalanced Baseline
**Dataset:** Credit Card Fraud Detection  
**Strategy:** Original Class Distribution (No Resampling, No Class Weighting)  
**Generated:** 2026-10-30 19:06:57  

## 1. Experimental Design & Leakage-Free Protocol

- **Data Integrity:** Raw CSV is completely untouched. Deduplication executed in memory.
- **Raw Record Count:** 284,807 (492 fraud, 284,315 legitimate).
- **Deduplicated Record Count:** 283,726 (1,081 exact full-row duplicates removed across all columns).
- **Stratified 60/20/20 Partitioning (Fixed Seed 42):**
  - **Training Split:** 170,235 samples (284 fraud, 0.167%).
  - **Validation Split:** 56,745 samples (94 fraud, 0.166%).
  - **Test Split:** 56,746 samples (95 fraud, 0.167%).
- **Zero Leakage Enforcement:** Numeric imputation, RobustScaler, and categorical imputation/OneHotEncoder are fitted strictly on training data. Resamplers omitted. Decision thresholds optimized strictly on validation data and applied frozen to test data.

## 2. Test Set Performance: Default Decision Threshold (θ = 0.50)

| Classifier | Average Precision (AP) | Recall | Precision | F1-Score | MCC | ROC-AUC | FPR | TP | FP | FN | TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7447 | 0.6000 | 0.8636 | 0.7081 | 0.7195 | 0.9678 | 0.000159 | 57 | 9 | 38 | 56,642 |
| **Decision Tree (CART)** | 0.6852 | 0.7684 | 0.8795 | 0.8202 | 0.8218 | 0.8630 | 0.000177 | 73 | 10 | 22 | 56,641 |
| **Random Forest** | 0.8420 | 0.7684 | 0.9733 | 0.8588 | 0.8646 | 0.9720 | 0.000035 | 73 | 2 | 22 | 56,649 |
| **XGBoost** | 0.8468 | 0.7895 | 0.9740 | 0.8721 | 0.8767 | 0.9784 | 0.000035 | 75 | 2 | 20 | 56,649 |
| **LightGBM** | 0.2189 | 0.6526 | 0.2279 | 0.3379 | 0.3841 | 0.7860 | 0.003707 | 62 | 210 | 33 | 56,441 |

## 3. Test Set Performance: Validation-Tuned Decision Threshold (Frozen θ*val)

| Classifier | Optimal θ*val | Average Precision (AP) | Recall | Precision | F1-Score | MCC | ROC-AUC | FPR | TP | FP | FN | TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.0800 | 0.7447 | 0.7790 | 0.7957 | 0.7872 | 0.7869 | 0.9678 | 0.000335 | 74 | 19 | 21 | 56,632 |
| **Decision Tree (CART)** | 0.3400 | 0.6852 | 0.7684 | 0.8795 | 0.8202 | 0.8218 | 0.8630 | 0.000177 | 73 | 10 | 22 | 56,641 |
| **Random Forest** | 0.3800 | 0.8420 | 0.8000 | 0.9744 | 0.8786 | 0.8827 | 0.9720 | 0.000035 | 76 | 2 | 19 | 56,649 |
| **XGBoost** | 0.3400 | 0.8468 | 0.8000 | 0.9500 | 0.8686 | 0.8716 | 0.9784 | 0.000071 | 76 | 4 | 19 | 56,647 |
| **LightGBM** | 0.9900 | 0.2189 | 0.6316 | 0.2752 | 0.3834 | 0.4155 | 0.7860 | 0.002789 | 60 | 158 | 35 | 56,493 |

## 4. Operational Latency and Computational Cost

| Classifier | Training Duration (s) | Test Inference Latency (ms / 1,000 tx) |
| :--- | :---: | :---: |
| **Logistic Regression** | 1.135 s | 0.51 ms |
| **Decision Tree (CART)** | 7.562 s | 0.57 ms |
| **Random Forest** | 21.997 s | 1.79 ms |
| **XGBoost** | 2.205 s | 0.81 ms |
| **LightGBM** | 1.525 s | 1.30 ms |

## 5. Methodological Observations

- **Impact of Extreme Skewness:** Without class weighting or resampling, models with uncalibrated probabilities frequently optimize for global accuracy, depressing Recall at the default threshold θ = 0.50.
- **Value of Validation-Tuned Thresholds:** Tuning θ strictly on validation data allows models to navigate the precision-recall trade-off without risking test-set data leakage.
- **Next Comparative Stage:** These original baseline scores establish the formal empirical benchmark against which Class Weighting, SMOTE, and Controlled Undersampling will be compared.
