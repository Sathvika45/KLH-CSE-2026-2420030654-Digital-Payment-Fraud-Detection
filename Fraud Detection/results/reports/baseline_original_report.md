# Benchmark Report: Original Imbalanced Baseline
**Dataset:** Fraud E-commerce  
**Strategy:** Original Class Distribution (No Resampling, No Class Weighting)  
**Generated:** 2026-10-30 18:44:30  

## 1. Experimental Design & Leakage-Free Protocol

- **Data Integrity:** Raw CSV is completely untouched. Deduplication executed in memory.
- **Raw Record Count:** 20,000 (1,873 fraud, 18,127 legitimate).
- **Deduplicated Record Count:** 19,558 (442 exact full-row duplicates removed across all columns).
- **Stratified 60/20/20 Partitioning (Fixed Seed 42):**
  - **Training Split:** 11,734 samples (859 fraud, 7.321%).
  - **Validation Split:** 3,912 samples (286 fraud, 7.311%).
  - **Test Split:** 3,912 samples (286 fraud, 7.311%).
- **Zero Leakage Enforcement:** Numeric imputation, RobustScaler, and categorical imputation/OneHotEncoder are fitted strictly on training data. Resamplers omitted. Decision thresholds optimized strictly on validation data and applied frozen to test data.

## 2. Test Set Performance: Default Decision Threshold (θ = 0.50)

| Classifier | Average Precision (AP) | Recall | Precision | F1-Score | MCC | ROC-AUC | FPR | TP | FP | FN | TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.2380 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6799 | 0.000000 | 0 | 0 | 286 | 3,626 |
| **Decision Tree (CART)** | 0.4209 | 0.3741 | 0.9554 | 0.5377 | 0.5818 | 0.7038 | 0.001379 | 107 | 5 | 179 | 3,621 |
| **Random Forest** | 0.4617 | 0.3671 | 1.0000 | 0.5371 | 0.5913 | 0.7014 | 0.000000 | 105 | 0 | 181 | 3,626 |
| **XGBoost** | 0.4322 | 0.3706 | 0.9298 | 0.5300 | 0.5702 | 0.6847 | 0.002206 | 106 | 8 | 180 | 3,618 |
| **LightGBM** | 0.4610 | 0.3671 | 1.0000 | 0.5371 | 0.5913 | 0.6881 | 0.000000 | 105 | 0 | 181 | 3,626 |

## 3. Test Set Performance: Validation-Tuned Decision Threshold (Frozen θ*val)

| Classifier | Optimal θ*val | Average Precision (AP) | Recall | Precision | F1-Score | MCC | ROC-AUC | FPR | TP | FP | FN | TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.1500 | 0.2380 | 0.4056 | 0.2915 | 0.3392 | 0.2823 | 0.6799 | 0.077772 | 116 | 282 | 170 | 3,344 |
| **Decision Tree (CART)** | 0.6700 | 0.4209 | 0.3741 | 0.9640 | 0.5390 | 0.5848 | 0.7038 | 0.001103 | 107 | 4 | 179 | 3,622 |
| **Random Forest** | 0.2900 | 0.4617 | 0.3671 | 1.0000 | 0.5371 | 0.5913 | 0.7014 | 0.000000 | 105 | 0 | 181 | 3,626 |
| **XGBoost** | 0.2400 | 0.4322 | 0.3706 | 0.8983 | 0.5248 | 0.5591 | 0.6847 | 0.003309 | 106 | 12 | 180 | 3,614 |
| **LightGBM** | 0.3400 | 0.4610 | 0.3671 | 1.0000 | 0.5371 | 0.5913 | 0.6881 | 0.000000 | 105 | 0 | 181 | 3,626 |

## 4. Operational Latency and Computational Cost

| Classifier | Training Duration (s) | Test Inference Latency (ms / 1,000 tx) |
| :--- | :---: | :---: |
| **Logistic Regression** | 0.070 s | 1.56 ms |
| **Decision Tree (CART)** | 0.079 s | 1.60 ms |
| **Random Forest** | 0.291 s | 17.47 ms |
| **XGBoost** | 0.437 s | 6.92 ms |
| **LightGBM** | 0.576 s | 6.41 ms |

## 5. Methodological Observations

- **Impact of Extreme Skewness:** Without class weighting or resampling, models with uncalibrated probabilities frequently optimize for global accuracy, depressing Recall at the default threshold θ = 0.50.
- **Value of Validation-Tuned Thresholds:** Tuning θ strictly on validation data allows models to navigate the precision-recall trade-off without risking test-set data leakage.
- **Next Comparative Stage:** These original baseline scores establish the formal empirical benchmark against which Class Weighting, SMOTE, and Controlled Undersampling will be compared.
