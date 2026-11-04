# Benchmark Report: Original Imbalanced Baseline
**Dataset:** PaySim Mobile Money Simulation  
**Strategy:** Original Class Distribution (No Resampling, No Class Weighting)  
**Generated:** 2026-10-30 19:08:21  

## 1. Experimental Design & Leakage-Free Protocol

- **Data Integrity:** Raw CSV is completely untouched. Deduplication executed in memory.
- **Raw Record Count:** 100,000 (129 fraud, 99,871 legitimate).
- **Deduplicated Record Count:** 100,000 (0 exact full-row duplicates removed across all columns).
- **Stratified 60/20/20 Partitioning (Fixed Seed 42):**
  - **Training Split:** 60,000 samples (77 fraud, 0.128%).
  - **Validation Split:** 20,000 samples (26 fraud, 0.130%).
  - **Test Split:** 20,000 samples (26 fraud, 0.130%).
- **Zero Leakage Enforcement:** Numeric imputation, RobustScaler, and categorical imputation/OneHotEncoder are fitted strictly on training data. Resamplers omitted. Decision thresholds optimized strictly on validation data and applied frozen to test data.

## 2. Test Set Performance: Default Decision Threshold (θ = 0.50)

| Classifier | Average Precision (AP) | Recall | Precision | F1-Score | MCC | ROC-AUC | FPR | TP | FP | FN | TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.6979 | 0.3462 | 1.0000 | 0.5143 | 0.5881 | 0.9986 | 0.000000 | 9 | 0 | 17 | 19,974 |
| **Decision Tree (CART)** | 0.6260 | 0.6539 | 0.8947 | 0.7556 | 0.7646 | 0.9906 | 0.000100 | 17 | 2 | 9 | 19,972 |
| **Random Forest** | 0.8194 | 0.6539 | 1.0000 | 0.7907 | 0.8084 | 0.9981 | 0.000000 | 17 | 0 | 9 | 19,974 |
| **XGBoost** | 0.8615 | 0.6923 | 1.0000 | 0.8182 | 0.8319 | 0.9994 | 0.000000 | 18 | 0 | 8 | 19,974 |
| **LightGBM** | 0.0316 | 0.1538 | 0.1379 | 0.1454 | 0.1445 | 0.5067 | 0.001252 | 4 | 25 | 22 | 19,949 |

## 3. Test Set Performance: Validation-Tuned Decision Threshold (Frozen θ*val)

| Classifier | Optimal θ*val | Average Precision (AP) | Recall | Precision | F1-Score | MCC | ROC-AUC | FPR | TP | FP | FN | TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.0800 | 0.6979 | 0.6539 | 0.5667 | 0.6071 | 0.6081 | 0.9986 | 0.000651 | 17 | 13 | 9 | 19,961 |
| **Decision Tree (CART)** | 0.0400 | 0.6260 | 0.6539 | 0.8947 | 0.7556 | 0.7646 | 0.9906 | 0.000100 | 17 | 2 | 9 | 19,972 |
| **Random Forest** | 0.1200 | 0.8194 | 0.7692 | 0.7143 | 0.7407 | 0.7409 | 0.9981 | 0.000401 | 20 | 8 | 6 | 19,966 |
| **XGBoost** | 0.2800 | 0.8615 | 0.6923 | 0.9474 | 0.8000 | 0.8097 | 0.9994 | 0.000050 | 18 | 1 | 8 | 19,973 |
| **LightGBM** | 0.9900 | 0.0316 | 0.1154 | 0.1500 | 0.1304 | 0.1306 | 0.5067 | 0.000851 | 3 | 17 | 23 | 19,957 |

## 4. Operational Latency and Computational Cost

| Classifier | Training Duration (s) | Test Inference Latency (ms / 1,000 tx) |
| :--- | :---: | :---: |
| **Logistic Regression** | 0.108 s | 0.55 ms |
| **Decision Tree (CART)** | 0.286 s | 0.56 ms |
| **Random Forest** | 0.919 s | 2.13 ms |
| **XGBoost** | 0.249 s | 0.79 ms |
| **LightGBM** | 0.316 s | 0.95 ms |

## 5. Methodological Observations

- **Impact of Extreme Skewness:** Without class weighting or resampling, models with uncalibrated probabilities frequently optimize for global accuracy, depressing Recall at the default threshold θ = 0.50.
- **Value of Validation-Tuned Thresholds:** Tuning θ strictly on validation data allows models to navigate the precision-recall trade-off without risking test-set data leakage.
- **Next Comparative Stage:** These original baseline scores establish the formal empirical benchmark against which Class Weighting, SMOTE, and Controlled Undersampling will be compared.
