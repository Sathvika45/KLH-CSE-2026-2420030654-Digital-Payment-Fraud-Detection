# Dataset Validation & Class-Imbalance Report: Credit Card Fraud Detection (European Cardholders)

**Generated:** 2026-10-30 16:19:40  
**Target Column:** `Class`  
**Integrity Status:** PASSED (Valid)  
**Data Scope:** Full Dataset (Zero Truncation)  

## 1. Source Files

- `creditcard.csv` (143.84 MB) - `C:\Users\sathvika2006\OneDrive\Documents\3-1\DAV\Fraud Detection\data\raw\creditcard.csv`

## 2. Structural & Integrity Diagnostics

| Diagnostic Metric | Value |
| :--- | :--- |
| Total Records Evaluated | 284,807 |
| Total Attributes (Columns) | 31 |
| Duplicate Rows Detected | 1,081 (0.38%) |
| Missing Target Values | 0 |
| Total Missing Feature Values | 0 |
| Columns with Missing Values | 0 |

## 3. Class Imbalance Quantification

| Class Name | Label | Transaction Count | Percentage |
| :--- | :--- | :--- | :--- |
| Legitimate (Majority) | `0` | 284,315 | 99.8273% |
| Fraudulent (Minority) | `1` | 492 | 0.1727% |
| **Total** | - | **284,807** | **100.00%** |

- **Imbalance Ratio:** 577.88 : 1 (legitimate transactions per fraud transaction)
- **Class Balance Assessment:** Severe Class Imbalance Detected

> [!IMPORTANT]
> Resampling (such as SMOTE or controlled undersampling) must **only** be fitted on training folds.
> Validation and test partitions must retain their natural class distribution to avoid data leakage.

## 4. Warnings & Observations

- [WARNING] Dataset contains 1081 duplicate rows (0.38%).
