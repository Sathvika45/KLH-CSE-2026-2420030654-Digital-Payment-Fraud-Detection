# Dataset Validation & Class-Imbalance Report: PaySim Mobile Money Simulation

**Generated:** 2026-10-30 16:20:28  
**Target Column:** `isFraud`  
**Integrity Status:** PASSED (Valid)  
**Data Scope:** Full Dataset (Zero Truncation)  

## 1. Source Files

- `paysim.csv` (470.67 MB) - `C:\Users\sathvika2006\OneDrive\Documents\3-1\DAV\Fraud Detection\data\raw\paysim.csv`

## 2. Structural & Integrity Diagnostics

| Diagnostic Metric | Value |
| :--- | :--- |
| Total Records Evaluated | 6,362,620 |
| Total Attributes (Columns) | 11 |
| Duplicate Rows Detected | 0 (0.00%) |
| Missing Target Values | 0 |
| Total Missing Feature Values | 0 |
| Columns with Missing Values | 0 |

## 3. Class Imbalance Quantification

| Class Name | Label | Transaction Count | Percentage |
| :--- | :--- | :--- | :--- |
| Legitimate (Majority) | `0` | 6,354,407 | 99.8709% |
| Fraudulent (Minority) | `1` | 8,213 | 0.1291% |
| **Total** | - | **6,362,620** | **100.00%** |

- **Imbalance Ratio:** 773.70 : 1 (legitimate transactions per fraud transaction)
- **Class Balance Assessment:** Severe Class Imbalance Detected

> [!IMPORTANT]
> Resampling (such as SMOTE or controlled undersampling) must **only** be fitted on training folds.
> Validation and test partitions must retain their natural class distribution to avoid data leakage.
