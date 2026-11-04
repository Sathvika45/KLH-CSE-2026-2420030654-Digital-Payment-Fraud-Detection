# Dataset Validation & Class-Imbalance Report: Fraud E-commerce

**Generated:** 2026-10-30 16:59:05  
**Target Column:** `class`  
**Integrity Status:** PASSED (Valid)  
**Data Scope:** Full Dataset (Zero Truncation)  

## 1. Source Files

- `Fraud_Data.csv` (15.33 MB) - `C:\Users\sathvika2006\OneDrive\Documents\3-1\DAV\Fraud Detection\data\raw\Fraud_Data.csv`
- `IpAddress_to_Country.csv` (4.67 MB) - `C:\Users\sathvika2006\OneDrive\Documents\3-1\DAV\Fraud Detection\data\raw\IpAddress_to_Country.csv`

## 2. Structural & Integrity Diagnostics

| Diagnostic Metric | Value |
| :--- | :--- |
| Total Records Evaluated | 151,112 |
| Total Attributes (Columns) | 12 |
| Duplicate Rows Detected | 0 (0.00%) |
| Missing Target Values | 0 |
| Total Missing Feature Values | 0 |
| Columns with Missing Values | 0 |

## 3. Class Imbalance Quantification

| Class Name | Label | Transaction Count | Percentage |
| :--- | :--- | :--- | :--- |
| Legitimate (Majority) | `0` | 136,961 | 90.6354% |
| Fraudulent (Minority) | `1` | 14,151 | 9.3646% |
| **Total** | - | **151,112** | **100.00%** |

- **Imbalance Ratio:** 9.68 : 1 (legitimate transactions per fraud transaction)
- **Class Balance Assessment:** Severe Class Imbalance Detected

> [!IMPORTANT]
> Resampling (such as SMOTE or controlled undersampling) must **only** be fitted on training folds.
> Validation and test partitions must retain their natural class distribution to avoid data leakage.
