# Comparative Analysis of Fraud Detection Techniques in Digital Payment Systems

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Project](https://img.shields.io/badge/Project-BTech%20DAV-orange)
![Status](https://img.shields.io/badge/Status-Implementation%20%26%20Evaluation-success)

## 1. Project Overview

Digital payment systems process large volumes of transactions, creating challenges in identifying fraudulent activities. Fraud detection is particularly challenging because legitimate transactions significantly outnumber fraudulent transactions, resulting in highly imbalanced datasets.

This project presents a comparative analysis of machine learning techniques for detecting fraudulent transactions across three digital payment datasets.

The study evaluates individual classification models, class-imbalance handling strategies, and a proposed soft-voting hybrid ensemble using multiple performance and efficiency metrics.

## 2. Research Objectives

1. **Class-Imbalance Mitigation:** Compare original imbalanced data, class weighting, SMOTE (Synthetic Minority Over-sampling Technique), and controlled undersampling.

2. **Baseline Model Benchmarking:** Evaluate the performance of:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - XGBoost
   - LightGBM

3. **Multi-Dataset Evaluation:** Conduct comparative experiments using:
   - Credit Card Fraud Detection
   - E-commerce Fraud Detection
   - PaySim

4. **Proposed Hybrid Model:** Implement a probability-based soft-voting ensemble combining Random Forest, LightGBM, and Logistic Regression. Evaluate its performance against individual classifiers.

5. **Comprehensive Evaluation:** Analyse model performance using PR-AUC, ROC-AUC, Precision, Recall, F1-score, Matthews Correlation Coefficient (MCC), False Positive Rate (FPR), confusion matrices, explainability, execution time, and stability.

## 3. Proposed Methodology

The project follows a structured experimental workflow:

1. Dataset loading and validation.
2. Data preprocessing and feature preparation.
3. Train, validation, and test data splitting.
4. Class-imbalance handling.
5. Baseline model training and evaluation.
6. Proposed hybrid soft-voting model implementation.
7. Comparative performance evaluation.
8. Explainability and feature-importance analysis.
9. Stability and computational-efficiency analysis.
10. Result visualisation and reporting.

### Class-Imbalance Strategies

- Original imbalanced dataset
- Class weighting
- SMOTE
- Controlled undersampling

### Machine Learning Models

| Model | Approach |
|---|---|
| Logistic Regression | Linear classification |
| Decision Tree | Tree-based classification |
| Random Forest | Ensemble learning |
| XGBoost | Gradient boosting |
| LightGBM | Gradient boosting |
| Proposed Hybrid | Random Forest + LightGBM + Logistic Regression |

## 4. Dataset Information

The project uses three datasets:

| Dataset | Description |
|---|---|
| Credit Card Fraud Detection | Credit card transaction records with fraudulent and legitimate labels |
| E-commerce Fraud Detection | E-commerce transactions and supporting IP-to-country information |
| PaySim | Simulated mobile money transactions |

Dataset download instructions, filenames, and folder requirements are provided in [`data/README.md`](data/README.md).

Large datasets are excluded from this repository because of GitHub file-size restrictions. They must be downloaded separately and placed in the `data/raw/` directory.

## 5. Evaluation Metrics

The project evaluates models using:

- **PR-AUC:** Primary metric for evaluating performance under class imbalance.
- **ROC-AUC:** Measures discrimination between classes.
- **Precision:** Measures the proportion of predicted fraud cases that are actually fraudulent.
- **Recall:** Measures the proportion of actual fraudulent transactions detected.
- **F1-score:** Harmonic mean of precision and recall.
- **MCC:** Evaluates classification quality across all confusion-matrix categories.
- **FPR:** Measures the proportion of legitimate transactions incorrectly classified as fraud.
- **Confusion Matrix:** Summarises classification outcomes.
- **SHAP:** Supports model explainability.
- **Execution Time:** Evaluates training and inference efficiency.
- **Stability Analysis:** Examines consistency across experimental runs or folds.

## 6. Project Structure

```text
Fraud-Detection-GitHub/
│
├── README.md
├── .gitignore
├── requirements.txt
├── main.py
├── dashboard.py
│
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
│
├── docs/
│   └── objectives_and_methodology.md
│
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── models/
│   ├── evaluation/
│   └── visualization/
│
└── results/
    ├── figures/
    ├── models/
    └── reports/
```

## 7. Implementation and Experimental Outputs

The repository includes implementation modules for:

- Dataset loading, validation, and analysis.
- Data preprocessing and imbalance handling.
- Baseline classification models.
- Hybrid soft-voting model experiments.
- Hybrid ablation and PaySim-focused experiments.
- Performance evaluation and interpretability.
- Stability analysis.
- Result visualisation.

### Experimental Outputs

The `results/` directory contains:

- Generated visualisations and feature-importance figures.
- Saved model artefacts in Joblib format.
- Model comparison reports.
- Evaluation metric CSV files.
- Hybrid model experiment reports.
- Imbalance comparison results.
- Interpretability and stability reports.

These outputs support the comparative analysis of the implemented techniques.

## 8. Installation and Execution

### Prerequisites

- Python 3.10 or later
- Visual Studio Code or another Python-compatible IDE
- Git (for repository operations)

### Step 1: Clone the Repository

```powershell
git clone https://github.com/Sathvika45/KLH-CSE-2026-2420030654-Digital-Payment-Fraud-Detection.git
cd Fraud-Detection-GitHub
```

### Step 2: Create a Virtual Environment

```powershell
python -m venv .venv
```

### Step 3: Activate the Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

Then activate the environment again.

### Step 4: Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Download and Place Datasets

Follow the instructions in `data/README.md` and place the required datasets in `data/raw/` using the specified filenames.

###