# Dataset Information

This project uses three datasets for digital payment fraud detection.

## 1. Credit Card Fraud Detection
- File name: `creditcard.csv`
- Download: [Credit Card Fraud Detection Dataset – Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- This dataset contains credit card transactions labelled as fraudulent or legitimate.
- The original file is approximately 143.8 MB and is not included in this GitHub repository.

## 2. E-commerce Fraud Detection
- File name: `Fraud_Data.csv`
- Supporting file: `IpAddress_to_Country.csv`
- These files are included in the repository, subject to their applicable dataset terms.

## 3. PaySim
- File name: `paysim.csv`
- Download: [PaySim Mobile Money Dataset – Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)
- This dataset contains simulated mobile money transactions.
- The original file is approximately 470.6 MB and is not included in this GitHub repository.

## Folder Structure

Place the downloaded datasets inside `data/raw/` using the exact filenames:

```text
data/
├── raw/
│   ├── creditcard.csv
│   ├── Fraud_Data.csv
│   ├── IpAddress_to_Country.csv
│   └── paysim.csv
└── processed/
```

## Important

The two large datasets (`creditcard.csv` and `paysim.csv`) are excluded from this repository because they exceed GitHub's regular file-size limit.

Download them from the provided sources and place them in `data/raw/` before running the project.

The original datasets are retained separately by the project owner.