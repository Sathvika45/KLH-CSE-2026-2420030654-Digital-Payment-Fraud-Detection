# Model Interpretability Report

## Method
Random Forest impurity-based feature importance was used to identify the features contributing most to the model's predictions.

## Top 15 Important Features

| feature      |   importance |
|:-------------|-------------:|
| numeric__V14 |    0.156312  |
| numeric__V17 |    0.148245  |
| numeric__V12 |    0.1307    |
| numeric__V10 |    0.0869056 |
| numeric__V16 |    0.0837    |
| numeric__V11 |    0.0506189 |
| numeric__V9  |    0.0372433 |
| numeric__V18 |    0.0324688 |
| numeric__V7  |    0.030638  |
| numeric__V4  |    0.0258565 |
| numeric__V26 |    0.0200159 |
| numeric__V21 |    0.0184222 |
| numeric__V3  |    0.015839  |
| numeric__V6  |    0.0146392 |
| numeric__V8  |    0.0133829 |

## SHAP Analysis

TreeSHAP was used to explain the Random Forest model's fraud-class predictions. All fraud cases and up to 200 randomly selected legitimate test cases were included.

### Top 15 SHAP Features

| feature      |   mean_absolute_shap |
|:-------------|---------------------:|
| numeric__V14 |           0.0532568  |
| numeric__V17 |           0.0444933  |
| numeric__V12 |           0.0387771  |
| numeric__V10 |           0.0224336  |
| numeric__V16 |           0.0182915  |
| numeric__V11 |           0.0111306  |
| numeric__V9  |           0.00779691 |
| numeric__V18 |           0.00700021 |
| numeric__V4  |           0.00694782 |
| numeric__V7  |           0.00515147 |
| numeric__V3  |           0.00379535 |
| numeric__V26 |           0.00365596 |
| numeric__V21 |           0.0026115  |
| numeric__V27 |           0.00250494 |
| numeric__V6  |           0.00240627 |

The SHAP summary plot shows the contribution of individual features to the model output. SHAP values describe model behaviour and do not establish causation.


## Limitation
Feature importance indicates how the model uses features. It does not establish causation. Correlated features can affect importance values.
