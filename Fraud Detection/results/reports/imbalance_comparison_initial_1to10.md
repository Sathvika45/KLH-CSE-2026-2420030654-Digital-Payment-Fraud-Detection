# Imbalance Handling Comparison

Dataset: Credit Card Fraud Detection.

The cleaned dataset was split into stratified 60% training, 20% validation, and 20% test partitions using random seed 42.

Duplicate removal was performed before splitting. Resampling and model fitting use training data only. Validation data selects the decision threshold; the test set remains untouched until evaluation.

The four strategies are original data, class-weighted Random Forest, SMOTE with a 1:10 minority-to-majority target, and random undersampling with a 1:10 minority-to-majority target.

The metric named `pr_auc` by the project's evaluator is calculated using scikit-learn Average Precision (AP); it is reported here as AP, not as trapezoidal PR-AUC.

## Results

| strategy              |   train_time_sec |   frozen_validation_threshold |   default_threshold |   default_total_samples |   default_precision |   default_recall |   default_f1 |   default_pr_auc |   default_roc_auc |   default_mcc |   default_false_positive_rate |   tuned_threshold |   tuned_total_samples |   tuned_precision |   tuned_recall |   tuned_f1 |   tuned_pr_auc |   tuned_roc_auc |   tuned_mcc |   tuned_false_positive_rate |
|:----------------------|-----------------:|------------------------------:|--------------------:|------------------------:|--------------------:|-----------------:|-------------:|-----------------:|------------------:|--------------:|------------------------------:|------------------:|----------------------:|------------------:|---------------:|-----------:|---------------:|----------------:|------------:|----------------------------:|
| original              |           50.776 |                        0.3765 |                 0.5 |                   56746 |             0.97368 |          0.77895 |      0.8655  |          0.8443  |           0.97776 |       0.8707  |                      3.5e-05  |            0.3765 |                 56746 |           0.94937 |        0.78947 |    0.86207 |        0.8443  |         0.97776 |     0.86554 |                    7.1e-05  |
| class_weighted        |           15.377 |                        0.3997 |                 0.5 |                   56746 |             0.87952 |          0.76842 |      0.82022 |          0.82912 |           0.97019 |       0.82182 |                      0.000177 |            0.3997 |                 56746 |           0.86047 |        0.77895 |    0.81768 |        0.82912 |         0.97019 |     0.8184  |                    0.000212 |
| smote_1_to_10         |           28.012 |                        0.5888 |                 0.5 |                   56746 |             0.89412 |          0.8     |      0.84444 |          0.86177 |           0.98461 |       0.84551 |                      0.000159 |            0.5888 |                 56746 |           0.96104 |        0.77895 |    0.86047 |        0.86177 |         0.98461 |     0.86502 |                    5.3e-05  |
| undersampling_1_to_10 |            2.373 |                        0.8029 |                 0.5 |                   56746 |             0.59854 |          0.86316 |      0.7069  |          0.82077 |           0.97457 |       0.71823 |                      0.000971 |            0.8029 |                 56746 |           0.83871 |        0.82105 |    0.82979 |        0.82077 |         0.97457 |     0.82955 |                    0.000265 |

## Interpretation notes

- Compare precision, recall, F1, AP, MCC, and false-positive rate together.
- Threshold-tuned results use thresholds selected from validation data.
- Test metrics may differ from validation metrics.
- Resampling strategies are compared on the same untouched test set.
