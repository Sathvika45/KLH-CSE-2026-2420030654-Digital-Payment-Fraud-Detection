# Proposed Hybrid Model Evaluation

Architecture: Random Forest + LightGBM base learners with Logistic Regression as the stacking meta-classifier.

## Experimental setup

- Dataset: PaySim Mobile Money Simulation
- Loaded rows: 100,000
- Prepared rows: 100,000
- Target column: isFraud
- Split: Stratified 60/20/20
- Random state: 42
- Stacking: 3-fold cross-validation
- Threshold selection: Validation set only
- Final evaluation: Untouched test set

## Test results

| dataset   | dataset_name                   | evaluation                 |   threshold |   precision |   recall |       f1 |   pr_auc |   roc_auc |      mcc |         fpr |   true_positives |   false_positives |   true_negatives |   false_negatives |   training_time_sec |   inference_ms_per_1000 |   sample_size |   prepared_rows |   train_rows |   validation_rows |   test_rows |
|:----------|:-------------------------------|:---------------------------|------------:|------------:|---------:|---------:|---------:|----------:|---------:|------------:|-----------------:|------------------:|-----------------:|------------------:|--------------------:|------------------------:|--------------:|----------------:|-------------:|------------------:|------------:|
| paysim    | PaySim Mobile Money Simulation | default_threshold          |    0.5      |    0.298701 | 0.884615 | 0.446602 | 0.832314 |  0.998005 | 0.513123 | 0.00270351  |               23 |                54 |            19920 |                 3 |             3.22198 |                 4.39777 |        100000 |          100000 |        60000 |             20000 |       20000 |
| paysim    | PaySim Mobile Money Simulation | validation_tuned_threshold |    0.999672 |    0.944444 | 0.653846 | 0.772727 | 0.832314 |  0.998005 | 0.785608 | 5.00651e-05 |               17 |                 1 |            19973 |                 9 |             3.22198 |                 4.39777 |        100000 |          100000 |        60000 |             20000 |       20000 |

## Methodological note

The test set was not used for threshold selection. The tuned threshold was selected using validation predictions and then frozen for test evaluation.
