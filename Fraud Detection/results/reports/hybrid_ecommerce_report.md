# Proposed Hybrid Model Evaluation

Architecture: Random Forest + LightGBM base learners with Logistic Regression as the stacking meta-classifier.

## Experimental setup

- Dataset: Fraud E-commerce
- Loaded rows: 151,112
- Prepared rows: 144,273
- Target column: class
- Split: Stratified 60/20/20
- Random state: 42
- Stacking: 3-fold cross-validation
- Threshold selection: Validation set only
- Final evaluation: Untouched test set

## Test results

| dataset   | dataset_name     | evaluation                 |   threshold |   precision |   recall |       f1 |   pr_auc |   roc_auc |      mcc |       fpr |   true_positives |   false_positives |   true_negatives |   false_negatives |   training_time_sec |   inference_ms_per_1000 |   sample_size |   prepared_rows |   train_rows |   validation_rows |   test_rows |
|:----------|:-----------------|:---------------------------|------------:|------------:|---------:|---------:|---------:|----------:|---------:|----------:|-----------------:|------------------:|-----------------:|------------------:|--------------------:|------------------------:|--------------:|----------------:|-------------:|------------------:|------------:|
| ecommerce | Fraud E-commerce | default_threshold          |    0.5      |    0.186472 | 0.139535 | 0.159624 | 0.175432 |   0.55706 | 0.122923 | 0.0324901 |              204 |               890 |            26503 |              1258 |             5.90383 |                 5.19941 |        151112 |          144273 |        86563 |             28855 |       28855 |
| ecommerce | Fraud E-commerce | validation_tuned_threshold |    0.994572 |    1        | 0.104651 | 0.189474 | 0.175432 |   0.55706 | 0.316035 | 0         |              153 |                 0 |            27393 |              1309 |             5.90383 |                 5.19941 |        151112 |          144273 |        86563 |             28855 |       28855 |

## Methodological note

The test set was not used for threshold selection. The tuned threshold was selected using validation predictions and then frozen for test evaluation.
