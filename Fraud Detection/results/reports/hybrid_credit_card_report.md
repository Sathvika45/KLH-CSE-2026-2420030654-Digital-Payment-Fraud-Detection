# Proposed Hybrid Model Evaluation

Architecture: Random Forest + LightGBM base learners with Logistic Regression as the stacking meta-classifier.

## Experimental setup

- Dataset: Credit Card Fraud
- Split: Stratified 60/20/20
- Random state: 42
- Stacking: 3-fold cross-validation
- Threshold selection: Validation set only
- Final evaluation: Untouched test set

## Test results

| evaluation                 |   threshold |   precision |   recall |       f1 |   pr_auc |   roc_auc |      mcc |         fpr |   true_positives |   false_positives |   true_negatives |   false_negatives |   training_time_sec |   inference_ms_per_1000 |
|:---------------------------|------------:|------------:|---------:|---------:|---------:|----------:|---------:|------------:|-----------------:|------------------:|-----------------:|------------------:|--------------------:|------------------------:|
| default_threshold          |    0.5      |    0.710526 | 0.852632 | 0.77512  | 0.839654 |  0.971404 | 0.777941 | 0.000582514 |               81 |                33 |            56618 |                14 |             120.237 |                 4.77964 |
| validation_tuned_threshold |    0.999957 |    0.974359 | 0.8      | 0.878613 | 0.839654 |  0.971404 | 0.882714 | 3.53039e-05 |               76 |                 2 |            56649 |                19 |             120.237 |                 4.77964 |

## Limitation

This experiment evaluates one proposed stacking architecture on the Credit Card dataset. Its performance must be compared with the individual base learners and other baselines before drawing conclusions about whether stacking improves results.
