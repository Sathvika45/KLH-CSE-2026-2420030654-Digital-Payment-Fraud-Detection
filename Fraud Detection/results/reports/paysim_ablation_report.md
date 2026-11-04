# PaySim Hybrid Ablation Study

## Experimental setup

- Dataset: PaySim
- Sample size: 100,000 rows
- Split: Stratified 60/20/20
- Random state: 42
- Threshold selection: Validation set only
- Final evaluation: Untouched test set

## Compared models

1. Random Forest
2. LightGBM
3. Random Forest + LightGBM stacking

## Results

| model                  |   training_time_sec |   inference_ms_per_1000 |   validation_threshold |   test_default_f1 |   test_tuned_f1 |   test_default_pr_auc |   test_tuned_pr_auc |   test_default_roc_auc |   test_tuned_roc_auc |   test_tuned_precision |   test_tuned_recall |   test_tuned_mcc |   test_tuned_fpr |   test_tuned_tp |   test_tuned_fp |   test_tuned_tn |   test_tuned_fn |
|:-----------------------|--------------------:|------------------------:|-----------------------:|------------------:|----------------:|----------------------:|--------------------:|-----------------------:|---------------------:|-----------------------:|--------------------:|-----------------:|-----------------:|----------------:|----------------:|----------------:|----------------:|
| Random Forest          |            1.51573  |                12.4973  |               0.126723 |          0.790698 |        0.754717 |              0.819349 |            0.819349 |               0.998142 |             0.998142 |               0.740741 |            0.769231 |         0.754526 |      0.000350456 |              20 |               7 |           19967 |               6 |
| LightGBM               |            0.843236 |                 2.10168 |               1        |          0.145455 |        0.139535 |              0.031577 |            0.031577 |               0.506661 |             0.506661 |               0.176471 |            0.115385 |         0.141797 |      0.000700911 |               3 |              14 |           19960 |              23 |
| RF + LightGBM Stacking |            6.59024  |                 6.91457 |               0.999848 |          0.389381 |        0.8      |              0.840438 |            0.840438 |               0.997841 |             0.997841 |               0.947368 |            0.692308 |         0.809658 |      5.00651e-05 |              18 |               1 |           19973 |               8 |

## Interpretation note

The ablation study compares individual base learners against the stacking model using the same data split. The results indicate whether the combination provides an improvement over its individual components.
