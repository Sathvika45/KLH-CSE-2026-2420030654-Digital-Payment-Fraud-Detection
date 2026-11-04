"""
Visualisation utilities for comparative model evaluation:
- Precision-Recall curves and ROC curves
- Normalised confusion matrix heatmaps
- Metric trade-off spider/radar charts and bar comparisons
- Class distribution and imbalance profiles
- SHAP feature importance and beeswarm summary plots
"""

from src.visualization.plots import plot_class_distribution

__all__ = ["plot_class_distribution"]
