# Phase 10: Advanced AI

## Executive Summary

Phase 10 adds advanced diagnostic AI around the production-safe Phase 6 model. The best Phase 10 experiment is **Failure trajectory prediction** with validation MCC **0.2419** and PR-AUC **0.1648**. The official production-safe benchmark remains **LightGBM** from Phase 6 with MCC **0.3386**.

These experiments are useful as engineering intelligence layers:

- **Isolation Forest** flags products whose numeric/categorical/date pattern looks unusual compared with normal products.
- **MLP reconstruction anomaly detection** finds products that a normal-pattern reconstruction model cannot reproduce well.
- **Graph message passing** spreads station risk through the production flow graph to estimate route-level risk exposure.
- **Failure trajectory prediction** combines ordered station exposure, timing, waiting, path, and product-family context to predict risk along the manufacturing path.

## Validation Results

| model                                |   threshold |    mcc |   precision |   recall |     f1 |   pr_auc | notes                                                                         |
|:-------------------------------------|------------:|-------:|------------:|---------:|-------:|---------:|:------------------------------------------------------------------------------|
| Phase 6 LightGBM reference           |      0.7851 | 0.3386 |      0.5743 |   0.2134 | 0.3111 |   0.2529 | Production-safe supervised benchmark from Phase 6.                            |
| Failure trajectory prediction        |      0.9897 | 0.2419 |      0.4615 |   0.1395 | 0.2143 |   0.1648 | LightGBM model using timing, path, family, and ordered station-risk exposure. |
| MLP reconstruction anomaly detection |      4.3655 | 0.0923 |      0.2535 |   0.0419 | 0.0719 |   0.0643 | Scikit-learn reconstruction model using 120 selected features.                |
| Graph message-passing risk model     |      0.5287 | 0.0611 |      0.0534 |   0.3017 | 0.0907 |   0.0411 | Logistic model using graph-propagated station risk exposure.                  |
| Isolation Forest anomaly detection   |     -0.1217 | 0.0537 |      0.0445 |   0.4349 | 0.0808 |   0.0395 | Unsupervised model trained on normal products only.                           |

## Top Propagated Station Risks

| station   |   base_station_risk |   one_hop_station_risk |   two_hop_station_risk |   message_passing_station_risk |
|:----------|--------------------:|-----------------------:|-----------------------:|-------------------------------:|
| L1_S24    |              1.0000 |                 0.6813 |                 0.5780 |                         1.0000 |
| L1_S25    |              0.9033 |                 0.6804 |                 0.5739 |                         0.9140 |
| L3_S29    |              0.8184 |                 0.6267 |                 0.6242 |                         0.8261 |
| L3_S30    |              0.7411 |                 0.6241 |                 0.6324 |                         0.7591 |
| L3_S32    |              0.7521 |                 0.6039 |                 0.5649 |                         0.7431 |
| L2_S27    |              0.6820 |                 0.6397 |                 0.5291 |                         0.6902 |
| L2_S26    |              0.6759 |                 0.6310 |                 0.5381 |                         0.6828 |
| L3_S33    |              0.6664 |                 0.6111 |                 0.5612 |                         0.6706 |

## Interpretation For Manufacturing Teams

The anomaly models should be treated as early-warning and triage tools. A high anomaly score means the product path or measurements look different from normal production history; it does not automatically prove a defect cause. The graph and trajectory models are closer to process intelligence: they show whether a product crossed stations that previous phases identified as central, bottlenecked, failure-associated, or downstream of risky stations.

## Recommended Use

1. Keep **Phase 6 LightGBM** as the main production-safe failure classifier.
2. Use **failure trajectory risk** as a second operational score for routing products to extra inspection.
3. Use **graph message-passing station risk** to explain whether a product's path crossed critical process nodes.
4. Use **Isolation Forest** and **MLP reconstruction error** for anomaly monitoring, alerting, and investigation queues.
5. Do not use these methods as causal proof without engineer review, sensor validation, and controlled process evidence.

## Outputs

- Validation score file: `data/processed/phase10_validation_advanced_ai_scores.csv`
- Test preview score file: `data/processed/phase10_test_preview_advanced_ai_scores.csv`
- Model comparison: `reports/phase10_advanced_ai_model_comparison.csv`
- Station message-passing risk: `reports/phase10_station_message_passing_risk.csv`
- Graph/trajectory feature sets: `data/processed/phase10_*_graph_trajectory_features.csv`
- Notebook: `notebooks/phase10_advanced_ai.ipynb`

## Caveats

- PyTorch/TensorFlow are not installed in the project environment, so the autoencoder is implemented as an MLP reconstruction-error model with scikit-learn.
- The graph neural network step is a lightweight graph message-passing experiment, not a full PyTorch Geometric GNN. It is appropriate for a portfolio-grade process graph prototype without adding a heavy deep-learning dependency.
- Test advanced-AI scores are produced for the existing Phase 6 test preview feature file. The validation metrics are the reliable comparison point because the Kaggle test labels are not available.
- The score levels are diagnostic and observational. They should be calibrated with real factory feedback before being used as production decision thresholds.
