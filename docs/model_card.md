# Model Card — Bosch Failure-Risk Classifier

## Intended use

Rank completed or scoreable production records for manual quality review. The
score supports prioritisation; it is not an automated disposition, safety
interlock, or root-cause determination.

## Out of scope

- Real-time line control or safety decisions.
- Causal explanations of failure.
- Generalisation to plants, products, sensors, or time periods not represented
  in the training data.

## Inputs and outputs

The pipeline uses selected numeric, categorical-presence, timing/path, and
product-family features keyed by `Id`. It produces a failure probability and a
binary flag only after a business-approved capacity/cost threshold is selected.

## Evaluation requirements

Release evidence must include PR-AUC, MCC, precision@K, recall@K, calibration,
confidence intervals, error analysis, and comparison with rules-only and
logistic-regression baselines on the locked holdout. The current repository
artifacts are experimental until this evidence is regenerated under the test-set
policy.

## Known limitations

- The source data is an historical Kaggle benchmark, so it is not proof of
  live-factory performance.
- Class imbalance makes accuracy misleading.
- Feature availability and sensor semantics must be verified against the actual
  inference-time process.
- Explanations identify associations, not causes; quality engineers retain the
  final decision.

## Responsible use and ownership

Quality Engineering owns the review workflow and false-negative tolerance. The
ML owner owns data validation, model versioning, monitoring, and retraining.
Any deployment must include a factory-specific fairness/safety assessment where
people are affected by the decision process.
