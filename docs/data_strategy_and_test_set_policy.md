# Data Strategy and Test-Set Policy

## Source inventory

| Source | Intended use | Refresh | Join key | Known limitation |
| --- | --- | --- | --- | --- |
| `train_numeric.csv` | Labels and numeric sensor/process features | Static Kaggle export | `Id` | Historical competition data; not live factory telemetry |
| `train_categorical.csv` | Station/category presence features | Static Kaggle export | `Id` | High cardinality and sparsity |
| `train_date.csv` | Timing and process-flow features | Static Kaggle export | `Id` | Relative timestamps require point-in-time review |
| `test_*.csv` | Unlabelled batch-scoring input only | Static Kaggle export | `Id` | **Not** a production test set and must not be used for model selection |

## Data limitations and approvals

This repository contains a public Kaggle dataset, not a factory system of
record. It cannot establish live-data freshness, consent, retention, or plant
privacy compliance. Production adoption requires a separate source inventory,
legal/privacy review, data-owner approval, and a documented label-latency
assessment.

## Split and isolation rule

1. Split the labelled population into training, validation, and a physically
   separate, immutable holdout before feature selection or EDA that uses the
   target.
2. Fit feature selection, encoders, imputers, scalers, and threshold selection
   using training folds only. Use validation only for model-selection decisions.
3. Evaluate the selected, frozen pipeline exactly once on the locked holdout.
4. Record split seed, row counts, class counts, data hash, code commit, and
   feature-list version in `reports/` for every release candidate.
5. Keep Kaggle's unlabelled `test_*.csv` separate from this holdout; it is an
   inference input, not evidence of generalisation.

## Leakage controls

- Every candidate feature needs a prediction-time availability check.
- Categorical level ranking and target-informed feature selection must be
  learned only from the training portion.
- Derived timing windows may only use events available before the declared
  scoring point.
- Any new upstream column requires schema validation and a leakage review
  before it is added to the feature registry.
