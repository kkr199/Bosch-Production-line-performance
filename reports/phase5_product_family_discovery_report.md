# Phase 5: Product Family Discovery

## Source Data

- Station presence matrix was built from raw `train_date.csv` and `test_date.csv`.
- The target `Response` was loaded from raw `train_numeric.csv`.
- Family-level models used `data/processed/phase4_train_engineered_features.csv` because Phase 4 created the timing and path features required for modeling.

## Dataset Size

- Train products clustered: 1,183,747
- Test products clustered: 1,183,748
- Product families selected for final labels: 8

## Clustering Diagnostics

| method       |   cluster_count |   noise_count |   rows_labeled |   unique_path_count |   sample_silhouette |
|:-------------|----------------:|--------------:|---------------:|--------------------:|--------------------:|
| KMeans       |               8 |             0 |        1183747 |                7927 |            0.190962 |
| DBSCAN       |               7 |           855 |        1183747 |                7927 |            0.231218 |
| Hierarchical |               8 |             0 |        1183747 |                7927 |            0.25825  |

## Highest-Risk Product Family

Family 4 has the highest observed failure rate at 0.737% across 128,104 products (1.27x the overall train failure rate).

## Family Model Training

- Trained family-specific models: 8
- Skipped families: 0
- Model type: `HistGradientBoostingClassifier` with balanced sample weights.
- Validation metric files include ROC AUC and average precision for each family.

## Output Files

- `data/processed/phase5_train_station_presence_matrix.csv`
- `data/processed/phase5_test_station_presence_matrix.csv`
- `data/processed/phase5_train_product_families.csv`
- `data/processed/phase5_test_product_families.csv`
- `reports/phase5_unique_path_cluster_map.csv`
- `reports/phase5_cluster_diagnostics.csv`
- `reports/phase5_product_family_profiles.csv`
- `reports/phase5_product_family_failure_rates.csv`
- `reports/phase5_family_model_metrics.csv`
- `models/phase5_family_*_hgb.joblib`