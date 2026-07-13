# Phase 4 Feature Engineering Report

All Phase 4 features are engineered from the raw date CSV files. `Response` is joined from `train_numeric.csv` for the train output only.

## Outputs Created

- `data/processed/phase4_train_engineered_features.csv`
- `data/processed/phase4_test_engineered_features.csv`
- `reports/phase4_feature_dictionary.csv`
- `reports/phase4_engineered_feature_summary.csv`

## Engineered Output Summary

| split   |    rows |   columns |   file_size_mb | has_response   |
|:--------|--------:|----------:|---------------:|:---------------|
| train   | 1183747 |        49 |         281.29 | True           |
| test    | 1183748 |        48 |         279.02 | False          |

## Feature Dictionary

| feature                        | description                                                                  |
|:-------------------------------|:-----------------------------------------------------------------------------|
| start_time                     | Earliest observed raw date value for a product.                              |
| end_time                       | Latest observed raw date value for a product.                                |
| cycle_time                     | Elapsed time between end_time and start_time.                                |
| processing_duration            | Sum of within-station active durations.                                      |
| waiting_time                   | Sum of positive gaps between consecutive station end/start times.            |
| mean_waiting_time              | Mean positive inter-station waiting gap.                                     |
| max_waiting_time               | Largest positive inter-station waiting gap.                                  |
| wait_event_count               | Number of positive waiting gaps between stations.                            |
| delay_ratio                    | waiting_time divided by cycle_time when cycle_time is positive.              |
| station_count                  | Number of stations with at least one observed date value.                    |
| line_count                     | Number of production lines with at least one observed date value.            |
| station_span                   | Distance from first observed station to last observed station.               |
| path_density                   | station_count divided by station_span.                                       |
| line_switch_count              | Number of production-line changes along the observed station path.           |
| path_complexity_score          | Composite path score using station_count, line_count, switches, and density. |
| date_completeness_pct          | Share of raw date features observed for a product.                           |
| line_{n}_present               | Whether production line n has at least one observed date value.              |
| line_{n}_start_time            | Earliest observed raw date value within production line n.                   |
| line_{n}_end_time              | Latest observed raw date value within production line n.                     |
| line_{n}_processing_duration   | Elapsed time between line_{n}_end_time and line_{n}_start_time.              |
| line_{n}_observed_date_values  | Observed raw date values within production line n.                           |
| line_{n}_station_count         | Stations visited within production line n.                                   |
| line_{n}_date_completeness_pct | Share of line n date features observed.                                      |
