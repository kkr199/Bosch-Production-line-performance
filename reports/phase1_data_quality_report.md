# Phase 1 Data Quality Report

`sample_submission.csv` is intentionally excluded because it is only a Kaggle submission-format reference file, not a modeling dataset.

## Dataset Inventory

| dataset           | path                                                                                            |   file_size_mb |    rows |   columns |   feature_columns | id_present   | target_present   |   total_missing_values |   missing_value_pct |
|:------------------|:------------------------------------------------------------------------------------------------|---------------:|--------:|----------:|------------------:|:-------------|:-----------------|-----------------------:|--------------------:|
| train_numeric     | C:\Users\karin\OneDrive\Desktop\Nexturn\Bosch Production Line Performance\train_numeric.csv     |        2040.77 | 1183747 |       970 |               968 | True         | True             |              929125166 |             80.9177 |
| train_categorical | C:\Users\karin\OneDrive\Desktop\Nexturn\Bosch Production Line Performance\train_categorical.csv |        2554.27 | 1183747 |      2141 |              2140 | True         | False            |             2465567643 |             97.284  |
| train_date        | C:\Users\karin\OneDrive\Desktop\Nexturn\Bosch Production Line Performance\train_date.csv        |        2759.33 | 1183747 |      1157 |              1156 | True         | False            |             1125431152 |             82.1725 |
| test_numeric      | C:\Users\karin\OneDrive\Desktop\Nexturn\Bosch Production Line Performance\test_numeric.csv      |        2038.27 | 1183748 |       969 |               968 | True         | False            |              929173660 |             81.0054 |
| test_categorical  | C:\Users\karin\OneDrive\Desktop\Nexturn\Bosch Production Line Performance\test_categorical.csv  |        2554.2  | 1183748 |      2141 |              2140 | True         | False            |             2465604793 |             97.2854 |
| test_date         | C:\Users\karin\OneDrive\Desktop\Nexturn\Bosch Production Line Performance\test_date.csv         |        2759.2  | 1183748 |      1157 |              1156 | True         | False            |             1125501041 |             82.1776 |

## Target Check

- `Response` is expected in `train_numeric.csv` and should be absent from test files.
- Missing values in `Response`: 0

## Top Missing Columns Per Dataset

| dataset           | column       |   missing_values |   missing_pct |
|:------------------|:-------------|-----------------:|--------------:|
| test_categorical  | L0_S15_F396  |          1183748 |      100      |
| test_categorical  | L0_S15_F399  |          1183748 |      100      |
| test_categorical  | L0_S15_F402  |          1183748 |      100      |
| test_categorical  | L0_S15_F405  |          1183748 |      100      |
| test_categorical  | L0_S15_F408  |          1183748 |      100      |
| test_categorical  | L0_S15_F411  |          1183748 |      100      |
| test_categorical  | L0_S15_F414  |          1183748 |      100      |
| test_categorical  | L0_S15_F417  |          1183748 |      100      |
| test_categorical  | L0_S15_F420  |          1183748 |      100      |
| test_categorical  | L1_S24_F1157 |          1183748 |      100      |
| test_date         | L3_S46_D4135 |          1183748 |      100      |
| test_date         | L1_S24_D1158 |          1183747 |       99.9999 |
| test_date         | L3_S42_D4045 |          1183731 |       99.9986 |
| test_date         | L3_S42_D4049 |          1183731 |       99.9986 |
| test_date         | L3_S42_D4053 |          1183731 |       99.9986 |
| test_date         | L3_S42_D4057 |          1183731 |       99.9986 |
| test_date         | L3_S42_D4029 |          1183728 |       99.9983 |
| test_date         | L3_S42_D4033 |          1183728 |       99.9983 |
| test_date         | L3_S42_D4037 |          1183728 |       99.9983 |
| test_date         | L3_S42_D4041 |          1183728 |       99.9983 |
| test_numeric      | L1_S25_F2181 |          1182552 |       99.899  |
| test_numeric      | L1_S25_F2184 |          1182552 |       99.899  |
| test_numeric      | L1_S25_F2187 |          1182552 |       99.899  |
| test_numeric      | L1_S25_F2190 |          1182552 |       99.899  |
| test_numeric      | L1_S25_F2193 |          1182552 |       99.899  |
| test_numeric      | L1_S25_F2196 |          1182552 |       99.899  |
| test_numeric      | L1_S25_F2199 |          1182552 |       99.899  |
| test_numeric      | L1_S25_F2202 |          1182552 |       99.899  |
| test_numeric      | L1_S25_F2672 |          1181719 |       99.8286 |
| test_numeric      | L1_S25_F2677 |          1181719 |       99.8286 |
| train_categorical | L0_S3_F69    |          1183747 |      100      |
| train_categorical | L0_S3_F71    |          1183747 |      100      |
| train_categorical | L0_S3_F73    |          1183747 |      100      |
| train_categorical | L0_S3_F75    |          1183747 |      100      |
| train_categorical | L0_S3_F77    |          1183747 |      100      |
| train_categorical | L0_S3_F79    |          1183747 |      100      |
| train_categorical | L0_S3_F81    |          1183747 |      100      |
| train_categorical | L0_S3_F83    |          1183747 |      100      |
| train_categorical | L0_S3_F85    |          1183747 |      100      |
| train_categorical | L0_S3_F87    |          1183747 |      100      |
| train_date        | L1_S24_D1158 |          1183746 |       99.9999 |
| train_date        | L3_S46_D4135 |          1183746 |       99.9999 |
| train_date        | L3_S42_D4045 |          1183739 |       99.9993 |
| train_date        | L3_S42_D4049 |          1183739 |       99.9993 |
| train_date        | L3_S42_D4053 |          1183739 |       99.9993 |
| train_date        | L3_S42_D4057 |          1183739 |       99.9993 |
| train_date        | L3_S42_D4029 |          1183732 |       99.9987 |
| train_date        | L3_S42_D4033 |          1183732 |       99.9987 |
| train_date        | L3_S42_D4037 |          1183732 |       99.9987 |
| train_date        | L3_S42_D4041 |          1183732 |       99.9987 |
| train_numeric     | L1_S25_F2181 |          1182504 |       99.895  |
| train_numeric     | L1_S25_F2184 |          1182504 |       99.895  |
| train_numeric     | L1_S25_F2187 |          1182504 |       99.895  |
| train_numeric     | L1_S25_F2190 |          1182504 |       99.895  |
| train_numeric     | L1_S25_F2193 |          1182504 |       99.895  |
| train_numeric     | L1_S25_F2196 |          1182504 |       99.895  |
| train_numeric     | L1_S25_F2199 |          1182504 |       99.895  |
| train_numeric     | L1_S25_F2202 |          1182504 |       99.895  |
| train_numeric     | L1_S25_F2712 |          1181630 |       99.8212 |
| train_numeric     | L1_S25_F2714 |          1181630 |       99.8212 |

## Notes

- Full per-column missing-value counts are saved in `reports/phase1_missing_values_by_column.csv`.
- The original Kaggle CSV files are ignored by git because they are large raw data assets.
- `sample_submission.csv` is ignored for Phase 1 profiling and downstream modeling.
