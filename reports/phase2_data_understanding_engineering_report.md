# Phase 2 Data Understanding and Engineering Report

## Outputs Created

- `reports/phase2_feature_metadata.csv`
- `reports/phase2_station_metadata.csv`
- `reports/phase2_line_metadata.csv`
- `reports/phase2_feature_completeness_metrics.csv`
- `reports/phase2_station_completeness_metrics.csv`
- `data/processed/manufacturing_flow_train.parquet`
- `data/processed/manufacturing_flow_test.parquet`

## Feature Metadata Summary

- Parsed feature columns: 4,264
- Production lines: 4
- Stations: 52

## Line Metadata

|   line |   station_count |   numeric_feature_count |   categorical_feature_count |   date_feature_count |   total_feature_count |
|-------:|----------------:|------------------------:|----------------------------:|---------------------:|----------------------:|
|      0 |              24 |                     168 |                         323 |                  184 |                   675 |
|      1 |               2 |                     513 |                        1227 |                  621 |                  2361 |
|      2 |               3 |                      42 |                         159 |                   78 |                   279 |
|      3 |              23 |                     245 |                         431 |                  273 |                   949 |

## Sparsest Station/Data-Type Groups

| split   | data_type   |   line |   station | station_key   |   feature_count |    rows |   missing_values |   observed_values |   possible_values |   completeness_pct |   missing_pct |
|:--------|:------------|-------:|----------:|:--------------|----------------:|--------:|-----------------:|------------------:|------------------:|-------------------:|--------------:|
| test    | categorical |      3 |        36 | L3_S36        |               8 | 1183748 |          9469984 |                 0 |           9469984 |             0      |      100      |
| test    | categorical |      3 |        39 | L3_S39        |               8 | 1183748 |          9469984 |                 0 |           9469984 |             0      |      100      |
| test    | categorical |      3 |        46 | L3_S46        |               3 | 1183748 |          3551244 |                 0 |           3551244 |             0      |      100      |
| train   | categorical |      3 |        46 | L3_S46        |               3 | 1183747 |          3551240 |                 1 |           3551241 |             0      |      100      |
| train   | categorical |      0 |        23 | L0_S23        |              30 | 1183747 |         35512410 |                 0 |          35512410 |             0      |      100      |
| train   | categorical |      0 |        18 | L0_S18        |              10 | 1183747 |         11837470 |                 0 |          11837470 |             0      |      100      |
| train   | categorical |      0 |         3 | L0_S3         |              18 | 1183747 |         21307446 |                 0 |          21307446 |             0      |      100      |
| test    | date        |      3 |        46 | L3_S46        |               1 | 1183748 |          1183748 |                 0 |           1183748 |             0      |      100      |
| test    | categorical |      0 |        15 | L0_S15        |               9 | 1183748 |         10653732 |                 0 |          10653732 |             0      |      100      |
| train   | categorical |      0 |        15 | L0_S15        |               9 | 1183747 |         10653714 |                 9 |          10653723 |             0.0001 |       99.9999 |
| train   | categorical |      3 |        36 | L3_S36        |               8 | 1183747 |          9469968 |                 8 |           9469976 |             0.0001 |       99.9999 |
| test    | categorical |      0 |        23 | L0_S23        |              30 | 1183748 |         35512410 |                30 |          35512440 |             0.0001 |       99.9999 |
| test    | categorical |      0 |         3 | L0_S3         |              18 | 1183748 |         21307446 |                18 |          21307464 |             0.0001 |       99.9999 |
| train   | date        |      3 |        46 | L3_S46        |               1 | 1183747 |          1183746 |                 1 |           1183747 |             0.0001 |       99.9999 |
| test    | categorical |      0 |        18 | L0_S18        |              10 | 1183748 |         11837460 |                20 |          11837480 |             0.0002 |       99.9998 |

## Most Complete Station/Data-Type Groups

| split   | data_type   |   line |   station | station_key   |   feature_count |    rows |   missing_values |   observed_values |   possible_values |   completeness_pct |   missing_pct |
|:--------|:------------|-------:|----------:|:--------------|----------------:|--------:|-----------------:|------------------:|------------------:|-------------------:|--------------:|
| train   | numeric     |      3 |        37 | L3_S37        |               4 | 1183747 |           253412 |           4481576 |           4734988 |            94.6481 |        5.3519 |
| train   | date        |      3 |        37 | L3_S37        |               6 | 1183747 |           380118 |           6722364 |           7102482 |            94.6481 |        5.3519 |
| test    | numeric     |      3 |        37 | L3_S37        |               4 | 1183748 |           253928 |           4481064 |           4734992 |            94.6372 |        5.3628 |
| test    | date        |      3 |        37 | L3_S37        |               6 | 1183748 |           380892 |           6721596 |           7102488 |            94.6372 |        5.3628 |
| train   | date        |      3 |        29 | L3_S29        |              63 | 1183747 |          4190349 |          70385712 |          74576061 |            94.3811 |        5.6189 |
| test    | date        |      3 |        29 | L3_S29        |              63 | 1183748 |          4203018 |          70373106 |          74576124 |            94.3641 |        5.6359 |
| train   | numeric     |      3 |        29 | L3_S29        |              53 | 1183747 |          3549169 |          59189422 |          62738591 |            94.3429 |        5.6571 |
| test    | numeric     |      3 |        29 | L3_S29        |              53 | 1183748 |          3559908 |          59178736 |          62738644 |            94.3258 |        5.6742 |
| train   | numeric     |      3 |        34 | L3_S34        |               4 | 1183747 |           274516 |           4460472 |           4734988 |            94.2024 |        5.7976 |
| train   | date        |      3 |        34 | L3_S34        |               5 | 1183747 |           343145 |           5575590 |           5918735 |            94.2024 |        5.7976 |
| test    | numeric     |      3 |        34 | L3_S34        |               4 | 1183748 |           275252 |           4459740 |           4734992 |            94.1869 |        5.8131 |
| test    | date        |      3 |        34 | L3_S34        |               5 | 1183748 |           344065 |           5574675 |           5918740 |            94.1869 |        5.8131 |
| train   | numeric     |      3 |        33 | L3_S33        |              10 | 1183747 |           690520 |          11146950 |          11837470 |            94.1667 |        5.8333 |
| train   | date        |      3 |        33 | L3_S33        |              10 | 1183747 |           690520 |          11146950 |          11837470 |            94.1667 |        5.8333 |
| test    | numeric     |      3 |        33 | L3_S33        |              10 | 1183748 |           692540 |          11144940 |          11837480 |            94.1496 |        5.8504 |

## Manufacturing Flow Datasets

- train: `C:\Users\karin\OneDrive\Desktop\Nexturn\Bosch Production Line Performance\data\processed\manufacturing_flow_train.parquet`
- test: `C:\Users\karin\OneDrive\Desktop\Nexturn\Bosch Production Line Performance\data\processed\manufacturing_flow_test.parquet`

Station presence indicators are derived from date features because date values capture whether a part appears to have passed through a station.
