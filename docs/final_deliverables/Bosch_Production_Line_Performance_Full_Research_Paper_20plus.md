# Manufacturing Failure Prediction, Root-Cause Analytics, and Decision Intelligence for Bosch Production-Line Data

**Author:** Nexturn Manufacturing Analytics Project  
**Format:** IEEE-style full technical research paper  
**Generated:** 2026-07-08 11:31  

## Abstract

This paper presents an end-to-end manufacturing analytics system developed from the Bosch Production Line Performance dataset. The project integrates data quality profiling, feature metadata extraction, exploratory analysis, feature engineering, product-family segmentation, production-safe failure prediction, root-cause analysis, process mining, knowledge-graph analytics, anomaly detection experiments, a manufacturing copilot, an executive dashboard, and final documentation deliverables. The primary production-safe classifier is a LightGBM model that achieved Matthews Correlation Coefficient (MCC)=0.339, precision=0.574, recall=0.213, F1=0.311, and PR-AUC=0.253 on the held-out validation sample. The project found that station L3_S32 had the highest observed station failure rate at 4.506%, product family 4 had the highest product-family failure rate at 0.737%, station L1_S25 was the top bottleneck by process-mining score, and station L1_S24 was the top knowledge-graph critical node. The final system is packaged as reproducible scripts, notebooks, model artifacts, SQLite tables, Streamlit applications, research documentation, and presentation material. The work is positioned as a future-client use-case architecture rather than an active Kaggle submission.

**Keywords:** manufacturing analytics, failure prediction, Bosch, LightGBM, SHAP, process mining, knowledge graph, product family segmentation, anomaly detection, Streamlit dashboard.

## 1. Introduction

Manufacturing quality systems increasingly depend on high-dimensional production data produced by sensors, workstations, routing systems, and inspection steps. Predicting a failure label is only one part of the decision problem. A useful production-line analytics system must also support traceability, engineering review, process bottleneck analysis, model explanation, operational prioritization, and communication to executives.

The Bosch Production Line Performance dataset is a suitable benchmark because it contains numeric, categorical, and date-derived station measurements for more than one million products. The original Kaggle competition rewarded predictive performance, including techniques that exploit row-order and neighborhood leakage. This project intentionally separates competition-style research from production-safe modeling because the project goal is future manufacturing deployment design, not leaderboard submission.

The main research question is: **Can a complete and explainable manufacturing analytics architecture be built from Bosch production-line data while preserving production-safe validation boundaries?** The answer is yes. The selected classifier provides practical prioritization performance, while the supporting modules provide segmentation, root-cause evidence, bottleneck evidence, graph context, anomaly signals, natural-language querying, and executive reporting.

## 2. Research Objectives

The project objectives are:

1. Build a reproducible project repository and Python workflow for raw Bosch train/test files.
2. Parse station-feature naming conventions into line, station, and feature metadata.
3. Engineer timing, waiting, station-count, line-count, and path-complexity features.
4. Discover product families from station-presence and production-path patterns.
5. Train and compare multiple production-safe failure prediction models.
6. Explain the selected model with feature importance and SHAP.
7. Reconstruct production flow for process mining and bottleneck scoring.
8. Build a manufacturing knowledge graph and identify critical nodes.
9. Experiment with advanced AI diagnostics without overstating their production readiness.
10. Build Streamlit applications for copilot querying, executive monitoring, and full-project KPI review.
11. Produce final deliverables, research documentation, and presentation artifacts.

## 3. Dataset Description and Measurement Scope

The project uses the raw Bosch train and test files: numeric, categorical, and date datasets. The training numeric file includes the target variable `Response`; test labels are not available. The sample submission file is intentionally excluded because it does not provide analytical value for this use case.

The labeled training population contains 1,183,747 products and 6,879 observed failures, implying a historical failure rate of 0.581%. The held-out validation population contains 56,720 products and 1,720 failures. The test population contains 1,183,748 products, all of which were scored by the production-safe model. At the selected threshold, 6,998 test products were flagged as predicted failures.

Several important scope constraints affect interpretation. First, test outcomes cannot be evaluated because test labels are absent. Second, Bosch date columns are relative production timestamps rather than normal calendar timestamps. Third, this dataset does not include actual plant intervention costs or realized savings. Fourth, production root cause cannot be confirmed from model explanations alone; engineering records are required.

## 4. Project Pipeline Overview

The project was implemented in twelve phases plus final deliverables. Phase 1 established the project structure and data-quality checks. Phase 2 parsed metadata from feature names. Phase 3 performed exploratory analysis. Phase 4 engineered production timing and path features. Phase 5 discovered product families. Phase 6 trained predictive failure models. Phase 7 generated root-cause explanations. Phase 8 performed process mining and bottleneck analysis. Phase 9 built a manufacturing knowledge graph. Phase 10 added advanced AI diagnostics. Phase 11 created a manufacturing copilot. Phase 12 built an executive dashboard. The final phase packaged models, dashboards, documentation, and presentation materials.

This design produces a layered architecture:

1. **Raw data layer:** Bosch CSV files.
2. **Metadata layer:** line, station, feature, completeness, and flow metadata.
3. **Feature layer:** timing, waiting, path, family, graph, and trajectory features.
4. **Model layer:** production-safe LightGBM plus benchmark and research models.
5. **Interpretation layer:** SHAP, station root-cause reports, bottleneck scores, graph centrality.
6. **Decision layer:** SQLite database, Streamlit dashboards, copilot query engine, final documentation.

## 5. Data Quality and Metadata Engineering

The raw Bosch data is sparse, high dimensional, and station structured. Phase 1 documented file sizes, row counts, feature counts, and missing values. Phase 2 parsed feature names such as `L3_S32_F3850` into line `L3`, station `S32`, and feature `F3850`. This parsing step made it possible to aggregate features by production line and station rather than treating every column as an unrelated variable.

Missingness was not treated as a simple data defect. In production-line data, missing measurements can indicate skipped operations, alternative routing, station-specific sensor availability, or product-family differences. Therefore the modeling pipeline preserved important missingness signals through missingness indicators and station-presence features.

## 6. Exploratory Data Analysis

Exploratory analysis showed that the global failure rate is low, but risk is unevenly distributed across stations, lines, time-derived bins, and product paths. Line-level failure rates differed by production line, and station-level analysis identified several elevated-risk stations. The most notable station-level result is L3_S32, with 4.506% failure rate and 7.75x lift over the overall rate.

Failure trend analysis used ordered relative production-time bins rather than calendar dates. This avoids incorrectly presenting Bosch timestamps as real weeks or months. The trend remains useful for detecting temporal concentration and production-order effects, but not for calendar-seasonality claims.

## 7. Feature Engineering

Feature engineering created production-relevant variables from the date and station structure. The primary engineered features included start time, end time, cycle time, processing duration, waiting time, mean waiting time, maximum waiting time, delay ratio, observed date values, station count, line count, first station, last station, station span, line-switch count, and path-complexity score.

Line-level aggregates were also created for each production line. These features allowed the model to learn timing and routing behavior at a manageable granularity. Feature engineering intentionally balanced predictive signal with explainability: timing, waiting, station count, line count, and path features are easier to discuss with manufacturing engineers than anonymous high-dimensional raw columns alone.

## 8. Product Family Discovery

Product-family discovery used station-presence matrices and clustering. KMeans, DBSCAN, and hierarchical clustering were tested, and the final family labels were used for profiling and downstream modeling. The segmentation produced 8 product families. Family 4 had the highest observed failure rate at 0.737% and 1.27x lift over the overall failure rate.

The family analysis supports an operational interpretation: failures are not only measurement-driven; they are also related to routing structure and product mix. Product-family segmentation is therefore a useful monitoring layer for future client deployments.

**Table 1. Product family failure-rate summary.**

|   final_product_family |   part_count |   failure_count |   failure_rate_pct |   failure_lift_vs_overall |   avg_station_count |   avg_line_count | top_stations                                                                                                      |
|-----------------------:|-------------:|----------------:|-------------------:|--------------------------:|--------------------:|-----------------:|:------------------------------------------------------------------------------------------------------------------|
|                      4 |       128104 |             944 |           0.736901 |                  1.26807  |             7.91947 |          2.90448 | L3_S35:100.0%, L3_S37:99.9%, L3_S30:99.8%, L3_S33:99.5%, L3_S34:99.5%, L3_S29:99.4%, L1_S24:68.3%, L2_S26:64.4%   |
|                      1 |       132878 |             974 |           0.733003 |                  1.26136  |             7.86902 |          2.88998 | L3_S36:99.3%, L3_S37:99.3%, L3_S30:99.2%, L3_S33:98.8%, L3_S34:98.8%, L3_S29:98.8%, L1_S24:67.9%, L2_S26:64.0%    |
|                      0 |       252037 |            1396 |           0.553887 |                  0.953136 |            13.2136  |          2.14689 | L0_S8:99.9%, L0_S1:99.9%, L0_S0:99.9%, L3_S29:99.8%, L3_S37:99.7%, L3_S30:99.7%, L3_S34:99.2%, L3_S33:99.2%       |
|                      3 |       125592 |             693 |           0.551787 |                  0.949522 |            13.1853  |          2.09333 | L0_S20:100.0%, L0_S13:100.0%, L0_S12:100.0%, L3_S29:99.8%, L3_S37:99.7%, L3_S30:99.7%, L3_S34:99.3%, L3_S33:99.2% |
|                      7 |       148569 |             799 |           0.537797 |                  0.925448 |            13.2117  |          2.14411 | L0_S0:99.9%, L0_S1:99.9%, L0_S8:99.9%, L3_S29:99.7%, L3_S37:99.7%, L3_S30:99.6%, L3_S34:99.2%, L3_S33:99.1%       |
|                      5 |        99296 |             533 |           0.536779 |                  0.923696 |            13.1855  |          2.09421 | L0_S20:100.0%, L0_S13:100.0%, L0_S12:100.0%, L3_S29:99.7%, L3_S37:99.7%, L3_S30:99.6%, L3_S34:99.2%, L3_S33:99.1% |
|                      2 |       237283 |            1236 |           0.520897 |                  0.896366 |            13.2187  |          2.14538 | L0_S1:99.9%, L0_S8:99.9%, L3_S29:99.9%, L0_S0:99.9%, L3_S37:99.8%, L3_S30:99.8%, L3_S34:99.4%, L3_S33:99.3%       |
|                      6 |        59988 |             304 |           0.506768 |                  0.872053 |            15.454   |          2.13089 | L3_S47:99.9%, L3_S45:99.9%, L3_S48:99.9%, L3_S40:99.9%, L3_S41:99.9%, L3_S39:99.9%, L3_S51:99.8%, L0_S1:60.8%     |

## 9. Predictive Failure Modeling

Phase 6 trained and compared Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost using production-safe validation. The selected model is LightGBM because it achieved the best validation MCC while maintaining a practical precision/recall balance. The selected LightGBM threshold is 0.785083.

**Table 2. Production-safe model comparison.**

| model               |   threshold |      mcc |   precision |   recall |       f1 |   pr_auc |   rank |
|:--------------------|------------:|---------:|------------:|---------:|---------:|---------:|-------:|
| LightGBM            |    0.785083 | 0.338642 |    0.574335 | 0.213372 | 0.311149 | 0.252867 |      1 |
| XGBoost             |    0.747938 | 0.335935 |    0.567442 | 0.212791 | 0.309514 | 0.246895 |      2 |
| CatBoost            |    0.796958 | 0.327926 |    0.557121 | 0.206977 | 0.301823 | 0.226068 |      3 |
| Random Forest       |    0.674328 | 0.325003 |    0.552426 | 0.205233 | 0.299279 | 0.239001 |      4 |
| Logistic Regression |    0.897303 | 0.304546 |    0.519562 | 0.193023 | 0.281475 | 0.165042 |      5 |

The best validation metrics were MCC=0.339, precision=0.574, recall=0.213, F1=0.311, and PR-AUC=0.253. The model is appropriate for risk prioritization and inspection triage, not autonomous accept/reject decisions. A manufacturing deployment would require forward-time validation, threshold calibration, quality-workflow design, and monitoring.

## 10. Model Improvement and Negative Results

After Phase 10, additional improvement experiments were performed. These included tuned LightGBM on the original Phase 6 features, tuned LightGBM with graph and trajectory features, product-family-aware LightGBM, and a validation-optimized blend. None outperformed the original Phase 6 LightGBM baseline. This is a valuable negative result: adding more complex features or family-specific models can fragment data, overfit validation conditions, or reduce generalization.

The accepted production-safe benchmark remains the Phase 6 LightGBM model. The validation-optimized blend is retained as a research artifact only because its weights were selected on validation data.

## 11. Root-Cause Analysis with SHAP

Root-cause analysis used model feature importance and SHAP values to explain the production-safe model. The strongest global drivers were timing and line-level timing signals, including start time, line start/end time, waiting-time measures, and selected station measurement or missingness signals. SHAP values explain model behavior; they do not prove physical cause.

**Table 3. Top global SHAP drivers.**

| feature                     |   mean_abs_shap |   mean_signed_shap | line   | station      | feature_family              |   model_importance |   driver_rank | driver_type                              |
|:----------------------------|----------------:|-------------------:|:-------|:-------------|:----------------------------|-------------------:|--------------:|:-----------------------------------------|
| start_time                  |       0.196968  |        -0.00390927 | timing | timing_level | timing                      |                512 |             1 | Timing or delay signal                   |
| line_0_start_time           |       0.114482  |         0.0109724  | L0     | line_level   | line_timing                 |                385 |             2 | Line-level timing signal                 |
| line_3_end_time             |       0.0947254 |        -0.003977   | L3     | line_level   | line_timing                 |                 88 |             3 | Line-level timing signal                 |
| line_3_start_time           |       0.0828658 |        -0.0271034  | L3     | line_level   | line_timing                 |                264 |             4 | Line-level timing signal                 |
| end_time                    |       0.076271  |         0.0129493  | timing | timing_level | timing                      |                474 |             5 | Timing or delay signal                   |
| L3_S32_F3850__is_missing    |       0.0655543 |        -0.0155487  | L3     | L3_S32       | raw_measurement_missingness |                111 |             6 | Missingness / skipped measurement signal |
| mean_waiting_time           |       0.0478386 |        -0.0207662  | timing | timing_level | timing                      |                875 |             7 | Timing or delay signal                   |
| line_0_end_time             |       0.0461221 |        -0.0131119  | L0     | line_level   | line_timing                 |                184 |             8 | Line-level timing signal                 |
| max_waiting_time            |       0.0438282 |         0.0177835  | timing | timing_level | timing                      |                772 |             9 | Timing or delay signal                   |
| cycle_time                  |       0.0432106 |        -0.00205111 | timing | timing_level | timing                      |                732 |            10 | Timing or delay signal                   |
| line_3_observed_date_values |       0.0396566 |         0.0030553  | L3     | line_level   | line_timing                 |                134 |            11 | Line-level timing signal                 |
| line_3_processing_duration  |       0.0386989 |         0.00144233 | L3     | line_level   | line_timing                 |                723 |            12 | Line-level timing signal                 |

The station-level root-cause report translated model drivers into engineering investigation recommendations. For timing drivers, the recommended action is to review queue buildup, waiting time, cycle-time drift, and maintenance windows. For missingness drivers, the recommended action is to check skipped measurements, sensor availability, and alternate routing. For raw numeric station measurements, the recommended action is to review distributions, calibration, tooling condition, and recent process changes.

## 12. Process Mining and Bottleneck Analysis

Process mining reconstructed station transitions from raw date features. Outputs included process-map nodes, process-map edges, station waiting times, bottleneck scores, critical process paths, and throughput-efficiency proxies. The top bottleneck station is L1_S25 with bottleneck score 88.34, average waiting time 30.31, and p90 waiting time 36.50.

**Table 4. Top process bottlenecks.**

| station   |   product_count |   avg_dwell |   p90_dwell |   avg_waiting_time |   p90_waiting_time |   positive_wait_rate |   failure_count |    labeled_count |   failure_rate_pct |   failure_lift |   bottleneck_score |   bottleneck_rank |
|:----------|----------------:|------------:|------------:|-------------------:|-------------------:|---------------------:|----------------:|-----------------:|-------------------:|---------------:|-------------------:|------------------:|
| L1_S25    |          167220 |   2.23603   |   8.6487    |           30.3054  |           36.4951  |           0.999251   |             424 |  83658           |           0.506825 |       0.903263 |            88.3371 |                 1 |
| L1_S24    |          366583 |   0.0969887 |   0.0629955 |           23.2534  |           41.0135  |           0.99417    |            1521 | 183727           |           0.827859 |       1.47541  |            74.372  |                 2 |
| L3_S38    |           54175 |   0         |   0         |           22.4587  |           32.765   |           1          |             212 |  27142           |           0.781077 |       1.39203  |            70.6311 |                 3 |
| L0_S13    |          484481 |   0         |   0         |           18.252   |           31.5788  |           0.247169   |            1323 | 242065           |           0.546547 |       0.974056 |            58.9562 |                 4 |
| L2_S28    |           19153 |   0         |   0         |            7.88453 |           13.4357  |           0.997879   |              67 |   9583           |           0.699155 |       1.24603  |            57.2127 |                 5 |
| L2_S27    |          240285 |   0         |   0         |            6.27098 |           12.2277  |           0.998524   |             822 | 120729           |           0.680864 |       1.21343  |            57.1323 |                 6 |
| L2_S26    |          454344 |   0         |   0         |            5.49317 |            9.86044 |           0.999148   |            1695 | 227011           |           0.74666  |       1.3307   |            55.5054 |                 7 |
| L0_S12    |          484476 |   0         |   0         |           21.7767  |           28.6266  |           0.00203478 |            1323 | 242061           |           0.546556 |       0.974072 |            55.4933 |                 8 |
| L3_S29    |         2239066 |   0         |   0         |            5.78747 |            9.25112 |           0.878491   |            6546 |      1.11963e+06 |           0.584658 |       1.04198  |            54.2751 |                 9 |
| L3_S39    |          120019 |   0         |   0         |            6.17152 |            8.51182 |           0.887157   |             303 |  59908           |           0.505776 |       0.901392 |            51.8482 |                10 |

The throughput efficiency proxy is calculated from available timestamps and should not be presented as formal Overall Equipment Effectiveness (OEE). It is useful for relative comparison inside the dataset, but a real client deployment would need plant-defined productive time, downtime, quality loss, and planned production time.

## 13. Knowledge Graph Construction

The knowledge graph links production lines, stations, features, process transitions, and failure evidence. The graph contains 4,321 nodes and 4,923 edges. Centrality and critical-node scoring combine graph position, transition evidence, bottleneck scores, failure rates, and root-cause importance.

The top critical node is L1_S24 with critical-node score 60.08. This does not mean the node is the single physical cause of failures. It means the node has high combined priority across network and operational signals.

**Table 5. Top knowledge-graph critical nodes.**

| station   |   in_degree_centrality |   out_degree_centrality |   weighted_inflow |   weighted_outflow |   pagerank |   betweenness_centrality |   closeness_centrality |   product_count |   avg_dwell |   p90_dwell |   avg_waiting_time |   p90_waiting_time |   positive_wait_rate |   failure_count |    labeled_count |   failure_rate_pct |   failure_lift |   bottleneck_score |   bottleneck_rank |   station_shap_importance |   centrality_score |   critical_node_score |   critical_rank | line   |
|:----------|-----------------------:|------------------------:|------------------:|-------------------:|-----------:|-------------------------:|-----------------------:|----------------:|------------:|------------:|-------------------:|-------------------:|---------------------:|----------------:|-----------------:|-------------------:|---------------:|-------------------:|------------------:|--------------------------:|-------------------:|----------------------:|----------------:|:-------|
| L1_S24    |               0.196078 |                0.333333 |    5833           |   366579           | 0.00312985 |              0.000784314 |                1.47839 |          366583 |   0.0969887 |   0.0629955 |          23.2534   |          41.0135   |             0.99417  |            1521 | 183727           |           0.827859 |       1.47541  |            74.372  |                 2 |                 0.261766  |          0.091378  |               60.0837 |               1 | L1     |
| L3_S29    |               0.490196 |                0.176471 |       2.2371e+06  |        2.23906e+06 | 0.0529309  |              0.0560784   |                3.4607  |         2239066 |   0         |   0         |           5.78747  |           9.25112  |             0.878491 |            6546 |      1.11963e+06 |           0.584658 |       1.04198  |            54.2751 |                 9 |                 0         |          0.640139  |               52.9543 |               2 | L3     |
| L3_S32    |               0.196078 |                0.117647 |   48678           |    47205           | 0.00412963 |              0           |                2.75962 |           48678 |   0         |   0         |           1.2056   |           1.22081  |             0.795405 |            1106 |  24543           |           4.50638  |       8.03125  |            40.1792 |                16 |                 0.0680442 |          0.126731  |               50.897  |               3 | L3     |
| L3_S30    |               0.411765 |                0.156863 |       2.23711e+06 |        2.23937e+06 | 0.0474883  |              0.0258824   |                2.76287 |         2239366 |   0         |   0         |           6.88751  |           8.89003  |             0.565179 |            6551 |      1.11981e+06 |           0.585009 |       1.0426   |            50.1608 |                11 |                 0         |          0.502128  |               47.3429 |               4 | L3     |
| L3_S34    |               0.235294 |                0.117647 |       2.23005e+06 |        2.2077e+06  | 0.0939865  |              0.00156863  |                3.21472 |         2230053 |   0         |   0         |           2.46923  |           2.48028  |             0.385507 |            5718 |      1.11512e+06 |           0.512771 |       0.913859 |            34.1502 |                21 |                 0         |          0.611678  |               46.018  |               5 | L3     |
| L3_S33    |               0.235294 |                0.117647 |       2.22919e+06 |        2.09366e+06 | 0.0680336  |              0.000392157 |                3.15096 |         2229189 |   0         |   0         |           4.54344  |           4.54399  |             0.443652 |            5546 |      1.1147e+06  |           0.497535 |       0.886706 |            41.447  |                15 |                 0         |          0.506582  |               44.6273 |               6 | L3     |
| L3_S39    |               0.431373 |                0.156863 |  119768           |   120019           | 0.0067733  |              0.119216    |                2.96034 |          120019 |   0         |   0         |           6.17152  |           8.51182  |             0.887157 |             303 |  59908           |           0.505776 |       0.901392 |            51.8482 |                10 |                 0         |          0.499602  |               44.2361 |               7 | L3     |
| L1_S25    |               0.215686 |                0.254902 |    1333           |   167220           | 0.00302658 |              0           |                1.31831 |          167220 |   2.23603   |   8.6487    |          30.3054   |          36.4951   |             0.999251 |             424 |  83658           |           0.506825 |       0.903263 |            88.3371 |                 1 |                 0.0227325 |          0.0684201 |               42.7627 |               8 | L1     |
| L3_S37    |               0.196078 |                0.117647 |       2.24065e+06 |        1.02219e+06 | 0.102561   |              0.00235294  |                3.08022 |         2240660 |   0         |   0         |           0.773873 |           0.776879 |             0.50794  |            6556 |      1.12039e+06 |           0.585151 |       1.04286  |            26.8599 |                25 |                 0         |          0.558886  |               42.5794 |               9 | L3     |
| L0_S13    |               0.117647 |                0.176471 |  484476           |   484481           | 0.00864762 |              0.0494118   |                1.00883 |          484481 |   0         |   0         |          18.252    |          31.5788   |             0.247169 |            1323 | 242065           |           0.546547 |       0.974056 |            58.9562 |                 4 |                 0         |          0.241084  |               39.5984 |              10 | L0     |

## 14. Advanced AI Experiments

Phase 10 tested four advanced diagnostic models: Isolation Forest, MLP reconstruction anomaly detection, graph message-passing risk, and failure trajectory prediction. The best advanced method was failure trajectory prediction, but it did not beat the production-safe Phase 6 LightGBM benchmark.

**Table 6. Advanced AI model comparison.**

| model                                |   threshold |       mcc |   precision |    recall |        f1 |    pr_auc | notes                                                                         |
|:-------------------------------------|------------:|----------:|------------:|----------:|----------:|----------:|:------------------------------------------------------------------------------|
| Phase 6 LightGBM reference           |    0.785083 | 0.338642  |   0.574335  | 0.213372  | 0.311149  | 0.252867  | Production-safe supervised benchmark from Phase 6.                            |
| Failure trajectory prediction        |    0.989711 | 0.24189   |   0.461538  | 0.139535  | 0.214286  | 0.164805  | LightGBM model using timing, path, family, and ordered station-risk exposure. |
| MLP reconstruction anomaly detection |    4.36552  | 0.0923335 |   0.253521  | 0.0418605 | 0.0718563 | 0.0642836 | Scikit-learn reconstruction model using 120 selected features.                |
| Graph message-passing risk model     |    0.52874  | 0.0611082 |   0.0533566 | 0.301744  | 0.0906788 | 0.0410834 | Logistic model using graph-propagated station risk exposure.                  |
| Isolation Forest anomaly detection   |   -0.121686 | 0.0537181 |   0.0445238 | 0.434884  | 0.0807775 | 0.0395153 | Unsupervised model trained on normal products only.                           |

These experiments are included because they are useful for future research and client demonstrations. However, they remain diagnostic add-ons. The official failure probability remains the Phase 6 LightGBM score.

## 15. Competition-Style Leaderboard Research and Why It Was Excluded

A separate leaderboard-style research workflow tested order and nearby-label features inspired by Kaggle discussions. This approach produced much higher validation scores, but it depends on nearby known training labels or row-order signals. In a live factory deployment, future product labels and neighboring failure outcomes are not available at prediction time. Therefore the leakage-style model is documented as research only and excluded from production-safe deliverables, dashboards, root-cause analysis, and copilot outputs.

This distinction is important for stakeholders. A high competition score can be mathematically valid inside a contest dataset but invalid for operational decision-making. The project preserves both artifacts: the research-only leaderboard model for learning and the production-safe LightGBM model for deployable architecture.

## 16. Manufacturing Copilot

Phase 11 stores model outputs and reviewed analytics tables in SQLite. The copilot database includes model metrics, failure drivers, station root causes, engineer actions, station failure rates, line failure rates, bottlenecks, throughput metrics, critical nodes, propagation routes, advanced-AI metrics, product predictions, and source catalog entries. The product prediction table contains 1,240,468 rows.

The Streamlit copilot supports questions such as: “Which stations have the highest failure rate?”, “Why is L3_S32 risky?”, “What are the top bottlenecks?”, “Show the most critical process nodes,” and “How good is the production-safe model?” The query layer uses reviewed deterministic intents and parameterized SQL rather than unrestricted generative SQL, which improves auditability and reduces hallucination risk.

## 17. Executive Dashboard and Unified Project Dashboard

Phase 12 created an executive dashboard for KPIs, relative failure trends, station heatmaps, bottleneck analytics, SHAP explanations, and business-impact scenarios. The business-impact calculator is explicitly assumption-driven; it uses production volume, historical failure rate, alert rate, validation precision, cost per failure, review cost, and intervention effectiveness.

After final deliverables, a unified Streamlit dashboard was also built for the whole project. It includes KPI Overview, Prediction Model, Product Families, Process Mining, Root Cause, Knowledge Graph, Copilot, Business Impact, and Deliverables pages. This consolidated app is the main reviewer interface for the complete project.

## 18. Final Deliverables

The project produced the following final deliverables:

1. Production Failure Prediction Model.
2. Product Family Segmentation Module.
3. Process Mining Engine.
4. Root Cause Analysis Engine.
5. Knowledge Graph.
6. Manufacturing Copilot.
7. Executive Dashboard.
8. Research Documentation and Presentation.
9. Unified Project Dashboard.

Every deliverable maps to concrete files in the repository: scripts, notebooks, models, reports, databases, or Streamlit applications. The final deliverables package contains a Word report, Markdown report, PowerPoint deck, manifest, and zip archive.

## 19. Robustness, Validation, and Production Boundaries

Validation was performed through script compilation, notebook JSON checks, model metric comparisons, SQLite integrity checks, Streamlit test harness runs, HTTP checks against local dashboard servers, and artifact-open checks for Word and PowerPoint files. The model validation remains a held-out dataset validation, not a live factory trial.

The main robustness boundary is temporal deployment realism. The project excludes features that rely on future labels or nearby known outcomes. It also labels SHAP and graph outputs as diagnostic rather than causal. Financial impact is scenario-based rather than observed. These boundaries make the final project more defensible for future manufacturing engagements.

## 20. Limitations

The project has several limitations:

1. Test labels are unavailable.
2. Bosch timestamps are relative rather than calendar dates.
3. The dataset lacks plant-specific maintenance logs, operator notes, inspection records, and financial outcomes.
4. Model explanations cannot prove physical root cause.
5. Business impact cannot be validated without intervention records.
6. Product-family segmentation is derived from observed station presence and may change on a different plant.
7. Advanced AI experiments were constrained by available local dependencies and were designed as diagnostics, not replacement production models.
8. A real deployment requires data contracts, feature monitoring, access control, retraining governance, and quality-workflow integration.

## 21. Recommendations for Future Manufacturing Client Work

For future projects, the recommended deployment path is:

1. Replace Kaggle files with governed plant data sources.
2. Define stable data contracts for product IDs, station events, timestamps, measurements, quality outcomes, and intervention outcomes.
3. Validate models using forward-time splits and post-alert quality results.
4. Calibrate thresholds based on operational capacity and cost.
5. Integrate alerts into engineering review workflows rather than autonomous disposition.
6. Connect root-cause analysis to maintenance, calibration, sensor, and quality records.
7. Monitor prediction drift, feature drift, alert precision, recall, and intervention effectiveness.
8. Maintain separate production-safe and research-only model registries.

## 22. Conclusion

This project demonstrates that the Bosch dataset can support a complete manufacturing intelligence architecture, not just a binary classifier. The production-safe LightGBM model provides practical failure-risk prioritization, while product-family segmentation, process mining, SHAP explanations, knowledge-graph analytics, advanced diagnostics, copilot querying, and dashboards make the result explainable and operationally usable. The work is suitable as a reference architecture for future manufacturing quality analytics projects, provided client-specific validation and deployment controls are applied.

## Appendix A. Phase-by-Phase Artifact Summary

Phase 1 generated data-quality summaries. Phase 2 generated feature, station, and line metadata. Phase 3 generated EDA charts and failure-rate reports. Phase 4 generated engineered timing and path features. Phase 5 generated product-family labels and family profiles. Phase 6 generated predictive models and validation metrics. Phase 7 generated feature importance, SHAP, and engineer action plans. Phase 8 generated process maps and bottleneck scores. Phase 9 generated a knowledge graph and critical-node rankings. Phase 10 generated anomaly, autoencoder-style reconstruction, graph-risk, and trajectory models. Phase 11 generated a SQLite database and copilot. Phase 12 generated the executive dashboard. The final stage generated project documentation and the unified dashboard.

## Appendix B. Supplemental Work Not Used as Production Inputs

The following completed work was intentionally not used as the official production-safe decision path:

1. **Leaderboard leakage modeling:** useful for understanding Kaggle competition behavior but excluded from deployable workflows.
2. **Validation-optimized blend:** retained as a research upper bound but excluded because validation weights can overfit.
3. **Family-aware model:** useful diagnostically but weaker overall than the global LightGBM baseline.
4. **Advanced AI anomaly models:** useful as supporting signals but weaker than the production-safe classifier.
5. **Graph message-passing risk:** useful for station risk propagation analysis but not a replacement supervised classifier.
6. **Business-impact scenario:** useful for executive planning but not observed realized financial savings.

## Appendix C. Reproducibility Notes

The project runs in a Python virtual environment with pandas, numpy, scikit-learn, LightGBM, XGBoost, CatBoost, SHAP, NetworkX, Streamlit, Plotly, python-docx, and python-pptx. The main dashboards can be started with:

```powershell
.\.venv\Scripts\streamlit.exe run app\project_streamlit_dashboard.py
.\.venv\Scripts\streamlit.exe run app\phase11_manufacturing_copilot.py
.\.venv\Scripts\streamlit.exe run app\phase12_executive_dashboard.py
```

## References

[1] Bosch Production Line Performance dataset, Kaggle competition dataset.  
[2] G. Ke et al., “LightGBM: A Highly Efficient Gradient Boosting Decision Tree,” Advances in Neural Information Processing Systems.  
[3] T. Chen and C. Guestrin, “XGBoost: A Scalable Tree Boosting System,” ACM KDD.  
[4] L. Prokhorenkova et al., “CatBoost: Unbiased Boosting with Categorical Features,” Advances in Neural Information Processing Systems.  
[5] S. Lundberg and S. Lee, “A Unified Approach to Interpreting Model Predictions,” Advances in Neural Information Processing Systems.  
[6] L. Breiman, “Random Forests,” Machine Learning.  
[7] W. van der Aalst, “Process Mining: Data Science in Action,” Springer.  
[8] M. Newman, “Networks: An Introduction,” Oxford University Press.  
[9] F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” Journal of Machine Learning Research.  
[10] Streamlit documentation for Python analytical applications.  
[11] Plotly documentation for interactive analytical visualization.  
[12] Internal project artifact: `reports/phase6_leaderboard_research_notes.md`.  
[13] Internal project artifact: `reports/phase6_model_improvement_report.md`.  
[14] Internal project artifact: `reports/phase10_advanced_ai_report.md`.  
[15] Internal project artifact: `reports/phase7_root_cause_analysis_report.md`.  
[16] Internal project artifact: `reports/phase8_process_mining_bottleneck_report.md`.  
[17] Internal project artifact: `reports/phase9_knowledge_graph_report.html`.  
[18] Internal project artifact: `reports/phase11_manufacturing_copilot_report.md`.  
[19] Internal project artifact: `reports/phase12_executive_dashboard_report.md`.  
[20] Internal project artifact: `docs/final_deliverables/final_deliverables_manifest.json`.
