# Phase 7: Root Cause Analysis

## Model Used

This phase uses the production-safe Phase 6 best model, not the Kaggle leaderboard/leak model. That makes the explanations more appropriate for future manufacturing projects and engineering discussions.

## Validation Population

- Validation rows explained: 56,720
- Validation failure rate: 3.032%
- SHAP sample size: 8,000

## Top Failure Driver

The strongest global SHAP driver is `start_time` with mean absolute SHAP 0.196968.

## Top Station / Root-Cause Area

The highest-priority root-cause area is `timing_level`, led by `start_time`.

## Top 15 Failure Drivers

|   driver_rank | feature                     | station      | driver_type                              |   mean_abs_shap |   mean_signed_shap |
|--------------:|:----------------------------|:-------------|:-----------------------------------------|----------------:|-------------------:|
|             1 | start_time                  | timing_level | Timing or delay signal                   |       0.196968  |       -0.00390927  |
|             2 | line_0_start_time           | line_level   | Line-level timing signal                 |       0.114482  |        0.0109724   |
|             3 | line_3_end_time             | line_level   | Line-level timing signal                 |       0.0947254 |       -0.003977    |
|             4 | line_3_start_time           | line_level   | Line-level timing signal                 |       0.0828658 |       -0.0271034   |
|             5 | end_time                    | timing_level | Timing or delay signal                   |       0.076271  |        0.0129493   |
|             6 | L3_S32_F3850__is_missing    | L3_S32       | Missingness / skipped measurement signal |       0.0655543 |       -0.0155487   |
|             7 | mean_waiting_time           | timing_level | Timing or delay signal                   |       0.0478386 |       -0.0207662   |
|             8 | line_0_end_time             | line_level   | Line-level timing signal                 |       0.0461221 |       -0.0131119   |
|             9 | max_waiting_time            | timing_level | Timing or delay signal                   |       0.0438282 |        0.0177835   |
|            10 | cycle_time                  | timing_level | Timing or delay signal                   |       0.0432106 |       -0.00205111  |
|            11 | line_3_observed_date_values | line_level   | Line-level timing signal                 |       0.0396566 |        0.0030553   |
|            12 | line_3_processing_duration  | line_level   | Line-level timing signal                 |       0.0386989 |        0.00144233  |
|            13 | waiting_time                | timing_level | Timing or delay signal                   |       0.0363432 |       -0.01175     |
|            14 | line_3_station_count        | line_level   | Line-level timing signal                 |       0.0279084 |        0.0139071   |
|            15 | line_0_processing_duration  | line_level   | Line-level timing signal                 |       0.0268708 |        0.000505324 |

## Top 15 Station-Level Root Cause Priorities

| line   | station      | feature_family              |   total_mean_abs_shap | top_driver               | recommended_action                                                                                                     |
|:-------|:-------------|:----------------------------|----------------------:|:-------------------------|:-----------------------------------------------------------------------------------------------------------------------|
| timing | timing_level | timing                      |            0.47539    | start_time               | Review cycle-time, waiting-time, and queue behavior for products following this path.                                  |
| L3     | line_level   | line_timing                 |            0.29254    | line_3_end_time          | Review cycle-time, waiting-time, and queue behavior for products following this path.                                  |
| L1     | L1_S24       | raw_measurement             |            0.258985   | L1_S24_F1844             | Review measurement distributions, tooling condition, calibration records, and recent process changes for this station. |
| L0     | line_level   | line_timing                 |            0.191997   | line_0_start_time        | Review cycle-time, waiting-time, and queue behavior for products following this path.                                  |
| path   | path_level   | path_or_family              |            0.0819223  | path_complexity_score    | Use this as a cross-station signal; compare affected product families and paths before station-specific action.        |
| L3     | L3_S32       | raw_measurement_missingness |            0.0656061  | L3_S32_F3850__is_missing | Check whether skipped measurements, sensor dropouts, or alternate routing through this station align with failures.    |
| L1     | line_level   | line_timing                 |            0.0469699  | line_1_end_time          | Review cycle-time, waiting-time, and queue behavior for products following this path.                                  |
| L2     | line_level   | line_timing                 |            0.0384039  | line_2_start_time        | Review cycle-time, waiting-time, and queue behavior for products following this path.                                  |
| other  | other        | categorical_or_other        |            0.0274471  | observed_date_values     | Use this as a cross-station signal; compare affected product families and paths before station-specific action.        |
| L1     | L1_S25       | raw_measurement             |            0.0222168  | L1_S25_F2990             | Check whether skipped measurements, sensor dropouts, or alternate routing through this station align with failures.    |
| L2     | L2_S27       | raw_measurement             |            0.0200769  | L2_S27_F3144             | Check whether skipped measurements, sensor dropouts, or alternate routing through this station align with failures.    |
| L2     | L2_S28       | raw_measurement             |            0.00734777 | L2_S28_F3248             | Check whether skipped measurements, sensor dropouts, or alternate routing through this station align with failures.    |
| L3     | L3_S49       | raw_measurement             |            0.00677214 | L3_S49_F4211             | Check whether skipped measurements, sensor dropouts, or alternate routing through this station align with failures.    |
| L2     | L2_S27       | raw_measurement_missingness |            0.00323982 | L2_S27_F3144__is_missing | Check whether skipped measurements, sensor dropouts, or alternate routing through this station align with failures.    |
| L1     | L1_S24       | raw_measurement_missingness |            0.00278039 | L1_S24_F1207__is_missing | Review measurement distributions, tooling condition, calibration records, and recent process changes for this station. |

## Recommended Engineering Actions

- Prioritize stations and path-level drivers with the largest total SHAP contribution.
- For missingness drivers, check sensor availability, skipped operations, routing differences, and whether missing measurements represent a known process branch.
- For timing drivers, inspect queue buildup, station waiting time, cycle-time drift, and recent maintenance windows.
- For raw numeric drivers, review calibration, tooling condition, process limits, and distribution shifts for the named station measurements.
- For path or product-family drivers, compare high-risk product routes against lower-risk routes and confirm whether routing policy or product mix explains the pattern.

## Output Files

- `reports/phase7_feature_importance.csv`
- `reports/phase7_shap_global_importance.csv`
- `reports/phase7_shap_local_values_sample.csv`
- `reports/phase7_top_failure_drivers.csv`
- `reports/phase7_station_root_cause_report.csv`
- `reports/phase7_engineer_action_plan.csv`
- `reports/figures/phase7_shap_summary.png`
- `reports/figures/phase7_top_shap_drivers.png`