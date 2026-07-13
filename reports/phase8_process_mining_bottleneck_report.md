# Phase 8: Process Mining & Bottleneck Analysis

## Data Source

Production routes were reconstructed from raw `train_date.csv` and `test_date.csv` station timestamps. Train failure labels were used only for failure-rate and failure-lift diagnostics.

## Process Map

- Unique station transitions: 531
- Stations represented: 52

## Highest Bottleneck Station

`L1_S25` ranks first with bottleneck score 88.34. Its average waiting time is 30.3054 and p90 waiting time is 36.4951.

## Highest Critical Process Path

The top path has critical-path score 70.61, throughput efficiency 0.20%, and failure rate 0.781%.

## Throughput Efficiency

| split   |   product_count |   avg_productive_time |   avg_waiting_time |   median_waiting_time |   p90_waiting_time |   avg_cycle_time |   avg_throughput_efficiency |   median_throughput_efficiency |   avg_wait_events |   failure_count |   labeled_count |   failure_rate_pct |
|:--------|----------------:|----------------------:|-------------------:|----------------------:|-------------------:|-----------------:|----------------------------:|-------------------------------:|------------------:|----------------:|----------------:|-------------------:|
| test    |         1183748 |              0.17296  |            10.5198 |               3.67999 |              33.83 |          10.698  |                  0.00467524 |                              0 |           5.41087 |               0 |               0 |         nan        |
| train   |         1183747 |              0.172941 |            10.5402 |               3.69    |              33.91 |          10.7184 |                  0.00471335 |                              0 |           5.40919 |            6879 |         1183747 |           0.581121 |

## Top 15 Bottlenecks

|   bottleneck_rank | station   |   bottleneck_score |   avg_waiting_time |   p90_waiting_time |   avg_dwell |   failure_rate_pct |   failure_lift |
|------------------:|:----------|-------------------:|-------------------:|-------------------:|------------:|-------------------:|---------------:|
|                 1 | L1_S25    |            88.3371 |           30.3054  |           36.4951  |   2.23603   |           0.506825 |       0.903263 |
|                 2 | L1_S24    |            74.372  |           23.2534  |           41.0135  |   0.0969887 |           0.827859 |       1.47541  |
|                 3 | L3_S38    |            70.6311 |           22.4587  |           32.765   |   0         |           0.781077 |       1.39203  |
|                 4 | L0_S13    |            58.9562 |           18.252   |           31.5788  |   0         |           0.546547 |       0.974056 |
|                 5 | L2_S28    |            57.2127 |            7.88453 |           13.4357  |   0         |           0.699155 |       1.24603  |
|                 6 | L2_S27    |            57.1323 |            6.27098 |           12.2277  |   0         |           0.680864 |       1.21343  |
|                 7 | L2_S26    |            55.5054 |            5.49317 |            9.86044 |   0         |           0.74666  |       1.3307   |
|                 8 | L0_S12    |            55.4933 |           21.7767  |           28.6266  |   0         |           0.546556 |       0.974072 |
|                 9 | L3_S29    |            54.2751 |            5.78747 |            9.25112 |   0         |           0.584658 |       1.04198  |
|                10 | L3_S39    |            51.8482 |            6.17152 |            8.51182 |   0         |           0.505776 |       0.901392 |
|                11 | L3_S30    |            50.1608 |            6.88751 |            8.89003 |   0         |           0.585009 |       1.0426   |
|                12 | L3_S43    |            48.1112 |            9.07361 |           10.4226  |   0         |           0.520441 |       0.927529 |
|                13 | L0_S14    |            42.3758 |            4.34254 |            4.34263 |   0         |           0.56456  |       1.00616  |
|                14 | L3_S31    |            41.9935 |            4.08246 |            5.82113 |   0         |           0.271774 |       0.484355 |
|                15 | L3_S33    |            41.447  |            4.54344 |            4.54399 |   0         |           0.497535 |       0.886706 |

## Recommended Operations Focus

- Investigate the top-ranked stations for queue buildup, uneven staffing, tooling delays, or maintenance-related slowdowns.
- Compare high-wait transitions with low-wait alternatives to identify routing or scheduling improvements.
- Prioritize paths that combine high volume, low throughput efficiency, high waiting time, and elevated failure lift.
- Validate whether long waits are true queues or overlapping/parallel production timestamps before operational changes.

## Output Files

- `reports/phase8_process_map_edges.csv`
- `reports/phase8_process_map_nodes.csv`
- `reports/phase8_station_waiting_times.csv`
- `reports/phase8_bottleneck_scores.csv`
- `reports/phase8_critical_process_paths.csv`
- `reports/phase8_throughput_efficiency.csv`
- `reports/figures/phase8_production_process_map.png`
- `reports/figures/phase8_bottleneck_scores.png`
- `reports/figures/phase8_critical_paths.png`