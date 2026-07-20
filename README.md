# Bosch Production Line Performance

Phase 1 sets up a reproducible Python project for the Kaggle Bosch Production Line Performance dataset.

## Project Structure

```text
.
+-- data/
|   +-- database/     # Database for streamlit app
+-- docs/             # Project documentation
+-- models/           # Trained model artifacts
+-- notebooks/        # Python notebooks for project phases
+-- reports/          # Generated reports and data quality outputs
+-- src/              # Reusable Python source code
+-- requirements.txt
```

The Kaggle CSV files are currently in the project root. The Phase 1 profiling script checks both the project root and `data/raw/`, so the files do not need to be moved.

`sample_submission.csv` is not used for Phase 1 analysis or model training. It is only a Kaggle submission-format reference file.

## Environment Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run Phase 1 Data Checks

```powershell
.\.venv\Scripts\python.exe src\data\phase1_data_quality.py
```

## Phase 1 Notebook

Open this notebook for the detailed, step-by-step version of the setup and data checks:

- `notebooks/phase1_project_setup_and_data_quality.ipynb`

## Phase 2 Notebook

Open this notebook for the detailed data-understanding and manufacturing-flow engineering workflow:

- `notebooks/phase2_data_understanding_and_engineering.ipynb`

Run the Phase 2 script directly with:

```powershell
.\.venv\Scripts\python.exe src\data\phase2_data_understanding_engineering.py
```

Phase 2 outputs:

- `reports/phase2_feature_metadata.csv`
- `reports/phase2_station_metadata.csv`
- `reports/phase2_line_metadata.csv`
- `reports/phase2_feature_completeness_metrics.csv`
- `reports/phase2_station_completeness_metrics.csv`
- `reports/phase2_data_understanding_engineering_report.md`
- `data/processed/manufacturing_flow_train.parquet`
- `data/processed/manufacturing_flow_test.parquet`

## Phase 3 Notebook

Open this notebook for raw-CSV exploratory analysis of line/station risk, product flow paths, relative-time patterns, correlations, and distributions:

- `notebooks/phase3_exploratory_data_analysis.ipynb`

Run the Phase 3 script directly with:

```powershell
.\.venv\Scripts\python.exe src\data\phase3_exploratory_data_analysis.py
```

Phase 3 outputs:

- `reports/phase3_line_failure_rates.csv`
- `reports/phase3_station_failure_rates.csv`
- `reports/phase3_high_risk_stations.csv`
- `reports/phase3_flow_path_summary.csv`
- `reports/phase3_first_time_failure_patterns.csv`
- `reports/phase3_duration_failure_patterns.csv`
- `reports/phase3_correlation_report.csv`
- `reports/phase3_distribution_report.csv`
- `reports/phase3_categorical_ohe_selected_columns.csv`
- `reports/phase3_categorical_ohe_failure_rates.csv`
- `reports/phase3_categorical_ohe_correlation_report.csv`
- `reports/phase3_exploratory_data_analysis_report.md`
- `reports/figures/phase3_*.png`
- `data/processed/phase3_train_time_features.csv`

## Phase 4 Notebook

Open this notebook for raw-date feature engineering of timing, delay, station-count, path-complexity, and line-level aggregate features:

- `notebooks/phase4_feature_engineering.ipynb`

Run the Phase 4 script directly with:

```powershell
.\.venv\Scripts\python.exe src\data\phase4_feature_engineering.py
```

Phase 4 outputs:

- `data/processed/phase4_train_engineered_features.csv`
- `data/processed/phase4_test_engineered_features.csv`
- `reports/phase4_feature_dictionary.csv`
- `reports/phase4_engineered_feature_summary.csv`
- `reports/phase4_feature_engineering_report.md`

## Phase 5 Notebook

Open this notebook for product-family discovery using station-presence clustering and family-specific baseline models:

- `notebooks/phase5_product_family_discovery.ipynb`

Run the Phase 5 script directly with:

```powershell
.\.venv\Scripts\python.exe src\data\phase5_product_family_discovery.py
```

Phase 5 outputs:

- `data/processed/phase5_train_station_presence_matrix.csv`
- `data/processed/phase5_test_station_presence_matrix.csv`
- `data/processed/phase5_train_product_families.csv`
- `data/processed/phase5_test_product_families.csv`
- `reports/phase5_unique_path_cluster_map.csv`
- `reports/phase5_cluster_diagnostics.csv`
- `reports/phase5_product_family_profiles.csv`
- `reports/phase5_product_family_failure_rates.csv`
- `reports/phase5_family_model_metrics.csv`
- `reports/phase5_product_family_discovery_report.md`
- `models/phase5_family_*_hgb.joblib`

## Phase 6 Notebook

Open this notebook for predictive failure modeling with raw numeric, categorical, and date-derived features:

- `notebooks/phase6_predictive_failure_modeling.ipynb`

Run the Phase 6 script directly with:

```powershell
.\.venv\Scripts\python.exe src\data\phase6_predictive_failure_modeling.py
```

Phase 6 outputs:

- `data/processed/phase6_sampled_training_ids.csv`
- `data/processed/phase6_train_dataset.csv`
- `data/processed/phase6_validation_dataset.csv`
- `data/processed/phase6_test_dataset_preview.csv`
- `data/processed/phase6_test_predictions.csv`
- `reports/phase6_numeric_correlation_report.csv`
- `reports/phase6_categorical_presence_report.csv`
- `reports/phase6_selected_numeric_features.csv`
- `reports/phase6_selected_categorical_features.csv`
- `reports/phase6_final_feature_correlation_report.csv`
- `reports/phase6_model_comparison_metrics.csv`
- `reports/phase6_predictive_failure_modeling_report.md`
- `models/phase6_best_model.joblib`
- `models/phase6_*.joblib`

## Phase 6 Leaderboard Boost

This optional Phase 6 extension uses public Kaggle-solution ideas: order/leak-style features, nearby known failure counts, production-time neighbor deltas, and MCC threshold tuning.

Run it with:

```powershell
.\.venv\Scripts\python.exe src\data\phase6_leaderboard_boost_modeling.py
```

Leaderboard boost outputs:

- `reports/phase6_leaderboard_research_notes.md`
- `reports/phase6_leaderboard_boost_report.md`
- `reports/phase6_leaderboard_boost_metrics.csv`
- `data/processed/phase6_leaderboard_boost_train_features.csv`
- `data/processed/phase6_leaderboard_boost_test_features.csv`
- `data/processed/phase6_leaderboard_boost_test_predictions.csv`
- `submissions/phase6_leaderboard_boost_submission.csv`
- `models/phase6_leaderboard_boost_best_model.joblib`

## Phase 7 Notebook

Open this notebook for production-safe root cause analysis, SHAP explanations, station-level reports, and recommended engineering actions:

- `notebooks/phase7_root_cause_analysis.ipynb`

Run the Phase 7 script directly with:

```powershell
.\.venv\Scripts\python.exe src\data\phase7_root_cause_analysis.py
```

Phase 7 outputs:

- `reports/phase7_feature_importance.csv`
- `reports/phase7_shap_global_importance.csv`
- `reports/phase7_shap_local_values_sample.csv`
- `reports/phase7_top_failure_drivers.csv`
- `reports/phase7_station_root_cause_report.csv`
- `reports/phase7_engineer_action_plan.csv`
- `reports/phase7_root_cause_analysis_report.md`
- `reports/figures/phase7_shap_summary.png`
- `reports/figures/phase7_top_shap_drivers.png`

## Phase 8 Notebook

Open this notebook for raw-date process mining, station waiting-time analysis, bottleneck scoring, critical-path identification, and throughput efficiency:

- `notebooks/phase8_process_mining_bottleneck_analysis.ipynb`

Run the Phase 8 script directly with:

```powershell
.\.venv\Scripts\python.exe src\data\phase8_process_mining_bottleneck_analysis.py
```

Phase 8 outputs:

- `reports/phase8_process_map_edges.csv`
- `reports/phase8_process_map_nodes.csv`
- `reports/phase8_station_waiting_times.csv`
- `reports/phase8_bottleneck_scores.csv`
- `reports/phase8_critical_process_paths.csv`
- `reports/phase8_throughput_efficiency.csv`
- `reports/phase8_process_mining_bottleneck_report.md`
- `reports/figures/phase8_production_process_map.png`
- `reports/figures/phase8_bottleneck_scores.png`
- `reports/figures/phase8_critical_paths.png`

## Phase 9 Notebook

Open this notebook for the manufacturing knowledge graph, station relationships, centrality metrics, candidate failure-propagation routes, and critical-node rankings:

- `notebooks/phase9_knowledge_graph.ipynb`

Run the Phase 9 script directly with:

```powershell
.\.venv\Scripts\python.exe src\data\phase9_knowledge_graph.py
```

Phase 9 outputs:

- `reports/phase9_knowledge_graph_nodes.csv`
- `reports/phase9_knowledge_graph_edges.csv`
- `reports/phase9_station_centrality_metrics.csv`
- `reports/phase9_critical_nodes.csv`
- `reports/phase9_failure_propagation_routes.csv`
- `reports/phase9_manufacturing_knowledge_graph.graphml`
- `reports/phase9_knowledge_graph_report.html`
- `reports/figures/phase9_critical_nodes.png`
- `reports/figures/phase9_station_relationship_network.png`
- `reports/figures/phase9_candidate_propagation_routes.png`

Outputs:

- `reports/phase1_data_quality_report.md`
- `reports/phase1_file_summary.csv`
- `reports/phase1_missing_values_by_column.csv`

Note: full Jupyter Lab is intentionally not in `requirements.txt` because this OneDrive path can exceed Windows package path limits during installation. `ipykernel` is included so the notebook can run from VS Code, Jupyter, or Google Colab. Add full Jupyter later from a shorter local path if needed.

## Phase 10 - Advanced AI

Phase 10 adds advanced diagnostic AI layers around the production-safe Phase 6 classifier:

- Isolation Forest anomaly detection.
- MLP reconstruction-error anomaly detection.
- Graph message-passing station risk using the Phase 9 knowledge graph.
- Failure trajectory prediction using station, timing, waiting, path, and family features.

Key files:

- `src/data/phase10_advanced_ai.py`
- `notebooks/phase10_advanced_ai.ipynb`
- `reports/phase10_advanced_ai_report.md`
- `reports/phase10_advanced_ai_model_comparison.csv`
- `reports/phase10_station_message_passing_risk.csv`
- `data/processed/phase10_validation_advanced_ai_scores.csv`
- `data/processed/phase10_test_preview_advanced_ai_scores.csv`

Important interpretation: Phase 10 scores are diagnostic add-ons. The production-safe Phase 6 LightGBM model remains the main failure prediction model unless future factory validation proves one of these advanced methods is more reliable.

## Phase 6 Model Improvement

Additional production-safe improvement experiments were added after Phase 10:

- Tuned LightGBM on the original Phase 6 feature matrix.
- Tuned LightGBM with Phase 10 graph and trajectory features.
- Product-family-aware LightGBM models with fallback rules.
- Validation-optimized blend as a research upper bound.

Key files:

- `src/data/phase6_model_improvement.py`
- `notebooks/phase6_model_improvement.ipynb`
- `reports/phase6_model_improvement_report.md`
- `reports/phase6_model_improvement_summary.csv`
- `data/processed/phase6_improvement_validation_scores.csv`

## Phase 11 - Manufacturing Copilot

Phase 11 consolidates reviewed outputs into SQLite and exposes them through an
interactive Streamlit application with offline natural-language querying.

Build or refresh the database:

```powershell
.\.venv\Scripts\python.exe src\data\phase11_manufacturing_copilot.py
```

Start the application:

```powershell
.\.venv\Scripts\streamlit.exe run app\phase11_manufacturing_copilot.py
```

Key files:

- `app/phase11_manufacturing_copilot.py`
- `src/copilot/query_engine.py`
- `src/data/phase11_manufacturing_copilot.py`
- `notebooks/phase11_manufacturing_copilot.ipynb`
- `data/database/manufacturing_copilot.db`
- `reports/phase11_manufacturing_copilot_report.md`

## Phase 12 - Executive Dashboard

Phase 12 provides an executive Streamlit dashboard for quality KPIs, relative
failure trends, station heatmaps, bottlenecks, SHAP explanations, and
assumption-driven business impact.

Prepare or refresh the executive tables:

```powershell
.\.venv\Scripts\python.exe src\data\phase12_executive_dashboard.py
```

Start the dashboard:

```powershell
.\.venv\Scripts\streamlit.exe run app\phase12_executive_dashboard.py
```

Key files:

- `app/phase12_executive_dashboard.py`
- `src/dashboard/business_impact.py`
- `src/data/phase12_executive_dashboard.py`
- `notebooks/phase12_executive_dashboard.ipynb`
- `reports/phase12_executive_dashboard_report.md`


## Unified Project Streamlit Dashboard

The unified dashboard brings the full project into one Streamlit application:
KPIs, model comparison, product families, process mining, root cause, knowledge
graph, copilot Q&A, business impact, and final deliverables.

Start the dashboard:

```powershell
.\.venv\Scripts\streamlit.exe run app\project_streamlit_dashboard.py
```

Key file:

- `app/project_streamlit_dashboard.py`
