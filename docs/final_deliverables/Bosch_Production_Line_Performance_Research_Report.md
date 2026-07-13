# Manufacturing Failure Prediction and Process Intelligence for Bosch Production-Line Data

**Author:** Nexturn Manufacturing Analytics Use Case  
**Document type:** IEEE-style technical research report  
**Generated:** 2026-07-08 11:07  

## Abstract

This project develops an end-to-end manufacturing analytics system using the Bosch Production Line Performance dataset. The system combines failure prediction, product-family segmentation, process mining, root-cause analysis, knowledge-graph analytics, anomaly detection, a manufacturing copilot, and an executive dashboard. The production-safe model is a LightGBM classifier trained from raw numeric, categorical, and date-derived features. On the held-out validation set, the selected model achieved MCC=0.339, precision=0.574, recall=0.213, F1=0.311, and PR-AUC=0.253. The analysis identifies L3_S32 as the highest observed risk station with 4.506% failure rate, L1_S25 as the top bottleneck, and L1_S24 as the top knowledge-graph critical node. The final system is packaged as reusable Python scripts, notebooks, model artifacts, a SQLite-backed copilot, and a Streamlit executive dashboard.

**Keywords:** manufacturing analytics, failure prediction, process mining, root-cause analysis, knowledge graph, SHAP, LightGBM, Bosch production-line data.

## I. Introduction

Modern production lines generate high-dimensional sensor, timestamp, and routing data. The business objective is not only to classify likely failures, but also to explain where quality risk concentrates, how products flow through the process, and which operational interventions should be prioritized. This project converts the Bosch dataset into a reusable manufacturing analytics architecture for future client engagements.

The implemented system covers the complete project roadmap: project setup, data understanding, exploratory analysis, feature engineering, product-family discovery, predictive modeling, root-cause analysis, process mining, knowledge graph construction, advanced AI experiments, a manufacturing copilot, and an executive dashboard.

## II. Dataset and Scope

The raw Bosch data includes numeric, categorical, and date files for train and test populations. The training numeric file contains the target variable `Response`; test labels are unavailable. The project uses raw CSV data for modeling and analysis, while selected processed files are created as reproducible phase outputs.

- Labeled train products: 1,183,747
- Train failures: 6,879
- Historical failure rate: 0.581%
- Test products scored: 1,183,748
- Validation products used for model assessment: 56,720
- Validation failures: 1,720

The Kaggle leaderboard-style leakage features are documented separately as research-only. They are not used for production-safe modeling, root-cause explanation, dashboards, or copilot outputs because nearby known labels would not be available in live deployment.

## III. Methodology

### A. Data Quality and Metadata Engineering

Phase 1 loaded raw numeric, categorical, date, and target datasets and documented dataset sizes, feature counts, missing values, and file-level quality checks. Phase 2 parsed feature names into production line, station, and feature identifiers, producing station metadata and completeness summaries.

### B. Exploratory Analysis

Phase 3 calculated failure rates by line and station, analyzed time-derived failure patterns, generated correlation and distribution reports, and used one-hot encoded categorical subsets where required for EDA. The most important station-level observation was that L3_S32 showed a materially elevated failure rate relative to the overall training baseline.

### C. Feature Engineering

Phase 4 constructed product-level timing and path features including start time, end time, cycle time, processing duration, waiting time, station counts, line counts, path complexity, and line-level aggregates. These features created a common interface for predictive modeling, process mining, and dashboarding.

### D. Product Family Discovery

Phase 5 created station-presence matrices and clustered products using KMeans, DBSCAN, and hierarchical clustering. The final segmentation produced 8 product families. Family 4 had the highest observed failure rate at 0.737%, indicating that product routes and family mix are meaningful risk segmentation dimensions.

### E. Predictive Modeling

Phase 6 compared logistic regression, random forest, XGBoost, LightGBM, and CatBoost. The selected production-safe model is LightGBM. It achieved MCC=0.339, precision=0.574, recall=0.213, F1=0.311, and PR-AUC=0.253. Later improvement experiments did not outperform the Phase 6 baseline, so the original LightGBM model remains the official failure prediction model.

### F. Root-Cause Analysis

Phase 7 generated model feature importance and SHAP explanations. The strongest global drivers included timing features, line timing features, station-specific numeric measurements, and missingness indicators. SHAP outputs are treated as model explanations and engineering investigation priorities, not as proof of physical causality.

### G. Process Mining and Bottleneck Analysis

Phase 8 reconstructed production process maps from raw date features, calculated waiting-time metrics, identified bottlenecks, and measured throughput efficiency proxies. L1_S25 ranked first with bottleneck score 88.34. The throughput efficiency metric is a relative timestamp-derived proxy and must not be presented as formal OEE.

### H. Knowledge Graph

Phase 9 created a Line-to-Station-to-Feature-to-Failure graph, built station relationship edges, calculated centrality metrics, identified candidate failure propagation routes, and highlighted critical nodes. The graph contains 4,321 nodes and 4,923 edges. L1_S24 is the top critical node by the combined graph and operational priority score.

### I. Advanced AI Experiments

Phase 10 evaluated Isolation Forest, MLP reconstruction anomaly scoring, graph message-passing station risk, and failure trajectory prediction. These methods were useful as diagnostic add-ons but did not replace the production-safe LightGBM model.

### J. Manufacturing Copilot and Executive Dashboard

Phase 11 stores model outputs and analytics artifacts in SQLite, then exposes them through a Streamlit manufacturing copilot. The copilot database contains 1,240,468 product prediction rows. Phase 12 adds an executive dashboard with KPI cards, relative failure trends, station heatmaps, bottleneck analytics, SHAP summaries, and scenario-based business impact calculations.

## IV. Key Results

### A. The official production-safe model is accurate enough for prioritization, not autonomous disposition.

The LightGBM model achieved MCC=0.339 and precision=0.574. This supports prioritizing inspection and engineering review. It should not be used as an autonomous scrap/release decision without plant-specific validation, monitoring, and operating procedures.

### B. Quality risk is concentrated by station and path.

L3_S32 is the highest observed risk station at 4.506% failure rate. Product family 4 has elevated risk at 0.737%, showing that route and family segmentation add operational context beyond a single global model score.

### C. Bottlenecks and critical nodes are not identical.

L1_S25 is the top bottleneck, while L1_S24 is the top knowledge-graph critical node. This distinction matters: bottleneck score emphasizes waiting and flow constraint, while graph criticality combines network position, failure rate, bottleneck evidence, and root-cause importance.

### D. Executive impact must be scenario-based.

The project includes a business-impact calculator using production volume, historical failure rate, model alert rate, validation precision, cost per failure, cost per alert review, and intervention effectiveness. The calculator is intentionally assumption-driven because the Kaggle dataset does not include plant-specific financial outcomes or measured intervention effects.

## V. System Architecture

The final architecture has five layers:

1. **Raw data and metadata layer:** Bosch raw CSV files, feature parsing, station metadata, line metadata, completeness metrics.
2. **Analytical feature layer:** timing, waiting, station count, path complexity, product-family labels, graph and trajectory features.
3. **Model layer:** production-safe LightGBM model plus benchmark models and research-only leaderboard variant.
4. **Diagnostic intelligence layer:** SHAP explanations, bottleneck scores, knowledge graph, anomaly scores, failure trajectory scores.
5. **Decision interface layer:** SQLite database, Manufacturing Copilot Streamlit app, Executive Dashboard Streamlit app, reports, notebooks, and presentation.

## VI. Deliverable Inventory

| ID | Deliverable | Status | Primary artifact |
|---:|---|---|---|
| 68 | Production Failure Prediction Model | Complete | `models/phase6_best_model.joblib` |
| 69 | Product Family Segmentation Module | Complete | `src/data/phase5_product_family_discovery.py` |
| 70 | Process Mining Engine | Complete | `src/data/phase8_process_mining_bottleneck_analysis.py` |
| 71 | Root Cause Analysis Engine | Complete | `src/data/phase7_root_cause_analysis.py` |
| 72 | Knowledge Graph | Complete | `reports/phase9_manufacturing_knowledge_graph.graphml` |
| 73 | Manufacturing Copilot | Complete | `app/phase11_manufacturing_copilot.py` |
| 74 | Executive Dashboard | Complete | `app/phase12_executive_dashboard.py` |
| 75 | Documentation and Presentation | Complete | `docs/final_deliverables/` |

## VII. Limitations and Robustness Notes

- Test labels are unavailable, so test predictions cannot be evaluated as measured failures.
- Kaggle competition leaderboard leakage strategies are excluded from production deliverables.
- SHAP and graph routes provide diagnostic associations, not causal proof.
- Bosch relative timestamps do not correspond to real calendar dates.
- Financial impact is scenario-based and requires client-specific cost validation.
- A production deployment would require retraining governance, feature drift monitoring, alert workflow design, security review, and integration with plant systems.

## VIII. Recommended Next Steps

1. Convert the current local pipeline into scheduled production jobs for a future client dataset.
2. Replace competition files with governed plant data sources and stable data contracts.
3. Validate the model with forward-time splits and post-alert quality outcomes.
4. Connect the copilot and dashboard to role-based access control.
5. Add monitoring for feature drift, prediction drift, alert precision, recall, and intervention outcomes.
6. Work with manufacturing engineers to confirm root causes using maintenance, sensor, routing, and inspection records.

## IX. Conclusion

The project delivers a complete manufacturing analytics use case rather than a single classifier. The selected production-safe model provides risk prioritization, while segmentation, process mining, root-cause analysis, graph analytics, copilot querying, and executive reporting make the results interpretable and operationally usable. The system is suitable as a reusable reference architecture for future production-line quality analytics projects, subject to client-specific validation and deployment controls.

## References

[1] Bosch Production Line Performance, Kaggle competition dataset.  
[2] LightGBM: A Highly Efficient Gradient Boosting Decision Tree.  
[3] SHAP: A Unified Approach to Interpreting Model Predictions.  
[4] Process mining and event-log analysis methods for production systems.  
[5] Network centrality and graph analytics methods for manufacturing process graphs.
