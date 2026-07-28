**PART X**

Dashboard Guide

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Chapters 68–75**

Overview • Prediction Model • Product Families • Process Mining • Explainability • Knowledge Graph • Business Impact • Manufacturing Copilot

![Image: image38.png](data:image/png;base64...)

*Unified dashboard architecture used in the Bosch manufacturing analytics project.*

**Bosch Production Line Performance — Technical Handbook**

A page-by-page operating guide for technical and non-technical reviewers

# Contents and Quick Start

| **Chapter** | **Dashboard view** | **Primary question** |
| --- | --- | --- |
| 68 | Overview | What is the current project-level quality and scoring position? |
| 69 | Prediction Model | Which production-safe model was selected and how does it perform? |
| 70 | Product Families | Which route-based product groups have different risk patterns? |
| 71 | Process Mining | Where are the strongest waiting-gap and bottleneck signals? |
| 72 | Explainability | Which predictive signals drive the model and what should engineers verify? |
| 73 | Knowledge Graph | Which stations are structurally and operationally critical? |
| 74 | Business Impact | Under explicit assumptions, what value could an intervention create? |
| 75 | Manufacturing Copilot | How can a reviewer ask project questions in plain English? |

## Start the unified dashboard

|  |
| --- |
| # Build or refresh the local SQLite evidence layer .\.venv\Scripts\python.exe src\data\phase11\_manufacturing\_copilot.py  # Optional: refresh executive dashboard tables and manifests .\.venv\Scripts\python.exe src\data\phase12\_executive\_dashboard.py  # Launch the unified Streamlit application .\.venv\Scripts\streamlit.exe run app\project\_streamlit\_dashboard.py |

![Image: image39.png](data:image/png;base64...)

*Figure X.1 — Recommended sequence for refreshing evidence and launching the dashboard.*

|  |
| --- |
| **Prerequisite:** The unified app stops if `data/database/manufacturing\_copilot.db` is missing. Refresh Phase 11 before launching after a clean checkout or after changing source reports. |

# Navigation and Evidence Boundaries

![Image: image40.png](data:image/png;base64...)

*Figure X.2 — Sidebar navigation, shared line filter and the main evidence boundary.*

| **Control or element** | **What it changes** | **What it does not change** |
| --- | --- | --- |
| Production-line multiselect | Filters line-aware bottleneck, root-cause and graph tables. | It does not retrain the model or change historical outcomes. |
| Sidebar page selector | Moves among analytical views. | It does not create a different data snapshot. |
| Cached SQLite queries | Improve interactive performance. | They do not replace the need to refresh the database when reports change. |
| CSV/JSON loaders | Bring model-improvement, family and manifest outputs into the app. | They do not validate source freshness automatically. |
| Warnings and captions | State interpretation boundaries. | They are not optional footnotes; they are part of the decision context. |

|  |
| --- |
| **Reading order:** Begin with Overview, open the relevant diagnostic page, inspect the supporting table, and then use Business Impact or Copilot only after the technical evidence and caveats are understood. |

**CHAPTER 68**

# Overview

*A summary-first page for quality, model, scoring volume and historical movement*

## Purpose of the overview page

The Overview page gives executives, mentors and plant stakeholders a stable first view of project performance. It combines the historical train failure rate, validation performance of the selected production-safe model, unlabelled test scoring volume, predicted alert volume, relative-time failure movement and line-level historical failure rates.

![Image: image41.png](data:image/png;base64...)

*Figure 68.1 — The five headline KPI cards displayed at the top of the overview page.*

| **KPI** | **Repository value** | **How to interpret it** |
| --- | --- | --- |
| Historical failure rate | 0.581% | Observed train-label baseline; not a live factory rate. |
| Validation MCC | 0.339 | Primary class-imbalance-aware model-selection metric. |
| Validation precision | 57.4% | Share of validation alerts that were true failures. |
| Test products scored | 1,183,748 | Unlabelled Kaggle test rows scored by the selected model. |
| Predicted alerts | 6,998 | Rows above the selected probability threshold; outcomes are unknown. |

## Failure movement and line comparison

![Image: image42.png](data:image/png;base64...)

*Figure 68.2 — Historical failure rate across 20 ordered relative production-time periods.*

![Image: image43.png](data:image/png;base64...)

*Figure 68.3 — Failure rate among products with recorded measurements on each production line.*

|  |
| --- |
| **Time-axis warning:** The period labels P01–P20 are ordered relative Bosch production-time segments, not calendar months, weeks or shifts. The chart is suitable for movement detection, not date-specific reporting. |

## Recommended review sequence

1. Check whether historical failure movement is stable or concentrated in specific relative periods.

2. Compare line rates, but remember that products can appear on more than one line.

3. Confirm that the official model card still lists LightGBM as the production-safe model.

4. Treat the alert card as workload estimation, not as confirmed defects.

5. Use the source summary table to identify the repository artifact behind each KPI.

**Repository evidence:** `app/project\_streamlit\_dashboard.py`, `reports/phase12\_executive\_dashboard\_report.md`, and `reports/phase12\_executive\_dashboard\_manifest.json`.

**CHAPTER 69**

# Prediction Model

*Comparing production-safe classifiers and understanding the selected operating point*

## What the page displays

* Selected model name and validation MCC, precision, recall and PR-AUC.
* MCC comparison across Logistic Regression, Random Forest, XGBoost, CatBoost and LightGBM.
* Precision-recall trade-off with threshold information.
* Production-safe improvement experiments.
* Advanced-AI diagnostics that remain supporting signals rather than the official model.

![Image: image44.png](data:image/png;base64...)

*Figure 69.1 — LightGBM leads the production-safe validation comparison.*

| **Rank** | **Model** | **Threshold** | **MCC** | **Precision** | **Recall** | **PR-AUC** |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | LightGBM | 0.785 | 0.339 | 0.574 | 0.213 | 0.253 |
| 2 | XGBoost | 0.748 | 0.336 | 0.567 | 0.213 | 0.247 |
| 3 | CatBoost | 0.797 | 0.328 | 0.557 | 0.207 | 0.226 |
| 4 | Random Forest | 0.674 | 0.325 | 0.552 | 0.205 | 0.239 |
| 5 | Logistic Regression | 0.897 | 0.305 | 0.520 | 0.193 | 0.165 |

## Operating trade-off

![Image: image45.png](data:image/png;base64...)

*Figure 69.2 — The selected models occupy similar precision-recall regions; MCC and PR-AUC separate the ranking.*

|  |
| --- |
| **Selection rule:** The dashboard keeps the Phase 6 LightGBM as the official production-safe model. Validation-optimized blends and competition-style leakage features are research-only and must not replace the governed model. |

## Questions to ask before changing the threshold

| **Question** | **Why it matters** |
| --- | --- |
| How many products can quality teams review? | The threshold determines alert workload. |
| What is the false-negative tolerance? | Higher recall can require more inspections and lower precision. |
| Are probabilities calibrated? | A ranking model may not provide literal probability estimates. |
| Was the threshold chosen on validation only? | Holdout outcomes must remain isolated from selection decisions. |
| Will the feature set exist at scoring time? | Unavailable future or neighbour outcomes create leakage. |

**Repository evidence:** `reports/phase6\_model\_comparison\_metrics.csv`, `reports/phase6\_model\_improvement\_summary.csv`, and `reports/phase10\_advanced\_ai\_model\_comparison.csv`.

**CHAPTER 70**

# Product Families

*Comparing route-based clusters by size, complexity and historical failure rate*

## What a family represents

A product family is a cluster of products that share similar station-presence patterns. The dashboard uses eight reviewed families to summarize route diversity without implying that family membership is a physical defect cause.

![Image: image46.png](data:image/png;base64...)

*Figure 70.1 — Family 4 has the highest observed failure rate, closely followed by Family 1.*

| **Family** | **Products** | **Failure rate** | **Lift** | **Avg stations** | **Avg lines** |
| --- | --- | --- | --- | --- | --- |
| 0 | 252,037 | 0.554% | 0.95× | 13.21 | 2.15 |
| 1 | 132,878 | 0.733% | 1.26× | 7.87 | 2.89 |
| 2 | 237,283 | 0.521% | 0.90× | 13.22 | 2.15 |
| 3 | 125,592 | 0.552% | 0.95× | 13.19 | 2.09 |
| 4 | 128,104 | 0.737% | 1.27× | 7.92 | 2.90 |
| 5 | 99,296 | 0.537% | 0.92× | 13.19 | 2.09 |
| 6 | 59,988 | 0.507% | 0.87× | 15.45 | 2.13 |
| 7 | 148,569 | 0.538% | 0.93× | 13.21 | 2.14 |

## Path complexity and family risk

![Image: image47.png](data:image/png;base64...)

*Figure 70.2 — Bubble size represents family population; labels identify family number.*

|  |
| --- |
| **Common mistake:** Do not conclude that a larger station count causes lower or higher risk. Family risk can reflect product mix, routing, measurement availability and unobserved process conditions. |

## How to use the page

1. Use the KPI cards to identify the highest-risk family and its population.

2. Compare failure rate with family size before prioritizing an investigation.

3. Use average station and line counts to understand route complexity.

4. Open the family-profile expander to review station-presence patterns.

5. Move to Process Mining or Knowledge Graph to examine the stations shared by high-risk families.

**Repository evidence:** `reports/phase5\_product\_family\_failure\_rates.csv` and `reports/phase5\_product\_family\_profiles.csv`.

**CHAPTER 71**

# Process Mining

*Reviewing timestamp-gap proxies, bottleneck priorities and candidate propagation routes*

## Page structure

* Top bottleneck KPI cards for station, score, average gap, P90 gap and failure rate.
* Bubble chart connecting observed gap, bottleneck score, volume and failure rate.
* Top-15 bottleneck table.
* Train/test throughput-efficiency table.
* Candidate failure-propagation route table.

![Image: image48.png](data:image/png;base64...)

*Figure 71.1 — The dashboard emphasizes both operational gap severity and product exposure.*

![Image: image49.png](data:image/png;base64...)

*Figure 71.2 — L1\_S25 is the highest-ranked bottleneck, while L1\_S24 combines high bottleneck and elevated failure evidence.*

## Read bottleneck and quality evidence separately

| **Station** | **Bottleneck score** | **Avg gap** | **P90 gap** | **Failure rate** | **Failure lift** |
| --- | --- | --- | --- | --- | --- |
| L1\_S25 | 88.3 | 30.31 | 36.50 | 0.507% | 0.90× |
| L1\_S24 | 74.4 | 23.25 | 41.01 | 0.828% | 1.48× |
| L3\_S38 | 70.6 | 22.46 | 32.77 | 0.781% | 1.39× |
| L0\_S13 | 59.0 | 18.25 | 31.58 | 0.547% | 0.97× |
| L2\_S28 | 57.2 | 7.88 | 13.44 | 0.699% | 1.25× |
| L2\_S27 | 57.1 | 6.27 | 12.23 | 0.681% | 1.21× |
| L2\_S26 | 55.5 | 5.49 | 9.86 | 0.747% | 1.33× |
| L0\_S12 | 55.5 | 21.78 | 28.63 | 0.547% | 0.97× |
| L3\_S29 | 54.3 | 5.79 | 9.25 | 0.585% | 1.04× |
| L3\_S39 | 51.8 | 6.17 | 8.51 | 0.506% | 0.90× |

|  |
| --- |
| **Proxy boundary:** Average and P90 waiting values are positive inter-station timestamp-gap proxies. They are not confirmed queue times. Throughput efficiency is not OEE. |

## Candidate routes

| **Rank** | **Route** | **Score** | **Minimum transition volume** | **Mean transition failure** |
| --- | --- | --- | --- | --- |
| 1 | L3\_S29 → L3\_S30 → L3\_S33 → L3\_S34 | 51.72 | 965,844 | 0.507% |
| 2 | L3\_S29 → L3\_S30 → L3\_S33 → L3\_S37 | 50.81 | 654,274 | 0.520% |
| 3 | L1\_S24 → L3\_S29 → L3\_S30 → L3\_S33 | 50.28 | 25,747 | 0.744% |
| 4 | L1\_S24 → L2\_S26 → L3\_S29 → L3\_S30 → L3\_S33 → L3\_S34 | 50.09 | 243,524 | 0.630% |
| 5 | L1\_S24 → L2\_S26 → L3\_S29 → L3\_S30 | 50.07 | 243,524 | 0.734% |

**Repository evidence:** `reports/phase8\_bottleneck\_scores.csv`, `reports/phase8\_throughput\_efficiency.csv`, and `reports/phase9\_failure\_propagation\_routes.csv`.

**CHAPTER 72**

# Explainability

*Translating model behavior into a prioritized engineering evidence review*

## What the page explains

The Explainability page displays global SHAP predictive signals, station-level root-cause evidence and a reviewed engineer action plan. It explains what influenced the model, not what physically caused a product failure.

![Image: image50.png](data:image/png;base64...)

*Figure 72.1 — Timing and line-level temporal indicators dominate the global SHAP ranking.*

![Image: image51.png](data:image/png;base64...)

*Figure 72.2 — Global SHAP evidence aggregated by reviewed signal type.*

## Top signals and safe language

| **Rank** | **Predictive signal** | **Signal type** | **Mean |SHAP|** | **Area** |
| --- | --- | --- | --- | --- |
| 1 | Earliest Measurement Timestamp | Timing or delay signal | 0.1970 | timing\_level |
| 2 | Line 0 Earliest Measurement Timestamp | Line-level timing signal | 0.1145 | line\_level |
| 3 | Line 3 Latest Measurement Timestamp | Line-level timing signal | 0.0947 | line\_level |
| 4 | Line 3 Earliest Measurement Timestamp | Line-level timing signal | 0.0829 | line\_level |
| 5 | Latest Measurement Timestamp | Timing or delay signal | 0.0763 | timing\_level |
| 6 | L3\_S32\_F3850\_\_is\_missing | Missingness / skipped measurement signal | 0.0656 | L3\_S32 |
| 7 | mean\_waiting\_time | Timing or delay signal | 0.0478 | timing\_level |
| 8 | line\_0\_end\_time | Line-level timing signal | 0.0461 | line\_level |
| 9 | max\_waiting\_time | Timing or delay signal | 0.0438 | timing\_level |
| 10 | cycle\_time | Timing or delay signal | 0.0432 | timing\_level |

|  |
| --- |
| **Required wording:** Use `predictive signal`, `association` and `recommended check`. Avoid saying that SHAP proved a root cause, that a timestamp feature is a verified delay, or that missingness is automatically a sensor failure. |

## Engineer review pattern

1. Read the top global signal and its feature type.

2. Check the station-level root-cause table for concentrated SHAP evidence.

3. Review the recommended action next to the operational bottleneck evidence.

4. Validate the signal against tooling, calibration, maintenance, routing and sensor records.

5. Document whether the evidence supports correlation, mechanism or controlled causal proof.

**Repository evidence:** `reports/phase7\_top\_failure\_drivers.csv`, `reports/phase7\_station\_root\_cause\_report.csv`, and `reports/phase7\_engineer\_action\_plan.csv`.

**CHAPTER 73**

# Knowledge Graph

*Combining structural centrality, bottleneck severity and failure evidence*

## Page purpose

The Knowledge Graph page identifies stations that are important for more than one reason. The critical-node score combines graph position, bottleneck severity, historical failure lift, station-level SHAP evidence and product volume.

![Image: image52.png](data:image/png;base64...)

*Figure 73.1 — L1\_S24 is the highest-ranked critical graph node.*

| **Rank** | **Station** | **Line** | **Critical score** | **Centrality** | **Bottleneck** | **Failure rate** |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | L1\_S24 | L1 | 60.1 | 0.091 | 74.4 | 0.828% |
| 2 | L3\_S29 | L3 | 53.0 | 0.640 | 54.3 | 0.585% |
| 3 | L3\_S32 | L3 | 50.9 | 0.127 | 40.2 | 4.506% |
| 4 | L3\_S30 | L3 | 47.3 | 0.502 | 50.2 | 0.585% |
| 5 | L3\_S34 | L3 | 46.0 | 0.612 | 34.2 | 0.513% |
| 6 | L3\_S33 | L3 | 44.6 | 0.507 | 41.4 | 0.498% |
| 7 | L3\_S39 | L3 | 44.2 | 0.500 | 51.8 | 0.506% |
| 8 | L1\_S25 | L1 | 42.8 | 0.068 | 88.3 | 0.507% |
| 9 | L3\_S37 | L3 | 42.6 | 0.559 | 26.9 | 0.585% |
| 10 | L0\_S13 | L0 | 39.6 | 0.241 | 59.0 | 0.547% |

## Structural influence versus operational constraint

![Image: image53.png](data:image/png;base64...)

*Figure 73.2 — Bubble size represents historical station failure rate.*

|  |
| --- |
| **Interpretation:** A structurally central station can be important because many high-volume routes pass through it. A high bottleneck score indicates operational gap severity. A high critical-node score combines these with failure evidence; none of the scores alone proves causation. |

## Best uses of the page

* Prioritize cross-functional investigations that involve routing, capacity and quality.
* Identify bridge stations whose disruption could affect several route communities.
* Compare high-risk stations with high-volume central stations.
* Select candidate routes for deeper Process Mining review.
* Use graph findings to organize evidence collection, not assign automatic blame.

**Repository evidence:** `reports/phase9\_critical\_nodes.csv`, `reports/phase9\_station\_centrality\_metrics.csv`, and `reports/phase9\_failure\_propagation\_routes.csv`.

**CHAPTER 74**

# Business Impact

*Converting model and workflow assumptions into a transparent scenario*

## Inputs controlled by the reviewer

| **Input** | **Default in dashboard** | **Meaning** |
| --- | --- | --- |
| Production volume | 1,183,748 | Population to which the scenario is applied. |
| Cost per failure | $500 | Assumed avoidable economic cost for one failure. |
| Cost per alert review | $20 | Assumed manual or automated review cost. |
| Intervention effectiveness | 25% | Share of true-positive alerts converted into prevented failures. |
| Failure rate | 0.581% | Historical train baseline supplied by the evidence table. |
| Alert rate | 0.591% | Selected-model alert share on the unlabelled test population. |
| Precision | 0.574 | Validation precision of the production-safe model. |

|  |
| --- |
| expected\_failures = volume × failure\_rate expected\_alerts = volume × alert\_rate true\_positive\_alerts = min(expected\_failures, expected\_alerts × precision) potentially\_prevented = true\_positive\_alerts × intervention\_effectiveness net\_impact = potentially\_prevented × failure\_cost − expected\_alerts × review\_cost |

![Image: image54.png](data:image/png;base64...)

*Figure 74.1 — Worked scenario using the dashboard defaults.*

## Worked scenario output

| **Output** | **Illustrative value** |
| --- | --- |
| Expected failures | 6,879 |
| Expected alerts | 6,998 |
| Expected true-positive alerts | 4,019 |
| Potentially prevented failures | 1,005 |
| Gross avoided failure cost | $502,399 |
| Alert review cost | $139,960 |
| Net estimated impact | $362,439 |
| Estimated ROI on review cost | 2.59× |

![Image: image55.png](data:image/png;base64...)

*Figure 74.2 — Net impact changes materially with the assumed intervention effectiveness.*

|  |
| --- |
| **Scenario, not savings:** The dashboard does not claim realized savings. Costs, effectiveness, precision transfer, operational capacity and model stability must be validated in the target factory. |

**Repository evidence:** `src/dashboard/business\_impact.py` and `app/project\_streamlit\_dashboard.py`.

**CHAPTER 75**

# Manufacturing Copilot

*Using an offline question-answering layer grounded in reviewed project artifacts*

## How the Copilot works

The unified dashboard includes an offline project Q&A agent. It routes common questions to curated explanations and uses TF-IDF retrieval over local reports and selected tables for supporting evidence. It requires no external language-model API and does not send factory data outside the local application.

![Image: image56.png](data:image/png;base64...)

*Figure 75.1 — Curated and retrieval-based answers share the same local evidence layer.*

| **Mode** | **Strength** | **Typical use** |
| --- | --- | --- |
| Curated topic answer | Consistent non-technical explanation with project-specific decisions. | Families, model choice, bottlenecks, leakage boundaries, advanced AI and dashboard overview. |
| Evidence retrieval | Searches report chunks and table previews using TF-IDF similarity. | Questions not covered by a curated explainer. |
| Legacy SQL template | Deterministic parameterized queries over the SQLite database. | Station risk, bottlenecks, routes, model metrics and production summary. |

## Example interaction and evidence review

![Image: image57.png](data:image/png;base64...)

*Figure 75.2 — Example answer with an explicit proxy and causation boundary.*

## Recommended questions

* Explain the eight product families in non-technical language.
* Why was LightGBM selected as the production-safe model?
* What are the main predictive signals linked to failure risk?
* What is the biggest bottleneck and why does it matter?
* Why is L3\_S32 risky?
* What are the most critical process nodes?
* What are the leading candidate failure-propagation routes?
* Explain the whole project to a plant manager.

|  |
| --- |
| **Copilot limitation:** The Copilot summarizes reviewed evidence. It does not observe live production, create new causal proof, approve corrective action or replace engineering judgement. |

## Evidence quality checklist

1. Read the offline topic route shown below the answer.

2. Open the supporting-evidence rows and verify that the source matches the question.

3. Prefer deterministic SQL-template answers for exact ranked tables.

4. Treat low retrieval match scores as a sign to refine the question.

5. Escalate decisions to the relevant dashboard page and source report.

**Repository evidence:** `src/copilot/offline\_agent.py`, `src/copilot/query\_engine.py`, and `reports/phase11\_manufacturing\_copilot\_report.md`.

# Dashboard Operations, Troubleshooting and Final Checklist

## Common issues

| **Symptom** | **Likely cause** | **Action** |
| --- | --- | --- |
| App stops with missing database | `manufacturing\_copilot.db` has not been built. | Run the Phase 11 database builder, then relaunch Streamlit. |
| A page shows stale values | SQLite or CSV/JSON artifacts were not refreshed. | Regenerate the upstream phase and rebuild the database. |
| Line filter returns no rows | The chosen page/table has no matching line evidence. | Restore all lines and check the source table. |
| Charts look correct but conclusions conflict | Different views answer different questions. | Separate prediction, bottleneck, graph and historical-rate evidence. |
| Copilot answer is too general | The question was routed to retrieval with weak overlap. | Use a station ID or a named topic such as model, bottleneck or family. |
| Business impact looks unexpectedly high | Cost or effectiveness assumptions dominate. | Review inputs and run sensitivity scenarios. |

## Final review checklist

1. Confirm the dashboard database and report manifests were refreshed from the same repository state.

2. Verify the Overview cards before presenting diagnostic pages.

3. Keep the production-safe LightGBM separate from research-only models.

4. Use family, bottleneck, SHAP and graph pages as complementary evidence.

5. Label test alerts as predictions with unknown outcomes.

6. Label timestamp measures as relative proxies and throughput efficiency as non-OEE.

7. Present Business Impact as an assumption-driven scenario.

8. Use the Copilot to navigate evidence, not to authorize actions.

## Primary dashboard files

| **File** | **Role** |
| --- | --- |
| app/project\_streamlit\_dashboard.py | Unified Streamlit interface and page navigation. |
| src/data/phase11\_manufacturing\_copilot.py | Builds the SQLite evidence database. |
| src/copilot/query\_engine.py | Reviewed parameterized SQL answers. |
| src/copilot/offline\_agent.py | Curated explanations and local TF-IDF retrieval. |
| src/dashboard/business\_impact.py | Transparent business-impact scenario calculations. |
| reports/phase11\_database\_manifest.json | Table and row-count audit. |
| reports/phase12\_executive\_dashboard\_manifest.json | Executive KPI audit. |

|  |
| --- |
| **Final principle:** The dashboard is a governed communication layer over reviewed project artifacts. Its value comes from making evidence understandable while preserving the limits of a public benchmark dataset. |