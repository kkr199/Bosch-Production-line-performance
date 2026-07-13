# Final Deliverables Index

Generated: 2026-07-08 11:07

## Headline Results

- Production-safe model: LightGBM
- Validation MCC: 0.339
- Validation precision: 0.574
- Validation recall: 0.213
- Highest-risk station: L3_S32 (4.506% failure rate)
- Top bottleneck: L1_S25 (score 88.34)
- Knowledge graph: 4,321 nodes, 4,923 edges

## Requested Deliverables

| ID | Deliverable | Status | Primary artifacts |
|---:|---|---|---|
| 68 | Production Failure Prediction Model | Complete | `models/phase6_best_model.joblib`<br>`src/data/phase6_predictive_failure_modeling.py`<br>`notebooks/phase6_predictive_failure_modeling.ipynb`<br>`reports/phase6_predictive_failure_modeling_report.md`<br>`data/processed/phase6_test_predictions.csv` |
| 69 | Product Family Segmentation Module | Complete | `src/data/phase5_product_family_discovery.py`<br>`notebooks/phase5_product_family_discovery.ipynb`<br>`reports/phase5_product_family_discovery_report.md`<br>`data/processed/phase5_train_product_families.csv`<br>`data/processed/phase5_test_product_families.csv` |
| 70 | Process Mining Engine | Complete | `src/data/phase8_process_mining_bottleneck_analysis.py`<br>`notebooks/phase8_process_mining_bottleneck_analysis.ipynb`<br>`reports/phase8_process_mining_bottleneck_report.md`<br>`reports/phase8_process_map_edges.csv`<br>`reports/phase8_bottleneck_scores.csv` |
| 71 | Root Cause Analysis Engine | Complete | `src/data/phase7_root_cause_analysis.py`<br>`notebooks/phase7_root_cause_analysis.ipynb`<br>`reports/phase7_root_cause_analysis_report.md`<br>`reports/phase7_top_failure_drivers.csv`<br>`reports/phase7_station_root_cause_report.csv` |
| 72 | Knowledge Graph | Complete | `src/data/phase9_knowledge_graph.py`<br>`notebooks/phase9_knowledge_graph.ipynb`<br>`reports/phase9_manufacturing_knowledge_graph.graphml`<br>`reports/phase9_knowledge_graph_report.html`<br>`reports/phase9_critical_nodes.csv` |
| 73 | Manufacturing Copilot | Complete | `app/phase11_manufacturing_copilot.py`<br>`src/copilot/query_engine.py`<br>`src/data/phase11_manufacturing_copilot.py`<br>`data/database/manufacturing_copilot.db`<br>`notebooks/phase11_manufacturing_copilot.ipynb` |
| 74 | Executive Dashboard | Complete | `app/phase12_executive_dashboard.py`<br>`src/dashboard/business_impact.py`<br>`src/data/phase12_executive_dashboard.py`<br>`notebooks/phase12_executive_dashboard.ipynb`<br>`reports/phase12_executive_dashboard_report.md` |
| 75 | Project Documentation and Presentation | Complete | `docs/final_deliverables/Bosch_Production_Line_Performance_Research_Report.docx`<br>`docs/final_deliverables/Bosch_Production_Line_Performance_Research_Report.md`<br>`docs/final_deliverables/Bosch_Production_Line_Performance_Presentation.pptx`<br>`docs/final_deliverables/final_deliverables_index.md`<br>`docs/final_deliverables/final_deliverables_manifest.json` |

## Documentation Artifacts

- Research Markdown: `docs\final_deliverables\Bosch_Production_Line_Performance_Research_Report.md`
- Research Word document: `docs\final_deliverables\Bosch_Production_Line_Performance_Research_Report.docx`
- Presentation deck: `docs\final_deliverables\Bosch_Production_Line_Performance_Presentation.pptx`
- Final package archive: `docs\final_deliverables\Bosch_Production_Line_Performance_Final_Deliverables.zip`
- Full 20+ page research paper: `docs\final_deliverables\Bosch_Production_Line_Performance_Full_Research_Paper_20plus.docx`
- Full research paper Markdown: `docs\final_deliverables\Bosch_Production_Line_Performance_Full_Research_Paper_20plus.md`

## Run Commands

```powershell
.\.venv\Scripts\python.exe src\data\phase11_manufacturing_copilot.py
.\.venv\Scripts\python.exe src\data\phase12_executive_dashboard.py
.\.venv\Scripts\streamlit.exe run app\phase11_manufacturing_copilot.py
.\.venv\Scripts\streamlit.exe run app\phase12_executive_dashboard.py
```
