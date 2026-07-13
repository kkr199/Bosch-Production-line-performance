# Phase 11: Manufacturing Copilot

## Delivered Capabilities

1. Model and analytics outputs are consolidated in SQLite.
2. A Streamlit app provides production summaries, risk review, root-cause evidence,
   bottleneck analysis, process-graph insights, and natural-language questions.
3. Natural-language answers use reviewed intent templates and parameterized SQL.
4. Root-cause answers distinguish model associations from confirmed physical causes.
5. The production-safe Phase 6 LightGBM remains the official failure-risk model.

## Database

- Path: `data/database/manufacturing_copilot.db`
- Tables: 14
- Product prediction rows: 1,240,468
- Test population: all rows from `phase6_test_predictions.csv`
- Validation population: official Phase 6 validation set with known outcomes

## Natural-Language Examples

- Which stations have the highest failure rate?
- What are the top bottlenecks?
- Why is L3_S32 risky?
- Show the most critical process nodes.
- What are the likely failure propagation routes?
- How good is the production-safe model?
- Summarize production performance.

## Governance Notes

- The question engine is deterministic and offline; it does not send factory data to
  an external language model.
- SHAP, graph routes, and anomaly scores are diagnostic evidence, not causal proof.
- Phase 10 advanced-AI scores are supporting signals and do not replace the official
  Phase 6 probability.
- Throughput efficiency is a timestamp-derived proxy and must not be presented as OEE.
