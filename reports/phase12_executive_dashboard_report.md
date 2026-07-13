# Phase 12: Executive Dashboard

## Executive Purpose

The dashboard provides a summary-first view of manufacturing quality, failure
movement, station risk, bottlenecks, model explanations, and scenario-based
business impact.

## Headline Baseline

- Historical failure rate: 0.581%
- Highest-risk station: `L3_S32`
- Highest bottleneck station: `L1_S25`
- Test products scored: 1,183,748
- Model alerts on test population: 6,998

## Dashboard Views

1. Executive overview and KPI scorecard.
2. Failure trend across ordered relative production-time periods.
3. Station heatmap by line and station.
4. Bottleneck analytics using waiting time, failure rate, and bottleneck score.
5. SHAP driver explanations from the production-safe Phase 6 model.
6. Business-impact scenario using user-controlled costs and intervention effectiveness.

## Interpretation Boundaries

- The failure trend uses relative Bosch production timestamps, not calendar dates.
- Test predictions do not have known outcomes.
- Business impact is a scenario, not realized savings.
- SHAP values explain model behavior and do not prove physical causation.
- Throughput efficiency is a timestamp-derived proxy and is not OEE.
