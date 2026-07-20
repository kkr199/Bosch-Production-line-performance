# Monitoring and Retraining Plan

## Monitoring contract

| Signal | Threshold | Owner | First response |
| --- | --- | --- | --- |
| Input schema or feature availability | Any missing required field or unexpected type | Data Engineering | Stop scoring; inspect upstream release; restore schema contract |
| Feature drift | PSI > 0.20 for a priority feature | ML owner | Check source and units; compare against training reference |
| Prediction drift | PSI > 0.20 for score distribution | ML owner | Inspect queue volume, feature drift, and threshold suitability |
| Operational health | p95 batch duration exceeds agreed SLO or scoring error rate > 1% | Platform owner | Roll back to the last registered artefact and investigate logs |
| Label-lagged performance | PR-AUC or precision@K below approved release floor | ML + Quality | Pause promotion; perform error analysis and consider retraining |

Training reference distributions, prediction logs, schema results, model
version, data-window ID, and code commit must be retained together so any score
is traceable.

## Retraining policy

Retraining is monthly when mature labels are available and also on confirmed
drift. A challenger is promoted only when data validation passes and it beats
both the incumbent and the approved minimum threshold on a recent locked
holdout. Failed challengers alert the ML owner; the incumbent stays live.

## Release and rollback

1. Validate schema and data freshness.
2. Run the reproducible training pipeline with recorded dependency versions and seed.
3. Evaluate the challenger against the incumbent on the locked holdout.
4. Shadow-score first, then use a staged rollout after business approval.
5. Roll back to the prior registered artefact immediately on an operational or
   quality guardrail breach.

The named owners, SLO values, review cadence, and registry location are
deployment-specific fields that must be completed before production release.
