# Problem Definition Charter — Bosch Production Line Performance

## Decision to improve

Prioritise products for engineering review before they leave the manufacturing
process. The model is a decision-support tool: it must not automatically reject
or release a product.

## Quantified objective and operating point

Reduce avoidable downstream quality-investigation effort by presenting a
capacity-constrained daily queue of the products most likely to fail. The
business owner must set the review capacity, false-positive handling cost, and
missed-failure cost before deployment. Until that sign-off exists, this project
is a research prototype and not a production release.

## ML framing

| Axis | Decision |
| --- | --- |
| Learning paradigm | Supervised |
| Task type | Binary classification: `Response` failure indicator |
| Inference mode | Batch scoring after the required station/timing data is available |
| Training mode | Offline training with scheduled and drift-triggered refreshes |

## Success measures

| Measure | Definition | Release gate |
| --- | --- | --- |
| Business | Estimated avoided investigation cost minus review cost | Defined and approved by the business owner |
| Technical | PR-AUC for ranking under severe class imbalance | Must beat the rules-only and logistic-regression baselines on the locked holdout |
| Operating | Precision@K and recall@K | K equals the agreed daily review capacity |
| Safety | False-negative rate for critical failures | Tolerance set by Quality Engineering |

Accuracy is explicitly not a release metric because failures are rare.

## Baselines and alternatives

The release comparison includes (1) a rules-only queue, (2) a logistic
regression pipeline, and (3) the selected tree-based model, all evaluated on
the same locked holdout. A rules-only alternative remains valid when it meets
the operating target with lower operational cost.

## Constraints

- Predictions are advisory and require a human quality-engineer review.
- Features must be available at the declared batch-scoring time; no
  post-outcome or downstream-quality features are permitted.
- Raw production data, model artefacts, and prediction logs are access
  controlled; identifiers are not shown in business dashboards.
- The production owner, review capacity, latency SLO, and cost matrix remain
  open items until documented sign-off is added to this charter.

## Approval record

| Role | Name | Decision | Date |
| --- | --- | --- | --- |
| Business owner | _TBD_ | _TBD_ | _TBD_ |
| Quality Engineering | _TBD_ | _TBD_ | _TBD_ |
| Data/ML owner | _TBD_ | _TBD_ | _TBD_ |
