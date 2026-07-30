**TECHNICAL MODEL REVIEW**

**Bosch Full-Data Model Performance**

Metrics, feature-screening decisions, model comparison, and final LightGBM selection

| **Evidence base** | **Selected configuration** |
| --- | --- |
| 1,183,747 labelled products | LightGBM |
| Stratified full-data benchmark | 80/20 train-validation split |
| Five model families | 283 final features |
| Primary selection metric | MCC = 0.2102 |

Prepared as a presentation-ready technical explanation
July 2026

# Executive summary

|  |
| --- |
| **Final recommendation Use** LightGBM from the leakage-safe full-data benchmark: 1,183,747 labelled rows, stratified 80/20 split, minimum numeric present rate 0.0025, 283 final model features, validation MCC 0.2102, and PR-AUC 0.1271. |

The selected result is lower than the earlier reported MCC of 0.3386, but it is more defensible for final prediction because it reflects the natural class distribution, uses the complete labelled population, learns feature-selection rules only from the training partition, and evaluates once on a large untouched validation partition.

## At-a-glance benchmark

| **Field** | **Selected value** | **Interpretation** |
| --- | --- | --- |
| Total labelled rows | 1,183,747 | All available products with known Response |
| Training rows | 946,997 (80%) | Used to learn screening rules and fit the model |
| Validation rows | 236,750 (20%) | Held out for performance measurement |
| Minimum present rate | 0.0025 (0.25%) | Numeric candidate needs about 2,367 observed training values |
| Candidate numeric features | 908 | Passed minimum coverage screening |
| Final model features | 283 | Actual inputs supplied to every compared model |
| Primary metric | MCC = 0.2102 | Best balanced binary-classification score |
| Ranking metric | PR-AUC = 0.1271 | Quality of failure ranking across thresholds |

## How to read this document

* Sections 1-3 explain the dataset, split and feature-screening fields.
* Sections 4-8 explain every performance metric with formulas and worked examples.
* Sections 9-10 compare all model families and justify LightGBM.
* Section 11 explains why the earlier MCC 0.3386 must not be compared without qualification.
* The appendices provide presentation wording and a technical checklist.

# 1. Dataset population and target

Each row represents one manufactured product. The labelled training population contains input measurements and the binary target Response. Response = 0 means the product passed; Response = 1 means it failed. The separate Kaggle test population contains inputs and Id values but does not expose Response, so it is used for final inference rather than local performance measurement.

## 1.1 Practical Bosch variable examples

The following examples use actual column names and committed dataset statistics. They show why variables at different lines and stations cannot be treated as interchangeable, even though their physical engineering meanings remain anonymous.

|  |
| --- |
| **Interpretation rule:** L, S and F/D identify where the variable belongs in the manufacturing hierarchy. They do not reveal the physical sensor name, unit, specification limit, operator, or machine condition. |

|  |  |  |  |
| --- | --- | --- | --- |
| **Variable** | **Type** | **Actual interpretation** | **Practical modelling role** |
| L0\_S1\_F24 | Numeric | Feature 24 at Line 0, Station 1; observed for 673,902 products. | Common measurement with broad coverage. |
| L1\_S24\_F867 | Numeric | Feature 867 at Line 1, Station 24; observed for 12,026 products. | Sparse station-specific value plus missingness signal. |
| L3\_S32\_F3854 | Categorical | Anonymous categories such as T2, T4 and T48 at Line 3, Station 32. | Supported levels are one-hot encoded. |
| L0\_S0\_D1 | Date | Relative timestamp associated with an early Line 0, Station 0 measurement. | Input to earliest/latest timestamp engineering. |
| Response | Target | 0 = pass; 1 = failure. | Training label only; absent from test. |

|  |  |  |
| --- | --- | --- |
| **Dataset field** | **Definition** | **Used by the final benchmark** |
| Id | Unique product identifier. It supports traceability but is excluded from model features. | No - excluded from X |
| Response | Known binary quality outcome: 0 = pass, 1 = failure. | Yes - target y |
| Raw measurements | Anonymous numeric, categorical and date fields identified by line/station/feature notation. | Screened and engineered |
| Labelled rows | Products for which Response is available. | 1,183,747 |
| Unlabelled test rows | Products without published Response. | Final probability and class prediction only |

## Why the complete labelled population matters

The original Phase 6 experiment used a down-sampled modelling population of 226,879 rows: all failures plus 220,000 sampled non-failures. The clean benchmark instead includes all 1,183,747 labelled products, preserving the real operating prevalence. This makes validation harder but closer to how the model will behave when deployed against the full production population.

|  |
| --- |
| **Important distinction Using** 100% of labelled rows for model development does not mean fitting on all rows before evaluation. The labelled population is first split; 80% fits the model and 20% remains untouched for validation. After selection, the chosen pipeline may be refit on all labelled rows for test-set inference. |

# 2. Train-validation split

|  |
| --- |
| **Total labelled rows = Training rows + Validation rows 1,183,747 = 946,997 + 236,750** |

A stratified split preserves approximately the same rare-failure proportion in both partitions. The 80% training partition is used to fit feature-screening rules, preprocessing and model parameters. The 20% validation partition is used only to estimate performance on unseen labelled products.

| **Partition** | **Rows** | **Share** | **Permitted use** |
| --- | --- | --- | --- |
| Training | 946,997 | 80% | Coverage screening, feature ranking, preprocessing fit, model fit, threshold development |
| Validation | 236,750 | 20% | Final comparison: MCC, PR-AUC, precision, recall and confusion matrix |
| Total | 1,183,747 | 100% | Complete labelled population |

## Worked example

Imagine 100 labelled products. An 80/20 split gives 80 training products and 20 validation products. The model may learn only from the 80. Once trained, it predicts the 20 unseen products; their known outcomes allow calculation of the confusion matrix and metrics.

## Why not use the Kaggle test set for evaluation?

A confusion matrix requires an actual label and a predicted label. The Kaggle test file has no published Response, so precision, recall, MCC and PR-AUC cannot be calculated locally. It can still be scored to produce a failure probability and final binary outcome for every test product.

|  |
| --- |
| **Leakage rule Any** feature-selection threshold, category map, imputation value or model setting that learns from labels must be fitted on the training partition only. Letting validation labels influence those decisions makes the reported score optimistic. |

# 3. Present rate and candidate features

## 3.1 Present rate

|  |
| --- |
| **Present rate = non-missing training observations / Training rows** |

Present rate is a data-coverage proportion, not a p-value. The selected threshold is 0.0025, or 0.25%. With 946,997 training rows, a numeric feature must have roughly 2,367 observed values to enter the candidate pool.

|  |
| --- |
| **Minimum observations = 946,997 x 0.0025 = 2,367.49 (about 2,367 products)** |

### Passing example

If feature L1\_S24\_F867 is observed in 3,000 training products, its present rate is 3,000 / 946,997 = 0.00317 = 0.317%. Because 0.317% is greater than 0.25%, it passes coverage screening.

### Failing example

If a feature is observed in only 1,000 training products, its present rate is 1,000 / 946,997 = 0.00106 = 0.106%. Because 0.106% is below 0.25%, it is excluded from the candidate pool.

### 3.1.1 Actual coverage and failure evidence

The next table replaces hypothetical examples with real full-data profiles. The final selection rules were learned on the 80% training partition, but these committed full-population counts clearly show which variables are broadly observed and which are extremely sparse.

Sparse-feature reliability example: L1\_S25\_F2181 has four failures among 1,243 observed products. If only four additional products had failed, its apparent failure rate would rise from 0.3218% to 0.6436%. L1\_S25\_F2712 has only two failures among 2,117 observations, so a few changed outcomes would alter its estimated rate substantially.

|  |
| --- |
| **What this proves:** The coverage gate does not prove that rare variables are useless. It prevents the global model from building raw-value rules on very small evidence bases. Route participation can still be represented through missingness, station, timing, path and product-family features. |

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| **Feature** | **Observed** | **Present rate** | **Failures / passes** | **Observed failure rate** | **Vs. overall 0.578%** |
| L0\_S1\_F24 | 673,902 | 56.9296% | 3,606 / 670,296 | 0.5351% | 0.93x |
| L1\_S24\_F1723 | 66,500 | 5.6178% | 609 / 65,891 | 0.9158% | 1.58x |
| L1\_S24\_F867 | 12,026 | 1.0159% | 75 / 11,951 | 0.6236% | 1.08x |
| L1\_S25\_F2761 | 2,477 | 0.2093% | 19 / 2,458 | 0.7671% | 1.33x |
| L1\_S25\_F2712 | 2,117 | 0.1788% | 2 / 2,115 | 0.0945% | 0.16x |
| L1\_S25\_F2181 | 1,243 | 0.1050% | 4 / 1,239 | 0.3218% | 0.56x |
| **Why such a permissive threshold?** Bosch data is structurally sparse because products follow different routes. A conventional rule that drops every highly missing feature could remove rare but meaningful station signals. The 0.25% value is an empirically benchmarked engineering threshold, not a universal statistical standard. | | | | | | |

## 3.2 Candidate features

Candidate features are raw numeric columns that pass the coverage rule. In the selected run, 908 numeric columns passed. Candidate status does not mean that all 908 were supplied to LightGBM. They entered the next ranking and selection stage.

## 3.3 Practical categorical example: L3\_S32\_F3854

L3\_S32\_F3854 demonstrates why categorical presence and level encoding are handled separately from raw numeric screening. It was observed in 21,582 products (1.8232% coverage). Approximately 1,087 of those products failed, producing a present-group failure rate of 5.0366%, compared with 0.4984% when the feature was missing.

|  |
| --- |
| **Causality caution:** The high failure rate proves association with an anonymous category, not a physical root cause. The dataset does not reveal what T2 or T4 means operationally. |

|  |  |  |
| --- | --- | --- |
| **Evidence** | **Actual value** | **Interpretation** |
| Present products | 21,582 | Products carrying a recorded category at this station feature. |
| Present failures / passes | 1,087 / 20,495 | Strong association in the observed group. |
| Present failure rate | 5.0366% | About 8.7 times the overall 0.578% failure rate. |
| Example supported levels | T4, T2, T48, T16, T128, T512, T8 | Anonymous states retained through one-hot encoding. |
| Missing failure rate | 0.4984% | Products without the category form a much lower-risk group. |

1. Calculate present rate on training rows only.
2. Retain numeric columns meeting present\_rate >= 0.0025.
3. Rank eligible signals using the implemented target-related selection logic on training rows.
4. Build the final merged feature matrix using the selected raw and engineered fields.
5. Apply the frozen feature contract to validation and test rows.

# 4. Final features

|  |
| --- |
| **Final feature count = number of columns actually passed to model.fit(X\_train, y\_train)** |

The selected benchmark uses 283 final features. This is different from the 908 candidate numeric features. Candidate count describes the pool eligible for ranking; final feature count describes the completed model matrix after selection, encoding and engineering.

## 4.1 Practical date and route feature examples

Individual raw D-columns are not presented as verified cycle-time or waiting-time measurements. They are combined into carefully named temporal and route features that describe the observed measurement record.

|  |  |  |  |
| --- | --- | --- | --- |
| **Feature** | **How it is calculated** | **Correct interpretation** | **Do not claim** |
| start\_time | Minimum available D-value for a product. | Earliest recorded measurement timestamp. | Official production start or delay. |
| cycle\_time | Maximum D-value minus minimum D-value. | Observed measurement time span. | Verified factory cycle time. |
| waiting\_time | Sum of positive gaps between ordered station timestamps. | Inter-station timestamp-gap proxy. | Confirmed queue waiting time. |
| station\_count | Stations containing at least one observed date value. | Number of observed station visits. | Complete physical route without validation. |
| path\_complexity\_score | Composite of station count, line count, switching and density. | Relative process-path complexity. | Direct cause of failure. |
| final\_product\_family | Cluster derived from station-presence routes. | One of eight route-based families. | Known Bosch product type. |

|  |  |  |
| --- | --- | --- |
| **Stage** | **Count** | **Meaning** |
| Raw Bosch feature space | Thousands | Original anonymous measurements before screening |
| Numeric candidates | 908 | Numeric columns meeting the 0.25% coverage threshold |
| Final merged features | 283 | Actual selected and engineered inputs supplied to each model |

## Why the count changed from roughly 323/325 to 283

## 4.2 Final 283-feature composition

The final model matrix can be explained as a combination of four feature groups rather than as 283 unrelated columns.

|  |  |  |  |
| --- | --- | --- | --- |
| **Feature group** | **Count** | **Practical example** | **Purpose** |
| Selected numeric values | 80 | L1\_S24\_F867 | Retain the strongest supported raw measurements. |
| Numeric missing indicators | 80 | L1\_S24\_F867\_\_is\_missing | Preserve whether the measurement was recorded. |
| Categorical indicators | 70 | L3\_S32\_F3854\_\_eq\_T2 and \_\_is\_missing | Represent supported anonymous states and absence. |
| Timing, route and family | 53 | start\_time, station\_count, final\_product\_family | Represent manufacturing behaviour and route context. |
| Total | 283 | 80 + 80 + 70 + 53 | Frozen feature contract supplied to every benchmark model. |

The newer leakage-safe pipeline refitted categorical selection and encoding using the training partition only and retained fewer supported one-hot levels. The final count is calculated dynamically from the completed model matrix. A lower count is not inherently worse: performance, stability and reproducibility matter more than retaining the largest possible number of columns.

|  |
| --- |
| **Accurate presentation wording the** final model uses 283 selected and engineered features. The benchmark first screened 908 numeric candidates, then constructed a consistent 283-column merged feature contract for training, validation and test inference. |

## Feature-count controls to verify

* Id and Response are excluded from the feature list.
* Train, validation and test matrices use identical names, order and data types.
* Duplicate columns created during merges are removed.
* Category levels are learned from training rows and unknown validation/test levels are handled safely.
* No future or target-derived feature is available at the intended prediction point.

# 5. Confusion matrix and threshold-dependent metrics

A probability model produces a score between 0 and 1. A decision threshold converts that score into 0 or 1. Every chosen threshold creates a confusion matrix. Precision, recall and MCC therefore depend on the operating threshold; PR-AUC evaluates ranking across many thresholds.

| **Outcome** | **Meaning** | **Operational interpretation** |
| --- | --- | --- |
| True Positive (TP) | Actual failure predicted as failure | Correctly prioritized defect |
| False Positive (FP) | Pass predicted as failure | Unnecessary inspection or alert |
| False Negative (FN) | Failure predicted as pass | Missed defect; often the costliest error |
| True Negative (TN) | Pass predicted as pass | Correctly left out of the failure queue |

## Worked confusion-matrix example

Assume a validation sample yields TP = 20, FP = 73, FN = 73 and TN = 9,834. Precision is 20 / (20 + 73) = 21.5%. Recall is 20 / (20 + 73) = 21.5%. MCC uses all four cells, so it remains informative even though the pass class dominates.

## Threshold trade-off

| **Threshold direction** | **Typical effect** | **Business consequence** |
| --- | --- | --- |
| Lower threshold | More alerts; recall often rises; precision may fall | Find more failures but inspect more passing products |
| Higher threshold | Fewer alerts; precision may rise; recall may fall | Reduce inspection load but miss more failures |

|  |
| --- |
| **Operating decision:** The model-selection score alone does not define the production threshold. Inspection capacity and the relative cost of false positives versus false negatives should determine the final operating point. |

# 6. Matthews Correlation Coefficient (MCC)

|  |
| --- |
| **MCC = (TP x TN - FP x FN) / sqrt[(TP+FP) (TP+FN) (TN+FP) (TN+FN)]** |

MCC summarizes the quality of binary predictions using all four cells of the confusion matrix. It is well suited to severe class imbalance because a model cannot obtain a strong MCC simply by predicting the majority class.

| **MCC value** | **Interpretation** |
| --- | --- |
| +1 | Perfect classification |
| 0 | No useful correlation between prediction and outcome |
| -1 | Perfect inverse classification |

## Selected result

|  |
| --- |
| **LightGBM validation MCC = 0.210198 (reported as 0.2102)** |

An MCC of 0.2102 indicates useful but moderate separation at the selected threshold. It does not mean 21.02% accuracy. MCC is a correlation-like balanced score, not a percentage of correct predictions.

## 6.1 Practical operating interpretation

At the selected operating point, precision is 21.52% and recall is 21.44%. If 1,000 products were flagged under comparable conditions, approximately 215 would be expected to be actual failures and approximately 785 would be unnecessary alerts. If the population contained 1,000 actual failures, the model would detect approximately 214 and miss approximately 786.

|  |
| --- |
| **Decision implication:** The current model is appropriate for ranking and prioritizing inspection. It is not appropriate for autonomous acceptance or rejection because most actual failures remain undetected at this operating point. |

## Why MCC was the primary selection metric

* The failure class is extremely rare.
* Accuracy would be dominated by correctly predicted passes.
* MCC penalizes both missed failures and false alarms.
* It provides a single threshold-specific comparison across model families.

|  |
| --- |
| **Common mistake:** Do not describe MCC 0.2102 as "the model is 21% accurate." Accuracy and MCC are different measures. Report the metric by name and explain that higher is better, with +1 representing perfect classification. |

# 7. Precision and recall

## 7.1 Precision

|  |
| --- |
| **Precision = TP / (TP + FP)** |

Precision answers: Of the products predicted as failures, how many actually failed? The selected LightGBM precision is 0.215171, or 21.52%. In an illustrative batch of 100 alerts, about 22 would be true failures and about 78 would be false alerts if the same rate held.

## 7.2 Recall

|  |
| --- |
| **Recall = TP / (TP + FN)** |

Recall answers: Of all actual failures, how many did the model identify? The selected LightGBM recall is 0.214390, or 21.44%. In an illustrative set of 100 actual failures, about 21 would be detected and about 79 would be missed at this operating threshold.

| **Metric** | **Optimizes** | **Risk when emphasized alone** |
| --- | --- | --- |
| Precision | Trustworthiness of alerts | A very high threshold can miss many failures |
| Recall | Coverage of actual failures | A very low threshold can overload inspection with false alerts |

## Precision@K and Recall@K

For a ranked inspection queue, K is the number of top-risk products the team can review. Precision@K is the fraction of those K that are actual failures; Recall@K is the fraction of all failures captured within those K. These metrics connect model ranking to a fixed inspection budget more directly than a generic threshold.

|  |
| --- |
| **Precision@K = failures in top K / K Recall@K = failures in top K / all validation failures** |

# 8. PR-AUC and runtime

## 8.1 PR-AUC

The precision-recall curve traces precision against recall as the classification threshold changes. PR-AUC summarizes this curve. It evaluates how effectively a model ranks rare failures above passes across many possible operating points.

|  |
| --- |
| **Random ranking baseline PR-AUC is approximately the positive-class prevalence** |

The selected LightGBM PR-AUC is 0.127076 (0.1271). Because the natural failure prevalence is roughly 0.58%, the model ranking is materially better than random. PR-AUC should not be read as 12.71% accuracy.

## Why PR-AUC and MCC can disagree

PR-AUC evaluates the entire ranking across thresholds, while MCC evaluates one binary operating point. One model can rank products slightly better overall but produce a slightly lower MCC at its chosen cutoff. The metrics answer different questions and should be reviewed together.

## 8.2 Runtime

|  |
| --- |
| **Runtime = elapsed benchmark time for fitting and evaluating one configuration** |

The selected LightGBM run took 99.58 seconds in the recorded benchmark environment. Runtime depends on hardware, library versions, thread count, data format and caching, so it is useful for relative comparison inside the same experiment but is not a guaranteed production latency.

|  |
| --- |
| **Do not confuse runtimes** Training runtime measures model development. Prediction latency measures how quickly new products can be scored. A model may take minutes to train but milliseconds per batch or request to infer. |

# 9. Model-by-model performance comparison

The following table reports the best-MCC configuration identified for each model family in the completed clean-split benchmark. All rows use the full labelled population and 283 final features, although the split and present-rate setting differ by the configuration that produced each model family's best MCC.

| **Model** | **Split** | **Present** | **Candidates** | **Features** | **MCC** | **PR-AUC** | **Precision** | **Recall** | **Runtime** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LightGBM | 80/20 | 0.0025 | 908 | 283 | 0.2102 | 0.1271 | 21.52% | 21.44% | 99.58 s |
| Random Forest | 90/10 | 0.0200 | 631 | 283 | 0.2089 | 0.1009 | 21.75% | 20.93% | 159.23 s |
| XGBoost | 90/10 | 0.0200 | 631 | 283 | 0.2023 | 0.1161 | 20.61% | 20.78% | 91.32 s |
| Logistic Regression | 80/20 | 0.0100 | 756 | 283 | 0.1923 | 0.0815 | 19.78% | 19.62% | 110.69 s |
| CatBoost | 80/20 | 0.0100 | 756 | 283 | 0.1860 | 0.0726 | 19.04% | 19.11% | 143.31 s |

## LightGBM

LightGBM produced the highest MCC and the strongest PR-AUC among the best-MCC configurations. Its histogram-based gradient boosting is efficient on large tabular datasets and captures nonlinear interactions among sparse process features.

## Random Forest

Random Forest produced a close MCC and the highest precision in this comparison, but its PR-AUC was materially lower and runtime was the longest. It is a strong challenger but not the best overall balance.

## XGBoost

XGBoost was the fastest recorded best configuration and achieved good PR-AUC, but its MCC remained below LightGBM. It remains useful as a robust gradient-boosting benchmark.

## Logistic Regression

Logistic Regression provides an interpretable linear baseline. Its lower metrics suggest the feature-to-failure relationship contains nonlinearities and interactions that a linear decision boundary does not capture sufficiently.

## CatBoost

CatBoost can be strong with raw categorical variables, but in this pipeline the categorical inputs were already encoded and the selected configuration did not outperform LightGBM. The result is specific to this feature construction and benchmark, not a universal judgement about CatBoost.

# 10. Why LightGBM is the final choice

| **Decision criterion** | **Evidence** | **Conclusion** |
| --- | --- | --- |
| Primary metric | Highest MCC: 0.2102 | Best threshold-specific balanced classification |
| Ranking quality | Highest PR-AUC: 0.1271 | Best failure-risk ordering among compared best-MCC runs |
| Validation strength | 236,750 held-out rows | Large evaluation sample under 80/20 split |
| Efficiency | 99.58 seconds | Faster than Random Forest and CatBoost in recorded runs |
| Pipeline discipline | Training-only selection; untouched validation | More defensible estimate than the earlier sampled result |

## Final model statement

|  |
| --- |
| **Selected model:** LightGBM; full labelled population of 1,183,747 rows; stratified 80/20 split; minimum numeric present rate 0.0025; 908 candidate numeric features; 283 final model features; validation MCC 0.2102; PR-AUC 0.1271; precision 21.52%; recall 21.44%; recorded runtime 99.58 seconds. |

## Final inference workflow

## 10.1 Full test-population scoring outcome

After the selected configuration was frozen, LightGBM was retrained on all 1,183,747 labelled products and applied to all 1,183,748 unlabelled test products. The final scoring threshold was 0.754167. It produced 7,003 predicted failure alerts, corresponding to an alert rate of 0.5916%.

|  |
| --- |
| **Metric boundary:** These are prediction outputs, not additional validation results. Test MCC, precision, recall and PR-AUC remain unknown because actual test Response labels are unavailable. |

|  |  |  |
| --- | --- | --- |
| **Output** | **Value** | **How to use it** |
| Test products scored | 1,183,748 | Complete unseen test population. |
| Failure probability | One value per product | Rank products from highest to lowest inspection priority. |
| Decision threshold | 0.754167 | Convert probability into a binary alert. |
| Predicted alerts | 7,003 | Products sent to the proposed inspection queue. |
| Alert rate | 0.5916% | Expected review workload relative to the test population. |

1. Freeze the selected feature-screening, encoding and preprocessing configuration.
2. Preserve the 80/20 validation result as the reported unbiased benchmark.
3. Refit the frozen pipeline on all 1,183,747 labelled products.
4. Apply the identical 283-feature contract to the complete unlabelled test dataset.
5. Save failure probability for ranking and a thresholded binary prediction for decisions.
6. Expose model version, feature-contract version and scoring timestamp in the Streamlit application.

# 11. Caution: the earlier MCC 0.3386

|  |
| --- |
| **Do not compare the two numbers as if they came from the same experiment:** The earlier MCC 0.3386 used a sampled modelling population and a different validation design. The full-data MCC 0.2102 uses the natural population and a cleaner training-only selection process. The lower number can therefore be the more trustworthy estimate. |

| **Item** | **Earlier Phase 6** | **Current clean full-data benchmark** |
| --- | --- | --- |
| Available labelled population | 1,183,747 | 1,183,747 |
| Rows entering modelling split | 226,879 (about 19.17%) | 1,183,747 (100%) |
| Training rows | 170,159 | 946,997 |
| Validation rows | 56,720 | 236,750 |
| Split | 75/25 | 80/20 |
| Failure prevalence | About 3.03% after down-sampling | Natural prevalence, about 0.58% |
| Reported MCC | 0.3386 | 0.2102 |
| Interpretation | Promising experimental result | Preferred defensible final benchmark |

## Why the earlier score may be higher

* All failures were retained while many non-failures were sampled, increasing failure prevalence.
* A smaller validation set may yield a less stable estimate.
* Feature selection or threshold tuning may have been influenced by data outside the fitting partition.
* Repeated tuning against the same validation set can indirectly overfit the evaluation.
* Different feature matrices and thresholds make the results non-comparable.

## When could the 0.3386 model become final?

Only if the earlier pipeline is reproduced under the exact clean protocol: identical full population, identical 80/20 split, all selection and preprocessing fitted on training rows only, threshold developed without peeking at final validation labels, and a single evaluation on the untouched 236,750-row validation partition. If it still achieves approximately 0.3386, it should replace the current model.

# 12. Interpretation guide for managers and reviewers

| **Question** | **Correct answer** |
| --- | --- |
| Did the final benchmark use all labelled products? | Yes. All 1,183,747 rows entered the 80/20 development split. |
| Did the fitted validation model train on all rows? | No. It fitted on 946,997 rows and evaluated on 236,750 untouched rows. |
| What does present rate 0.0025 mean? | A numeric candidate needs observations in at least 0.25% of training rows, about 2,367 products. |
| Are 908 candidate features all used by the model? | No. They are the eligible numeric pool; the completed model uses 283 final inputs. |
| Is MCC 0.2102 the same as 21.02% accuracy? | No. MCC is a balanced correlation-like score using all confusion-matrix cells. |
| Can the Kaggle test set produce a confusion matrix? | No, because its actual Response labels are not published. |
| Why select LightGBM? | Highest MCC and PR-AUC with a large validation set and reasonable runtime. |

## Presentation-ready narrative

The final benchmark used all 1.18 million labelled products with a stratified 80/20 split. Feature eligibility and selection were learned only from the 946,997 training products, while 236,750 products remained untouched for validation. Numeric features needed at least 0.25% training coverage, approximately 2,367 observations, producing 908 candidates. After selection and engineering, each model used 283 final features. LightGBM achieved the best balanced performance with MCC 0.2102 and PR-AUC 0.1271, so it was selected for final test-set inference. The earlier MCC 0.3386 is retained as an experimental result because it came from a down-sampled population and a different evaluation setup.

## 12.1 Practical interpretation guardrails

The most important review distinction is predictive evidence versus physical causation. The following wording keeps explanations technically defensible.

|  |
| --- |
| **Validation needed for physical root cause:** Machine IDs, measurement units, engineering limits, maintenance history, calibration records, defect codes, operator or shift data, and process-engineer confirmation. |

|  |  |  |
| --- | --- | --- |
| **Model signal** | **Defensible explanation** | **Incorrect claim** |
| start\_time | Certain earliest-measurement time windows carry different predicted risk. | Production started late and caused failure. |
| cycle\_time | Observed measurement spans differ between risk groups. | Verified cycle-time delay caused failure. |
| Missing indicator | Measurement absence may identify a route or station-participation pattern. | The sensor malfunctioned. |
| final\_product\_family | A route-based cluster has a different failure profile. | This is a known Bosch physical product type. |
| L3\_S32\_F3854 category | An anonymous state is strongly associated with failures. | A known machine mode is the root cause. |

# Appendix A. Metric reference sheet

| **Term** | **Formula or definition** | **Example interpretation** |
| --- | --- | --- |
| Present rate | non-missing training observations / training rows | 3,000 / 946,997 = 0.317%; passes 0.25% |
| Candidate features | features passing eligibility screening | 908 numeric columns entered ranking |
| Final features | columns supplied to model.fit | 283 model inputs |
| Precision | TP / (TP + FP) | Of 100 alerts, about 22 are true failures at 21.52% |
| Recall | TP / (TP + FN) | About 21 of 100 failures are found at 21.44% |
| MCC | balanced function of TP, TN, FP and FN | 0.2102; useful moderate predictive signal |
| PR-AUC | area under precision-recall curve | 0.1271; ranking quality across thresholds |
| Runtime | elapsed fit/evaluation time | 99.58 seconds in recorded environment |

# Appendix B. Reproducibility checklist

* Fix random seed and split indices.
* Store training-row IDs and validation-row IDs.
* Fit coverage thresholds and feature ranking only on training rows.
* Persist exact feature names and ordering.
* Persist categorical maps, missingness rules and model parameters.
* Record library versions and hardware.
* Save the probability threshold separately from the model.
* Hash the final model and feature contract.
* Re-run golden-set predictions before deployment.
* Keep probability outputs even when binary predictions are required.

# Appendix C. Evidence and scope note

This document is grounded in the completed Phase 6 clean-split benchmark values supplied in the conversation and the public Bosch project repository. The model-family table summarizes the best-MCC configuration reported for each family. Performance remains a historical-data validation result; it does not demonstrate live factory performance, causal root causes, official OEE, business savings or current production drift.

The practical examples added in this revision use committed project profiles and final scoring summaries. Full-data feature counts are used to demonstrate coverage and failure evidence; the final leakage-safe feature-selection rules themselves remain fitted only on the 80% training partition.