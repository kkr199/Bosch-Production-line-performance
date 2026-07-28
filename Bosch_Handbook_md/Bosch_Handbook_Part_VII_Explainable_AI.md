**PART VII**

Explainable AI

**Chapters 53–57**

Feature Importance • SHAP • Global & Local Explanations • Correlation vs Causation • Interpretation Controls

![Image: image1.png](data:image/png;base64...)

**Bosch Production Line Performance — Technical Handbook**

Repository-grounded explanation methods for the Phase 6 LightGBM benchmark

# Contents and Explainability Context

| **Chapter** | **Topic** | **Purpose** |
| --- | --- | --- |
| **53** | Feature Importance | Understand model-native importance and its limitations. |
| **54** | SHAP | Explain additive feature contributions with global and local views. |
| **55** | Global & Local Explanations | Translate explanations into engineering review priorities. |
| **56** | Correlation vs Causation | Separate predictive evidence from physical root-cause claims. |
| **57** | Common Interpretation Mistakes | Avoid misleading plots, wording, and decisions. |

## Phase 7 evidence base

| **Item** | **Repository value** | **Meaning** |
| --- | --- | --- |
| **Model explained** | Phase 6 LightGBM | The project’s non-leakage benchmark model, not a live-factory validated controller. |
| **Validation rows** | 56,720 | 1,720 labelled failures; 3.032% failure rate. |
| **Model features** | 323 | Numeric, categorical-presence, timing/path, missingness, and family features. |
| **SHAP global sample** | 8,000 | Random-state-42 sample from the validation population. |
| **Local records exported** | 1,000 | Row-level SHAP contribution vectors for audit and examples. |
| **Primary output** | Mean absolute SHAP | Ranks features by average magnitude of contribution to model output. |

|  |
| --- |
| **Interpretation boundary:** Phase 7 explains the behaviour of the trained model. It does not prove that a feature, station, route, or timestamp physically caused a defect. |

## Learning objectives

* Distinguish model-native feature importance from SHAP importance.
* Read a SHAP beeswarm, global bar chart, and local contribution chart correctly.
* Connect predictive signals to safe engineering investigation steps.
* Recognize confounding, proxy variables, correlated features, and missingness semantics.
* Apply an interpretation checklist before presenting conclusions to plant stakeholders.

**CHAPTER 53**

# Feature Importance

*Using model-native importance as a fast overview—not as a causal verdict*

## What feature importance means

Feature importance is a model-specific summary of how much a trained algorithm used each input. For the selected LightGBM estimator, the repository reads the model’s feature\_importances\_ array. With the default LightGBM setting, this is a split-count measure: it records how often a feature was selected for a tree split across the ensemble. Frequent use indicates that the variable was useful to the fitted model, but it does not directly show the direction of effect, the size of a row-level contribution, or whether the feature is physically causal.

![Image: image2.png](data:image/png;base64...)

*Figure 53.1 — Model-native split-count importance emphasizes frequently used timing and duration features.*

## Repository implementation

|  |
| --- |
| if hasattr(estimator, 'feature\_importances\_'):  importance = estimator.feature\_importances\_ elif hasattr(estimator, 'coef\_'):  importance = abs(estimator.coef\_).ravel() report = DataFrame({'feature': feature\_cols,  'model\_importance': importance}) |

## Top model-native features

| **Rank** | **Reader-facing feature** | **Model key** | **Split count** | **Feature family** |
| --- | --- | --- | --- | --- |
| **1** | Mean Inter-station Timestamp Gap | mean\_waiting\_time | 875 | timing |
| **2** | Maximum Inter-station Timestamp Gap | max\_waiting\_time | 772 | timing |
| **3** | Observed Measurement Time Span | cycle\_time | 732 | timing |
| **4** | Line 3 Observed Measurement Time Span | line\_3\_processing\_duration | 723 | line\_timing |
| **5** | Earliest Measurement Timestamp | start\_time | 512 | timing |
| **6** | Line 0 Observed Measurement Time Span | line\_0\_processing\_duration | 495 | line\_timing |
| **7** | Latest Measurement Timestamp | end\_time | 474 | timing |
| **8** | Relative Timestamp-Gap Ratio | delay\_ratio | 406 | timing |
| **9** | Line 0 Earliest Measurement Timestamp | line\_0\_start\_time | 385 | line\_timing |
| **10** | L2\_S27 Feature 3144 Measurement | L2\_S27\_F3144 | 330 | raw\_measurement |
| **11** | Observed Inter-station Timestamp Gaps | waiting\_time | 314 | timing |
| **12** | Line 2 Earliest Measurement Timestamp | line\_2\_start\_time | 299 | line\_timing |

## Strengths and limitations

| **Strength** | **Limitation** |
| --- | --- |
| Very fast to extract from the trained model. | Scale and meaning differ across algorithm families. |
| Useful for an initial shortlist and feature-registry review. | A high count does not show whether values raise or lower risk. |
| Covers all 323 model inputs. | Correlated features can divide or redirect importance. |
| Supports model debugging and version comparison. | Split-based importance can favour variables offering many useful thresholds. |
| Provides a coarse audit trail. | It is not a local explanation for any one product. |

|  |
| --- |
| **Recommended use:** Treat model-native importance as a navigation tool. Use SHAP, distribution checks, and engineering evidence before drawing process conclusions. |

## Why importance methods disagree

![Image: image3.png](data:image/png;base64...)

*Figure 53.2 — Split frequency and average SHAP magnitude may rank the same feature differently because they measure different aspects of model use.*

A feature may appear in many small tree splits yet make modest contributions to final outputs. Conversely, a feature used in fewer, high-impact regions can have a larger mean absolute SHAP value. Disagreement is therefore diagnostic rather than an error. It should prompt questions about feature interactions, correlated alternatives, sparse availability, and whether the feature matters only for a narrow product family or route.

| **Question** | **Model-native importance** | **Mean absolute SHAP** |
| --- | --- | --- |
| **Core question** | How often or how strongly did the model use this variable? | How large was this variable’s contribution across explained rows? |
| **Direction** | Not available | Available locally; global mean absolute values remove sign. |
| **Row-specific** | No | Yes |
| **Cross-model comparability** | Limited | Still model- and output-scale dependent, but more consistent conceptually. |
| **Best role** | Fast screening and model debugging | Global ranking and local explanation. |

**CHAPTER 54**

# SHAP

*Using Shapley-value attribution to decompose tree-model outputs*

## Additive explanation idea

SHAP assigns each feature a contribution value for a particular prediction. The explanation starts from a baseline model output and adds feature contributions until it reaches the model output for that product. A positive contribution moves the output toward the failure class; a negative contribution moves it away. The numerical unit is the model-output scale used by the explainer and must not automatically be read as a probability percentage.

![Image: image4.png](data:image/png;base64...)

*Figure 54.1 — A local SHAP explanation decomposes one model output into a baseline and feature contributions.*

|  |
| --- |
| explainer = shap.TreeExplainer(estimator) shap\_values = explainer.shap\_values(x\_sample) mean\_abs\_shap = abs(shap\_array).mean(axis=0) mean\_signed\_shap = shap\_array.mean(axis=0) local\_values = shap\_array[:1000] |

## Why SHAP is useful

* Local accuracy: contributions add to the explained model output under the explainer’s formulation.
* Consistency: when a model changes so a feature contributes more, its attribution should not decrease under the SHAP framework.
* Global aggregation: local absolute contributions can be averaged to rank signals across a population.
* Direction and heterogeneity: beeswarm plots reveal whether high and low values have different effects for different products.
* Auditability: exported row-level values allow explanation review without recomputing every chart.

## Repository sampling design

| **Step** | **Implementation** | **Reason** |
| --- | --- | --- |
| **Select rows** | Sample up to 8,000 validation rows with random\_state=42. | Controls compute cost while preserving reproducibility. |
| **Prepare matrix** | Apply the fitted pipeline imputer before TreeExplainer. | Explains the same numerical representation used by the estimator. |
| **Compute values** | Use shap.TreeExplainer on the unwrapped LightGBM estimator. | Efficient exact/optimized tree explanations. |
| **Aggregate** | Calculate mean absolute and mean signed SHAP by feature. | Produces global rankings and directional summaries. |
| **Export local data** | Write the first 1,000 contribution vectors. | Supports row-level audits and examples. |

|  |
| --- |
| **Scope:** The SHAP sample explains the validation population used in Phase 7. It is not evidence that the same feature relationships will remain stable in another plant, product generation, or production period. |

## Reading the SHAP summary plot

![Image: image5.png](data:image/png;base64...)

*Figure 54.2 — Repository-generated SHAP beeswarm for the top 25 predictive signals.*

1. Read features from top to bottom: ranking is based on average absolute contribution.
2. Read the horizontal axis as contribution to model output: right increases the model output for the failure class; left decreases it.
3. Read colour as the feature value: high values are shown toward the warm end and low values toward the cool end.
4. Look for spread: a wide distribution means the feature can have large effects for some products.
5. Look for colour reversal or mixing: nonlinear effects and interactions can make the same value behave differently across contexts.

For Earliest Measurement Timestamp, high and low relative values occupy different parts of the contribution distribution. This indicates a production-window association in the model. Because the Bosch timestamps are anonymized and relative, the correct conclusion is that the model distinguishes production periods or sequences—not that a calendar time or verified delay causes failure.

**CHAPTER 55**

# Global & Local Explanations

*Moving from population-level signal ranking to product-level review*

## Global explanation

Global explanations summarize what the model relies on across many products. The primary Phase 7 ranking is mean absolute SHAP, which removes sign before averaging so positive and negative contributions do not cancel. The strongest signal is Earliest Measurement Timestamp with mean absolute SHAP 0.196968. Line-level timing, measurement availability at L3\_S32, gap features, and path complexity also appear among the leading signals.

![Image: image6.png](data:image/png;base64...)

*Figure 55.1 — Global mean absolute SHAP ranks predictive signals by average magnitude, not by causality.*

| **Rank** | **Predictive signal** | **Model key** | **Mean |SHAP|** | **Mean signed** | **Defensible interpretation** |
| --- | --- | --- | --- | --- | --- |
| **1** | Earliest Measurement Timestamp | start\_time | 0.196968 | -0.003909 | Temporal production indicator; may proxy batch, routing, or latent process conditions. |
| **2** | Line 0 Earliest Measurement Timestamp | line\_0\_start\_time | 0.114482 | 0.010972 | Temporal production indicator; may proxy batch, routing, or latent process conditions. |
| **3** | Line 3 Latest Measurement Timestamp | line\_3\_end\_time | 0.094725 | -0.003977 | Temporal production indicator; may proxy batch, routing, or latent process conditions. |
| **4** | Line 3 Earliest Measurement Timestamp | line\_3\_start\_time | 0.082866 | -0.027103 | Temporal production indicator; may proxy batch, routing, or latent process conditions. |
| **5** | Latest Measurement Timestamp | end\_time | 0.076271 | 0.012949 | Temporal production indicator; may proxy batch, routing, or latent process conditions. |
| **6** | L3\_S32 Feature 3850 Missingness Indicator | L3\_S32\_F3850\_\_is\_missing | 0.065554 | -0.015549 | Measurement-availability or routing indicator; not a direct physical cause. |
| **7** | Mean Inter-station Timestamp Gap | mean\_waiting\_time | 0.047839 | -0.020766 | Derived timestamp-gap feature; not verified physical processing, queue, or delay time. |
| **8** | Line 0 Latest Measurement Timestamp | line\_0\_end\_time | 0.046122 | -0.013112 | Temporal production indicator; may proxy batch, routing, or latent process conditions. |
| **9** | Maximum Inter-station Timestamp Gap | max\_waiting\_time | 0.043828 | 0.017783 | Derived timestamp-gap feature; not verified physical processing, queue, or delay time. |
| **10** | Observed Measurement Time Span | cycle\_time | 0.043211 | -0.002051 | Derived timestamp-gap feature; not verified physical processing, queue, or delay time. |
| **11** | Line 3 Observed Measurement Timestamps | line\_3\_observed\_date\_values | 0.039657 | 0.003055 | Predictive association requiring process-record validation before causal interpretation. |
| **12** | Line 3 Observed Measurement Time Span | line\_3\_processing\_duration | 0.038699 | 0.001442 | Derived timestamp-gap feature; not verified physical processing, queue, or delay time. |
| **13** | Observed Inter-station Timestamp Gaps | waiting\_time | 0.036343 | -0.011750 | Derived timestamp-gap feature; not verified physical processing, queue, or delay time. |
| **14** | Line 3 Stations with Recorded Measurements | line\_3\_station\_count | 0.027908 | 0.013907 | Predictive association requiring process-record validation before causal interpretation. |
| **15** | Line 0 Observed Measurement Time Span | line\_0\_processing\_duration | 0.026871 | 0.000505 | Derived timestamp-gap feature; not verified physical processing, queue, or delay time. |

## Signal families

![Image: image7.png](data:image/png;base64...)

*Figure 55.2 — Timing and line-level timing dominate the aggregated explanatory mass in this model.*

Aggregating by feature family helps prevent a long list of similar timestamp variables from being mistaken for many independent causes. The cluster of timing signals may reflect common latent conditions such as batch, route, production window, maintenance state, or product mix. A family-level view therefore guides the next analysis: compare windows, paths, families, and sensor availability rather than taking station action from one variable alone.

## Local explanation: labelled failure example

![Image: image8.png](data:image/png;base64...)

*Figure 55.3 — Largest positive and negative SHAP contributions for one validation record labelled as a failure.*

| **Feature** | **SHAP contribution** | **Direction** | **Interpretation** |
| --- | --- | --- | --- |
| **L3\_S32 Feature 3850 Missingness Indicator** | 1.96389 | Raises model output | Measurement-availability or routing indicator; not a direct physical cause. |
| **Line 3 Stations with Recorded Measurements** | 1.70752 | Raises model output | Predictive association requiring process-record validation before causal interpretation. |
| **Line 3 Observed Measurement Time Span** | 0.24212 | Raises model output | Derived timestamp-gap feature; not verified physical processing, queue, or delay time. |
| **Stations with Recorded Measurements** | 0.20221 | Raises model output | Predictive association requiring process-record validation before causal interpretation. |
| **Line 3 Observed Measurement Timestamps** | 0.16962 | Raises model output | Predictive association requiring process-record validation before causal interpretation. |
| **L1\_S24 Feature 1844 Measurement** | -0.08053 | Lowers model output | Predictive association requiring process-record validation before causal interpretation. |
| **L1\_S24 Feature 1778 Measurement** | -0.10188 | Lowers model output | Predictive association requiring process-record validation before causal interpretation. |
| **L1\_S24 Feature 1667 Measurement** | -0.10758 | Lowers model output | Predictive association requiring process-record validation before causal interpretation. |
| **L1\_S24 Feature 1609 Measurement** | -0.23940 | Lowers model output | Predictive association requiring process-record validation before causal interpretation. |
| **L1\_S24 Feature 1842 Measurement** | -0.26075 | Lowers model output | Predictive association requiring process-record validation before causal interpretation. |

|  |
| --- |
| **Local-reading rule:** A labelled failure can still contain features that lower the model output, and a non-failure can contain features that raise it. A prediction is the sum of all contributions plus the baseline—not a one-feature rule. |

## Local explanation: labelled non-failure example

![Image: image9.png](data:image/png;base64...)

*Figure 55.4 — Largest positive and negative contributions for one validation record labelled as a non-failure.*

| **Feature** | **SHAP contribution** | **Direction** | **Review note** |
| --- | --- | --- | --- |
| **L3\_S32 Feature 3850 Missingness Indicator** | 2.33424 | Raises model output | Compare with route, family, and neighbouring records. |
| **Line 3 Stations with Recorded Measurements** | 1.31874 | Raises model output | Compare with route, family, and neighbouring records. |
| **Hierarchical Family** | 0.23403 | Raises model output | Compare with route, family, and neighbouring records. |
| **Earliest Measurement Timestamp** | 0.15677 | Raises model output | Compare with route, family, and neighbouring records. |
| **Line 3 Observed Measurement Timestamps** | 0.12428 | Raises model output | Compare with route, family, and neighbouring records. |
| **Kmeans Family** | -0.11129 | Lowers model output | Compare with route, family, and neighbouring records. |
| **Line 0 Earliest Measurement Timestamp** | -0.19956 | Lowers model output | Compare with route, family, and neighbouring records. |
| **Line 3 Earliest Measurement Timestamp** | -0.20648 | Lowers model output | Compare with route, family, and neighbouring records. |
| **L3\_S32 Feature 3850 Measurement** | -0.33926 | Lowers model output | Compare with route, family, and neighbouring records. |
| **Latest Measurement Timestamp** | -0.47218 | Lowers model output | Compare with route, family, and neighbouring records. |

## From explanation to engineering review

![Image: image10.png](data:image/png;base64...)

*Figure 55.5 — Feature-level SHAP values aggregated into predictive-signal areas for investigation prioritization.*

| **Area / station** | **Feature family** | **Total mean |SHAP|** | **Leading signal** | **Recommended validation** |
| --- | --- | --- | --- | --- |
| **timing\_level** | timing | 0.475390 | Earliest Measurement Timestamp | Review cycle-time, waiting-time, and queue behavior for products following this path. |
| **line\_level** | line\_timing | 0.292540 | Line 3 Latest Measurement Timestamp | Review cycle-time, waiting-time, and queue behavior for products following this path. |
| **L1\_S24** | raw\_measurement | 0.258985 | L1\_S24 Feature 1844 Measurement | Review measurement distributions, tooling condition, calibration records, and recent process changes for this station. |
| **line\_level** | line\_timing | 0.191997 | Line 0 Earliest Measurement Timestamp | Review cycle-time, waiting-time, and queue behavior for products following this path. |
| **path\_level** | path\_or\_family | 0.081922 | Path Complexity Score | Use this as a cross-station signal; compare affected product families and paths before station-specific action. |
| **L3\_S32** | raw\_measurement\_missingness | 0.065606 | L3\_S32 Feature 3850 Missingness Indicator | Check whether skipped measurements, sensor dropouts, or alternate routing through this station align with failures. |
| **line\_level** | line\_timing | 0.046970 | Line 1 Latest Measurement Timestamp | Review cycle-time, waiting-time, and queue behavior for products following this path. |
| **line\_level** | line\_timing | 0.038404 | Line 2 Earliest Measurement Timestamp | Review cycle-time, waiting-time, and queue behavior for products following this path. |
| **other** | categorical\_or\_other | 0.027447 | Observed Measurement Timestamps | Use this as a cross-station signal; compare affected product families and paths before station-specific action. |
| **L1\_S25** | raw\_measurement | 0.022217 | L1\_S25 Feature 2990 Measurement | Check whether skipped measurements, sensor dropouts, or alternate routing through this station align with failures. |

|  |
| --- |
| **Naming caution:** The report uses the legacy filename station\_root\_cause\_report.csv, but its contents should be communicated as predictive-signal priorities until physical process records confirm a cause. |

**CHAPTER 56**

# Correlation vs Causation

*Preventing predictive explanations from becoming unsupported root-cause claims*

## Three different statements

| **Statement type** | **Example** | **Evidence required** |
| --- | --- | --- |
| **Correlation** | Products in one relative time window show a different failure rate. | Labelled observational data and a valid comparison. |
| **Model attribution** | Earliest Measurement Timestamp contributes strongly to the LightGBM output. | A frozen model and a valid explanation method. |
| **Causal effect** | Changing the production timing would reduce failures. | Controlled intervention, quasi-experimental design, or strong causal identification with process evidence. |

SHAP strengthens understanding of the model-attribution statement. It does not automatically upgrade an association into a causal effect. A variable can be important because it is a proxy for an unobserved condition, because it is measured after the decision point, or because it shares information with other correlated variables.

![Image: image11.png](data:image/png;base64...)

*Figure 56.1 — Confounding can make timestamp and station features predictive without being physical causes.*

## Project-specific confounding examples

| **Predictive signal** | **Possible non-causal explanation** | **Validation action** |
| --- | --- | --- |
| **Earliest / latest timestamps** | Batch, production period, product mix, maintenance window, or route sequence. | Compare product families and paths within the same time windows; review maintenance history. |
| **Inter-station timestamp gaps** | Anonymized event spacing, parallel operations, sparse measurement capture, or route variation. | Verify event semantics against real process records before calling the value waiting time. |
| **L3\_S32\_F3850\_\_is\_missing** | Optional operation, alternate routing, sensor capture policy, or genuine dropout. | Cross-tabulate routing, sensor availability, and failures; inspect station logs. |
| **Path complexity** | Product variant complexity or planned route differences. | Compare like-for-like products and approved routing policies. |
| **Raw station measurement** | Calibration, tooling, environment, product specification, or correlated measurement. | Review control limits, calibration, distributions, and nearby correlated sensors. |

## Causal validation ladder

1. Confirm that the feature is available before the scoring and intervention decision.
2. Verify the physical meaning, units, sensor ownership, and data-capture rules.
3. Check stratified associations within product family, route, line, and production window.
4. Test stability across time slices and operational regimes.
5. Review maintenance, calibration, tooling, and operator records for competing explanations.
6. Design a controlled process trial or approved quasi-experiment when intervention is feasible.
7. Measure whether the intervention changes the suspected mechanism and the failure outcome.
8. Update the model and explanation report only after the process change is independently validated.

|  |
| --- |
| **Safe language:** “The model uses this signal” is a valid explainability statement. “This station caused the defect” requires separate manufacturing evidence. |

## Temporal leakage and post-outcome signals

Even a perfectly calculated SHAP value can explain an invalid model. Every feature must pass a prediction-time availability check. A timestamp, inspection result, rework flag, or measurement recorded after the failure decision could create leakage. The current repository deliberately explains the production-safe Phase 6 model rather than the leaderboard/leak model, but future factory implementations still require a declared scoring point and feature-time audit.

| **Control** | **Question** |
| --- | --- |
| **Scoring point** | At exactly what process event must the risk score be available? |
| **Feature timestamp** | Was each feature known at or before that event? |
| **Transformation fit** | Were imputers, selected levels, and feature selection learned only from training data? |
| **Holdout isolation** | Was the final labelled holdout protected from explanation-led model tuning? |
| **Operational action** | Can the signal lead to a safe, approved intervention rather than merely describe completed production? |

**CHAPTER 57**

# Common Interpretation Mistakes

*A practical review checklist for charts, reports, and stakeholder presentations*

![Image: image12.png](data:image/png;base64...)

*Figure 57.1 — Replace causal-sounding shortcuts with defensible model and process language.*

## Mistakes and corrections

| **Mistake** | **Why it is wrong** | **Correction** |
| --- | --- | --- |
| **Highest importance = root cause** | Importance measures model use or attribution, not intervention effect. | Call it a leading predictive signal and define a validation plan. |
| **Positive mean signed SHAP = universally risky** | A mean can hide different local directions and interactions. | Inspect distributions, dependence plots, and local cases. |
| **Mean absolute SHAP gives direction** | Absolute values deliberately remove sign. | Use local values or directional plots for sign. |
| **Missingness means sensor failure** | Missing values can encode planned routing or optional operations. | Verify capture rules and station presence. |
| **Timestamp span is real cycle time** | Bosch timestamps are anonymized relative measurements. | Use “observed measurement span” unless factory semantics are confirmed. |
| **SHAP value is a probability change** | Tree SHAP may be on a raw model-output scale. | Label the axis as model-output contribution unless probability-space settings are verified. |
| **Top 15 features are independent causes** | Correlated variables and feature families share information. | Group related signals and test collinearity. |
| **One local chart explains all failures** | A local explanation applies to one feature vector and one model version. | Use population summaries and representative case cohorts. |
| **Unlabelled test alerts prove performance** | The Kaggle test set lacks outcome labels. | Use a locked labelled holdout for performance claims. |
| **Explanation remains valid after retraining** | Attributions depend on the exact model, data, and preprocessing. | Version and regenerate explanation artifacts with every release. |

## Presentation language guide

| **Avoid** | **Use instead** |
| --- | --- |
| **“The root cause is start\_time.”** | “Earliest Measurement Timestamp is the strongest global predictive signal in the explained validation sample.” |
| **“L3\_S32 causes failures.”** | “Measurement availability at L3\_S32 is associated with model output and should be checked against routing and sensor records.” |
| **“Long waiting time increases defects.”** | “Derived timestamp gaps contribute to model predictions; their physical waiting-time meaning is not yet verified.” |
| **“The model explains why the product failed.”** | “The local chart explains which features moved this model’s output for the product.” |
| **“SHAP proves the process change will work.”** | “SHAP prioritizes a hypothesis that requires controlled process validation.” |

## Explanation review checklist

1. State the exact model version, feature-list version, validation population, and explanation sample.
2. Label model-native importance separately from SHAP importance.
3. Use reader-facing feature names while retaining the original model keys for traceability.
4. Describe mean absolute SHAP as magnitude, not direction.
5. Confirm the output scale before translating SHAP values into probability language.
6. Show at least one global distribution and more than one local example.
7. Group related timing, path, missingness, and raw-measurement signals.
8. Check correlated features and explain why rankings may shift between retrains.
9. State that anonymized timestamps are not verified queue, delay, or official cycle-time measures.
10. Separate predictive association, engineering hypothesis, and confirmed cause in the conclusion.
11. Attach a process-validation action for each high-priority signal.
12. Require Quality Engineering review before using explanations for disposition or process changes.

|  |
| --- |
| **Final principle:** A trustworthy explanation is not the most confident story. It is the clearest statement of what the model shows, what the data cannot establish, and what evidence must be collected next. |

# Part VII Summary

Part VII converts the Phase 7 artifacts into a disciplined explanation workflow. Model-native feature importance supplies a fast overview; SHAP provides additive global and local attributions; aggregation organizes signals into engineering review areas; and causal safeguards prevent predictive patterns from being misrepresented as confirmed manufacturing causes.

| **Topic** | **Project conclusion** |
| --- | --- |
| **Feature Importance** | Useful for fast screening, but split frequency has no direction and is not causal. |
| **SHAP** | Mean absolute SHAP ranks global contribution magnitude; local values explain one model output. |
| **Global & Local Explanations** | Timing, line timing, measurement availability, raw station features, and path signals guide investigation. |
| **Correlation vs Causation** | Attribution explains the model; process records and interventions are required to confirm causes. |
| **Common Mistakes** | Safe wording, output-scale checks, feature-time audits, cohort review, and versioning are mandatory. |

## Repository traceability

| **Purpose** | **Repository artifact** |
| --- | --- |
| **Explainability implementation** | src/data/phase7\_root\_cause\_analysis.py |
| **Model-native importance** | reports/phase7\_feature\_importance.csv |
| **Global SHAP importance** | reports/phase7\_shap\_global\_importance.csv |
| **Local SHAP records** | reports/phase7\_shap\_local\_values\_sample.csv |
| **Top predictive signals** | reports/phase7\_top\_failure\_drivers.csv |
| **Station / signal aggregation** | reports/phase7\_station\_root\_cause\_report.csv |
| **Engineering action plan** | reports/phase7\_engineer\_action\_plan.csv |
| **SHAP visualizations** | reports/figures/phase7\_shap\_summary.png and phase7\_top\_shap\_drivers.png |
| **Technical report** | reports/phase7\_root\_cause\_analysis\_report.md |

|  |
| --- |
| **Intended use:** Use explanations to prioritize human review and investigation. Do not use them as autonomous product disposition, safety control, or causal proof. |