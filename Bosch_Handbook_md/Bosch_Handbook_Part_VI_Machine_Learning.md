**PART VI**

Machine Learning

**Chapters 47–52**

Linear Baseline • Tree Ensembles • Gradient Boosting • Model Selection

![Image: image1.png](data:image/png;base64...)

*Figure VI.1 — Repository modeling workflow from the 323-feature matrix to the selected LightGBM operating point.*

| **Evaluation boundary:** The reported scores come from a stratified validation split used for experimentation and model selection. The Kaggle test files are unlabelled scoring inputs; they are not an independent production holdout. |
| --- |

**Author: Krishnakanth Reddy Karingula**

Repository-driven technical handbook edition

# Contents and Modeling Context

Part VI explains the five supervised classifiers compared in Phase 6 and the rule used to select the repository’s production-safe benchmark. Each chapter connects algorithm theory to the exact pipeline configuration, validation result, and manufacturing interpretation used in the project.

| **Chapter** | **Topic** | **Purpose** |
| --- | --- | --- |
| **47** | Logistic Regression | Interpretable linear probability baseline |
| **48** | Random Forest | Bagged decision-tree ensemble |
| **49** | XGBoost | Regularized gradient-boosted trees |
| **50** | CatBoost | Boosted trees with category-aware design |
| **51** | LightGBM | Efficient leaf-wise gradient boosting |
| **52** | Model Selection | MCC threshold tuning, ranking, and governance |

## Phase 6 modeling population

| **Item** | **Repository value** | **Interpretation** |
| --- | --- | --- |
| **Sampled labeled rows** | 226,879 | All 6,879 failures plus a reproducible sample of non-failures |
| **Training split** | 170,159 rows | 75% of sampled rows after stratification |
| **Validation split** | 56,720 rows | 25% of sampled rows; 1,720 observed failures |
| **Final feature set** | 323 features | Numeric, categorical indicators, timing/path, missingness, and product family |
| **Kaggle test scoring** | 1,183,748 rows | Unlabelled inference population |
| **Selected test alerts** | 6,998 | 0.591% of the test scoring population |

## Common preprocessing and evaluation

* Important missingness patterns are preserved as explicit binary indicators when selected numeric features exceed 5% missingness.
* All five model pipelines use median imputation; Logistic Regression additionally standardizes the feature matrix.
* Class imbalance is addressed through balanced class weights or an equivalent positive-class weight.
* Each model produces probabilities. The binary operating threshold is selected from 80 score quantiles to maximize Matthews correlation coefficient (MCC).
* Models are ranked by MCC first and PR-AUC second. Accuracy is not used as the primary metric because the defect class is rare.

![Image: image2.png](data:image/png;base64...)

*Figure VI.2 — MCC, F1, and PR-AUC for the five models at their selected validation thresholds.*

| **Class-imbalance warning:** The validation failure rate is 3.032%. A classifier predicting every product as non-failure would look accurate but would detect no failures, which is why balanced metrics are essential. |
| --- |

**CHAPTER 47**

# Logistic Regression

*Establishing a transparent linear baseline before using nonlinear tree ensembles*

## Model concept

Logistic Regression estimates a linear score from the 323 model features and transforms that score through the sigmoid function. The output is a probability between zero and one. Coefficients describe how a one-unit feature change shifts the log-odds when all other variables are held constant, although heavy sparsity, interactions, and correlated manufacturing signals can make direct causal interpretation inappropriate.

![Image: image3.png](data:image/png;base64...)

*Figure 47.1 — The sigmoid maps a weighted feature score into a bounded failure probability.*

## Repository configuration

| Pipeline([  ('imputer', SimpleImputer(strategy='median')),  ('scaler', StandardScaler()),  ('model', LogisticRegression(  max\_iter=700, class\_weight='balanced', solver='saga',  n\_jobs=-1, random\_state=42  )) ]) |
| --- |

| **Setting** | **Value** | **Reason** |
| --- | --- | --- |
| **Median imputation** | Enabled | Provides a complete numeric matrix while missing indicators preserve important absence patterns |
| **StandardScaler** | Enabled | Places differently scaled sensor and engineered variables on comparable numerical ranges |
| **Solver** | SAGA | Works with large sparse-style feature sets and supports scalable iterative optimization |
| **Class weighting** | Balanced | Increases the influence of rare failures during loss minimization |
| **Maximum iterations** | 700 | Allows convergence on the large, high-dimensional matrix |

## Validation result

| **Model** | **Threshold** | **MCC** | **Precision** | **Recall** | **F1** | **PR-AUC** |
| --- | --- | --- | --- | --- | --- | --- |
| **Logistic Regression** | 0.8973 | 0.3045 | 0.5196 | 0.1930 | 0.2815 | 0.1650 |

The baseline reaches MCC 0.3045. Its precision of 0.5196 means just over half of its selected alerts correspond to labeled failures on this validation split, while recall 0.1930 means it identifies fewer than one in five failures. The model is useful because it demonstrates that the engineered feature set contains learnable signal even without nonlinear splits.

## Strengths in this project

* Fast to fit and straightforward to reproduce.
* Provides a defensible baseline against which more complex models must improve.
* Can support coefficient review after careful standardization and collinearity checks.
* Its simpler decision surface is less likely to memorize narrow route patterns than an unconstrained tree ensemble.

## Limitations and failure modes

* A linear log-odds assumption cannot naturally represent threshold effects, route interactions, or non-monotonic timing patterns.
* Highly correlated sparse station features can produce unstable coefficients even when predictive performance is stable.
* Balanced weighting changes the optimization target; raw probabilities may require calibration before operational use.
* A strong coefficient is a predictive association, not proof that the corresponding station or measurement caused failure.

| **Release practice:** Retain Logistic Regression in every release as a rules-light baseline. If a new complex model cannot materially outperform it on a locked holdout, the additional operational complexity may not be justified. |
| --- |

**CHAPTER 48**

# Random Forest

*Combining many decorrelated decision trees to learn nonlinear manufacturing patterns*

## Model concept

Random Forest trains multiple decision trees on bootstrapped samples while exposing each split to only a subset of features. Averaging across trees reduces the variance of a single deep tree and allows the ensemble to learn interactions among sensor values, missingness indicators, temporal features, product families, and route structure.

![Image: image4.png](data:image/png;base64...)

*Figure 48.1 — Bagging and random feature selection create diverse trees whose probabilities are averaged.*

## Repository configuration

| RandomForestClassifier(  n\_estimators=80,  max\_depth=14,  min\_samples\_leaf=10,  class\_weight='balanced\_subsample',  n\_jobs=-1,  random\_state=42 ) |
| --- |

| **Hyperparameter** | **Value** | **Effect** |
| --- | --- | --- |
| **Trees** | 80 | Balances ensemble diversity against memory and run time |
| **Maximum depth** | 14 | Restricts individual-tree complexity and overfitting |
| **Minimum leaf size** | 10 | Prevents leaves from representing only a few products |
| **Class weighting** | Balanced per bootstrap sample | Re-estimates imbalance weights within each sampled tree |
| **Parallel jobs** | All cores | Speeds independent tree training |

## Validation result

| **Model** | **Threshold** | **MCC** | **Precision** | **Recall** | **F1** | **PR-AUC** |
| --- | --- | --- | --- | --- | --- | --- |
| **Random Forest** | 0.6743 | 0.3250 | 0.5524 | 0.2052 | 0.2993 | 0.2390 |

Random Forest improves on Logistic Regression, reaching MCC 0.3250 and PR-AUC 0.2390. This improvement supports the expectation that failure risk depends on nonlinear feature interactions. Its selected operating point retains precision above 0.55, but recall remains near 0.205 because threshold optimization prioritizes balanced correlation rather than maximizing the number of alerts.

## Why the ensemble helps

* Trees can split directly on sparse numeric values and missingness flags without requiring a linear relationship.
* Interactions such as “specific station measurement + product family + late timestamp window” arise naturally through split sequences.
* Bagging reduces sensitivity to any one sample of non-failure rows.
* Feature importance can provide a coarse diagnostic, although permutation or SHAP methods are preferred for more reliable explanations.

## Operational limitations

* Large forests consume more memory than a linear model and can be slower for batch scoring.
* Probability estimates may be conservative or poorly calibrated under class weighting.
* Standard impurity-based feature importance can favor variables with many possible split points.
* The fixed depth and tree count are practical project settings, not a fully tuned production configuration.

| **Project conclusion:** Random Forest is a valuable nonlinear benchmark. In this project, boosted trees deliver slightly stronger MCC at similar precision and recall, so the forest is not selected as the primary model. |
| --- |

**CHAPTER 49**

# XGBoost

*Using regularized sequential trees to correct residual classification errors*

## Model concept

XGBoost builds trees sequentially. Each new tree is trained to reduce the gradient of the current loss, so later trees concentrate on products the existing ensemble predicts poorly. Regularization, row sampling, and feature sampling control complexity. This approach is well suited to sparse, high-dimensional manufacturing data with nonlinear interactions.

![Image: image5.png](data:image/png;base64...)

*Figure 49.1 — Gradient boosting adds regularized trees that correct the ensemble’s remaining errors.*

## Repository configuration

| XGBClassifier(  n\_estimators=250, max\_depth=5, learning\_rate=0.05,  subsample=0.85, colsample\_bytree=0.85,  eval\_metric='aucpr', scale\_pos\_weight=negative\_count/positive\_count,  n\_jobs=-1, random\_state=42 ) |
| --- |

| **Setting** | **Value** | **Modeling role** |
| --- | --- | --- |
| **Boosting rounds** | 250 | Adds many small corrective steps |
| **Tree depth** | 5 | Allows interactions without extremely deep trees |
| **Learning rate** | 0.05 | Shrinks each tree’s contribution for smoother learning |
| **Row sampling** | 0.85 | Adds stochastic regularization |
| **Column sampling** | 0.85 | Reduces dependence on a narrow set of dominant features |
| **Positive weight** | negatives ÷ positives | Compensates for rare failures |
| **Evaluation metric** | AUC-PR | Matches the imbalanced nature of the target during training diagnostics |

## Validation result

| **Model** | **Threshold** | **MCC** | **Precision** | **Recall** | **F1** | **PR-AUC** |
| --- | --- | --- | --- | --- | --- | --- |
| **XGBoost** | 0.7479 | 0.3359 | 0.5674 | 0.2128 | 0.3095 | 0.2469 |

XGBoost ranks second with MCC 0.3359, only 0.0027 below LightGBM. Precision reaches 0.5674 and recall 0.2128. The small performance difference means the model is a credible alternative; operational selection should therefore also consider run time, package stability, reproducibility, model size, and monitoring support rather than treating the ranking as an absolute scientific verdict.

## Advantages

* Strong performance on heterogeneous tabular features.
* Explicit regularization and controlled depth make the learning process auditable.
* Robust handling of nonlinear thresholds and interactions.
* Broad ecosystem support for feature importance, SHAP explanations, and deployment.

## Risks and controls

* Boosting can exploit subtle leakage or unstable timestamp patterns more aggressively than a linear baseline.
* A positive-class weight improves learning but does not automatically yield calibrated probabilities.
* Hyperparameters should be selected inside training folds; repeated tuning against one validation split can overfit the model-selection process.
* Prediction-time feature availability must be verified for every numeric, categorical, and engineered input.

| **Selection caution:** The MCC gap between XGBoost and LightGBM is small. A physically isolated holdout could reverse their order; both should remain reproducible candidates until the final evaluation policy is completed. |
| --- |

**CHAPTER 50**

# CatBoost

*Comparing a category-aware boosting architecture on the common engineered matrix*

## Model concept

CatBoost is designed to reduce prediction shift when categorical variables are converted into target statistics and to train stable symmetric decision trees. In this repository, however, categorical inputs are first reduced to selected missingness and level-equality indicators. CatBoost therefore receives the same numeric matrix as the other models; its comparison mainly measures the value of its boosting and regularization strategy rather than its native raw-category processing advantage.

![Image: image6.png](data:image/png;base64...)

*Figure 50.1 — CatBoost’s category-aware design is evaluated here on the common pre-encoded feature matrix.*

## Repository configuration

| CatBoostClassifier(  iterations=250, depth=6, learning\_rate=0.05,  loss\_function='Logloss', eval\_metric='PRAUC',  auto\_class\_weights='Balanced',  random\_seed=42, verbose=False ) |
| --- |

| **Setting** | **Value** | **Purpose** |
| --- | --- | --- |
| **Iterations** | 250 | Controls boosting length |
| **Depth** | 6 | Uses balanced symmetric trees with moderate interaction depth |
| **Learning rate** | 0.05 | Applies gradual corrective updates |
| **Loss** | Logloss | Optimizes probabilistic binary classification |
| **Evaluation metric** | PRAUC | Tracks minority-class ranking quality |
| **Class weights** | Automatic balanced | Offsets severe target imbalance |

## Validation result

| **Model** | **Threshold** | **MCC** | **Precision** | **Recall** | **F1** | **PR-AUC** |
| --- | --- | --- | --- | --- | --- | --- |
| **CatBoost** | 0.7970 | 0.3279 | 0.5571 | 0.2070 | 0.3018 | 0.2261 |

CatBoost ranks third with MCC 0.3279. Its precision is 0.5571 and recall is 0.2070. The model outperforms Logistic Regression but falls behind both XGBoost and LightGBM. The PR-AUC of 0.2261 is also lower than Random Forest’s 0.2390, illustrating why model ranking should consider more than a single architecture’s reputation.

## Interpretation of the result

* The common one-hot-style matrix limits CatBoost’s native advantage on raw high-cardinality categories.
* Symmetric trees may offer predictable inference and compact representations, but that advantage was not enough to lead the validation ranking.
* Its performance confirms that the feature set is suitable for several boosting libraries rather than being tied to one implementation.
* A future factory pipeline with raw categorical variables could justify a separate native-CatBoost experiment learned strictly within training folds.

## Deployment considerations

* Pin the CatBoost package version because serialized model compatibility can change across releases.
* Validate CPU inference latency and model size for the intended batch or API environment.
* Calibrate the probability output before interpreting it as an expected real-world failure frequency.
* Document whether categories are supplied natively or as engineered binary indicators; the two pipelines are not interchangeable.

| **Project conclusion:** CatBoost remains a useful challenger model, especially if a future production dataset exposes stable raw categorical fields. For the present repository matrix it does not provide the strongest validation score. |
| --- |

**CHAPTER 51**

# LightGBM

*Selecting an efficient leaf-wise boosted-tree model as the Phase 6 benchmark*

## Model concept

LightGBM grows trees leaf-wise, expanding the leaf expected to produce the largest loss reduction. This can capture complex interactions efficiently, but it must be regularized because unconstrained leaf-wise growth can overfit small regions. The repository combines 300 trees, a low learning rate, a moderate leaf limit, and row/column sampling.

![Image: image7.png](data:image/png;base64...)

*Figure 51.1 — Leaf-wise growth allocates new splits where the current ensemble can reduce loss most.*

## Repository configuration

| LGBMClassifier(  n\_estimators=300, learning\_rate=0.04, num\_leaves=48,  subsample=0.85, colsample\_bytree=0.85,  class\_weight='balanced', random\_state=42,  n\_jobs=-1, verbose=-1 ) |
| --- |

| **Setting** | **Value** | **Purpose** |
| --- | --- | --- |
| **Trees** | 300 | Provides sufficient boosting capacity |
| **Learning rate** | 0.04 | Reduces the contribution of each tree |
| **Maximum leaves** | 48 | Controls leaf-wise model complexity |
| **Row sampling** | 0.85 | Adds stochastic regularization |
| **Column sampling** | 0.85 | Prevents repeated dependence on the same features |
| **Class weighting** | Balanced | Increases the contribution of rare failures |

## Validation result

| **Model** | **Threshold** | **MCC** | **Precision** | **Recall** | **F1** | **PR-AUC** |
| --- | --- | --- | --- | --- | --- | --- |
| **LightGBM** | 0.7851 | 0.3386 | 0.5743 | 0.2134 | 0.3111 | 0.2529 |

LightGBM ranks first with MCC 0.3386 and PR-AUC 0.2529. At the selected threshold of 0.7851 it achieves precision 0.5743 and recall 0.2134. The operating point is intentionally conservative: it prioritizes a manageable, higher-confidence alert queue rather than attempting to flag every possible failure.

## Why it leads the comparison

* Efficient histogram-based training suits the large, mostly numeric 323-feature matrix.
* Leaf-wise growth captures nonlinear interactions among timing, route, family, missingness, and sensor variables.
* The model achieves the best MCC and best PR-AUC in the comparison, not merely the best value of one metric.
* The saved pipeline includes its median imputer and exact feature list, supporting repeatable batch or API scoring.

## Scoring output

| **Output** | **Value** | **Meaning** |
| --- | --- | --- |
| **Kaggle test products scored** | 1,183,748 | Every unlabelled test row receives a probability |
| **Selected threshold** | 0.7851 | Threshold optimized for MCC on validation |
| **Predicted alerts** | 6,998 | 0.591% of test rows |
| **Probability range** | 0.0031–0.9968 | Model score range; not proven real-world failure frequency |

## Important limits

* The selected score is experimental until a locked, physically isolated labelled holdout is evaluated once.
* The public dataset does not demonstrate live-factory accuracy, drift robustness, latency, or business impact.
* SHAP explanations describe model behavior and associations; quality engineering records are required for causal conclusions.
* The operational threshold must be revisited when inspection capacity, false-negative cost, and label latency are known.

| **Intended use:** Use LightGBM as the official production-safe benchmark for this project, not as an autonomous rejection system. The score should rank products for human quality review. |
| --- |

**CHAPTER 52**

# Model Selection

*Choosing a model and operating threshold under severe class imbalance*

## Selection procedure

![Image: image8.png](data:image/png;base64...)

*Figure 52.1 — Phase 6 selection sequence and the boundary between validation and unlabelled test scoring.*

1. Construct the common 323-feature matrix and preserve the same train/validation rows for all algorithms.
2. Fit each full preprocessing-and-model pipeline on the stratified training split.
3. Generate validation probabilities rather than accepting each library’s default class threshold.
4. Evaluate 80 unique probability quantiles between the 50th and 99.5th percentiles.
5. Choose the threshold with the highest Matthews correlation coefficient for that model.
6. Rank models by MCC first and PR-AUC second; save every candidate and package the winner with its feature list and threshold.
7. Use the frozen winner to score the unlabelled Kaggle test population.

| thresholds = unique(quantile(scores, linspace(0.50, 0.995, 80))) for threshold in thresholds:  prediction = (scores >= threshold)  mcc = matthews\_corrcoef(y\_valid, prediction) select threshold with maximum MCC rank models by MCC, then PR-AUC |
| --- |

## Complete comparison

| **Rank** | **Model** | **Threshold** | **MCC** | **Precision** | **Recall** | **F1** | **PR-AUC** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **1** | LightGBM | 0.7851 | 0.3386 | 0.5743 | 0.2134 | 0.3111 | 0.2529 |
| **2** | XGBoost | 0.7479 | 0.3359 | 0.5674 | 0.2128 | 0.3095 | 0.2469 |
| **3** | CatBoost | 0.7970 | 0.3279 | 0.5571 | 0.2070 | 0.3018 | 0.2261 |
| **4** | Random Forest | 0.6743 | 0.3250 | 0.5524 | 0.2052 | 0.2993 | 0.2390 |
| **5** | Logistic Regression | 0.8973 | 0.3045 | 0.5196 | 0.1930 | 0.2815 | 0.1650 |

![Image: image9.png](data:image/png;base64...)

*Figure 52.2 — The selected operating points cluster tightly; LightGBM provides the strongest combined validation result.*

## Why MCC is primary

Matthews correlation coefficient uses all four confusion-matrix cells and remains informative when one class dominates. It reaches +1 for perfect classification, 0 for chance-level correlation, and −1 for complete disagreement. Unlike accuracy, it penalizes a model that obtains a high percentage merely by predicting the majority class. PR-AUC is retained as a secondary ranking measure because it evaluates how well probabilities rank rare failures across thresholds.

| **Metric** | **What it answers** | **Use in this project** |
| --- | --- | --- |
| **MCC** | Are predicted and observed classes correlated across both classes? | Primary model and threshold ranking |
| **Precision** | How many alerts are true failures? | Inspection efficiency and alert credibility |
| **Recall** | How many failures are detected? | False-negative exposure |
| **F1** | How balanced are precision and recall? | Secondary threshold summary |
| **PR-AUC** | How well are rare failures ranked over all thresholds? | Tie-breaker and probability-ranking quality |
| **Accuracy** | How many total labels are correct? | Not suitable as a primary metric under this imbalance |

## Selection result

| **Decision:** LightGBM is selected because it has the highest validation MCC (0.3386) and the highest PR-AUC (0.2529). XGBoost is a close challenger with MCC 0.3359. The difference is small enough that locked-holdout evidence and operational constraints remain decisive. |
| --- |

## Evaluation governance

* Create training, validation, and a physically separate immutable holdout before target-informed feature selection for any production release.
* Learn selected features, categorical levels, imputers, scalers, hyperparameters, and thresholds from training folds only.
* Use validation for model-selection decisions; evaluate the frozen pipeline exactly once on the locked holdout.
* Record split seed, row and class counts, data hash, code commit, feature-list version, model hash, and threshold for every release candidate.
* Keep Kaggle test files separate: they are inference inputs and cannot prove generalization because labels are unavailable.

## Operational threshold design

The best statistical threshold is not automatically the best factory threshold. Production deployment should convert the probability ranking into a review policy based on inspection capacity, cost per review, cost of a missed failure, intervention effectiveness, label delay, and safety requirements. Multiple operating points may be needed—for example, a high-risk mandatory review tier and a lower-risk sampling tier.

| **Decision input** | **Question to resolve before deployment** |
| --- | --- |
| **Inspection capacity** | How many products can be reviewed per shift without delaying production? |
| **False-negative tolerance** | What is the consequence of a missed defect for quality, warranty, and safety? |
| **False-positive cost** | What labor, test, and throughput cost is created by an unnecessary review? |
| **Score timing** | Are all 323 inputs available before the review decision must be made? |
| **Calibration** | Does a score of 0.80 correspond to a stable observed failure frequency in current factory data? |
| **Drift response** | Which feature, score, and outcome changes trigger investigation or retraining? |

## Release checklist

1. Rebuild the pipeline from pinned dependencies and verify the 323-feature input contract.
2. Run unit tests and the golden-set regression test; confirm serialized-model hashes.
3. Evaluate the selected candidate on the locked labelled holdout and publish confidence intervals and error analysis.
4. Review false negatives by product family, route, station presence, and production-time window.
5. Validate probability calibration and choose a capacity-aware operating threshold with Quality Engineering.
6. Deploy initially in shadow mode with audit logging and no automated disposition.
7. Monitor input schema, missingness, score distribution, alert rate, precision, recall, label delay, and operator overrides.
8. Define rollback criteria and retain the Logistic Regression and XGBoost challengers for comparison.

# Part VI Summary

The Phase 6 comparison demonstrates a clear progression from a linear baseline to nonlinear ensembles. Every model learns useful signal, but tree methods improve the balance of precision, recall, and minority-class ranking. LightGBM provides the strongest validation result and becomes the repository’s production-safe benchmark, while XGBoost remains a close challenger.

| **Model** | **Role in the handbook** | **Validation conclusion** |
| --- | --- | --- |
| **Logistic Regression** | Transparent linear baseline | Confirms learnable signal; limited nonlinear capacity |
| **Random Forest** | Bagged-tree benchmark | Improves MCC and PR-AUC over the baseline |
| **XGBoost** | Regularized boosting challenger | Second-best MCC; close to the selected model |
| **CatBoost** | Category-aware boosting challenger | Strong but below XGBoost and LightGBM on the common matrix |
| **LightGBM** | Selected benchmark | Best MCC and PR-AUC in Phase 6 |
| **Model Selection** | Threshold and governance layer | MCC-first ranking with a locked-holdout requirement |

## Repository traceability

| **Purpose** | **Repository artifact** |
| --- | --- |
| **Training implementation** | src/data/phase6\_predictive\_failure\_modeling.py |
| **Model comparison** | reports/phase6\_model\_comparison\_metrics.csv |
| **Technical report** | reports/phase6\_predictive\_failure\_modeling\_report.md |
| **Train/validation matrices** | data/processed/phase6\_train\_dataset.csv and phase6\_validation\_dataset.csv |
| **Test scores** | data/processed/phase6\_test\_predictions.csv |
| **Selected model bundle** | models/phase6\_best\_model.joblib |
| **Evaluation policy** | docs/data\_strategy\_and\_test\_set\_policy.md |
| **Intended-use limits** | docs/model\_card.md |

| **Final boundary:** The model is a decision-support tool for prioritizing manual quality review. It is not a safety interlock, automated product disposition, or causal root-cause engine. |
| --- |