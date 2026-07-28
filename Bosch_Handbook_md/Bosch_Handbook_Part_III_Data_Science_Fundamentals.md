**BOSCH PRODUCTION LINE
PERFORMANCE PROJECT HANDBOOK**

**PART III**

**Data Science Fundamentals Through the Implemented Project**

Chapters 16-26

**Author: Krishnakanth Reddy Karingula**

Project repository: kkr199/Bosch-Production-line-performance

*Repository-grounded edition*

**CHAPTER PART III**

**How This Part Was Written**

*A strict repository-only learning guide*

|  |
| --- |
| **Scope rule:** From Part III onward, the handbook uses only the work implemented in the Bosch project repository and the reports, datasets, model metrics, and artifacts produced by that project. External books, unrelated examples, and third-party implementations are intentionally excluded. |

This part explains data-science fundamentals by tracing the exact decisions made in the project. A concept is introduced only through a project operation, output, limitation, or control. When a requested topic was not fully implemented, the handbook says so directly instead of presenting it as completed work.

# What the reader will learn

* How the project converted the Bosch quality task into a binary classification problem.
* How extreme class imbalance changed sampling, model weighting, thresholding, and metric selection.
* Why missing values were preserved as process information while also being imputed for model compatibility.
* Which duplicate and outlier controls are present, and which controls are not claimed.
* How the project engineered and selected the final 323 model features.
* How the train-validation split works and why Kaggle test data is not a labelled holdout.
* What cross-validation was and was not performed.
* How the project separated production-safe results from competition-style leakage experiments.
* How MCC, precision, recall, F1, PR-AUC, and the selected threshold describe the final model.

# Contents

| **Chapter** | **Topic** | **Project focus** |
| --- | --- | --- |
| 16 | Classification Problems | Response target, probabilities, five classifiers |
| 17 | Class Imbalance | Sampling, weights, stratification and threshold tuning |
| 18 | Missing Values | Profiling, missing indicators and median imputation |
| 19 | Duplicates | Id integrity, split isolation and duplicate columns |
| 20 | Outliers | Distribution analysis without blanket deletion |
| 21 | Feature Engineering | Phases 2, 4 and 5 engineered manufacturing signals |
| 22 | Feature Selection | Numeric, categorical and final feature selection |
| 23 | Train/Validation/Test Split | Actual Phase 6 split and Kaggle test separation |
| 24 | Cross-Validation | Current one-split design and documented limits |
| 25 | Data Leakage | Base-model caveat and leaderboard leak experiment |
| 26 | Evaluation Metrics | Confusion matrix, MCC and model comparison |

**CHAPTER 16**

**Classification Problems**

*Turning manufacturing quality records into a binary prediction task*

|  |
| --- |
| **Learning objectives:** By the end of this chapter, the reader should be able to: identify the prediction target and unit of analysis used by the project; distinguish a probability prediction from a final binary alert; explain why the Phase 6 task is classification rather than regression or clustering; describe the five algorithms compared in the official modeling phase. |

|  |
| --- |
| **Project evidence used:** src/data/phase6\_predictive\_failure\_modeling.py; reports/phase6\_predictive\_failure\_modeling\_report.md; reports/phase6\_model\_comparison\_metrics.csv. No external dataset, textbook, or third-party implementation is used in this chapter. |

# 16.1 The exact question answered by the model

Every modelling row represents one manufactured part identified by Id. The labelled training data contains the binary column Response. In this project, Response = 0 means that the row is not labelled as an internal failure, while Response = 1 means that an internal failure was recorded. The machine-learning task is therefore to learn a mapping from the available measurements and engineered manufacturing indicators to the probability of Response = 1.

|  |
| --- |
| **Project question:** Given the measurements and process indicators available for one Id, what is the estimated probability that Response equals 1? |

![](data:image/png;base64...)

*Figure 16.1. The classification workflow implemented in Phase 6.*

# 16.2 The input, target and output

| **Element** | **What the project uses** | **Meaning in the pipeline** |
| --- | --- | --- |
| Unit of analysis | Id | One manufactured part or product record |
| Target | Response | Binary label used only for labelled training rows |
| Raw numeric inputs | Selected Lx\_Sy\_Fz columns | Anonymous numerical measurements |
| Categorical inputs | Selected values expanded into indicators | Anonymous categorical states and missingness |
| Engineered inputs | Phase 4 timing/path fields and Phase 5 family labels | Compact process-structure indicators derived by the project |
| Model output | failure\_probability | Continuous score between 0 and 1 |
| Final output | predicted\_failure\_at\_selected\_threshold | Binary class obtained by comparing probability with the selected threshold |

# 16.3 Why this is not regression

Regression predicts a continuous target such as a measured quantity. The project does produce a continuous failure probability, but that probability is the output of a classifier; it is not the original target. The original target has only two values. The final decision is also binary after thresholding. For that reason the project evaluates classification metrics rather than regression measures.

# 16.4 Why this is not the same as the project clustering work

Phase 5 applies clustering to station-presence patterns to discover product families. That is an unsupervised task because it creates groups without using Response as the group label. Phase 6 then uses the resulting family labels as additional inputs to a supervised classifier. The same project therefore contains both clustering and classification, but they solve different problems.

| **Phase** | **Learning type** | **Output** | **How it is used** |
| --- | --- | --- | --- |
| Phase 5 | Unsupervised clustering | Product-family labels | Describe recurring station-presence patterns |
| Phase 6 | Supervised classification | Failure probability and binary alert | Estimate Response using labelled examples |

# 16.5 Models actually compared

The official Phase 6 script trains five classifiers on the same engineered model table. Each model is wrapped in a pipeline so preprocessing and prediction use the same feature order. The project compares Logistic Regression, Random Forest, XGBoost, LightGBM and CatBoost. LightGBM is selected because it ranks first by MCC and then by PR-AUC.

| **Model** | **Project preprocessing** | **Imbalance control** |
| --- | --- | --- |
| Logistic Regression | Median imputation and StandardScaler | class\_weight="balanced" |
| Random Forest | Median imputation | class\_weight="balanced\_subsample" |
| XGBoost | Median imputation | scale\_pos\_weight calculated from training class counts |
| LightGBM | Median imputation | class\_weight="balanced" |
| CatBoost | Median imputation | auto\_class\_weights="Balanced" |

**Simplified Phase 6 classification sequence**

|  |
| --- |
| feature\_cols = [c for c in train\_frame.columns if c not in {"Id", "Response"}] x\_train = train\_frame[feature\_cols].replace([np.inf, -np.inf], np.nan) y\_train = train\_frame["Response"].astype(np.int8) model.fit(x\_train, y\_train) scores = model.predict\_proba(x\_valid)[:, 1] pred = (scores >= selected\_threshold).astype(int) |

# 16.6 Training and inference are different operations

During training, Response is available and the algorithms learn relationships between inputs and the target. During inference on the Kaggle test files, Response is unavailable. The saved model produces failure\_probability and a thresholded binary prediction. The project therefore keeps the target outside the feature list and saves feature\_cols inside the model bundle so test data is aligned to the training schema.

|  |
| --- |
| **Important:** A probability is not a confirmed defect. It is a model score derived from patterns learned in historical labelled data. |

## Common mistakes to avoid

|  |
| --- |
| **Mistake:** Calling the project a regression problem because the classifier outputs a probability. The target itself is binary. |

|  |
| --- |
| **Mistake:** Using Response as an input feature. The Phase 6 feature list explicitly excludes Id and Response. |

|  |
| --- |
| **Mistake:** Treating a thresholded alert as a physical diagnosis. The model predicts the label; it does not reveal a confirmed failure mechanism. |

|  |
| --- |
| **Mistake:** Confusing Phase 5 product-family clustering with the Phase 6 supervised failure classifier. |

## Review questions

**1.** What is the unit of analysis in the project?

**2.** What is the difference between failure\_probability and predicted\_failure\_at\_selected\_threshold?

**3.** Why are Id and Response excluded from feature\_cols?

**4.** Which five classifiers were compared in Phase 6?

## Practical exercise using this project

|  |
| --- |
| **Exercise:** Open phase6\_train\_dataset.csv and write down the number of input columns after excluding Id and Response. Then classify five sample columns as raw numeric, missingness indicator, categorical expansion, or engineered feature. |

## Chapter summary

* The project solves a binary classification problem at the Id level.
* Response is the target; the model outputs a probability and then a binary class.
* Phase 5 clustering and Phase 6 classification are separate but connected tasks.
* Five classifiers were compared, and LightGBM was selected using project metrics.

**CHAPTER 17**

**Class Imbalance**

*Learning from a dataset in which failures are extremely rare*

|  |
| --- |
| **Learning objectives:** By the end of this chapter, the reader should be able to: calculate and interpret the full and sampled failure rates; explain why accuracy is not used as the model-selection metric; describe sampling, stratification, class weighting and threshold tuning in the project; read the final LightGBM confusion matrix. |

|  |
| --- |
| **Project evidence used:** reports/phase3\_exploratory\_data\_analysis\_report.md; src/data/phase6\_predictive\_failure\_modeling.py; reports/phase6\_model\_comparison\_metrics.csv; data/processed/phase6\_sampled\_training\_ids.csv. No external dataset, textbook, or third-party implementation is used in this chapter. |

# 17.1 The imbalance in the original labelled data

The full labelled training population contains 1,183,747 rows and only 6,879 recorded failures. The remaining 1,176,868 rows have Response = 0. The resulting failure rate is 0.5811%. This means that fewer than six rows in every thousand are positive examples.

![](data:image/png;base64...)

*Figure 17.1. Actual target counts reported by the project.*

|  |
| --- |
| **Why this matters:** A classifier that predicts Response = 0 for every row would achieve about 99.4189% accuracy while finding none of the 6,879 failures. |

# 17.2 The project sampling strategy

Training every experiment on all 1,183,747 labelled rows and thousands of candidate features would be expensive. The Phase 6 script therefore retains all 6,879 positive rows and randomly samples 220,000 negative rows using random\_state = 42. The resulting modeling population contains 226,879 rows and has a positive rate of 3.0320%.

![](data:image/png;base64...)

*Figure 17.2. The three imbalance controls used by the project.*

| **Population** | **Rows** | **Failures** | **Failure rate** |
| --- | --- | --- | --- |
| Full labelled Bosch training data | 1,183,747 | 6,879 | 0.5811% |
| Phase 6 modeling sample | 226,879 | 6,879 | 3.0320% |
| Phase 6 training split | 170,159 | 5,159 | 3.0319% |
| Phase 6 validation split | 56,720 | 1,720 | 3.0324% |

# 17.3 Stratification preserves the class ratio

The train\_test\_split call uses stratify=frame["Response"]. This makes the training and validation partitions contain nearly identical positive rates. Without stratification, a random split could place too few failure examples in one partition, making evaluation unstable.

# 17.4 Model-specific weighting

Sampling changes the modeling population, but the classes remain unequal. The project additionally uses class-weighting options supplied by each algorithm. XGBoost receives scale\_pos\_weight calculated from the negative-to-positive ratio in the training split. The other classifiers use their balanced weighting options. These settings increase the cost of incorrectly classifying a failure during model fitting.

**Implemented class-weight calculations**

|  |
| --- |
| pos = int(y\_train.sum()) neg = int(len(y\_train) - pos) scale\_pos\_weight = neg / max(1, pos)  LGBMClassifier(class\_weight="balanced", ...) XGBClassifier(scale\_pos\_weight=scale\_pos\_weight, ...) |

# 17.5 The threshold is not fixed at 0.5

The selected LightGBM threshold is 0.785083. The project searches 80 candidate thresholds based on validation-score quantiles from 0.50 to 0.995 and selects the threshold with the highest MCC. This is essential because the best binary decision point in an imbalanced problem is not automatically 0.5.

# 17.6 What the final confusion matrix means

![](data:image/png;base64...)

*Figure 17.3. Validation outcomes corresponding to the reported LightGBM metrics.*

| **Outcome** | **Count** | **Meaning** |
| --- | --- | --- |
| True positive | 367 | A recorded failure was correctly alerted |
| False positive | 272 | A non-failure row was alerted |
| False negative | 1353 | A recorded failure was not alerted |
| True negative | 54728 | A non-failure row was correctly left without an alert |

The model detects 367 of the 1,720 failures in validation, which produces recall of 0.2134. Among the 639 predicted alerts, 367 are true failures, producing precision of 0.5743. These results explain why the project reports multiple metrics rather than relying on one number.

## Common mistakes to avoid

|  |
| --- |
| **Mistake:** Reporting the majority-class accuracy as evidence that the model works. |

|  |
| --- |
| **Mistake:** Removing positive examples to make the classes equal. The project keeps every known failure. |

|  |
| --- |
| **Mistake:** Assuming class weighting replaces the need for appropriate evaluation metrics. |

|  |
| --- |
| **Mistake:** Applying a default 0.5 threshold without checking MCC on validation predictions. |

|  |
| --- |
| **Mistake:** Comparing the sampled failure rate with the full population rate as though they describe the same population. |

## Review questions

**1.** How many failures are in the full labelled training data?

**2.** Why does the modeling sample have a higher failure rate than the full data?

**3.** What does stratification preserve?

**4.** Why is the final threshold greater than 0.5?

**5.** What is the difference between a false positive and a false negative in this project?

## Practical exercise using this project

|  |
| --- |
| **Exercise:** Using the four confusion-matrix counts in Figure 17.3, calculate precision and recall manually. Compare your answers with phase6\_model\_comparison\_metrics.csv. |

## Chapter summary

* The original failure rate is only 0.5811%.
* The project keeps all failures and samples 220,000 non-failures for computationally manageable modeling.
* Stratification and model weights address different parts of the imbalance problem.
* Threshold tuning by MCC turns probabilities into binary alerts.
* Accuracy is not suitable as the primary selection metric for this project.

**CHAPTER 18**

**Missing Values**

*Treating absence as both a data condition and a manufacturing signal*

|  |
| --- |
| **Learning objectives:** By the end of this chapter, the reader should be able to: describe the missing-value profiling completed in Phase 1 and Phase 2; explain why missingness can represent product routing; distinguish explicit missing indicators from median imputation; identify how missing categorical values are encoded. |

|  |
| --- |
| **Project evidence used:** src/data/phase1\_data\_quality.py; reports/phase1\_missing\_values\_by\_column.csv; reports/phase2\_data\_understanding\_engineering\_report.md; src/data/phase6\_predictive\_failure\_modeling.py. No external dataset, textbook, or third-party implementation is used in this chapter. |

# 18.1 Missingness was measured before modeling

Phase 1 reads each large CSV in chunks and counts missing values for every column. It records total rows, columns, feature columns, file size, target presence, total missing cells and missing percentage. This produces phase1\_file\_summary.csv and phase1\_missing\_values\_by\_column.csv rather than loading the complete multi-file dataset into memory at once.

**Phase 1 chunked missing-value profiling**

|  |
| --- |
| reader = pd.read\_csv(path, chunksize=chunksize, low\_memory=False) for chunk in reader:  rows += len(chunk)  missing\_counts = missing\_counts.add(chunk.isna().sum(), fill\_value=0)  missing\_pct = total\_missing / (rows \* number\_of\_columns) \* 100 |

# 18.2 Missing does not always mean broken

The project discovered that many station and data-type groups are extremely sparse. Phase 2 uses date-feature presence to infer whether a part appears to have passed through a station. A missing measurement may therefore mean that the part followed a different route, skipped a station, or did not receive a particular measurement. Deleting every sparse column or filling all missing values without an indicator would remove this process information.

| **Observed project example** | **Completeness** | **Safe interpretation** |
| --- | --- | --- |
| Train categorical L3\_S46 | approximately 0.0000% | Almost no observed values; not enough evidence to assign a physical reason |
| Train numeric L3\_S37 | 94.6481% | Most rows have measurements at this station group |
| Train date L3\_S29 | 94.3811% | Station timing is present for most rows |
| L3\_S32 missingness indicator | selected as a strong model signal | Missingness pattern is predictive, but not a confirmed root cause |

![](data:image/png;base64...)

*Figure 18.1. The two missing-value channels used in Phase 6.*

# 18.3 Explicit missing indicators

For each selected numeric feature whose missing rate exceeds 5%, the project adds a binary column ending in \_\_is\_missing. A value of 1 means the original measurement is absent; a value of 0 means it is present. The final model matrix contains 80 numeric missingness indicators. The 20 selected categorical columns also receive explicit \_\_is\_missing indicators.

**Missing-indicator creation in the project**

|  |
| --- |
| for column in selected\_numeric:  if report.loc[column, "missing\_rate"] > 0.05:  frame[f"{column}\_\_is\_missing"] = frame[column].isna().astype(np.uint8)  values = categorical\_chunk[column].fillna("\_\_MISSING\_\_") encoded[f"{column}\_\_is\_missing"] = (values == "\_\_MISSING\_\_").astype(np.uint8) |

# 18.4 Median imputation inside every model pipeline

The explicit indicator preserves whether the value was missing, but most classifiers still require a numeric value in the original feature position. The project therefore uses SimpleImputer(strategy="median") inside every model pipeline. The median is learned during model fitting and applied consistently during validation and test scoring.

|  |
| --- |
| **Two different purposes:** The indicator answers “was the original value absent?” The imputed value answers “what numeric placeholder can the algorithm process?” They should not be confused. |

# 18.5 Composition of the final feature matrix

| **Feature group** | **Count** | **Missing-value role** |
| --- | --- | --- |
| Selected raw numeric measurements | 80 | Original values may be missing and are median-imputed |
| Numeric \_\_is\_missing indicators | 80 | Preserve predictive absence patterns |
| Categorical indicator features | 110 | Include category-level and missing-category indicators |
| Phase 4/5 engineered features | 53 | Include completeness, station count, line count and family/path information |
| Total | 323 | Final Phase 6 model input count |

# 18.6 What the project does not claim

The project does not claim that every missing value is meaningful. Some absence may be ordinary sparsity, unavailable measurement, file structure, or true data-quality loss. The model can learn an association, but engineering documentation would be required to determine the operational reason.

## Common mistakes to avoid

|  |
| --- |
| **Mistake:** Dropping every column with a high missing rate before checking whether absence describes station routing. |

|  |
| --- |
| **Mistake:** Median-imputing without retaining an indicator when missingness itself may be predictive. |

|  |
| --- |
| **Mistake:** Calling a missingness signal a confirmed physical root cause. |

|  |
| --- |
| **Mistake:** Fitting an imputer separately on validation or test data. |

|  |
| --- |
| **Mistake:** Confusing the categorical token \_\_MISSING\_\_ with an original Bosch category value. |

## Review questions

**1.** What files are produced by Phase 1 missing-value profiling?

**2.** Why can date-feature absence help infer station presence?

**3.** What condition causes a numeric \_\_is\_missing feature to be created?

**4.** Why does the pipeline still need median imputation after creating indicators?

**5.** How many numeric missingness indicators are in the final model table?

## Practical exercise using this project

|  |
| --- |
| **Exercise:** Choose five columns ending in \_\_is\_missing from phase6\_train\_dataset.csv. For each column, locate its original feature and compare the average Response for indicator value 0 versus 1. |

## Chapter summary

* Missingness was profiled across all six raw datasets in chunks.
* Station and route structure makes some missing values informative.
* The model preserves absence using indicators and supplies processable values using median imputation.
* Missingness is predictive evidence, not proof of a physical cause.

**CHAPTER 19**

**Duplicates**

*Protecting identifier, split and feature-table integrity*

|  |
| --- |
| **Learning objectives:** By the end of this chapter, the reader should be able to: distinguish duplicate identifiers, duplicate rows, duplicate columns and repeated values; describe the duplicate controls present in the generated artifacts and code; explain why repeated measurement values are not automatically deleted; state the limits of the project duplicate audit. |

|  |
| --- |
| **Project evidence used:** src/data/phase3\_exploratory\_data\_analysis.py; src/data/phase6\_predictive\_failure\_modeling.py; data/processed/phase6\_sampled\_training\_ids.csv; data/processed/phase6\_train\_dataset.csv; data/processed/phase6\_validation\_dataset.csv. No external dataset, textbook, or third-party implementation is used in this chapter. |

# 19.1 Duplicate can mean four different things

| **Duplicate type** | **Project meaning** | **Required action** |
| --- | --- | --- |
| Duplicate Id | The same manufactured-part identifier appears more than once | Investigate before modeling because Id is the join key |
| Duplicate complete row | All stored values for two rows are identical | Check whether it is a repeated record or legitimate repeated observation |
| Duplicate column name | A merge creates two columns with the same name | Remove or rename to prevent ambiguous model input |
| Repeated feature value | Many parts have the same measurement or category | Usually retain; identical values are not automatically duplicate records |

![](data:image/png;base64...)

*Figure 19.1. Duplicate controls verified in the current project.*

# 19.2 Identifier alignment across raw files

Phase 3 reads train\_date.csv and train\_numeric.csv in matching chunks and verifies that their Id sequences are identical before combining target and date information. The categorical EDA performs the same order check between train\_categorical.csv and train\_numeric.csv. If the Id order does not match, the script raises an error instead of silently combining unrelated rows.

**Row-alignment protection used in Phase 3**

|  |
| --- |
| if not date\_chunk["Id"].astype("int64").equals(  target\_chunk["Id"].astype("int64")):  raise ValueError("Id order mismatch between train\_date.csv and train\_numeric.csv") |

# 19.3 Duplicate column protection after merges

Phase 6 merges raw numeric selections, engineered Phase 4/5 features and categorical indicators by Id. After the merge, it checks frame.columns.duplicated() and keeps one copy of each duplicated name. This protects the model from ambiguous feature names created by overlapping source tables.

**Duplicate-column control in Phase 6**

|  |
| --- |
| frame = numeric.merge(engineered, on="Id", how="left").merge(  categorical, on="Id", how="left") duplicated = frame.columns[frame.columns.duplicated()].tolist() if duplicated:  frame = frame.loc[:, ~frame.columns.duplicated()] |

# 19.4 Artifact verification

The generated project artifacts were checked for this handbook. phase6\_sampled\_training\_ids.csv, phase6\_train\_dataset.csv and phase6\_validation\_dataset.csv each contain zero duplicated Id values. The training and validation Id sets also have zero overlap. These checks confirm the integrity of the stored Phase 6 split artifacts.

| **Artifact** | **Rows** | **Duplicate Id values** | **Duplicate Id overlap** |
| --- | --- | --- | --- |
| phase6\_sampled\_training\_ids.csv | 226,879 | 0 | Not applicable |
| phase6\_train\_dataset.csv | 170,159 | 0 | 0 with validation |
| phase6\_validation\_dataset.csv | 56,720 | 0 | 0 with training |

# 19.5 What is not claimed

The Phase 1 script profiles size, shape, target presence and missing values, but it does not implement a full raw-file duplicate-row report. The handbook therefore does not claim that every possible duplicate pattern in the six original CSV files was exhaustively audited. It only documents the controls and artifact checks that exist.

|  |
| --- |
| **Why repeated values remain:** Manufacturing measurements can legitimately repeat because multiple parts can receive the same category, follow the same route, or record the same anonymous value. Removing rows merely because values repeat would risk deleting valid data. |

## Common mistakes to avoid

|  |
| --- |
| **Mistake:** Treating every repeated numeric value as a duplicate row. |

|  |
| --- |
| **Mistake:** Joining raw files by their current row position without checking Id alignment. |

|  |
| --- |
| **Mistake:** Allowing duplicate feature names after merging multiple engineered tables. |

|  |
| --- |
| **Mistake:** Claiming a complete raw duplicate audit when only generated model artifacts were verified. |

|  |
| --- |
| **Mistake:** Removing rows before understanding whether Id represents a unique part or a repeated event. |

## Review questions

**1.** What are the four duplicate types discussed in this chapter?

**2.** How does Phase 3 protect against row-order mismatch?

**3.** What does Phase 6 do with duplicated column names?

**4.** How many Ids overlap between the saved training and validation files?

**5.** Why are repeated feature values not sufficient evidence for row deletion?

## Practical exercise using this project

|  |
| --- |
| **Exercise:** Read Id from the Phase 6 training and validation CSV files. Confirm that each file has unique Id values and that the intersection of the two Id sets is empty. |

## Chapter summary

* The project treats Id as the core row and join key.
* Raw-file chunk alignment is checked before target information is combined.
* Duplicate column names are removed after model-table merges.
* Saved Phase 6 training and validation artifacts contain unique, non-overlapping Ids.
* A complete raw duplicate-row audit is not claimed.

**CHAPTER 20**

**Outliers**

*Describing extreme observations without inventing physical limits*

|  |
| --- |
| **Learning objectives:** By the end of this chapter, the reader should be able to: explain how distributions were examined in Phase 3; distinguish statistical extremes from physically invalid values; describe the project treatment of infinity and missing values before modeling; state why blanket outlier deletion was not used. |

|  |
| --- |
| **Project evidence used:** src/data/phase3\_exploratory\_data\_analysis.py; reports/phase3\_distribution\_report.csv as generated by Phase 3; reports/figures/phase3\_flow\_metric\_distributions.png; src/data/phase6\_predictive\_failure\_modeling.py. No external dataset, textbook, or third-party implementation is used in this chapter. |

# 20.1 The project first describes distributions

Phase 3 creates a distribution report for station\_count, line\_count, observed\_date\_values, date\_completeness\_pct, first\_event\_time, last\_event\_time and process\_duration. For Response = 0 and Response = 1 separately, it calculates count, mean, median, standard deviation, 10th percentile and 90th percentile. It also generates boxplots for selected flow metrics and divides temporal indicators into quantile bins for failure-rate analysis.

![](data:image/png;base64...)

*Figure 20.1. Actual Phase 3 boxplots generated by the project.*

| **Implemented diagnostic** | **Purpose in the project** |
| --- | --- |
| Mean and median | Compare central tendency and detect skewed distributions |
| Standard deviation | Describe spread around the mean |
| p10 and p90 | Describe the central 80% without defining values outside it as invalid |
| Boxplots by Response | Compare distribution shape between non-failure and failure rows |
| Quantile bins | Compare failure rates across relative-time and measurement-span ranges |

# 20.2 No blanket outlier removal is claimed

The project does not globally delete observations using z-scores, IQR fences, percentile clipping or winsorisation. This is an appropriate limitation for the anonymised Bosch data because the actual engineering units, sensor specifications and permissible operating ranges are unavailable. A large anonymous value may be a valid measurement, a rare process state, or a useful failure signal.

![](data:image/png;base64...)

*Figure 20.2. The implemented treatment of extreme observations.*

# 20.3 Invalid mathematical values are normalised

Before fitting each model, the Phase 6 code replaces positive and negative infinity with NaN. The pipeline then handles those values using median imputation. This operation prevents invalid floating-point values from breaking the algorithms without claiming that ordinary large finite values are incorrect.

**Implemented numerical sanitisation**

|  |
| --- |
| x\_train = train\_frame[feature\_cols].replace([np.inf, -np.inf], np.nan) x\_valid = valid\_frame[feature\_cols].replace([np.inf, -np.inf], np.nan)  Pipeline([  ("imputer", SimpleImputer(strategy="median")),  ("model", selected\_classifier), ]) |

# 20.4 Different models react differently

Logistic Regression is the only Phase 6 model that additionally uses StandardScaler. Random Forest, XGBoost, LightGBM and CatBoost are tree-based models in the implemented comparison and do not use the scaler in their pipelines. The project does not add a separate outlier-transform stage for these models.

# 20.5 Safe interpretation rule

|  |
| --- |
| **Interpretation rule:** Use terms such as “extreme observed value,” “rare distribution region,” or “high measurement-span bin.” Do not call a value physically abnormal unless a factory specification or engineering range confirms it. |

The date-derived process\_duration used in Phase 3 was later renamed more carefully in Phase 4 as an observed measurement span. This correction illustrates why statistical extremes must not be presented as confirmed delays or process violations.

## Common mistakes to avoid

|  |
| --- |
| **Mistake:** Deleting every point outside an IQR fence without knowing the physical measurement scale. |

|  |
| --- |
| **Mistake:** Calling an anonymous extreme value a sensor fault. |

|  |
| --- |
| **Mistake:** Treating a long observed measurement span as a confirmed production delay. |

|  |
| --- |
| **Mistake:** Applying StandardScaler to all models when the project only uses it in Logistic Regression. |

|  |
| --- |
| **Mistake:** Confusing replacement of infinity with general outlier removal. |

## Review questions

**1.** Which seven metrics are included in the Phase 3 distribution report?

**2.** Why does the project avoid blanket outlier deletion?

**3.** How are positive and negative infinity handled before modeling?

**4.** Which model pipeline includes StandardScaler?

**5.** What wording should be used for extreme timestamp-derived values?

## Practical exercise using this project

|  |
| --- |
| **Exercise:** Using phase3\_train\_time\_features.csv, reproduce the median, p10 and p90 of process\_duration separately for Response 0 and Response 1. Do not remove any finite observations. |

## Chapter summary

* Phase 3 describes distributions using summary statistics, boxplots and quantile bins.
* No universal outlier-removal rule is claimed.
* Infinity is converted to missing and handled by the existing imputation pipelines.
* Physical abnormality cannot be inferred from anonymous feature values alone.

**CHAPTER 21**

**Feature Engineering**

*Converting anonymous measurements into compact process indicators*

|  |
| --- |
| **Learning objectives:** By the end of this chapter, the reader should be able to: trace feature engineering across Phases 2, 4 and 5; explain the main engineered feature groups; describe how product-family labels enter the Phase 6 model; use the corrected timestamp terminology. |

|  |
| --- |
| **Project evidence used:** reports/phase2\_data\_understanding\_engineering\_report.md; reports/phase4\_feature\_engineering\_report.md; reports/phase4\_feature\_dictionary.csv; reports/phase5\_product\_family\_discovery\_report.md; src/data/phase6\_predictive\_failure\_modeling.py. No external dataset, textbook, or third-party implementation is used in this chapter. |

# 21.1 Why engineering was required

The parsed Bosch schema contains 4,264 anonymous features spread across 4 lines and 52 stations. Training only on raw columns would leave the model with many sparse signals and little direct process structure. The project therefore converts raw date presence and relative timestamps into compact indicators that describe station participation, line participation, observed measurement spans, path complexity and recurring product families.

![](data:image/png;base64...)

*Figure 21.1. Feature engineering as implemented across the project phases.*

# 21.2 Phase 2: structure from names and presence

Phase 2 parses every Lx\_Sy\_Fz and Lx\_Sy\_Dz name into line, station and feature metadata. It creates feature, station and line metadata tables, completeness reports, and manufacturing\_flow\_train.parquet and manufacturing\_flow\_test.parquet. Station presence is derived from whether any date feature for the station is observed.

| **Phase 2 output** | **Project use** |
| --- | --- |
| phase2\_feature\_metadata.csv | Maps 4,264 columns to line, station, feature number and data type |
| phase2\_station\_metadata.csv | Summarises the 52 stations |
| phase2\_line\_metadata.csv | Summarises four production lines and feature counts |
| station completeness metrics | Shows observed and missing values by station and data type |
| manufacturing flow parquet files | Compact station-presence and route-oriented data for train and test |

# 21.3 Phase 4: date-derived and path features

Phase 4 creates 47 engineered input columns from the raw date CSV files. The training output has 49 columns because it also contains Id and Response; the test output has 48 columns because Response is unavailable. The feature dictionary explicitly states that timestamps are relative and anonymised and that gap-based fields are temporal proxies rather than verified delays.

| **Feature group** | **Implemented examples** | **Correct interpretation** |
| --- | --- | --- |
| Measurement-time bounds | start\_time, end\_time | Earliest and latest observed measurement timestamp |
| Measurement spans | cycle\_time, processing\_duration | Observed timestamp span, not verified production cycle time |
| Gap proxies | waiting\_time, mean\_waiting\_time, max\_waiting\_time | Positive inter-station timestamp gaps, not confirmed queues |
| Route size | station\_count, line\_count, station\_span | Number and range of observed stations/lines |
| Route complexity | path\_density, line\_switch\_count, path\_complexity\_score | Composite description of the observed path |
| Completeness | date\_completeness\_pct and line-level completeness | Share of raw date observations present |
| Line aggregates | line\_n\_present, start/end, spans and station count | Same concepts calculated within each line |

|  |
| --- |
| **Reader-facing terminology:** The dashboard translates start\_time to “Earliest Measurement Timestamp,” cycle\_time to “Observed Measurement Span,” and waiting\_time to “Observed Measurement Gap.” These names describe what was calculated without inventing a physical cause. |

# 21.4 Phase 5: product-family features

Phase 5 clusters products using station-presence patterns and creates kmeans\_family, dbscan\_family, hierarchical\_family and final\_product\_family labels. It also carries station\_count and line\_count. Phase 6 merges selected Phase 5 columns with the Phase 4 engineered table. The family labels provide a compact representation of recurring route patterns.

# 21.5 Final engineered contribution to Phase 6

The final 323-feature table contains 53 Phase 4/5 engineered fields. These are combined with 80 selected raw numeric measurements, 80 numeric missingness indicators and 110 categorical expansion columns. Engineering and selection therefore work together: engineering adds process-oriented variables, while selection reduces the raw feature space.

**How Phase 6 combines engineered and selected raw features**

|  |
| --- |
| engineered = load\_engineered\_date\_and\_path\_features(ids, split) categorical = load\_categorical\_ohe(ids, selected\_cat, level\_map, split) frame = numeric.merge(engineered, on="Id", how="left").merge(  categorical, on="Id", how="left") |

# 21.6 Prediction is not causal interpretation

Engineered features make patterns easier to model and explain, but they do not create physical meaning that is absent from the source data. A high SHAP value for Earliest Measurement Timestamp means that the model uses that temporal indicator. It does not prove that a late production start caused failure.

## Common mistakes to avoid

|  |
| --- |
| **Mistake:** Renaming start\_time as an official factory start time. |

|  |
| --- |
| **Mistake:** Calling measurement gaps confirmed waiting time or delay. |

|  |
| --- |
| **Mistake:** Assuming a cluster label is a known Bosch product model. |

|  |
| --- |
| **Mistake:** Using engineered features without documenting their source and calculation. |

|  |
| --- |
| **Mistake:** Claiming that engineered predictive variables are confirmed root causes. |

## Review questions

**1.** How many raw parsed features, production lines and stations were found in Phase 2?

**2.** How is station presence derived?

**3.** What is the safe interpretation of cycle\_time?

**4.** Which Phase 5 labels are merged into Phase 6?

**5.** How many engineered Phase 4/5 fields are in the final model table?

## Practical exercise using this project

|  |
| --- |
| **Exercise:** Open phase4\_feature\_dictionary.csv. Group every feature into measurement bounds, spans, gap proxies, route structure, completeness or line-level aggregation. Write one safe reader-facing name for each group. |

## Chapter summary

* Phase 2 exposes manufacturing structure from feature names and date presence.
* Phase 4 creates documented temporal, route and completeness indicators.
* Phase 5 adds product-family labels derived from station-presence clustering.
* Phase 6 merges 53 engineered fields with selected raw signals.
* Engineered variables remain predictive indicators rather than confirmed causes.

**CHAPTER 22**

**Feature Selection**

*Reducing thousands of sparse candidates to a manageable model table*

|  |
| --- |
| **Learning objectives:** By the end of this chapter, the reader should be able to: describe the numeric and categorical selection rules used in Phase 6; explain the final 323-feature composition; distinguish feature selection from feature engineering; identify the target-informed selection limitation. |

|  |
| --- |
| **Project evidence used:** src/data/phase6\_predictive\_failure\_modeling.py; reports/phase6\_numeric\_correlation\_report.csv; reports/phase6\_selected\_numeric\_features.csv; reports/phase6\_selected\_categorical\_features.csv; reports/phase6\_final\_feature\_correlation\_report.csv. No external dataset, textbook, or third-party implementation is used in this chapter. |

# 22.1 Why selection was necessary

The raw feature space is large, sparse and split across multiple files. The Phase 6 source code states that a naive all-column one-hot model would be too large. The project profiles candidate features in chunks, keeps the strongest raw signals, merges engineered variables and trains the comparison models on a 323-feature matrix.

![](data:image/png;base64...)

*Figure 22.1. The actual Phase 6 selection funnel.*

# 22.2 Numeric selection

Every numeric feature is profiled for observed-value correlation with Response and for the difference between failure rates when the feature is missing versus present. Candidates must have present\_rate of at least 0.005. The selection score is the larger of absolute value correlation and absolute missingness signal. The top 80 features are retained.

**Implemented numeric selection rule**

|  |
| --- |
| candidates = numeric\_report[numeric\_report["present\_rate"] >= 0.005] candidates["selection\_score"] = candidates[  ["abs\_value\_corr", "missing\_signal\_abs\_diff"] ].max(axis=1) selected = candidates.sort\_values("selection\_score", ascending=False).head(80) |

| **Rank** | **Feature** | **Selection score** |
| --- | --- | --- |
| 1 | L1\_S24\_F867 | 0.227125 |
| 2 | L1\_S24\_F1723 | 0.185300 |
| 3 | L1\_S24\_F839 | 0.140475 |
| 4 | L1\_S24\_F1695 | 0.134357 |
| 5 | L1\_S24\_F1632 | 0.127364 |
| 6 | L1\_S24\_F1604 | 0.104619 |
| 7 | L1\_S24\_F1758 | 0.093545 |
| 8 | L1\_S24\_F902 | 0.084625 |
| 9 | L1\_S24\_F1667 | 0.083643 |
| 10 | L1\_S24\_F1846 | 0.062711 |

# 22.3 Categorical selection and expansion

Categorical columns are first profiled by presence. The selection compares failure rates when a categorical column is present versus missing. The project initially chooses 20 columns with at least 200 observed values and a present rate between 0.0001 and 0.9999. It then ranks values within each chosen column and keeps up to 12 levels whose count is at least 50. Each chosen level becomes a binary \_\_eq\_ indicator, and each chosen column also receives an \_\_is\_missing indicator.

**Implemented categorical selection and level retention**

|  |
| --- |
| selected = categorical\_report[  (categorical\_report["present\_count"] >= 200)  & (categorical\_report["present\_rate"].between(0.0001, 0.9999)) ].head(20)  level\_map[column] = [  value for value, stat in ranked[:12] if stat["count"] >= 50 ] |

# 22.4 Final feature composition

| **Component** | **Feature count** | **How it enters the model** |
| --- | --- | --- |
| Selected raw numeric features | 80 | Top numeric signals |
| Numeric missingness indicators | 80 | Added for selected numeric columns with high missingness |
| Categorical level and missing indicators | 110 | Binary expansion of 20 selected categorical columns |
| Phase 4 and Phase 5 engineered fields | 53 | Temporal, path, completeness and family indicators |
| Final total | 323 | Stored feature\_cols in the model bundle |

![](data:image/png;base64...)

*Figure 22.2. Actual project correlation plot from Phase 3.*

# 22.5 Feature engineering versus feature selection

| **Operation** | **Project example** | **Purpose** |
| --- | --- | --- |
| Feature engineering | Create path\_complexity\_score from station, line, switch and density information | Add a new compact process descriptor |
| Feature selection | Keep the top 80 raw numeric features by selection\_score | Reduce the number of candidate raw measurements |
| Categorical expansion | Create \_\_eq\_ and \_\_is\_missing columns | Convert selected category information into numeric indicators |
| Final correlation report | Rank final model columns by correlation with Response | Document relationships before training |

# 22.6 Important validation limitation

In the current Phase 6 script, numeric profiling, categorical presence profiling and category-level ranking use the labelled dataset before the saved train-validation split is created. This means that feature-selection decisions have seen target information from rows that later enter validation. The repository governance policy subsequently states that target-informed selection should be fitted using training folds only. A stricter future refactor should therefore split first and perform selection inside the training partition.

|  |
| --- |
| **Honest project status:** The official model remains the project’s production-safe benchmark relative to the explicit leaderboard leak experiment, but its validation estimate should still be read with the target-informed selection limitation documented above. |

## Common mistakes to avoid

|  |
| --- |
| **Mistake:** Saying that all 4,264 raw features are used by the final model. |

|  |
| --- |
| **Mistake:** Confusing high correlation with physical causation. |

|  |
| --- |
| **Mistake:** Selecting categorical levels on validation data and then treating validation as unseen. |

|  |
| --- |
| **Mistake:** Counting engineered features and raw selected features as the same operation. |

|  |
| --- |
| **Mistake:** Dropping missingness indicators when missingness contributed to the selection score. |

## Review questions

**1.** What minimum present\_rate is applied to numeric candidates?

**2.** How is the numeric selection\_score calculated?

**3.** How many categorical columns are selected before level expansion?

**4.** What is the final feature count?

**5.** Why should target-informed selection be moved after the split in a stricter implementation?

## Practical exercise using this project

|  |
| --- |
| **Exercise:** Recreate the numeric selection ranking from phase6\_numeric\_correlation\_report.csv using the implemented filters and compare your top 20 with phase6\_selected\_numeric\_features.csv. |

## Chapter summary

* Phase 6 selects 80 raw numeric features and 20 categorical columns.
* Categorical levels are expanded into binary features, including missing indicators.
* The final matrix contains 323 features from four groups.
* Feature engineering creates new variables; feature selection reduces candidates.
* Target-informed selection before splitting is documented as a validation limitation.

**CHAPTER 23**

**Train, Validation and Test Split**

*Understanding the actual Phase 6 data partitions*

|  |
| --- |
| **Learning objectives:** By the end of this chapter, the reader should be able to: describe how the modelling sample is divided; explain stratification and random\_state; distinguish internal validation from the Kaggle test files; state the missing immutable-holdout limitation. |

|  |
| --- |
| **Project evidence used:** src/data/phase6\_predictive\_failure\_modeling.py; reports/phase6\_predictive\_failure\_modeling\_report.md; docs/data\_strategy\_and\_test\_set\_policy.md; data/processed/phase6\_train\_dataset.csv; data/processed/phase6\_validation\_dataset.csv. No external dataset, textbook, or third-party implementation is used in this chapter. |

# 23.1 The split happens after sampling and dataset construction

Phase 6 first creates a labelled modeling sample of 226,879 rows. This population retains all 6,879 known failures and samples 220,000 non-failures. The selected raw and engineered features are then assembled into one frame. train\_test\_split divides that frame into 170,159 training rows and 56,720 validation rows.

![](data:image/png;base64...)

*Figure 23.1. The saved Phase 6 split and class counts.*

**Actual split call in Phase 6**

|  |
| --- |
| train\_frame, valid\_frame = train\_test\_split(  frame,  test\_size=0.25,  random\_state=42,  stratify=frame["Response"], ) |

# 23.2 Role of each partition

| **Partition** | **Contains Response?** | **Project use** |
| --- | --- | --- |
| Phase 6 training split | Yes | Fit imputers, scaler where applicable, and model parameters |
| Phase 6 validation split | Yes | Compare five models and tune the binary threshold |
| Kaggle test\_numeric/categorical/date | No | Build features and generate batch predictions for 1,183,748 Ids |
| Future immutable holdout | Not implemented in current Phase 6 | Required by the later governance policy for a stronger release estimate |

# 23.3 What random\_state = 42 does

The negative sample and train-validation split both use random\_state = 42. This makes the generated sample and split reproducible when the same source data and software behavior are used. Reproducibility does not guarantee that the split is the best possible representation; it guarantees that the same procedure can be repeated.

# 23.4 What stratify does

The sampled positive rate is 3.0320%. The training rate is 3.0319% and the validation rate is 3.0324%. These nearly identical values show the effect of stratification. The validation partition contains 1,720 positive examples, enough to calculate the reported metrics.

# 23.5 The Kaggle test set is not a labelled evaluation set

The project scores 1,183,748 test Ids and writes phase6\_test\_predictions.csv. Because Response is unavailable for these rows, the project cannot calculate MCC, recall, precision or PR-AUC on the Kaggle test files locally. The repository data policy explicitly states that test\_\*.csv is an inference input and must not be treated as evidence of generalisation.

|  |
| --- |
| **Terminology rule:** “Validation” refers to the labelled 25% internal split. “Kaggle test” refers to the unlabelled competition scoring files. They are not interchangeable. |

# 23.6 Split integrity and current limitation

The saved training and validation files contain unique Id values and have zero Id overlap. However, the official Phase 6 workflow does not create a third labelled immutable holdout. The later data-strategy policy specifies a stronger future workflow: training, validation and a physically separate holdout, with the holdout evaluated once after model and threshold selection are frozen.

| **Check** | **Current result** |
| --- | --- |
| Training Id duplicates | 0 |
| Validation Id duplicates | 0 |
| Train-validation Id overlap | 0 |
| Class stratification | Implemented |
| Immutable labelled holdout | Not implemented in current Phase 6 |
| Kaggle test labels | Unavailable locally |

## Common mistakes to avoid

|  |
| --- |
| **Mistake:** Calling the Kaggle test files the internal validation set. |

|  |
| --- |
| **Mistake:** Calculating test metrics without true test labels. |

|  |
| --- |
| **Mistake:** Ignoring stratification in a rare-event split. |

|  |
| --- |
| **Mistake:** Changing random\_state repeatedly and reporting only the best result. |

|  |
| --- |
| **Mistake:** Claiming a locked holdout was completed when the current Phase 6 has only train and validation partitions. |

## Review questions

**1.** How many rows are in the saved training and validation files?

**2.** What percentage of the modelling sample is assigned to validation?

**3.** Why is stratify=Response important?

**4.** Why can local MCC not be calculated for the Kaggle test files?

**5.** What additional labelled partition is required by the governance policy?

## Practical exercise using this project

|  |
| --- |
| **Exercise:** Verify the training and validation row counts, positive counts and Id overlap using the saved CSV files. Write the three commands or Pandas expressions you used. |

## Chapter summary

* The modeling sample is divided into a stratified 75% training and 25% validation split.
* random\_state = 42 makes sampling and splitting reproducible.
* The Kaggle test files are unlabelled inference inputs, not a validation set.
* Saved train and validation Ids are unique and non-overlapping.
* A separate immutable labelled holdout remains a documented future requirement.

**CHAPTER 24**

**Cross-Validation**

*What the project validates today and what it does not claim*

|  |
| --- |
| **Learning objectives:** By the end of this chapter, the reader should be able to: state the exact validation design currently implemented; explain why the project does not claim k-fold cross-validation results; identify the risk of selecting a model and threshold on one validation split; describe the stronger validation plan documented in the repository. |

|  |
| --- |
| **Project evidence used:** src/data/phase6\_predictive\_failure\_modeling.py; docs/data\_strategy\_and\_test\_set\_policy.md; reports/phase6\_model\_comparison\_metrics.csv. No external dataset, textbook, or third-party implementation is used in this chapter. |

# 24.1 Current implementation: one fixed stratified split

The official Phase 6 modeling code imports train\_test\_split and does not use KFold, StratifiedKFold, cross\_val\_score or an equivalent repeated-fold procedure. Every model is fitted once on the same saved training partition and evaluated on the same saved validation partition. Threshold selection is also performed using that validation partition.

![](data:image/png;base64...)

*Figure 24.1. Cross-validation status of the current repository.*

# 24.2 Why this still provides useful evidence

A fixed split creates a consistent comparison: all five algorithms see the same training rows, validation rows and feature columns. It produces reproducible artifacts and allows the project to compare model families within a computationally manageable workflow. The result is useful as a benchmark and portfolio POC estimate.

# 24.3 Why it is weaker than repeated validation

The reported scores depend on one particular allocation of rows. A different stratified split could produce a different ranking or threshold. In addition, the project searches the best MCC threshold on the same validation predictions used for model comparison. This makes the validation set part of the decision process rather than a final untouched estimate.

|  |
| --- |
| **Current evidence level:** The Phase 6 metrics are validation metrics from one reproducible split. They are not averaged k-fold cross-validation results and not final locked-holdout results. |

# 24.4 What the later governance policy requires

The repository’s data strategy states that the labelled population should be split into training, validation and a physically separate immutable holdout before target-informed feature work. Selection, encoders, imputers, scalers and threshold choice should be fitted using training folds only. Validation should support model decisions, and the frozen pipeline should be evaluated once on the holdout.

**1.** Create training, validation and immutable labelled holdout partitions.

**2.** Perform target-informed feature selection inside training data only.

**3.** Fit preprocessing using training folds only.

**4.** Use validation for model and threshold decisions.

**5.** Freeze the selected pipeline.

**6.** Evaluate exactly once on the locked holdout and record hashes, row counts and code version.

# 24.5 Why this chapter does not invent cross-validation scores

No fold-level files or cross-validation metrics are present in the official Phase 6 artifacts. This handbook therefore does not present average MCC, standard deviation across folds or repeated-validation confidence intervals. Adding such values without running the experiments would misrepresent the work completed.

| **Validation element** | **Status in current project** |
| --- | --- |
| Fixed stratified train-validation split | Completed |
| Same split used for five-model comparison | Completed |
| Validation threshold search | Completed |
| K-fold or repeated cross-validation | Not implemented |
| Immutable labelled holdout | Documented policy, not completed in Phase 6 |
| Temporal factory validation | Unavailable with the public historical dataset |

## Common mistakes to avoid

|  |
| --- |
| **Mistake:** Describing a single train-validation split as k-fold cross-validation. |

|  |
| --- |
| **Mistake:** Reporting an average or standard deviation that was never calculated. |

|  |
| --- |
| **Mistake:** Using the validation partition repeatedly and then calling it untouched test evidence. |

|  |
| --- |
| **Mistake:** Fitting feature selection on all labels before fold creation. |

|  |
| --- |
| **Mistake:** Treating the public Kaggle test files as an immutable labelled holdout. |

## Review questions

**1.** Which scikit-learn split function is used in Phase 6?

**2.** How many times is each official Phase 6 model fitted in the main comparison?

**3.** Why can the selected threshold be optimistic?

**4.** What three labelled partitions are required by the later policy?

**5.** Why are no fold-average metrics shown in this handbook?

## Practical exercise using this project

|  |
| --- |
| **Exercise:** Create a design-only notebook cell that defines StratifiedKFold for the saved sampled Ids, but do not report results until all target-informed selection and preprocessing steps are moved inside each training fold. |

## Chapter summary

* The current project uses one reproducible stratified train-validation split.
* K-fold and repeated cross-validation are not claimed as completed.
* Model and threshold selection use the same validation partition.
* The repository documents a stronger training-validation-holdout policy for future releases.
* The handbook reports only experiments that actually exist.

**CHAPTER 25**

**Data Leakage**

*Separating valid prediction evidence from competition-only shortcuts*

|  |
| --- |
| **Learning objectives:** By the end of this chapter, the reader should be able to: identify the leakage controls and limitations in the project; distinguish the official model from the leaderboard-style experiment; explain why target-informed feature selection before splitting is a concern; describe the repository policy for future leakage-safe releases. |

|  |
| --- |
| **Project evidence used:** docs/data\_strategy\_and\_test\_set\_policy.md; src/data/phase6\_predictive\_failure\_modeling.py; reports/phase6\_leaderboard\_research\_notes.md; reports/phase6\_model\_comparison\_metrics.csv. No external dataset, textbook, or third-party implementation is used in this chapter. |

# 25.1 What leakage means inside this project

Leakage occurs when a model-development step uses information that would not legitimately be available at the declared prediction point or when evaluation rows influence decisions that are supposed to be learned only from training data. Leakage can produce excellent validation numbers while weakening confidence on truly future unseen production data.

![](data:image/png;base64...)

*Figure 25.1. Leakage lessons documented by the project itself.*

# 25.2 Safe separation already documented

The data-strategy policy separates Kaggle test\_\*.csv from labelled evaluation. These files contain no Response and are used only for batch scoring. The final feature list excludes Id and Response. The official model bundle stores feature\_cols and the selected threshold rather than using test labels that are unavailable locally.

# 25.3 Base Phase 6 target-informed selection limitation

The current Phase 6 sequence profiles numeric correlations, categorical presence and category failure rates using the labelled training CSV before it creates the saved train-validation split. It also ranks selected categorical levels using Response from the labelled population. Validation rows therefore influence which raw features and category levels enter the model. This is not the explicit production-order leak, but it is a form of validation contamination that can make the reported validation estimate somewhat optimistic.

|  |
| --- |
| **Required refactor:** Split labelled Ids first. Fit numeric selection, categorical selection, level ranking, imputation and any target-informed transformation on the training portion or training folds only. Apply the learned decisions unchanged to validation and holdout rows. |

# 25.4 The explicit leaderboard-style leak experiment

The project separately implements phase6\_leaderboard\_boost\_modeling.py based on competition observations. It creates previous and next known failure indicators, nearby failure counts, distances to known failures and production-order deltas. Because train and test were sampled from the same competition production period, these features exploit neighbour information that would not be available in the same way for a genuinely future factory batch.

| **Result set** | **MCC** | **Precision** | **Recall** | **PR-AUC** | **Project status** |
| --- | --- | --- | --- | --- | --- |
| Official Phase 6 LightGBM | 0.338642 | 0.574335 | 0.213372 | 0.252867 | Official production-safe benchmark |
| Leaderboard XGBoost | 0.834966 | 0.876982 | 0.804070 | 0.905169 | Competition-style and intentionally optimistic |

The project correctly keeps these two result sets separate. The higher competition-style score is not used as evidence of live-factory performance.

# 25.5 Timestamp and prediction-time availability

The data policy also requires every derived timing window to use only events available before the declared scoring point. In the historical project, date-derived features can use the full row of timestamps. A real early-warning deployment would need to define exactly when prediction occurs and exclude later measurements that would not yet exist.

# 25.6 Leakage checklist derived from the repository policy

| **Check** | **Question for this project** |
| --- | --- |
| Target exclusion | Is Response excluded from model inputs? |
| Identifier exclusion | Is Id excluded as a direct predictive feature in the official model? |
| Selection fitting | Were target-informed feature decisions learned on training data only? |
| Preprocessing fitting | Were imputers and scalers fitted only on training data? |
| Threshold fitting | Was the threshold selected before final holdout evaluation? |
| Timestamp availability | Were only measurements available at scoring time used? |
| Test isolation | Were Kaggle test files kept separate from model selection? |
| Competition leak separation | Are order/neighbour leak results reported as research-only? |

## Common mistakes to avoid

|  |
| --- |
| **Mistake:** Assuming that excluding Response alone eliminates every form of leakage. |

|  |
| --- |
| **Mistake:** Selecting features with all labels and then treating the validation score as fully untouched. |

|  |
| --- |
| **Mistake:** Reporting the leaderboard boost MCC as production-safe performance. |

|  |
| --- |
| **Mistake:** Using measurements recorded after the intended prediction time in a real early-warning system. |

|  |
| --- |
| **Mistake:** Treating an unlabelled Kaggle test file as proof of generalisation. |

## Review questions

**1.** What target-informed steps occur before the current Phase 6 split?

**2.** Which explicit leak-style features are used in the leaderboard experiment?

**3.** Why is the leaderboard MCC not a factory-readiness metric?

**4.** What does prediction-time availability mean for date features?

**5.** Which controls should move inside training folds?

## Practical exercise using this project

|  |
| --- |
| **Exercise:** Draw the current Phase 6 sequence and a leakage-safer sequence. Mark the point at which the split occurs, where feature selection is learned, where threshold tuning occurs and where the final holdout would be evaluated. |

## Chapter summary

* The project separates Kaggle test scoring from labelled evaluation.
* Current target-informed selection before splitting is a documented validation limitation.
* The leaderboard boost intentionally uses competition order leakage and is excluded from production claims.
* A real deployment must enforce prediction-time availability for timestamp-derived inputs.
* The repository policy defines the leakage-safe direction for future releases.

**CHAPTER 26**

**Evaluation Metrics**

*Reading the final model results correctly*

|  |
| --- |
| **Learning objectives:** By the end of this chapter, the reader should be able to: derive precision, recall and F1 from the validation confusion matrix; explain why MCC is the primary ranking metric; interpret PR-AUC and the selected threshold; compare all five official Phase 6 models. |

|  |
| --- |
| **Project evidence used:** reports/phase6\_model\_comparison\_metrics.csv; reports/phase6\_predictive\_failure\_modeling\_report.md; src/data/phase6\_predictive\_failure\_modeling.py; app/project\_streamlit\_dashboard.py. No external dataset, textbook, or third-party implementation is used in this chapter. |

# 26.1 The project does not rank models by accuracy

Because Response = 1 is rare, a high accuracy score can be achieved by predicting almost every row as non-failure. The Phase 6 model table therefore reports MCC, precision, recall, F1 and PR-AUC. Models are sorted by MCC first and PR-AUC second.

![](data:image/png;base64...)

*Figure 26.1. Actual Phase 6 comparison of MCC, F1 and PR-AUC.*

# 26.2 The confusion matrix behind the LightGBM scores

![](data:image/png;base64...)

*Figure 26.2. LightGBM validation confusion matrix at threshold 0.785083.*

| **Metric** | **Formula using project counts** | **Project value** | **Interpretation** |
| --- | --- | --- | --- |
| Precision | 367 / (367 + 272) | 0.574335 | Share of alerts that are recorded failures |
| Recall | 367 / (367 + 1353) | 0.213372 | Share of recorded failures detected |
| F1 | 2 × precision × recall / (precision + recall) | 0.311149 | Harmonic balance of precision and recall |
| MCC | Uses TP, TN, FP and FN together | 0.338642 | Balanced binary correlation measure |
| PR-AUC | Area under the precision-recall curve | 0.252867 | Ranking quality across probability thresholds |

# 26.3 Precision

The LightGBM precision is 0.5743. Of 639 validation alerts, 367 correspond to Response = 1 and 272 correspond to Response = 0. Higher precision means fewer unnecessary alerts, but precision alone does not reveal how many failures were missed.

# 26.4 Recall

The recall is 0.2134. The model detects 367 of 1,720 validation failures and misses 1353. A lower threshold would usually increase recall but can also create more false positives. The selected threshold reflects the project’s decision to maximize MCC rather than maximize recall alone.

# 26.5 F1

The F1 score is 0.3111. It becomes high only when both precision and recall are reasonably high. In this project precision is substantially higher than recall, so the F1 score remains below precision.

# 26.6 Matthews Correlation Coefficient

The MCC is 0.3386. MCC includes all four confusion-matrix cells and is therefore suitable for the project’s rare positive class. A value near zero would indicate little binary association between prediction and actual label; the reported positive value shows useful separation while leaving substantial room for improvement.

**Metrics calculated by Phase 6**

|  |
| --- |
| metrics = {  "mcc": matthews\_corrcoef(y\_true, pred),  "precision": precision\_score(y\_true, pred, zero\_division=0),  "recall": recall\_score(y\_true, pred, zero\_division=0),  "f1": f1\_score(y\_true, pred, zero\_division=0), } metrics["pr\_auc"] = average\_precision\_score(y\_true, scores) |

# 26.7 PR-AUC

The LightGBM PR-AUC is 0.2529. Unlike the single-threshold precision, recall, F1 and MCC values, PR-AUC evaluates the ranking of failure probabilities across many possible thresholds. It is used as the second ranking criterion after MCC.

# 26.8 Threshold selection

The project does not test every number between 0 and 1. It calculates score quantiles from 0.50 to 0.995 and obtains 80 candidate threshold values. For each threshold it converts probabilities to classes, calculates MCC, precision, recall and F1, and keeps the candidate with the highest MCC. PR-AUC is then added as a probability-ranking metric.

# 26.9 Full official model comparison

| **Rank** | **Model** | **Threshold** | **MCC** | **Precision** | **Recall** | **F1** | **PR-AUC** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | LightGBM | 0.785083 | 0.338642 | 0.574335 | 0.213372 | 0.311149 | 0.252867 |
| 2 | XGBoost | 0.747938 | 0.335935 | 0.567442 | 0.212791 | 0.309514 | 0.246895 |
| 3 | CatBoost | 0.796958 | 0.327926 | 0.557121 | 0.206977 | 0.301823 | 0.226068 |
| 4 | Random Forest | 0.674328 | 0.325003 | 0.552426 | 0.205233 | 0.299279 | 0.239001 |
| 5 | Logistic Regression | 0.897303 | 0.304546 | 0.519562 | 0.193023 | 0.281475 | 0.165042 |

LightGBM ranks first, but the difference from XGBoost is small. The selected model is therefore not described as universally superior; it is the best result under this project’s current data construction, validation split, parameters and ranking rule.

# 26.10 Dashboard presentation

The dashboard shows the selected model, MCC, precision, recall and PR-AUC and displays the model comparison and precision-recall trade-off. It also warns that leaderboard leakage and validation-optimised blends are research-only. This keeps metric presentation aligned with the project’s governance position.

## Common mistakes to avoid

|  |
| --- |
| **Mistake:** Using accuracy as the primary score for this rare-event target. |

|  |
| --- |
| **Mistake:** Interpreting precision as the percentage of all failures detected. |

|  |
| --- |
| **Mistake:** Interpreting recall as the percentage of alerts that were correct. |

|  |
| --- |
| **Mistake:** Comparing threshold-dependent MCC with PR-AUC without noting that they measure different things. |

|  |
| --- |
| **Mistake:** Presenting the selected threshold as a universal factory setting. |

|  |
| --- |
| **Mistake:** Claiming that the small LightGBM-XGBoost difference proves one algorithm is always better. |

## Review questions

**1.** How many validation alerts were produced by LightGBM?

**2.** What is the difference between precision and recall?

**3.** Why does MCC use true negatives as well as positive-class outcomes?

**4.** What is the difference between PR-AUC and F1?

**5.** How is the threshold selected?

**6.** Which model ranks second by the official Phase 6 rule?

## Practical exercise using this project

|  |
| --- |
| **Exercise:** Using phase6\_model\_comparison\_metrics.csv, reproduce the rank ordering by sorting MCC descending and PR-AUC descending. Then write a two-sentence non-technical interpretation of the selected LightGBM result. |

## Chapter summary

* The official comparison reports MCC, precision, recall, F1 and PR-AUC.
* LightGBM produces 367 true positives, 272 false positives, 1,353 false negatives and 54,728 true negatives on validation.
* MCC is the primary selection metric because all four confusion-matrix outcomes matter.
* PR-AUC evaluates probability ranking across thresholds.
* The selected threshold maximizes validation MCC and is not a universal operational setting.
* LightGBM ranks first under the current project design, narrowly ahead of XGBoost.

# Part III Conclusion

Part III explained data-science fundamentals only through work present in the Bosch project. The project begins with a rare-event binary target, measures missingness at scale, preserves structural absence, engineers process-oriented indicators, reduces the feature space, trains five weighted classifiers and evaluates them on one reproducible stratified validation split. It also records important limitations: no complete raw duplicate-row audit, no blanket outlier removal, no k-fold cross-validation, no immutable labelled holdout in the current Phase 6, and target-informed feature selection before splitting.

|  |
| --- |
| **Most important lesson:** A strong project explanation includes both what was implemented and what remains unverified. Clear limitations make the work more trustworthy, not less valuable. |

# Part III Project Checklist

| **Area** | **Implemented evidence** | **Current limitation** |
| --- | --- | --- |
| Classification | Five classifiers and probability-to-class thresholding | Historical benchmark, not live factory validation |
| Class imbalance | All positives retained, negative sampling, weights, stratification, MCC tuning | Sampled class rate differs from full population |
| Missing values | Chunked profiling, indicators and median imputation | Physical reason for missingness is unknown |
| Duplicates | Id alignment, unique split artifacts, duplicate-column removal | No exhaustive raw duplicate-row report |
| Outliers | Distribution reports, boxplots and infinity handling | No engineering limits or physical validation |
| Feature engineering | Phases 2, 4 and 5 features | Engineered signals are not causal proof |
| Feature selection | 80 numeric + categorical selection + final 323 features | Target-informed selection occurs before split |
| Split | Stratified 75/25 with unique non-overlapping Ids | No immutable labelled holdout |
| Cross-validation | Consistent one-split model comparison | No k-fold or repeated CV |
| Leakage | Test separation and research-only leak model labelling | Base selection sequence requires stricter refactor |
| Metrics | MCC-first model comparison and threshold tuning | Validation metrics are not factory performance |

# Glossary for Part III

| **Term** | **Meaning in this project** |
| --- | --- |
| Binary classification | Prediction problem with two target outcomes: Response 0 or Response 1. |
| Class imbalance | A large difference between the number of non-failure and failure rows. |
| Stratification | Splitting data while preserving the target-class ratio. |
| Imputation | Replacing a missing model input with a learned placeholder such as the median. |
| Missingness indicator | Binary feature recording whether the original value was absent. |
| Duplicate Id | Repeated use of the same part identifier. |
| Outlier | Observation that is extreme relative to a distribution; not automatically invalid. |
| Feature engineering | Creation of new variables from existing project data. |
| Feature selection | Retention of a smaller candidate set from the available features. |
| Validation split | Labelled rows used for model and threshold comparison. |
| Kaggle test set | Unlabelled rows used for competition scoring and local inference output. |
| Cross-validation | Repeated fitting and evaluation across multiple data folds; not implemented in official Phase 6. |
| Data leakage | Use of unavailable or evaluation information during training or model decisions. |
| Threshold | Probability cut-off used to convert a model score into class 0 or 1. |
| Precision | Fraction of predicted alerts that are actual Response 1 rows. |
| Recall | Fraction of actual Response 1 rows that are alerted. |
| F1 | Harmonic mean of precision and recall. |
| MCC | Binary correlation metric calculated from all four confusion-matrix cells. |
| PR-AUC | Area under the precision-recall curve across probability thresholds. |

# Repository Evidence Map

| **Topic** | **Project files used** |
| --- | --- |
| Data quality and missingness | src/data/phase1\_data\_quality.py; reports/phase1\_file\_summary.csv; reports/phase1\_missing\_values\_by\_column.csv |
| Feature metadata and flow | reports/phase2\_data\_understanding\_engineering\_report.md; manufacturing\_flow\_train.parquet |
| EDA and distributions | src/data/phase3\_exploratory\_data\_analysis.py; reports/phase3\_exploratory\_data\_analysis\_report.md; reports/figures/phase3\_\*.png |
| Feature engineering | reports/phase4\_feature\_engineering\_report.md; reports/phase4\_feature\_dictionary.csv; phase4\_train\_engineered\_features.csv |
| Product families | reports/phase5\_product\_family\_discovery\_report.md; reports/phase5\_product\_family\_profiles.csv |
| Official modeling | src/data/phase6\_predictive\_failure\_modeling.py; reports/phase6\_predictive\_failure\_modeling\_report.md |
| Model metrics | reports/phase6\_model\_comparison\_metrics.csv |
| Competition leak research | reports/phase6\_leaderboard\_research\_notes.md; src/data/phase6\_leaderboard\_boost\_modeling.py |
| Governance policy | docs/data\_strategy\_and\_test\_set\_policy.md |
| Dashboard metrics and warnings | app/project\_streamlit\_dashboard.py |

**End of Part III — Data Science Fundamentals**