**PART IV**

Complete Project Walkthrough

**Chapters 27–30**

*Data Quality • Data Understanding • Exploratory Analysis • Feature Engineering*

![](data:image/png;base64...)

*The first four implementation phases convert raw, highly sparse files into validated manufacturing intelligence and compact engineered features.*

**Author: Krishnakanth Reddy Karingula**

Repository-driven condensed edition

# Contents and Scope

This volume documents the implementation choices and evidence produced in Phases 1–4. It is intentionally concise: each chapter explains the operational objective, the most important code path, the outputs, the engineering decisions and the hand-off to the next phase.

| **Chapter** | **Topic** | **Core contribution** |
| --- | --- | --- |
| 27 | Data Quality Assessment | Inventory, schema checks, missingness and scalable profiling |
| 28 | Data Understanding & Engineering | Feature-name parsing, metadata, completeness and manufacturing-flow tables |
| 29 | Exploratory Data Analysis | Line/station risk, paths, relative-time patterns, categorical levels and distributions |
| 30 | Feature Engineering | Timing, waiting proxies, path structure, completeness and line-level aggregates |

## Repository execution order

|  |
| --- |
| python src/data/phase1\_data\_quality.py python src/data/phase2\_data\_understanding\_engineering.py python src/data/phase3\_exploratory\_data\_analysis.py python src/data/phase4\_feature\_engineering.py |

|  |
| --- |
| **Data contract:** Every phase validates Id alignment before combining files. This is essential because silent row-order drift would attach the Response label to the wrong product and invalidate all later analysis. |

## Reading convention

* “Observed” means present in the anonymized Bosch files; it does not automatically mean a physical sensor reading was taken at a known clock time.
* Date-derived durations and waits are analytical proxies from relative timestamps, not verified plant cycle-time or queue-time measurements.
* Association with failure is evidence for prioritization, not proof that a line, station or category caused the defect.

**CHAPTER 27**

# Phase 1 — Data Quality Assessment

*Establishing a trustworthy foundation for a 14 GB, high-dimensional manufacturing dataset*

## Phase objective

Phase 1 answers a basic but decisive question: can the six Bosch train/test files be processed consistently and safely? The implementation inventories file size, row and column counts, checks the Id and Response fields, and measures missingness without attempting to load each complete multi-gigabyte CSV into memory.

## Dataset inventory

| **Dataset** | **Size (MB)** | **Rows** | **Columns** | **Missing values** |
| --- | --- | --- | --- | --- |
| train\_numeric | 2,040.77 | 1,183,747 | 970 | 80.9177% |
| train\_categorical | 2,554.27 | 1,183,747 | 2,141 | 97.2840% |
| train\_date | 2,759.33 | 1,183,747 | 1,157 | 82.1725% |
| test\_numeric | 2,038.27 | 1,183,748 | 969 | 81.0054% |
| test\_categorical | 2,554.20 | 1,183,748 | 2,141 | 97.2854% |
| test\_date | 2,759.20 | 1,183,748 | 1,157 | 82.1776% |

|  |
| --- |
| **Key observation:** The near-identical train/test shapes and missingness levels are reassuring, while the 80–97% missingness rates show that sparsity is a defining property of the data rather than a minor cleaning issue. |

![](data:image/png;base64...)

*Figure 27.1 — Missingness across the six raw Bosch datasets.*

## Scalable profiling workflow

1. Search both the repository root and data/raw/ so the pipeline remains usable before and after data reorganization.
2. Read only the CSV header to determine schema, feature count and the presence of Id or Response.
3. Stream rows in chunks of 20,000 and accumulate row counts plus per-column null counts.
4. Write a compact file summary, a column-level missingness file and a human-readable Markdown report.

|  |
| --- |
| for chunk in pd.read\_csv(path, chunksize=20\_000, low\_memory=False):  row\_count += len(chunk)  missing\_counts = missing\_counts.add(chunk.isna().sum(), fill\_value=0)  missing\_pct = missing\_counts / row\_count \* 100 |

The chunk loop makes runtime proportional to the number of cells while limiting peak memory to one chunk plus the accumulated statistics. The DatasetProfile data class keeps the summary schema explicit and easy to serialize.

## Quality gates and boundaries

* Response exists only in train\_numeric.csv and contains no missing values.
* Id is treated as a join key, never as an ordinary predictive feature.
* Missingness is measured, not automatically imputed or deleted.
* Phase 1 does not claim to perform duplicate detection, outlier detection, drift testing, target-distribution modeling or leakage analysis.

|  |
| --- |
| **Design decision:** Preserving missing values is deliberate. In this dataset, absence often indicates that a part did not traverse a station or did not receive a measurement; deleting sparse columns too early would remove process-routing information. |

## Outputs and hand-off

| **Artifact** | **Purpose** |
| --- | --- |
| phase1\_file\_summary.csv | Dataset-level inventory and overall missingness. |
| phase1\_missing\_values\_by\_column.csv | Per-column rows, null counts and missing percentages. |
| phase1\_data\_quality\_report.md | Readable evidence and implementation notes. |

Phase 2 consumes the per-column missingness output directly, demonstrating a clear dependency rather than repeating the same expensive scan.

## Chapter checkpoint

* Evidence established: the six files are structurally consistent and the target is complete.
* Constraint established: sparsity is extreme and must be treated as process information as well as data quality.
* Reusable contract: Phase 1 exposes per-column quality metrics instead of hiding them inside notebook output.
* Interview message: explain why chunked profiling and explicit limitations are more trustworthy than a one-line pandas summary.

**CHAPTER 28**

# Phase 2 — Data Understanding & Engineering

*Transforming anonymous feature names and missingness patterns into manufacturing metadata and flow indicators*

## Phase objective

Phase 2 adds structure to the anonymous dataset. It parses each feature into production line, station, feature kind and identifier; aggregates this information into station and line metadata; combines it with Phase 1 missingness; and builds compact product-level manufacturing-flow datasets from date-feature presence.

![](data:image/png;base64...)

*Figure 28.1 — The naming pattern exposes a hierarchy even though the measurements themselves remain anonymous.*

## Feature-name parsing and metadata

|  |
| --- |
| FEATURE\_RE = re.compile(  r"^L(?P<line>\d+)\_S(?P<station>\d+)\_(?P<kind>[FD])(?P<feature>\d+)$" ) |

A name such as L3\_S32\_F3854 becomes a structured record with line=3, station=32, station\_key=L3\_S32, kind=F and feature\_id=3854. The same parser is applied to the numeric, categorical and date headers in both train and test. Non-feature columns such as Id and Response are ignored.

## Manufacturing metadata produced

| **Line** | **Stations** | **Numeric** | **Categorical** | **Date** | **Total features** |
| --- | --- | --- | --- | --- | --- |
| L0 | 24 | 168 | 323 | 184 | 675 |
| L1 | 2 | 513 | 1,227 | 621 | 2,361 |
| L2 | 3 | 42 | 159 | 78 | 279 |
| L3 | 23 | 245 | 431 | 273 | 949 |

Across the raw headers, the project identifies 4,264 unique feature columns, four production lines and 52 stations. L1 is notable: it has only two stations but 2,361 features, more than half of all parsed features. Feature count therefore cannot be interpreted as station count or route length.

## Completeness metrics

The Phase 1 column-level file is merged with feature metadata. For each column, observed\_values = rows − missing\_values and completeness\_pct = 100 − missing\_pct. A second aggregation summarizes possible, observed and missing values for each split × data type × station group.

|  |
| --- |
| possible\_values = rows \* feature\_count completeness\_pct = observed\_values / possible\_values \* 100 |

Completeness ranges from effectively zero for several categorical station groups to about 94.65% for the date and numeric groups at L3\_S37. The variation confirms that missingness reflects routing and measurement availability at specific stations, not one uniform data-quality mechanism.

## Manufacturing-flow datasets

Date features are used as route evidence: if any date feature at a station is observed for a product, present\_Lx\_Sy is set to 1. Station indicators are then aggregated into line-presence flags and product-level flow descriptors.

| **Field group** | **Meaning** |
| --- | --- |
| Id / Response | Product identifier; target is present only for training. |
| present\_Lx\_Sy | 52 binary station-presence indicators. |
| line\_n\_present | Four production-line presence indicators. |
| station\_count / line\_count | Breadth of the observed route. |
| first\_station / last\_station | Minimum and maximum observed station numbers. |
| observed / possible date values | Coverage counts. |
| date\_completeness\_pct | Product-level date-feature completeness. |

The writer processes 20,000-row chunks and appends PyArrow tables to Snappy-compressed Parquet. Train and test outputs contain approximately 1.18 million products; the train schema has 65 columns including Response, while the test schema has 64.

|  |
| --- |
| **Integrity check:** Before Response is joined, the implementation verifies that Id order in train\_date.csv exactly matches train\_numeric.csv. A mismatch raises an exception instead of silently corrupting the target. |

## Engineering decisions and limitations

* Anonymous IDs are preserved; metadata adds context without pretending to know sensor names or operations.
* Station presence is inferred from date observations because those values provide the clearest evidence that the product appeared at a station.
* First and last station are based on anonymized station numbers, not a guaranteed physical route order.
* Parquet is used for compact typed storage and downstream analytical reads; CSV remains appropriate for small metadata tables.

## Outputs

| **Artifact group** | **Files** |
| --- | --- |
| Metadata | phase2\_feature\_metadata.csv; phase2\_station\_metadata.csv; phase2\_line\_metadata.csv |
| Completeness | phase2\_feature\_completeness\_metrics.csv; phase2\_station\_completeness\_metrics.csv |
| Flow | manufacturing\_flow\_train.parquet; manufacturing\_flow\_test.parquet |
| Report | phase2\_data\_understanding\_engineering\_report.md |

## Chapter checkpoint

* The anonymous feature space now has a machine-readable line–station hierarchy.
* Completeness is available at column and station/data-type levels for train and test.
* The route representation is compact enough for later clustering, EDA and modelling.
* The central assumption—date presence as station evidence—is documented and testable rather than hidden.

Quick review: Why should L1 not be called the “longest” line simply because it owns the most features? How could station-presence inference be validated if plant event logs later became available?

**CHAPTER 29**

# Phase 3 — Exploratory Data Analysis

*Locating failure concentration across lines, stations, routes, relative time and categorical states*

## Phase objective

Phase 3 uses the raw train files and raw headers to develop evidence about failure risk. It deliberately rebuilds the required compact flow features instead of reading the Phase 2 Parquet file, making the EDA independently reproducible. The analysis covers line and station failure rates, high-risk ranking, common routes, relative-time bins, correlations, distributions and selected one-hot categorical levels.

## Class imbalance and analytical baseline

The training set contains 6,879 failed products out of 1,183,747, an overall failure rate of 0.5811%. Because failures are rare, raw failure counts can mislead: a very common station may accumulate many failures while having only average risk. Phase 3 therefore reports failure rate, lift over the baseline and a volume-aware risk score.

|  |
| --- |
| failure\_rate\_lift = station\_failure\_rate / overall\_failure\_rate risk\_score = excess\_failure\_rate\_pct \* log10(part\_count) |

## Line and station findings

| **Line** | **Products** | **Failures** | **Failure rate** | **Lift** |
| --- | --- | --- | --- | --- |
| L1 | 267,273 | 1,940 | 0.7259% | 1.249× |
| L2 | 357,019 | 2,575 | 0.7213% | 1.241× |
| L3 | 1,183,159 | 6,875 | 0.5811% | 1.000× |
| L0 | 916,029 | 4,924 | 0.5375% | 0.925× |

L1 has the highest line-level failure rate, but the strongest localized signal appears at L3\_S32: 1,106 failures among 24,543 observed products, a 4.506% failure rate and 7.75× lift. L1\_S24 and L2\_S26 are also high-volume risk candidates, although their lifts are much smaller.

![](data:image/png;base64...)

*Figure 29.1 — Phase 3 triangulates risk using rate, volume, time and categorical-state evidence.*

## Product paths and temporal patterns

A path signature is created by joining the active station keys for each product in station order. The two most common routes both begin at L1\_S24, continue through L2\_S26 and terminate through the L3\_S29–L3\_S37 region; each has a failure rate near 0.675%. Closely related routes that start at L1\_S25 show materially lower rates near 0.32–0.38%, suggesting that route context contains predictive information.

Relative date values are not calendar timestamps, so the script uses quantile bins rather than months or weekdays. Failed products begin earlier on average (760.3 versus 848.9 relative units) and have longer observed process spans (13.91 versus 10.70). These are distributional differences, not proof of physical delay causation.

## Categorical one-hot exploration

One-hot analysis is deliberately constrained. A 50,000-row variability sample identifies columns with enough observed values and more than one level. Up to 30 columns are selected using completeness and priority stations; chunks are then one-hot encoded and aggregated without retaining the full dummy matrix.

The most striking level is L3\_S32\_F3854\_T2: 667 failures among 4,136 occurrences, a 16.13% failure rate, 27.75× lift and phi correlation of 0.121 with Response. This supports deeper modelling and root-cause investigation, while still requiring caution because a categorical level can be a marker of a route, product family or inspection outcome rather than a causal defect source.

## Outputs and interpretation rules

| **Output family** | **Examples** |
| --- | --- |
| Risk tables | line/station failure rates; high-risk stations |
| Process patterns | flow paths; first-time and duration bins; distribution report |
| Association | flow correlations; selected categorical columns; one-hot lift and correlations |
| Visuals | nine phase3\_\*.png charts |
| Reusable data | phase3\_train\_time\_features.csv |

|  |
| --- |
| **Interpretation rule:** High lift with small support is unstable; high volume with tiny excess rate may be operationally important but weakly predictive. The project therefore uses multiple lenses rather than ranking by one number. |

## Chapter checkpoint

* L1 and L2 have elevated line-level rates, while L3\_S32 carries the strongest localized lift.
* Routes beginning at L1\_S24 and L1\_S25 show different failure rates despite sharing much of the downstream path.
* Failed products have longer observed date spans on average, but the values remain anonymized temporal proxies.
* L3\_S32\_F3854\_T2 is an important marker for modelling and investigation, not a standalone causal conclusion.

Interview talking point: describe why rate, lift, support, correlation and route context must be considered together in a highly imbalanced manufacturing problem.

**CHAPTER 30**

# Phase 4 — Feature Engineering

*Compressing raw relative-time measurements into timing, delay-proxy, route and completeness features*

## Phase objective

Phase 4 converts 1,156 sparse raw date columns into a compact, interpretable table for every train and test product. The engineered variables capture when observations occur, how widely a part appears to travel, temporal gaps between observed stations, route complexity, data completeness and line-specific summaries.

![](data:image/png;base64...)

*Figure 30.1 — The 48 predictive columns are grouped into six manufacturing-oriented feature families.*

## Core transformations

| **Feature family** | **Definition / role** |
| --- | --- |
| start\_time / end\_time | Earliest and latest observed date values. |
| cycle\_time | end\_time − start\_time; an observed span, not verified plant cycle time. |
| processing\_duration | Sum of non-negative within-station date spans. |
| waiting\_time | Sum of positive gaps between consecutive station end and start values. |
| delay\_ratio | waiting\_time ÷ cycle\_time when cycle\_time > 0. |
| station\_count / line\_count | Number of observed stations and lines. |
| station\_span / path\_density | Route extent and observed-station density. |
| line\_switch\_count | Changes between line IDs across active stations. |
| path\_complexity\_score | station\_count + 2×line\_count + switches + (1−density). |
| line\_n\_\* | Presence, time span, observations, station count and completeness per line. |

## Station timing and waiting proxies

For each station, the earliest and latest non-null date values define an observed station span. Consecutive active stations are compared row by row; only positive start(next) − end(previous) gaps contribute to waiting\_time. The function also records mean gap, maximum gap and number of positive gaps. Negative or overlapping gaps are ignored rather than forced into a waiting interpretation.

|  |
| --- |
| gaps = starts[1:] - ends[:-1] gaps = gaps[gaps > 0] waiting\_time = gaps.sum() max\_waiting\_time = gaps.max() |

## Route geometry and line aggregates

The station-presence matrix yields first\_station, last\_station and station\_span. Path density distinguishes a compact observed route from a sparse route across the same station-number span. Line-switch count records changes in line identifiers along the ordered active stations. Each production line then receives its own presence flag, start/end values, observed span, date count, station count and completeness percentage.

|  |
| --- |
| **Caution:** The feature dictionary should use “observed measurement span” and “temporal gap proxy” language. Anonymized relative date values do not prove actual workstation processing time, queue time or transport delay. |

## Chunked train/test engineering

The same engineer\_chunk function is applied to train and test in 20,000-row batches. For train only, Response is read from train\_numeric.csv and Id order is verified before it is attached. Each engineered chunk is appended to CSV, preventing a full 1.18-million-row date table plus all intermediate station matrices from remaining in memory.

| **Split** | **Rows** | **Columns** | **File size** | **Response** |
| --- | --- | --- | --- | --- |
| Train | 1,183,747 | 49 | 281.29 MB | Yes |
| Test | 1,183,748 | 48 | 279.02 MB | No |

## Engineering strengths and improvement opportunities

* Strength: identical feature logic for train and test reduces training-serving skew.
* Strength: compact outputs reduce 1,156 raw date columns to 48 predictors while retaining interpretable route and timing signals.
* Strength: explicit low-width integer and float dtypes reduce intermediate memory.
* Improvement: row-wise waiting and line-switch loops could be vectorized or accelerated if runtime becomes a bottleneck.
* Improvement: feature validation tests should assert finite ratios, valid ranges and train/test schema parity.
* Improvement: downstream cross-validation must fit any learned transformations only on training folds; these Phase 4 rules are deterministic and therefore safe to apply globally.

## Outputs and hand-off

| **Artifact** | **Role** |
| --- | --- |
| phase4\_train\_engineered\_features.csv | Id, Response and 48 engineered predictors. |
| phase4\_test\_engineered\_features.csv | Id and the identical 48 predictor columns. |
| phase4\_feature\_dictionary.csv | Definitions and interpretation boundaries. |
| phase4\_engineered\_feature\_summary.csv | Rows, columns, file sizes and target presence. |
| phase4\_feature\_engineering\_report.md | Run-level documentation. |

These outputs become reusable inputs for product-family discovery and predictive modelling. They also provide interpretable features for later explainability, process mining and executive dashboards.

## Chapter checkpoint

* The raw date space is reduced to 48 deterministic predictors with identical train/test definitions.
* Feature names and the dictionary preserve the boundary between an observed timestamp span and a verified plant KPI.
* Waiting, density and complexity features encode route behaviour that raw measurements alone do not express directly.
* The output is now suitable for clustering, family-specific modelling and failure prediction without repeatedly scanning all date columns.

Interview talking point: explain how you would unit-test cycle-time, waiting, line-switch and completeness features before allowing them into a production model.

# Chapters 27–30: Integrated Data Contract

The four phases form one data-engineering chain. Each phase produces explicit artifacts rather than passing hidden notebook state to the next stage.

| **Chapter** | **Primary input** | **Transformation** | **Principal hand-off** |
| --- | --- | --- | --- |
| 27 | Raw CSVs | File and column quality evidence | phase1\_missing\_values\_by\_column.csv |
| 28 | Raw headers + Phase 1 | Manufacturing metadata and flow indicators | manufacturing\_flow\_\*.parquet |
| 29 | Raw train CSVs | Risk evidence and EDA outputs | phase3 reports, figures, time features |
| 30 | Raw date CSVs + target | Model-ready deterministic features | phase4\_\*\_engineered\_features.csv |

## End-to-end controls

* Reproducibility: every chapter has a command-line script and a companion notebook.
* Scalability: large CSVs are processed in chunks instead of full-memory loads.
* Traceability: reports, dictionaries and summaries document each artifact.
* Schema safety: train/test logic is shared and Id alignment is checked before target joins.
* Interpretability: anonymous columns are never renamed to unsupported physical meanings.

## Review questions

1. Why is extreme missingness not automatically evidence that a Bosch feature is useless?
2. How does Phase 2 transform a feature such as L3\_S32\_F3854 into structured metadata?
3. Why does the Phase 3 risk score combine excess failure rate with product volume?
4. Why are relative-time bins appropriate while calendar seasonality is not?
5. What is the difference between cycle\_time in Phase 4 and a verified factory cycle-time KPI?
6. Which safeguards prevent Response from being attached to the wrong products?
7. Why are deterministic Phase 4 features less vulnerable to cross-validation leakage than learned encodings?
8. What evidence from Chapters 27–30 supports the need for product-family discovery in the next phase?

## Repository references

* src/data/phase1\_data\_quality.py and reports/phase1\_\*
* src/data/phase2\_data\_understanding\_engineering.py and reports/phase2\_\*
* src/data/phase3\_exploratory\_data\_analysis.py, reports/phase3\_\* and reports/figures/phase3\_\*.png
* src/data/phase4\_feature\_engineering.py, reports/phase4\_\* and data/processed/phase4\_\*
* README.md — execution commands, repository structure and phase output inventory

|  |
| --- |
| **Next chapter:** Chapter 31 uses station-presence patterns to discover product families and build family-specific baselines, addressing the reality that different routes may represent distinct manufacturing populations. |