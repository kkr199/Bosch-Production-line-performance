**PART V**

Feature Engineering

**Chapters 39–46**

Temporal Signals • Manufacturing Flow • Product Segmentation • Path Structure

![Image: image1.png](data:image/png;base64...)

*Figure V.1 — Repository feature-engineering pipeline from anonymized raw timestamps to model-ready variables.*

|  |
| --- |
| **Interpretation boundary:** Bosch timestamps are relative and anonymized. Every duration or gap in this part is an analytical measurement-time proxy, not a verified physical cycle time, queue time, delay, or root cause. |

**Author: Krishnakanth Reddy Karingula**

Repository-driven technical handbook edition

# Contents and Scope

Part V isolates the most important engineered variables used by the Bosch project and explains how they are calculated, validated, interpreted, and passed into later modeling phases. The focus is implementation rather than generic feature-engineering theory.

| **Chapter** | **Topic** | **Core question** |
| --- | --- | --- |
| **39** | Earliest Measurement Timestamp | Relative temporal anchor and production-window signal |
| **40** | Observed Measurement Span | Difference between the latest and earliest observed timestamps |
| **41** | Observed Measurement Gaps | Positive inter-station timestamp gaps and their summaries |
| **42** | Station Count | Number of stations with at least one observed date value |
| **43** | Line Count | Number of production lines represented in the observed route |
| **44** | Manufacturing Flow | Line–station hierarchy, route signatures, and flow datasets |
| **45** | Product Families | Clustering station-presence patterns into eight route-based groups |
| **46** | Path Complexity | Composite route structure from station count, line count, switches, and density |

## Part-level output summary

| **Artifact** | **Train** | **Test** |
| --- | --- | --- |
| Phase 4 engineered features | 1,183,747 rows × 49 columns | 1,183,748 rows × 48 columns |
| Manufacturing-flow dataset | 1,183,747 products | 1,183,748 products |
| Product-family labels | 8 final families | 8 family labels transferred |
| Parsed manufacturing structure | 4 lines • 52 stations • 4,264 features | Same schema |

## Reading convention

* “Observed” means present in the public Bosch data, not independently verified against plant event logs.
* Failure-rate differences are descriptive associations. Sparse groups can have unstable percentages and must be interpreted with their product counts.
* Features are keyed by Id and engineered identically for train and test; Response is attached only to the labeled train output.
* The pipeline streams raw CSV files in 20,000-row chunks to keep memory bounded.

**CHAPTER 39**

# Earliest Measurement Timestamp

*Using the first available relative timestamp as a temporal anchor without treating it as an official production start time*

## Definition and purpose

For each product, the pipeline scans every raw date feature and retains the minimum observed value. In the repository this variable is stored as start\_time and presented to readers as Earliest Measurement Timestamp. It anchors a product within the anonymized production-time ordering and can capture batch windows, route regimes, maintenance periods, or other latent operating conditions.

|  |
| --- |
| **Do not over-interpret:** The value is not a calendar date and is not guaranteed to be the moment physical production started. It is the earliest timestamp visible in the supplied feature matrix. |

## Repository implementation

|  |
| --- |
| date\_values = date\_chunk.drop(columns=['Id']) earliest = date\_values.min(axis=1, skipna=True).astype('float32') features['start\_time'] = earliest |

The operation is row-wise, ignores missing values, preserves the float32 representation used throughout Phase 4, and is repeated with the same logic for every 20,000-row chunk. Per-line variants are also created so the model can distinguish the earliest observed timestamp on L0, L1, L2, and L3.

## Observed dataset profile

| **Statistic** | **Value** |
| --- | --- |
| **Train products with value** | 1,183,165 |
| **Mean** | 848.41 |
| **Median** | 877.71 |
| **10th–90th percentile** | 198.86 – 1,476.23 |
| **99th percentile** | 1,666.98 |
| **Correlation with Response** | -0.0144 |
| **Mean for non-failures** | 848.93 |
| **Mean for failures** | 760.34 |

![Image: image2.png](data:image/png;base64...)

*Figure 39.1 — Failure rate by ordered earliest-timestamp decile. The non-monotonic pattern suggests production-window context rather than a simple linear effect.*

## Engineering interpretation

* Useful as an ordering feature: products produced in similar relative windows may share unobserved conditions.
* Useful in interaction with product family and line-level timestamp features, because the same time window can contain different routes.
* Potentially leakage-prone when future data collection rules differ; a deployment must confirm the value exists at the intended scoring time.
* Non-monotonic failure rates make tree models more appropriate than assuming a single linear relationship.

## Validation checks

1. Verify that a product with all date values missing receives a missing earliest timestamp rather than an invented zero.
2. Confirm train and test use the same feature headers and the same row-wise minimum rule.
3. Track the distribution by data window; a large timestamp-range shift can indicate a different production period or extraction convention.
4. Use SHAP or partial-dependence analysis only as predictive evidence, not proof that time itself caused failure.

**CHAPTER 40**

# Observed Measurement Span

*Representing the distance between the earliest and latest recorded timestamps*

## Definition

Observed Measurement Span is stored as cycle\_time for backward compatibility with the original pipeline, but the reader-facing name is deliberately more conservative. It is calculated as the latest observed timestamp minus the earliest observed timestamp for the same product.

|  |
| --- |
| latest = date\_values.max(axis=1, skipna=True) earliest = date\_values.min(axis=1, skipna=True) observed\_span = (latest - earliest).astype('float32') |

|  |
| --- |
| **Interpretation boundary:** Observed Measurement Span is not verified end-to-end manufacturing cycle time. Parallel operations, sparse timestamps, skipped features, and anonymization can change its physical meaning. |

## Profile and comparison

| **Metric** | **Observed value** |
| --- | --- |
| **Mean span** | 10.718 |
| **Median span** | 3.700 |
| **90th percentile** | 35.200 |
| **99th percentile** | 62.950 |
| **Maximum** | 699.200 |
| **Mean, non-failures** | 10.700 |
| **Mean, failures** | 13.914 |
| **Correlation with Response** | 0.0144 |

![Image: image3.png](data:image/png;base64...)

*Figure 40.1 — Descriptive failure rates across observed-span deciles. The highest decile contains a materially higher rate but still requires route and product-mix validation.*

## Why the feature is useful

A longer observed span may reflect a more complex route, a large inter-station gap, a different product family, or a production window with a different timestamp pattern. The feature therefore compresses many raw date columns into one interpretable scalar. It also supports engineered ratios and line-level comparisons.

## Related span features

| **Feature** | **Meaning in this project** |
| --- | --- |
| **processing\_duration** | Sum of within-station observed timestamp spans; not verified active processing time. |
| **line\_n\_processing\_duration** | Latest minus earliest observed timestamp inside line n. |
| **date\_completeness\_pct** | Share of raw date fields observed; helps distinguish a short span from sparse recording. |
| **delay\_ratio** | Observed gaps divided by the overall observed span when the span is positive. |

## Recommended modeling treatment

* Keep the continuous value and allow tree-based models to learn nonlinear cut points.
* Retain missingness explicitly if scoring can occur before any timestamp is recorded.
* Compare distribution stability by product family and route; global drift can hide stable within-family behavior.
* Avoid winsorizing extreme values until the corresponding route and timestamp completeness have been inspected.

**CHAPTER 41**

# Observed Measurement Gaps

*Summarizing positive differences between one station’s last timestamp and the next station’s first timestamp*

## Calculation logic

The pipeline first derives a start and end timestamp for every station. For each product, stations are held in manufacturing order. Positive differences between the next station start and the previous station end are retained; zero or negative differences are ignored. Four summaries are produced: total gap, mean gap, maximum gap, and number of positive gap events.

|  |
| --- |
| gaps = station\_starts[1:] - station\_ends[:-1] positive = gaps[gaps > 0] waiting\_time = positive.sum() mean\_waiting\_time = positive.mean() max\_waiting\_time = positive.max() wait\_event\_count = len(positive) |

|  |
| --- |
| **Terminology safeguard:** The stored names contain “waiting,” but the evidence is only a gap between anonymized timestamps. It may represent queueing, transport, unrecorded operations, sparse measurement, or overlapping process logic. |

## Observed profile

| **Feature** | **Mean** | **Median** | **P90** | **Failure mean** | **Corr. with Response** |
| --- | --- | --- | --- | --- | --- |
| **Total observed gap** | 10.540 | 3.690 | 33.910 | 13.729 | 0.0146 |
| **Mean observed gap** | 2.378 | 0.652 | 7.987 | 3.087 | 0.0130 |
| **Maximum observed gap** | 8.887 | 3.320 | 26.820 | 11.639 | 0.0145 |
| **Gap event count** | 5.409 | 5.000 | 7.000 | 5.397 | -0.0007 |
| **Relative gap ratio** | 0.995 | 1.000 | 1.000 | 0.996 | 0.0015 |

![Image: image4.png](data:image/png;base64...)

*Figure 41.1 — Failure rate across ordered observed-gap bands. The last band deserves investigation but is not by itself causal evidence.*

## How the four summaries complement one another

| **Question** | **Best feature** |
| --- | --- |
| **How much positive gap was recorded overall?** | Total observed gap |
| **Were gaps generally large or concentrated in one event?** | Mean and maximum observed gap |
| **Did the route contain many separated timestamp intervals?** | Gap event count |
| **How dominant were gaps relative to the complete observed span?** | Relative gap ratio |

## Operational use

* Rank products or routes for review when unusually large gaps co-occur with elevated failure probability.
* Compare the same station transition inside different product families before proposing capacity or scheduling changes.
* Validate candidate delays against real event logs, transport records, staffing, maintenance, and process-control history.
* Monitor the gap distribution separately from model performance; an upstream timestamp change can alter the feature without changing the physical process.

**CHAPTER 42**

# Station Count

*Compressing sparse station-level observations into a route-length feature*

## Definition

Station Count is the number of station groups for which at least one date value is present for a product. The raw date schema is first grouped by station key such as L1\_S24. A station receives a presence value of one when any of its date columns is non-null; Station Count is the row-wise sum of these binary indicators.

|  |
| --- |
| present\_station = station\_values.notna().any(axis=1).astype('uint8') station\_count = station\_presence\_matrix.sum(axis=1).astype('int16') |

## Dataset behavior

| **Statistic** | **Value** |
| --- | --- |
| **Mean** | 12.15 |
| **Median** | 13 |
| **90th percentile** | 14 |
| **99th percentile** | 16 |
| **Mean, failures** | 11.73 |
| **Mean, non-failures** | 12.15 |
| **Correlation with Response** | -0.0133 |

![Image: image5.png](data:image/png;base64...)

*Figure 42.1 — Bubble area represents product count. Small station-count groups can show high rates but require sample-size caution.*

## Why count is not the same as complexity

Two products can visit the same number of stations but follow completely different lines or branches. Station Count measures route breadth, not route identity. It therefore works best together with line count, first and last station, station span, line switches, path density, and product family.

## Quality checks

* A zero count is allowed only when no date feature is observed; the dataset contains a small such group.
* Presence must be derived from date observations consistently in train and test, rather than from the target-bearing numeric file.
* Counts should use integer dtypes to reduce memory and make invalid fractional values impossible.
* Rare count values should not be interpreted through raw failure percentages without confidence intervals or minimum-volume thresholds.

## Model role

The feature is directly included in Phase 4 and is also repeated in Phase 5 product-family outputs. In Phase 6, the merged dataset can therefore contain both the original date-derived count and a suffixed family-phase count. The pipeline preserves both only when names are unique and removes accidental duplicate columns before training.

**CHAPTER 43**

# Line Count

*Representing how many of the four production lines appear in a product’s observed path*

## Definition and implementation

For each of the four line groups, the pipeline checks whether any date feature is observed. The binary line-presence flags are summed to create Line Count. The feature ranges from zero to four, although the overwhelming majority of products contain two or three observed lines.

|  |
| --- |
| line\_present = date\_chunk[line\_columns].notna().any(axis=1) line\_count += line\_present.astype('int8') |

| **Line count** | **Products** | **Failures** | **Failure rate** |
| --- | --- | --- | --- |
| **0** | 582 | 2 | 0.344% |
| **1** | 2,531 | 30 | 1.185% |
| **2** | 823,147 | 4,267 | 0.518% |
| **3** | 355,294 | 2,570 | 0.723% |
| **4** | 2,193 | 10 | 0.456% |

![Image: image6.png](data:image/png;base64...)

*Figure 43.1 — Product count is printed above each bar; percentages are descriptive and not adjusted for product family or route.*

## Interpretation

* Two-line routes dominate the dataset and have a lower observed failure rate than the large three-line group.
* A higher line count can represent extra routing breadth, but it does not prove rework or inefficiency.
* Line Count is tightly related to line-switch count because products are ordered through line-numbered station groups; both should be checked for redundancy.
* Per-line presence and per-line timestamp features provide the route identity that the count alone cannot provide.

## Per-line companion features

| **Pattern** | **Created fields** |
| --- | --- |
| **Presence** | line\_0\_present … line\_3\_present |
| **Temporal anchor** | line\_n\_start\_time and line\_n\_end\_time |
| **Observed span** | line\_n\_processing\_duration |
| **Measurement density** | line\_n\_observed\_date\_values and line\_n\_date\_completeness\_pct |
| **Route breadth** | line\_n\_station\_count |

## Deployment check

A live scoring system must define when line count is considered complete. Scoring too early can systematically undercount downstream lines and change the meaning of every route-based feature. The production contract should therefore specify the scoring point and reject or label incomplete records when necessary.

**CHAPTER 44**

# Manufacturing Flow

*Turning 4,264 anonymized features into a line–station hierarchy and reusable route representation*

## Structural parsing

Bosch feature names encode manufacturing structure. The pattern L{line}\_S{station}\_{type}{feature} is parsed into line, station key, data type, and feature identifier. Across the repository this yields 4 production lines, 52 stations, and 4,264 parsed feature columns.

| **Line** | **Stations** | **Numeric** | **Categorical** | **Date** | **Total features** |
| --- | --- | --- | --- | --- | --- |
| **L0** | 24 | 168 | 323 | 184 | 675 |
| **L1** | 2 | 513 | 1,227 | 621 | 2,361 |
| **L2** | 3 | 42 | 159 | 78 | 279 |
| **L3** | 23 | 245 | 431 | 273 | 949 |

![Image: image7.png](data:image/png;base64...)

*Figure 44.1 — Manufacturing hierarchy and example route signatures constructed from date-derived station presence.*

## Manufacturing-flow dataset

Phase 2 creates train and test manufacturing-flow Parquet datasets. Station presence is derived from date features because a non-null date indicates that the product appears to have passed through the station. The resulting flow representation is compact, model-independent, and reusable by exploratory analysis, product-family discovery, process mining, and knowledge-graph construction.

## Flow representation components

| **Component** | **Purpose** |
| --- | --- |
| **Station presence matrix** | Binary product × station view used for route comparison and clustering. |
| **Ordered station list** | Sequence used to calculate first/last station, switches, and inter-station gaps. |
| **Path signature** | Packed binary station-presence pattern used to identify unique routes efficiently. |
| **Line–station metadata** | Connects anonymous features to manufacturing structure without requiring real sensor names. |
| **Train/test parity** | Ensures the same structural map is applied to both labeled and unlabeled products. |

## Engineering advantages

* Reduces thousands of sparse columns to a small set of route features while preserving the original hierarchy.
* Supports station-level diagnostics even though the sensor semantics are anonymized.
* Creates a common backbone for clustering, prediction, explainability, process mining, and graph analytics.
* Allows future factory mappings to replace anonymous station labels without rewriting every downstream analytical step.

## Limits

Presence indicates that at least one date value is available, not that the station definitely performed a complete physical operation. Missing dates can represent routing, sensor availability, extraction rules, or true absence. Any live implementation should reconcile the inferred route with the manufacturing execution system or event log.

**CHAPTER 45**

# Product Families

*Grouping products by station-presence patterns before building family-aware models*

## Why product families are engineered features

A single global model must learn many distinct manufacturing routes. Phase 5 uses the station-presence matrix to discover route-based product families and adds the final family label to the model dataset. The family becomes a categorical path feature and also supports specialized family-level baseline models.

## Clustering workflow

1. Convert each product to a binary vector over the 52 stations.
2. Pack the vector into a compact hexadecimal path signature so repeated routes can be aggregated.
3. Fit MiniBatch KMeans with eight clusters on all train station-presence vectors and predict the same labels for test products.
4. Run DBSCAN and hierarchical clustering as diagnostics on unique path patterns.
5. Use KMeans as final\_product\_family because it provides complete, stable train/test assignment without noise labels.

| **Method** | **Clusters** | **Noise rows** | **Unique paths** | **Sample silhouette** |
| --- | --- | --- | --- | --- |
| **KMeans** | 8 | 0 | 7,927 | 0.191 |
| **DBSCAN** | 7 | 855 | 7,927 | 0.231 |
| **Hierarchical** | 8 | 0 | 7,927 | 0.258 |

![Image: image8.png](data:image/png;base64...)

*Figure 45.1 — Failure rate and product volume for the eight final route-based product families. Dashed line: overall train failure rate.*

## Family findings

* Family 4 has the highest observed failure rate at 0.737% across 128,104 products.
* Family 1 is similar at 0.733% and is dominated by a different L3 branch.
* Family 6 has the greatest average station count (15.45) but the lowest observed family failure rate (0.507%).
* The result demonstrates that route identity is more informative than assuming that more stations always mean higher failure risk.

## Family-specific models

| **Family** | **Rows used** | **ROC AUC** | **Average precision** |
| --- | --- | --- | --- |
| **0** | 150,000 | 0.663 | 0.028 |
| **1** | 132,878 | 0.725 | 0.133 |
| **2** | 150,000 | 0.676 | 0.024 |
| **3** | 125,592 | 0.702 | 0.039 |
| **4** | 128,104 | 0.758 | 0.115 |
| **5** | 99,296 | 0.705 | 0.042 |
| **6** | 59,988 | 0.657 | 0.028 |
| **7** | 148,569 | 0.695 | 0.037 |

|  |
| --- |
| **Leakage control:** The family label is learned from route structure, not from Response. Failure rates are used only after clustering to profile risk and train labeled family models. |

**CHAPTER 46**

# Path Complexity

*Combining route breadth, line transitions, and station density into a compact structural score*

## Formula

The repository defines Path Complexity Score as a deterministic composite of Station Count, Line Count, Line Switch Count, and Path Density:

|  |
| --- |
| **Implemented formula:** Path Complexity = Station Count + 2 × Line Count + Line Switch Count + (1 − Path Density) |

Path Density is Station Count divided by Station Span, where Station Span is the inclusive distance between the first and last observed station numbers. Dense routes receive a smaller sparsity penalty; broad multi-line routes receive higher complexity through station and line terms.

## Component definitions

| **Component** | **Calculation** | **Interpretation** |
| --- | --- | --- |
| **Station Count** | Number of stations present | Route breadth |
| **Line Count** | Number of lines present | Cross-line breadth |
| **Line Switch Count** | Adjacent active-line changes | Observed line transitions |
| **Path Density** | Station Count ÷ Station Span | How densely the station range is occupied |
| **1 − Path Density** | Sparse-span penalty | Branches or skipped station numbers |

## Observed profile

| **Statistic** | **Value** |
| --- | --- |
| **Mean** | 18.624 |
| **Median** | 18.658 |
| **90th percentile** | 21.692 |
| **99th percentile** | 23.423 |
| **Mean, failures** | 18.412 |
| **Mean, non-failures** | 18.625 |
| **Correlation with Response** | -0.0078 |

![Image: image9.png](data:image/png;base64...)

*Figure 46.1 — Descriptive failure rates across path-complexity bands. Complexity is not monotonic and should be interpreted jointly with route identity.*

## Why the score is useful

* Provides a single model feature that summarizes several related route attributes.
* Supports quick segmentation and visual diagnostics without expanding every station presence flag.
* Creates an interpretable cross-feature interaction for tree models.
* Can be recalculated deterministically for test or future data without target information.

## Why the score is not sufficient alone

The lower-complexity band contains a higher observed failure rate than some middle bands because it includes specific high-risk route families. A single composite score cannot identify which stations or branches were traversed. The project therefore retains the original components, per-line features, and product-family labels alongside the score.

## Integrated feature-engineering checklist

| **Gate** | **Required evidence** |
| --- | --- |
| **Schema** | Date features parse to the expected line and station hierarchy. |
| **Parity** | Train and test receive the same formulas, dtypes, and feature ordering. |
| **Timing** | All features are available at the chosen scoring point. |
| **Missingness** | Missing timestamps remain distinguishable from valid zeros. |
| **Stability** | Distributions are monitored by route and product family. |
| **Interpretation** | Timestamp spans/gaps are described as proxies, not verified physical times. |
| **Governance** | Feature changes are versioned and revalidated before model promotion. |

# Part V Summary

The feature-engineering layer converts an extremely sparse, anonymized manufacturing dataset into a compact representation of temporal position, observed timestamp span, inter-station gaps, route breadth, production-line coverage, flow structure, route families, and path complexity. These features form the bridge between raw data understanding and predictive failure modeling.

| **Feature group** | **Primary value** | **Main safeguard** |
| --- | --- | --- |
| **Temporal anchor** | Captures relative production windows | Not an official start time |
| **Observed span** | Compresses first-to-last timestamp distance | Not verified cycle time |
| **Observed gaps** | Summarizes positive inter-station timestamp gaps | Not confirmed waiting or delay |
| **Station and line counts** | Represent route breadth | Do not encode route identity alone |
| **Manufacturing flow** | Creates reusable line–station structure | Presence is inferred from date availability |
| **Product families** | Encodes recurring route patterns | Cluster without target leakage |
| **Path complexity** | Summarizes breadth, switching, and density | Retain original components and family labels |

## Repository traceability

| **Topic** | **Primary repository artifact** |
| --- | --- |
| **Feature calculations** | src/data/phase4\_feature\_engineering.py |
| **Feature dictionary and summary** | reports/phase4\_feature\_dictionary.csv; reports/phase4\_engineered\_feature\_summary.csv |
| **Manufacturing metadata and flow** | src/data/phase2\_data\_understanding\_engineering.py; data/processed/manufacturing\_flow\_\*.parquet |
| **Product-family discovery** | src/data/phase5\_product\_family\_discovery.py |
| **Family profiles and diagnostics** | reports/phase5\_product\_family\_\*.csv; reports/phase5\_cluster\_diagnostics.csv |
| **Model integration** | src/data/phase6\_predictive\_failure\_modeling.py |

## Terminology note

The source code retains historical variable names such as start\_time, cycle\_time, waiting\_time, and processing\_duration. This handbook uses reader-facing labels that better match the evidence available from the public data. Renaming the stored model inputs would break existing artifacts, so the safer practice is to preserve technical keys while improving documentation and dashboard labels.