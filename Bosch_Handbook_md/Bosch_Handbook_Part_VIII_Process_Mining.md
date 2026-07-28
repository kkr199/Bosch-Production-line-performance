**PART VIII**

Process Mining

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Chapters 58–62**

Event Logs • Waiting Times • Bottlenecks • Throughput • Failure Propagation

![](data:image/png;base64...)

*Process-mining workflow used in the Bosch manufacturing analytics project.*

**Bosch Production Line Performance — Technical Handbook**

Repository-grounded reconstruction of observed production paths and temporal proxies

# Contents and Process-Mining Context

| **Chapter** | **Topic** | **Purpose** |
| --- | --- | --- |
| 58 | Event Logs | Transform sparse station timestamps into ordered product-event sequences. |
| 59 | Waiting Times | Quantify positive inter-station timestamp gaps without overstating physical delay. |
| 60 | Bottlenecks | Rank operational constraints with a transparent multi-signal score. |
| 61 | Throughput | Summarize productive-span and waiting-gap proxies across train and test populations. |
| 62 | Failure Propagation | Prioritize candidate station routes for investigation, not causal attribution. |

## Phase 8–9 evidence base

| **Item** | **Repository value** | **Interpretation** |
| --- | --- | --- |
| Train products | 1,183,747 | Labelled products used for failure-rate diagnostics. |
| Test products | 1,183,748 | Unlabelled products used to assess route structure and scoring. |
| Stations represented | 52 | Nodes in the direct-follow manufacturing graph. |
| Unique transitions | 531 | Directed station-to-station relationships. |
| Aggregated path records | 34,909 | Distinct reconstructed path strings after split aggregation. |
| Top bottleneck | L1\_S25 — 88.34 | Composite operational constraint, not the highest defect-risk station. |
| Top candidate route | L3\_S29 → L3\_S30 → L3\_S33 → L3\_S34 | Highest ranked observational propagation hypothesis. |

|  |
| --- |
| **Interpretation boundary:** Bosch timestamps are relative and anonymized. Waiting, dwell, cycle and throughput measures in this part are temporal proxies. They must not be described as verified seconds, queues, official cycle times or OEE without factory-system validation. |

## Learning objectives

* Construct a defensible event-log representation from sparse station timestamps.
* Separate observed timestamp gaps from confirmed physical waiting or queue time.
* Explain why a bottleneck can be operationally severe without having the highest failure rate.
* Read throughput-efficiency proxies while respecting their limitations.
* Use candidate propagation routes to organize investigations rather than claim causal transmission.

**CHAPTER 58**

# Event Logs

*Converting wide station timestamp tables into product-centric process histories*

## Why an event log is necessary

The raw Bosch date files are wide tables: each row is a product and each date column belongs to a line, station and anonymized measurement identifier. Process mining requires a different view. It needs an ordered sequence of events for every product so that direct-follow relationships, paths and temporal gaps can be calculated.

|  |
| --- |
| **Core transformation:** For each product and station, the repository takes the minimum observed station timestamp as the station start and the maximum as the station end. Stations are then ordered by start time to form the observed product route. |

![](data:image/png;base64...)

*Figure 58.1 — Illustrative station-event sequence. The long gap before L2\_S26 is a timestamp-gap proxy, not a confirmed physical queue.*

## Conceptual event-log schema

| **Field** | **Definition** | **Use** |
| --- | --- | --- |
| product\_id | Bosch product Id | Case identifier in process-mining terminology. |
| event\_index | Order after sorting station starts | Defines the observed path sequence. |
| station | Line/station key such as L1\_S24 | Activity or process node. |
| station\_start | Minimum timestamp observed at station | Orders events and starts gap calculation. |
| station\_end | Maximum timestamp observed at station | Ends the observed station span. |
| dwell\_span | max(station\_end − station\_start, 0) | Within-station measurement-span proxy. |
| response | Train label when available | Failure-rate diagnostics only. |

## Repository implementation

|  |
| --- |
| starts[:, station\_i] = station\_data.min(axis=1, skipna=True) ends[:, station\_i] = station\_data.max(axis=1, skipna=True)  present = np.isfinite(starts) safe\_starts = np.where(present, starts, np.inf) event\_order = np.argsort(safe\_starts, axis=1) |

The implementation processes the large date files in 20,000-row chunks. It does not need to materialize a multi-million-row event table on disk. Instead, each chunk is converted into station start/end matrices and aggregated into transition, station, path and throughput summaries.

## Direct-follow graph

When station B follows station A in a product sequence, the pipeline records a directed A → B transition. Repeated sequences are aggregated to produce transition count, total positive gap, positive-gap rate and train-only failure diagnostics.

![](data:image/png;base64...)

*Figure 58.2 — Highest-volume directed transitions reconstructed from train and test products.*

| **Transition** | **Count** | **Positive-gap rate** | **Train failure rate** |
| --- | --- | --- | --- |
| L3\_S29 → L3\_S30 | 1,971,196 | 50.7% | 0.574% |
| L0\_S0 → L0\_S1 | 1,346,767 | 7.6% | 0.535% |
| L3\_S34 → L3\_S33 | 1,087,398 | 0.0% | 0.511% |
| L3\_S30 → L3\_S33 | 970,910 | 91.8% | 0.481% |
| L3\_S33 → L3\_S34 | 965,844 | 42.6% | 0.466% |
| L0\_S1 → L0\_S2 | 678,332 | 36.4% | 0.534% |
| L0\_S6 → L0\_S8 | 675,449 | 19.5% | 0.531% |
| L0\_S1 → L0\_S3 | 669,098 | 40.1% | 0.535% |

## Event-log quality checks

* Keep the product Id as the case key and confirm train/date Id order before combining labels.
* Treat missing station timestamps as possible route semantics, not automatic data corruption.
* Order by observed station start rather than station number because routing is not always numerically monotonic.
* Define deterministic tie handling for stations with identical start timestamps.
* Preserve split identity because test products have no verified outcome labels.

**Repository evidence:** `src/data/phase8\_process\_mining\_bottleneck\_analysis.py`, `reports/phase8\_process\_map\_edges.csv`, and `reports/phase8\_process\_map\_nodes.csv`.

**CHAPTER 59**

# Waiting Times

*Measuring positive inter-station timestamp gaps while preserving semantic caution*

## Definition used by the repository

For two consecutive observed events, the raw gap is the next station's start minus the current station's end. Only positive values are accumulated as waiting-time proxies. Negative values are clipped to zero because overlapping station spans can occur.

|  |
| --- |
| raw\_gap = next\_station\_start - current\_station\_end waiting\_gap = max(raw\_gap, 0)  product\_waiting\_time = sum(waiting\_gap) wait\_event\_count = count(waiting\_gap > 0) |

|  |
| --- |
| **Important:** A positive timestamp gap can represent queueing, transport, inspection, batching, shift boundaries, data-capture timing or anonymization effects. A zero gap can represent immediate flow, overlap or identical encoded timestamps. |

## Transition-level waiting metrics

* Average gap: mean positive-clipped gap across all observed A → B transitions.
* Median gap: robust central value, useful when rare long gaps distort the average.
* P90 gap: upper-tail operational signal used in bottleneck scoring.
* Positive-gap rate: share of transitions whose raw next-start minus prior-end was positive.
* Maximum gap: useful for investigation but too unstable to dominate ranking.

![](data:image/png;base64...)

*Figure 59.1 — Large average gaps among transitions observed at least 20,000 times.*

## Station-level aggregation

Inbound transition gaps are aggregated to the destination station. This answers a practical question: when products arrive at a station, how large are the observed gaps before that station begins recording measurements?

| **Station** | **Average gap** | **P90 gap** | **Positive-gap rate** | **Products** |
| --- | --- | --- | --- | --- |
| L1\_S25 | 30.305 | 36.495 | 99.9% | 167,220 |
| L1\_S24 | 23.253 | 41.014 | 99.4% | 366,583 |
| L3\_S38 | 22.459 | 32.765 | 100.0% | 54,175 |
| L0\_S13 | 18.252 | 31.579 | 24.7% | 484,481 |
| L2\_S28 | 7.885 | 13.436 | 99.8% | 19,153 |
| L2\_S27 | 6.271 | 12.228 | 99.9% | 240,285 |
| L2\_S26 | 5.493 | 9.860 | 99.9% | 454,344 |
| L0\_S12 | 21.777 | 28.627 | 0.2% | 484,476 |
| L3\_S29 | 5.787 | 9.251 | 87.8% | 2,239,066 |
| L3\_S39 | 6.172 | 8.512 | 88.7% | 120,019 |

## What the strongest gap signals mean

| **Finding** | **Observed evidence** | **Safe interpretation** |
| --- | --- | --- |
| L1\_S25 | Average 30.305; P90 36.495; positive-gap rate 99.9% | Persistent large inbound gap signal. Validate transfer, queue, batch and timestamp semantics. |
| L1\_S24 | Average 23.253; P90 41.014; positive-gap rate 99.4% | High and variable upper-tail gap, with above-average failure lift. |
| L3\_S38 | Average 22.459; P90 32.765; positive-gap rate 100% | Consistent positive-gap pattern, but dwell span is recorded as zero. |
| L0\_S12 | Average 21.777 but positive-gap rate only 0.2% | Rare positive gaps are extremely large; the mean alone would mislead. |

|  |
| --- |
| **Interpretation lesson:** Always read average gap together with P90, positive-gap rate, volume and route context. A large mean caused by a tiny fraction of events is different from a consistently positive gap. |

## Validation checklist

* Confirm timestamp units and the physical event represented by each date feature.
* Compare reconstructed gaps with MES queue, conveyor, inspection or buffer logs.
* Check whether stations run in parallel or record data asynchronously.
* Review product-family and route mix before comparing stations.
* Use a controlled operational change before claiming that reducing a gap improves quality.

**Repository evidence:** `reports/phase8\_station\_waiting\_times.csv` and `reports/phase8\_process\_map\_edges.csv`.

**CHAPTER 60**

# Bottlenecks

*Combining delay, consistency, volume and failure context into an investigation ranking*

## Why one metric is not enough

The station with the largest average gap is not automatically the most important bottleneck. A useful ranking also considers upper-tail waiting, how consistently a positive gap occurs, observed dwell span, product volume and whether failure rate is elevated.

![](data:image/png;base64...)

*Figure 60.1 — Components and weights in the repository bottleneck score.*

|  |
| --- |
| score = 100 \* (  0.30 \* norm(log1p(p90\_wait))  + 0.20 \* norm(log1p(avg\_wait))  + 0.15 \* norm(positive\_wait\_rate)  + 0.15 \* norm(log1p(p90\_dwell))  + 0.10 \* norm(log1p(product\_count))  + 0.10 \* norm(failure\_lift) ) |

Each component is min–max normalized across stations before weighting. Log transforms limit the dominance of extreme time and volume values. The resulting score is relative to this dataset and will change if the station population, time window or weights change.

## Bottleneck ranking

![](data:image/png;base64...)

*Figure 60.2 — Top 15 composite station bottleneck scores.*

| **Rank** | **Station** | **Score** | **Avg gap** | **P90 gap** | **Failure rate** | **Lift** |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | L1\_S25 | 88.3 | 30.31 | 36.50 | 0.507% | 0.90× |
| 2 | L1\_S24 | 74.4 | 23.25 | 41.01 | 0.828% | 1.48× |
| 3 | L3\_S38 | 70.6 | 22.46 | 32.77 | 0.781% | 1.39× |
| 4 | L0\_S13 | 59.0 | 18.25 | 31.58 | 0.547% | 0.97× |
| 5 | L2\_S28 | 57.2 | 7.88 | 13.44 | 0.699% | 1.25× |
| 6 | L2\_S27 | 57.1 | 6.27 | 12.23 | 0.681% | 1.21× |
| 7 | L2\_S26 | 55.5 | 5.49 | 9.86 | 0.747% | 1.33× |
| 8 | L0\_S12 | 55.5 | 21.78 | 28.63 | 0.547% | 0.97× |
| 9 | L3\_S29 | 54.3 | 5.79 | 9.25 | 0.585% | 1.04× |
| 10 | L3\_S39 | 51.8 | 6.17 | 8.51 | 0.506% | 0.90× |
| 11 | L3\_S30 | 50.2 | 6.89 | 8.89 | 0.585% | 1.04× |
| 12 | L3\_S43 | 48.1 | 9.07 | 10.42 | 0.520% | 0.93× |

## Operational bottleneck versus quality hotspot

| **Station** | **Bottleneck evidence** | **Quality evidence** | **Management implication** |
| --- | --- | --- | --- |
| L1\_S25 | Rank 1; score 88.34; average gap 30.31 | Failure lift 0.90× | Strong flow constraint, but not an above-average failure hotspot. |
| L1\_S24 | Rank 2; score 74.37; P90 gap 41.01 | Failure lift 1.48× | Combined operational and quality investigation priority. |
| L3\_S38 | Rank 3; score 70.63 | Failure lift 1.39× | High gap and elevated failure association. |
| L3\_S32 | Rank 16; score 40.18 | Failure rate 4.506%; lift 8.03× | Major quality-risk hotspot that is not the largest flow bottleneck. |

|  |
| --- |
| **Key distinction:** Bottleneck ranking answers where observed flow constraints concentrate. Failure lift answers where labelled failures are disproportionately represented. They are related diagnostic views, not interchangeable measures. |

## Critical process paths

The project also ranks complete route patterns. The critical-path score combines path volume, P90 waiting, smoothed failure lift, low throughput efficiency and the maximum bottleneck score among stations on the path.

![](data:image/png;base64...)

*Figure 60.3 — Top route patterns by critical-path score.*

| **Rank** | Path ID | **Products** | **Avg gap** | **Efficiency** | **Failure** | **Score** |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Path 1 | 31,964 | 28.37 | 0.197% | 0.781% | 70.6 |
| 2 | Path 2 | 20,156 | 28.44 | 0.197% | 0.740% | 69.3 |
| 3 | Path 3 | 19,767 | 28.41 | 0.196% | 0.708% | 69.2 |
| 4 | Path 4 | 20,797 | 28.30 | 0.197% | 0.621% | 69.2 |
| 5 | Path 5 | 12,042 | 34.44 | 7.546% | 0.413% | 68.7 |

|  |  |
| --- | --- |
| **Path ID** | **Observed station sequence** |
| Path 1 | L1\_S24 → L2\_S26 → L3\_S30 → L3\_S29 → L3\_S34 → L3\_S33 → L3\_S36 → L3\_S37 |
| Path 2 | L1\_S24 → L2\_S26 → L3\_S30 → L3\_S29 → L3\_S34 → L3\_S33 → L3\_S37 → L3\_S35 |
| Path 3 | L1\_S24 → L2\_S26 → L3\_S29 → L3\_S30 → L3\_S34 → L3\_S33 → L3\_S36 → L3\_S37 |
| Path 4 | L1\_S24 → L2\_S26 → L3\_S29 → L3\_S30 → L3\_S37 → L3\_S35 → L3\_S34 → L3\_S33 |
| Path 5 | L1\_S25 → L2\_S26 → L3\_S30 → L3\_S29 → L3\_S34 → L3\_S33 → L3\_S36 → L3\_S37 |

|  |
| --- |
| **Population note:** Path product\_count combines train and test observations; labelled\_count and failure rate use train products only. Failure rates are smoothed for ranking so rare paths do not dominate. Top path: L1\_S24 → L2\_S26 → L3\_S30 → L3\_S29 → L3\_S34 → L3\_S33 → L3\_S36 → L3\_S37. |

**Repository evidence:** `reports/phase8\_bottleneck\_scores.csv` and `reports/phase8\_critical\_process\_paths.csv`.

**CHAPTER 61**

# Throughput

*Summarizing observed active spans, timestamp gaps and route-level efficiency proxies*

## Repository definitions

| **Metric** | **Formula** | **Meaning** |
| --- | --- | --- |
| Productive time | Σ max(station\_end − station\_start, 0) | Sum of observed within-station measurement spans. |
| Waiting time | Σ max(next\_start − current\_end, 0) | Sum of positive inter-station timestamp gaps. |
| Cycle span | latest station end − earliest station start | Observed product measurement span. |
| Throughput efficiency | productive / (productive + waiting) | Proxy share of accounted time represented by station spans. |
| Wait-event count | count(gap > 0) | Number of positive timestamp-gap events. |

|  |
| --- |
| **Not OEE:** The throughput-efficiency field is not Overall Equipment Effectiveness. It has no direct availability, performance or quality components and is based on anonymized measurement timestamps. |

![](data:image/png;base64...)

*Figure 61.1 — Waiting-gap time is much larger than observed within-station span on the repository scale. A logarithmic axis is used.*

## Train and test comparison

| **Split** | **Products** | **Productive** | **Waiting** | **P90 wait** | **Cycle** | **Efficiency** | **Wait events** | **Failure** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Test | 1,183,748 | 0.1730 | 10.5198 | 33.8300 | 10.6980 | 0.4675% | 5.411 |  |
| Train | 1,183,747 | 0.1729 | 10.5402 | 33.9100 | 10.7184 | 0.4713% | 5.409 | 0.581% |

![](data:image/png;base64...)

*Figure 61.2 — The train and test populations have very similar timestamp-derived efficiency.*

## What similarity across splits supports

* The route/timestamp structure used by the process-mining pipeline is broadly similar between the two Kaggle splits.
* The transformation is not producing an obvious train-only distribution artifact at the aggregate level.
* The result supports stable batch scoring for this benchmark, not live-factory stability or future generalization.

## What it does not support

* It does not establish a real production rate, takt time, utilization or OEE.
* It does not prove that a timestamp gap is waste or that all within-station span is productive work.
* It does not show that reducing a gap will reduce failures.
* It does not replace capacity studies based on actual cycle counters, downtime reasons and line calendars.

**Repository evidence:** `reports/phase8\_throughput\_efficiency.csv`.

**CHAPTER 62**

# Failure Propagation

*Ranking candidate multi-station routes for evidence-driven investigation*

## From process graph to candidate route

Phase 9 extends the Phase 8 direct-follow graph by combining transition volume and failure association with station-level critical-node evidence. It searches for multi-station routes that repeatedly carry products through influential, bottlenecked or failure-associated nodes.

|  |
| --- |
| **Terminology:** Failure propagation is used here as an investigation label. The route scores do not demonstrate that a defect physically originated at one station and travelled downstream. |

![](data:image/png;base64...)

*Figure 62.1 — Highest-ranked candidate route in the repository.*

## Top route findings

| **Rank** | **Candidate route** | **Score** | **Min volume** | **Mean failure** | **Mean node score** |
| --- | --- | --- | --- | --- | --- |
| 1 | L3\_S29 → L3\_S30 → L3\_S33 → L3\_S34 | 51.72 | 965,844 | 0.507% | 47.74 |
| 2 | L3\_S29 → L3\_S30 → L3\_S33 → L3\_S37 | 50.81 | 654,274 | 0.520% | 46.88 |
| 3 | L1\_S24 → L3\_S29 → L3\_S30 → L3\_S33 | 50.28 | 25,747 | 0.744% | 51.25 |
| 4 | L1\_S24 → L2\_S26 → L3\_S29 → L3\_S30 → L3\_S33 → L3\_S34 | 50.09 | 243,524 | 0.630% | 48.10 |
| 5 | L1\_S24 → L2\_S26 → L3\_S29 → L3\_S30 | 50.07 | 243,524 | 0.734% | 49.49 |
| 6 | L2\_S26 → L3\_S29 → L3\_S30 → L3\_S33 | 49.89 | 257,914 | 0.610% | 45.63 |
| 7 | L1\_S24 → L3\_S29 → L3\_S30 → L3\_S33 → L3\_S34 → L3\_S37 | 49.69 | 25,747 | 0.636% | 48.93 |
| 8 | L3\_S29 → L3\_S30 → L3\_S34 → L3\_S37 | 49.57 | 264,184 | 0.576% | 47.22 |
| 9 | L1\_S24 → L2\_S26 → L3\_S29 → L3\_S30 → L3\_S33 → L3\_S37 | 49.51 | 243,524 | 0.638% | 47.53 |
| 10 | L1\_S24 → L3\_S29 → L3\_S30 → L3\_S32 | 49.43 | 25,747 | 2.040% | 52.82 |

## How route ranking should be read

![](data:image/png;base64...)

*Figure 62.2 — Candidate-route scores are tightly grouped, so ranking should guide review rather than imply certainty.*

![](data:image/png;base64...)

*Figure 62.3 — Some routes are supported by very high volume, while others show higher observed failure association with lower minimum volume.*

## Examples of the trade-off

| **Route** | **Evidence pattern** | **Interpretation** |
| --- | --- | --- |
| R1: L3\_S29 → L3\_S30 → L3\_S33 → L3\_S34 | Minimum volume 965,844; mean transition failure 0.507% | Strongest high-volume structural route. |
| R3: L1\_S24 → L3\_S29 → L3\_S30 → L3\_S33 | Minimum volume 25,747; mean transition failure 0.744% | Lower support at the first transition but higher observed failure association. |
| R10: route ending at L3\_S32 | Minimum volume 25,747; mean transition failure 2.040%. | Higher-risk candidate ending at the strongest station-level failure hotspot. |

## Engineering investigation workflow

1. Confirm that the candidate route represents a real physical sequence in the plant routing master.
2. Separate product families, variants, batches and shifts so product mix does not create a false route association.
3. Align station alarms, inspection results, maintenance history and tooling changes to the same product population.
4. Check whether the earliest abnormal signal occurs upstream of the detected failure and is available at decision time.
5. Compare the route with a matched lower-risk route that handles similar products.
6. Use controlled trials or natural experiments before assigning causal responsibility.
7. Document evidence, uncertainty and the decision owner for every recommended action.

## Common propagation mistakes

| **Mistake** | **Why it fails** | **Better practice** |
| --- | --- | --- |
| Treating arrows as causal direction | A direct-follow edge only records temporal order. | Use arrows as routing context; validate mechanisms separately. |
| Using train labels on test products | The Kaggle test population has no verified outcomes. | Keep route structure and labelled risk diagnostics separate. |
| Ignoring route frequency | Rare routes can have unstable failure rates. | Use minimum transition count and smoothed estimates. |
| Blaming the end station | The final station may only detect an upstream issue. | Search for the earliest process or sensor deviation. |
| Acting on a score alone | Composite scores depend on normalization and weights. | Review raw evidence and sensitivity before action. |

|  |
| --- |
| **Decision rule:** Use candidate routes to decide where to look first, what data to align and which comparison groups to build. Do not use them to automate blame, disposition or safety decisions. |

**Repository evidence:** `reports/phase9\_failure\_propagation\_routes.csv`, `reports/phase9\_critical\_nodes.csv`, and `reports/phase9\_knowledge\_graph\_report.html`.

# Part VIII Summary and Operational Checklist

Process mining converts sparse manufacturing timestamps into a structured view of product paths, direct-follow transitions, temporal gaps and station-level priorities. Its greatest value is not a single score. It is the ability to connect product routing, delay proxies, quality associations and network context in one evidence chain.

## Five conclusions

| **Area** | **Repository conclusion** |
| --- | --- |
| Event logs | 52 stations and 531 direct-follow transitions were reconstructed from relative date columns. |
| Waiting times | Positive gaps are useful operational signals but require MES and process validation. |
| Bottlenecks | L1\_S25 is the strongest flow constraint; L3\_S32 is a much stronger quality hotspot. |
| Throughput | Train and test timing proxies are similar, but the efficiency measure is not OEE. |
| Propagation | Candidate routes prioritize investigation; they do not establish physical defect transmission. |

## Before presenting a process-mining conclusion

* State the event and timestamp definitions.
* Show volume, central tendency and upper-tail measures together.
* Separate operational constraints from failure-risk hotspots.
* Identify whether metrics use train-only labels or train-plus-test structure.
* Use the words observed, associated, candidate and proxy where appropriate.
* Name the factory evidence required to confirm the interpretation.

## Primary repository artifacts

| **Artifact** | **Purpose** |
| --- | --- |
| src/data/phase8\_process\_mining\_bottleneck\_analysis.py | Chunked event reconstruction, transition aggregation, path and throughput analysis. |
| reports/phase8\_process\_map\_edges.csv | Direct-follow transition volume, waiting and failure diagnostics. |
| reports/phase8\_station\_waiting\_times.csv | Station-level dwell and inbound waiting summaries. |
| reports/phase8\_bottleneck\_scores.csv | Composite station bottleneck ranking. |
| reports/phase8\_critical\_process\_paths.csv | Route-level priority ranking with smoothed failure evidence. |
| reports/phase8\_throughput\_efficiency.csv | Train/test aggregate time and efficiency proxies. |
| reports/phase9\_failure\_propagation\_routes.csv | Candidate multi-station investigation routes. |

|  |
| --- |
| **Final interpretation boundary:** The public Bosch benchmark supports a technically complete process-mining prototype. Live operational recommendations require actual station semantics, plant calendars, MES event logs, maintenance records, validated labels and engineering approval. |