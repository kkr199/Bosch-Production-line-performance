**PART IX**

Knowledge Graph

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Chapters 63–67**

Nodes & Edges • Critical Stations • Graph Centrality • Communities • Failure Relationships

![](data:image/png;base64...)

*Knowledge-graph workflow used to connect manufacturing structure, model evidence and operational risk.*

**Bosch Production Line Performance — Technical Handbook**

Repository-grounded graph construction with an additional documented community analysis

# Contents and Knowledge-Graph Context

| **Chapter** | **Topic** | **Purpose** |
| --- | --- | --- |
| 63 | Nodes & Edges | Define the graph schema and explain how manufacturing evidence becomes linked entities. |
| 64 | Critical Stations | Rank stations by combined structural, operational and model evidence. |
| 65 | Graph Centrality | Interpret influence, bridge position, reachability and flow exposure. |
| 66 | Communities | Identify densely connected station groups in the weighted transition network. |
| 67 | Failure Relationships | Trace feature and station associations to failure without claiming causality. |

## Phase 9 evidence base

| **Item** | **Repository value** | **Interpretation** |
| --- | --- | --- |
| Total graph nodes | 4,321 | Lines, stations, features and one failure target node. |
| Total graph relationships | 4,923 | Hierarchy, process-flow and model-association links. |
| Station nodes | 52 | Manufacturing process nodes used for network analysis. |
| Feature nodes | 4,264 | Deduplicated raw feature metadata from the Bosch dataset. |
| Transition relationships | 531 | Observed directed station-to-station process links. |
| Failure-association relationships | 76 | Feature links with non-zero SHAP evidence. |
| Top critical station | L1\_S24 — 60.08 | Highest composite evidence priority in Phase 9. |

|  |
| --- |
| **Interpretation boundary:** The graph records hierarchy, temporal order and predictive association. It does not prove physical causation, defect transmission or responsibility for failure. |

## Learning objectives

* Explain the node and relationship types in the manufacturing knowledge graph.
* Distinguish structural importance from operational severity and model attribution.
* Interpret PageRank, betweenness, closeness and weighted flow in manufacturing terms.
* Use communities to understand route structure without treating them as official product families.
* Read feature-to-failure relationships as predictive evidence rather than causal mechanisms.

**CHAPTER 63**

# Nodes & Edges

*Representing production structure and analytical evidence as a connected graph*

## Why a knowledge graph is useful

Traditional tables store lines, stations, features, transition counts and model explanations in separate files. The knowledge graph links them in one navigable structure. A station can be connected to its production line, its raw measurements, downstream stations and the failure target through evidence-bearing relationships.

![](data:image/png;base64...)

*Figure 63.1 — Core graph schema used by Phase 9.*

## Node types

![](data:image/png;base64...)

*Figure 63.2 — Feature nodes dominate the graph because every raw measurement is represented individually.*

| **Node type** | **Count** | **Meaning** |
| --- | --- | --- |
| Line | 4 | Production lines L0–L3. |
| Station | 52 | Manufacturing process stations linked by observed transitions. |
| Feature | 4,264 | Raw numeric, categorical and date metadata linked to their station. |
| Failure | 1 | Target node representing Response = 1. |

## Relationship types

![](data:image/png;base64...)

*Figure 63.3 — Relationship composition of the 4,923 graph edges.*

| **Relationship** | **Count** | **Direction** | **Meaning** |
| --- | --- | --- | --- |
| CONTAINS\_STATION | 52 | Line → Station | Production hierarchy. |
| HAS\_FEATURE | 4,264 | Station → Feature | Station ownership of raw measurements. |
| TRANSITIONS\_TO | 531 | Station → Station | Observed direct-follow process relationship. |
| MODEL\_ASSOCIATED\_WITH\_FAILURE | 76 | Feature → Failure | Non-zero SHAP evidence from the selected model. |

|  |
| --- |
| **Edge semantics:** TRANSITIONS\_TO means station B followed station A in reconstructed timestamp order. MODEL\_ASSOCIATED\_WITH\_FAILURE means the feature influenced the model prediction on average. Neither relationship is a causal statement. |

## Graph construction logic

|  |
| --- |
| line:L1 --CONTAINS\_STATION--> station:L1\_S24 station:L1\_S24 --HAS\_FEATURE--> feature:L1\_S24\_F1844 station:L1\_S24 --TRANSITIONS\_TO--> station:L2\_S26 feature:L1\_S24\_F1844 --MODEL\_ASSOCIATED\_WITH\_FAILURE--> failure:Response\_1 |

The station subgraph is a directed graph for centrality analysis, while the complete graph is a MultiDiGraph because different relationship types can coexist. Node attributes store line, station, critical score and evidence value. Edge attributes store relationship type, weight, transition count and failure evidence.

**Repository evidence:** `src/data/phase9\_knowledge\_graph.py`, `reports/phase9\_knowledge\_graph\_nodes.csv`, and `reports/phase9\_knowledge\_graph\_edges.csv`.

**CHAPTER 64**

# Critical Stations

*Combining graph position, process constraints, failure concentration and model evidence*

## Critical-node scoring

A station can be important for several different reasons. It may be structurally central, operationally bottlenecked, associated with elevated failure, heavily used or strongly represented in SHAP explanations. Phase 9 combines these signals into one investigation-priority score.

![](data:image/png;base64...)

*Figure 64.1 — Components of the composite critical-node score.*

|  |
| --- |
| critical\_node\_score = 100 \* (  0.25 \* normalized\_bottleneck\_score  + 0.20 \* normalized\_failure\_lift  + 0.20 \* normalized\_station\_shap\_importance  + 0.20 \* normalized\_centrality\_score  + 0.15 \* normalized\_log\_product\_volume ) |

|  |
| --- |
| **Relative score:** The score is normalized across the 52 stations in this dataset. It is a prioritization measure, not an absolute risk probability or engineering acceptance limit. |

## Critical station ranking

![](data:image/png;base64...)

*Figure 64.2 — Top 15 stations by composite critical-node score.*

| **Rank** | **Station** | **Critical** | **Centrality** | **Bottleneck** | **Failure lift** | **SHAP** |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | L1\_S24 | 60.1 | 0.091 | 74.4 | 1.48× | 0.262 |
| 2 | L3\_S29 | 53.0 | 0.640 | 54.3 | 1.04× | 0.000 |
| 3 | L3\_S32 | 50.9 | 0.127 | 40.2 | 8.03× | 0.068 |
| 4 | L3\_S30 | 47.3 | 0.502 | 50.2 | 1.04× | 0.000 |
| 5 | L3\_S34 | 46.0 | 0.612 | 34.2 | 0.91× | 0.000 |
| 6 | L3\_S33 | 44.6 | 0.507 | 41.4 | 0.89× | 0.000 |
| 7 | L3\_S39 | 44.2 | 0.500 | 51.8 | 0.90× | 0.000 |
| 8 | L1\_S25 | 42.8 | 0.068 | 88.3 | 0.90× | 0.023 |
| 9 | L3\_S37 | 42.6 | 0.559 | 26.9 | 1.04× | 0.000 |
| 10 | L0\_S13 | 39.6 | 0.241 | 59.0 | 0.97× | 0.000 |
| 11 | L3\_S38 | 38.5 | 0.142 | 70.6 | 1.39× | 0.002 |
| 12 | L2\_S26 | 37.6 | 0.184 | 55.5 | 1.33× | 0.000 |

## Why the top stations differ

| **Station** | **Dominant evidence** | **Interpretation** |
| --- | --- | --- |
| L1\_S24 | Highest station SHAP evidence, bottleneck rank 2, failure lift 1.48× | Balanced quality, model and operational priority. |
| L3\_S29 | Highest centrality score among the top nodes and very high product flow | Major transfer and influence node in the observed process network. |
| L3\_S32 | Failure rate 4.506% and lift 8.03× | Strongest quality hotspot despite only moderate network centrality. |
| L3\_S30 | High centrality and very high flow | Core routing node with broad downstream influence. |
| L1\_S25 | Bottleneck rank 1 but failure lift below 1.0× | Strong operational constraint, not an above-average quality hotspot. |

|  |
| --- |
| **Management lesson:** A critical-station list should be decomposed before action. The same score can arise from different combinations of flow, delay, failure concentration and model evidence. |

## Recommended investigation sequence

1. Check whether the station is critical because of centrality, bottleneck severity, failure lift, SHAP evidence or volume.
2. Review the top incoming and outgoing transitions to understand route context.
3. Compare affected product families and production windows.
4. Validate sensor names, measurement semantics, maintenance history and inspection records.
5. Use a controlled change or matched comparison before assigning causal responsibility.

**Repository evidence:** `reports/phase9\_critical\_nodes.csv` and `reports/phase9\_station\_centrality\_metrics.csv`.

**CHAPTER 65**

# Graph Centrality

*Understanding influence, bridge position, reachability and flow exposure*

## Centrality measures used by Phase 9

| **Metric** | **Manufacturing interpretation** | **Important caution** |
| --- | --- | --- |
| In-degree centrality | Number of different upstream stations entering a station. | Counts route variety, not volume. |
| Out-degree centrality | Number of different downstream stations leaving a station. | May reflect routing alternatives or data artifacts. |
| Weighted inflow/outflow | Total transition volume entering or leaving the station. | High flow does not automatically imply high defect risk. |
| PageRank | Importance gained from receiving flow from other influential stations. | Sensitive to graph direction and edge weights. |
| Betweenness | Share of weighted shortest paths that pass through the station. | Highlights bridges, not necessarily bottlenecks. |
| Closeness | How efficiently the station can reach the rest of the graph. | Depends on network connectivity and distance definition. |

![](data:image/png;base64...)

*Figure 65.1 — PageRank and betweenness reveal different kinds of network importance.*

## Centrality score used by the project

|  |
| --- |
| centrality\_score = (  0.35 \* normalized\_pagerank  + 0.35 \* normalized\_betweenness  + 0.15 \* normalized\_closeness  + 0.15 \* normalized\_weighted\_outflow ) |

PageRank and betweenness receive the largest weights because the project emphasizes influential receiving points and bridge stations. Closeness and weighted outflow add reachability and production-volume context.

![](data:image/png;base64...)

*Figure 65.2 — The top critical stations have different centrality profiles rather than one uniform pattern.*

| **Station** | **Score** | **PageRank** | **Betweenness** | **Closeness** | **Weighted outflow** |
| --- | --- | --- | --- | --- | --- |
| L3\_S29 | 0.640 | 0.0529 | 0.0561 | 3.461 | 2,239,065 |
| L3\_S34 | 0.612 | 0.0940 | 0.0016 | 3.215 | 2,207,703 |
| L3\_S37 | 0.559 | 0.1026 | 0.0024 | 3.080 | 1,022,189 |
| L3\_S33 | 0.507 | 0.0680 | 0.0004 | 3.151 | 2,093,661 |
| L3\_S30 | 0.502 | 0.0475 | 0.0259 | 2.763 | 2,239,366 |
| L3\_S39 | 0.500 | 0.0068 | 0.1192 | 2.960 | 120,019 |
| L3\_S47 | 0.411 | 0.0456 | 0.0498 | 2.464 | 119,985 |
| L0\_S9 | 0.390 | 0.0084 | 0.1012 | 0.997 | 451,406 |
| L3\_S36 | 0.370 | 0.0607 | 0.0000 | 3.021 | 540,072 |
| L3\_S35 | 0.343 | 0.0498 | 0.0000 | 2.736 | 887,932 |

## Interpreting key examples

| **Station** | **Centrality pattern** | **Operational reading** |
| --- | --- | --- |
| L3\_S29 | Highest composite centrality; strong PageRank, betweenness and flow | Major transfer point whose behavior can influence many routes. |
| L3\_S34 | High PageRank and closeness with very large flow | Influential receiving node embedded in the central Line 3 cluster. |
| L3\_S39 | High betweenness relative to its volume | Bridge-like station linking parts of the later Line 3 network. |
| L0\_S9 | High betweenness and outflow | Gateway from early Line 0 toward downstream cross-line routes. |
| L1\_S24 | Low centrality score but highest overall critical score | Quality and bottleneck evidence dominate its priority. |

|  |
| --- |
| **Interpretation mistake to avoid:** The most central station is not automatically the root cause of failure. Centrality describes graph position; failure lift and SHAP describe statistical association; engineering evidence is still required. |

**Repository evidence:** `src/data/phase9\_knowledge\_graph.py` centrality calculations and `reports/phase9\_station\_centrality\_metrics.csv`.

**CHAPTER 66**

# Communities

*Finding densely connected station groups in the weighted process network*

## Community method used for this handbook

The repository does not export a community label. For this handbook, the 531 directed transition relationships were converted into an undirected weighted station graph. NetworkX greedy modularity optimization was then applied using transition count as the edge weight. The result is a reproducible structural segmentation of the station network.

|  |
| --- |
| **Derived analysis:** These communities are an additional handbook analysis derived from repository transition data. They are not product families, official routing groups or plant organizational units. |

![](data:image/png;base64...)

*Figure 66.1 — Five weighted station communities identified from the highest-volume transition network.*

## Community summary

![](data:image/png;base64...)

*Figure 66.2 — Community 5 has the highest average critical and centrality scores, while Community 4 contains the highest single critical station.*

| **ID** | **Structural interpretation** | **Stations** | **Leading members** | **Avg critical** | **Avg centrality** | **Max failure** |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | Late Line 3 inspection/transfer cluster | 13 | L3\_S39, L3\_S47, L3\_S45, L3\_S41, L3\_S43 | 26.5 | 0.213 | 0.520% |
| C2 | Mid-to-late Line 0 cluster | 12 | L0\_S13, L0\_S12, L0\_S17, L0\_S14, L0\_S20 | 26.7 | 0.141 | 0.566% |
| C3 | Early Line 0 cluster | 10 | L0\_S5, L0\_S8, L0\_S4, L0\_S1, L0\_S2 | 25.4 | 0.153 | 0.545% |
| C4 | Cross-line transfer core | 10 | L1\_S24, L3\_S29, L3\_S30, L1\_S25, L2\_S26 | 39.2 | 0.236 | 0.828% |
| C5 | Core Line 3 quality cluster | 7 | L3\_S32, L3\_S34, L3\_S33, L3\_S37, L3\_S38 | 42.1 | 0.380 | 4.506% |

## Community interpretations

| **Community** | **Main stations** | **Interpretation** |
| --- | --- | --- |
| C1 | L3\_S39–L3\_S51 | Later Line 3 inspection/transfer structure with several bridge-like stations. |
| C2 | L0\_S12–L0\_S23 | Mid-to-late Line 0 sequence with broadly stable failure rates near the baseline. |
| C3 | L0\_S0–L0\_S10 | Early Line 0 process block dominated by high-volume sequential transitions. |
| C4 | L1\_S24, L1\_S25, L2\_S26–S28, L3\_S29–S31, L0\_S9–S11 | Cross-line transfer core containing the overall top critical station and several bottlenecks. |
| C5 | L3\_S32–L3\_S38 | Core Line 3 quality cluster containing L3\_S32 and high-centrality stations L3\_S33–S37. |

|  |
| --- |
| **Why communities matter:** Station communities provide a useful unit for monitoring route changes, comparing failure concentration and organizing engineering review. They should be validated against the actual routing master and product mix before operational use. |

## Community validation checklist

* Compare the graph-derived group with the physical line layout and routing master.
* Check whether the group is stable across time windows, product families and shifts.
* Measure internal transition volume versus external transition volume.
* Review whether one high-volume route dominates the community assignment.
* Recompute communities after major process or routing changes.

**Repository evidence:** `reports/phase8\_process\_map\_edges.csv`; derived outputs saved as `part\_ix\_station\_communities.csv` and `part\_ix\_community\_summary.csv`.

**CHAPTER 67**

# Failure Relationships

*Connecting predictive model evidence to the manufacturing graph without converting association into causation*

## How failure relationships enter the graph

Phase 7 produces global SHAP evidence for the selected LightGBM model. Phase 9 maps each raw model feature back to its base feature metadata. When a feature has non-zero SHAP evidence, the graph creates a directed MODEL\_ASSOCIATED\_WITH\_FAILURE relationship from that feature node to the failure node.

|  |
| --- |
| **Meaning of edge weight:** The relationship weight is aggregated mean absolute SHAP evidence. It measures average model contribution magnitude, not defect probability, physical force or causal effect. |

![](data:image/png;base64...)

*Figure 67.1 — Nine strongest raw feature-to-failure associations represented in the graph.*

| **Feature** | **Station** | **Line** | **Graph edge weight** |
| --- | --- | --- | --- |
| L3\_S32\_F3850 | L3\_S32 | L3 | 0.06799 |
| L2\_S27\_F3144 | L2\_S27 | L2 | 0.02332 |
| L1\_S24\_F1844 | L1\_S24 | L1 | 0.02089 |
| L1\_S24\_F1846 | L1\_S24 | L1 | 0.01784 |
| L1\_S24\_F1723 | L1\_S24 | L1 | 0.01380 |
| L1\_S24\_F1778 | L1\_S24 | L1 | 0.01257 |
| L1\_S24\_F1667 | L1\_S24 | L1 | 0.01168 |
| L1\_S24\_F1498 | L1\_S24 | L1 | 0.01041 |
| L1\_S24\_F1713 | L1\_S24 | L1 | 0.00902 |

## Example evidence chain: L1\_S24

![](data:image/png;base64...)

*Figure 67.2 — Selected L1\_S24 feature nodes connected to the failure target through model evidence.*

L1\_S24 becomes the highest critical station because multiple evidence layers align: elevated failure lift, high bottleneck severity and the largest station-level SHAP contribution. The graph makes this evidence chain visible, but the individual feature identifiers remain anonymized and require factory metadata before any physical interpretation.

![](data:image/png;base64...)

*Figure 67.3 — Structural influence and failure concentration are related but distinct dimensions.*

## Failure relationship hierarchy

| **Evidence layer** | **Question answered** | **What it cannot prove** |
| --- | --- | --- |
| Transition failure rate | Which observed station transitions contain more labelled failures? | That either station caused the failure. |
| Station failure lift | Which station-exposed products fail more often than average? | That the station is the physical origin. |
| SHAP feature relationship | Which inputs drive the model prediction? | That changing the feature changes the outcome. |
| Critical-node score | Where should investigation start? | A calibrated failure probability or causal ranking. |
| Candidate route score | Which multi-station paths deserve review? | Physical propagation of a defect. |

## Responsible engineering workflow

1. Use the graph to identify the station, route and features that should be reviewed together.

2. Translate anonymized feature identifiers using sensor and process metadata.

3. Align maintenance, inspection, tooling and batch records to the same products.

4. Check temporal availability so no post-failure information is treated as a predictor.

5. Compare matched products that share route and family but differ in outcome.

6. Confirm suspected mechanisms through controlled tests or approved process experiments.

7. Record confidence level and unresolved alternatives in every root-cause recommendation.

|  |
| --- |
| **Final rule:** The graph supports evidence navigation and investigation prioritization. Quality engineers retain the final decision, and causal claims require process records plus controlled evidence. |

**Repository evidence:** `reports/phase7\_shap\_global\_importance.csv`, `reports/phase9\_knowledge\_graph\_edges.csv`, and `reports/phase9\_failure\_propagation\_routes.csv`.

# Part IX Summary and Knowledge-Graph Checklist

The manufacturing knowledge graph unifies process hierarchy, station flow, raw feature metadata, model explanations and failure evidence. Its value comes from preserving the links between these evidence types while keeping their meanings separate.

## Five conclusions

| **Area** | **Repository conclusion** |
| --- | --- |
| Nodes & edges | 4,321 nodes and 4,923 relationships connect lines, stations, features, transitions and failure evidence. |
| Critical stations | L1\_S24 ranks first because model, bottleneck and failure evidence align. |
| Centrality | L3\_S29 is the strongest network influence node, while L3\_S32 is the strongest quality hotspot. |
| Communities | Five weighted station groups summarize the process network; Communities 4 and 5 contain the main priority clusters. |
| Failure relationships | 76 feature-to-failure links encode SHAP association and must not be interpreted as causal mechanisms. |

## Before presenting a graph conclusion

* Name the node type, relationship type and edge weight being interpreted.
* Separate graph position from operational bottleneck and failure association.
* State whether the result is a repository output or an additional derived analysis.
* Use labels such as observed, associated, candidate and proxy where appropriate.
* Show the supporting station, route and feature evidence rather than only the composite score.
* Identify the plant data required to confirm the interpretation.

## Primary artifacts

| **Artifact** | **Purpose** |
| --- | --- |
| src/data/phase9\_knowledge\_graph.py | Graph construction, centrality, critical-node scoring and route search. |
| reports/phase9\_knowledge\_graph\_nodes.csv | Node inventory with type, label, location and evidence attributes. |
| reports/phase9\_knowledge\_graph\_edges.csv | Relationship inventory with weights and evidence fields. |
| reports/phase9\_station\_centrality\_metrics.csv | Station-level centrality, operational and failure metrics. |
| reports/phase9\_critical\_nodes.csv | Composite station priority ranking. |
| reports/phase9\_failure\_propagation\_routes.csv | Candidate multi-station investigation routes. |
| reports/phase9\_manufacturing\_knowledge\_graph.graphml | Portable graph representation for graph tools. |

|  |
| --- |
| **Final interpretation boundary:** The public Bosch benchmark supports a portfolio-grade manufacturing knowledge graph. Production adoption requires real feature semantics, routing governance, live system integration, validated labels and engineering approval. |