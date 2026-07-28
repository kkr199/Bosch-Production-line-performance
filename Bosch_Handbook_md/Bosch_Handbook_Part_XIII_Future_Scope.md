**PART XIV**

Future Scope

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Chapters 90-95**

MES Integration • IoT • PLC • Digital Twins • LLMs • Manufacturing Copilot Roadmap

![Target evolution from machine data and manufacturing systems to governed decision support.](data:image/png;base64...)

*Target evolution from machine data and manufacturing systems to governed decision support.*

**Bosch Production Line Performance - Technical Handbook**

A staged roadmap from the public benchmark to a future automotive POC/MVP

# Contents and Future-Scope Boundary

| **Chapter** | **Topic** | **Future capability** |
| --- | --- | --- |
| 90 | MES Integration | Connect product, order, route, operation and quality outcomes. |
| 91 | IoT | Collect trustworthy high-frequency machine and sensor telemetry. |
| 92 | PLC | Expose approved control-system tags without entering safety logic. |
| 93 | Digital Twins | Synchronize plant state and test what-if quality and flow scenarios. |
| 94 | LLMs | Add grounded natural-language assistance with tools, citations and guardrails. |
| 95 | Manufacturing Copilot Roadmap | Move from offline Q&A to a governed engineering workspace. |

![Figure XIV.1 - Each stage requires new data, validation and operational ownership.](data:image/png;base64...)

*Figure XIV.1 - Each stage requires new data, validation and operational ownership.*

|  |
| --- |
| **Current boundary:** The repository currently uses historical anonymized Kaggle data, reviewed reports, SQLite, Streamlit and an offline deterministic Copilot. MES, live IIoT, PLC, digital-twin and LLM integrations are proposed future work, not delivered factory connections. |

## Guiding principles

* Keep the official predictive model advisory and human-reviewed.
* Declare the exact scoring point before collecting new features.
* Prefer standard interfaces and governed semantic models over device-specific point solutions.
* Treat timestamps, units, data quality and asset identity as first-class data.
* Introduce decision impact gradually: offline replay, shadow mode, assisted pilot and staged scale.
* Require evidence, citations and explicit uncertainty from any generative-AI layer.

**CHAPTER 94**

# MES Integration

*Connecting manufacturing operations context to predictive-quality decisions*

## Learning objectives

* Explain the role of MES between control systems and enterprise planning.
* Define the events and master data required by the Bosch predictive-quality use case.
* Design a point-in-time-safe MES-to-feature pipeline.
* Return model recommendations and mature quality outcomes without bypassing human authority.
* Use ISA-95-style boundaries to clarify system ownership.

![Figure 94.1 - MES sits in the manufacturing-operations layer and connects plant execution with business context.](data:image/png;base64...)

*Figure 94.1 - MES sits in the manufacturing-operations layer and connects plant execution with business context.*

The future MES integration should turn anonymized station and timing features into a governed manufacturing record. A scoreable record needs product identity, order and material context, planned and actual route, operation completion, equipment, quality status and the time when each fact became available.

| **Information domain** | **Minimum fields** | **Project use** |
| --- | --- | --- |
| Product | product\_id, type, variant, serial or batch | Case identity and product-family context |
| Order | order\_id, quantity, priority, planned dates | Production context and demand constraints |
| Route | route\_id, operation sequence, alternate path | Manufacturing-flow and path features |
| Operation | start, complete, station, status, rework | Point-in-time event log |
| Equipment | asset\_id, state, tool and configuration | Station condition and maintenance context |
| Quality | inspection, result, defect code, label maturity | Training target and operational outcome |
| Material | lot, supplier, material status | Batch and upstream-risk analysis |
| Personnel / shift | approved pseudonymous context | Shift-level analysis where legally and ethically approved |

## Target MES data flow

![Figure 94.2 - The model consumes a point-in-time feature contract and returns an advisory review priority.](data:image/png;base64...)

*Figure 94.2 - The model consumes a point-in-time feature contract and returns an advisory review priority.*

![Figure 94.3 - Product, route, operation and quality domains are the first integration priorities.](data:image/png;base64...)

*Figure 94.3 - Product, route, operation and quality domains are the first integration priorities.*

|  |
| --- |
| **Point-in-time rule:** A feature may be used only when its source event is available before or at the declared scoring point. Final inspection, downstream failure and future operation data must not leak into the score. |

## Canonical MES event

|  |
| --- |
| {  "event\_id": "uuid",  "event\_type": "operation\_completed",  "event\_time\_utc": "2026-07-21T08:30:14.125Z",  "available\_time\_utc": "2026-07-21T08:30:16.002Z",  "product\_id": "pseudonymous-id",  "order\_id": "order-identifier",  "route\_id": "route-version",  "operation\_id": "OP-320",  "asset\_id": "L3\_S32",  "status": "complete",  "quality": {"code": "GOOD", "source": "MES"},  "schema\_version": "mes-event-v1" } |

## Integration patterns

| **Pattern** | **Best use** | **Control** |
| --- | --- | --- |
| Batch extract | Historical POC and backfill | Snapshot date, hash and reconciliation |
| REST/API | On-demand product and operation context | Authentication, timeout and idempotency |
| Message/event bus | Near-real-time operation and quality events | Schema registry, ordering and replay |
| ISA-95/B2MML mapping | Cross-vendor manufacturing semantics | Governed object and attribute mapping |
| Database replication | High-volume analytical access | Read-only access and source-system ownership |

**Project evidence:** The current project already defines advisory batch scoring, a strict feature contract, audit logging and mature-label monitoring. MES integration supplies the missing plant system of record and label lifecycle.

## MES implementation roadmap

1. Create a source inventory and data-owner approval for MES product, route, operation and quality objects.

2. Map plant equipment and operations to the existing Line-Station representation.

3. Build a historical extract and reconcile product counts, event order and quality labels.

4. Declare the scoring event and produce a point-in-time feature view.

5. Run offline replay against mature historical outcomes.

6. Shadow-write predictions to a review table without changing the factory workflow.

7. Add quality-engineer feedback and final outcome timestamps.

8. Promote only after holdout, capacity, security and operator acceptance gates pass.

## Common mistakes

| **Mistake** | **Why it fails** | **Better practice** |
| --- | --- | --- |
| Using current MES state for historical rows | Creates future-data leakage | Reconstruct state as of each scoring time. |
| Treating operation completion as event time only | Late-arriving data changes availability | Store event time and available time. |
| Hard-coding station names | Plant routing and assets change | Use versioned master-data mappings. |
| Writing model scores back as quality truth | Prediction becomes confused with outcome | Separate score, review action and final label. |
| Ignoring rework | A product may repeat an operation or route | Model event occurrence and route version explicitly. |

## Interview questions

* Why should event\_time and available\_time be stored separately?
* How would ISA-95 help an MES integration project?
* What makes a feature point-in-time safe?
* How should rework loops be represented in an event log?
* Why must prediction, inspection action and final defect outcome remain separate records?

**CHAPTER 95**

# IoT

*Collecting trustworthy machine telemetry through edge and message-oriented architectures*

## Learning objectives

* Distinguish raw telemetry from model-ready features.
* Use OPC UA for industrial information access and MQTT for scalable publish/subscribe transport.
* Preserve timestamp, unit, quality code, asset identity and calibration context.
* Design buffering and replay for intermittent factory connectivity.
* Prevent high-frequency data volume from overwhelming storage and review systems.

![Figure 95.1 - Edge normalization separates industrial protocols from downstream analytical services.](data:image/png;base64...)

*Figure 95.1 - Edge normalization separates industrial protocols from downstream analytical services.*

The Kaggle dataset contains sparse, already-extracted measurements. A future IIoT implementation must handle the much harder problem of creating trustworthy records from live devices: clock alignment, units, quality flags, duplicate messages, out-of-order delivery, calibration, buffering and equipment metadata.

| **Telemetry field** | **Purpose** |
| --- | --- |
| asset\_id | Stable equipment identity tied to a governed asset hierarchy |
| tag\_id | Versioned measurement or state definition |
| event\_time\_utc | When the device or gateway observed the value |
| ingest\_time\_utc | When the analytical platform received it |
| value and unit | Numerical or categorical observation with engineering unit |
| quality\_code | Good, uncertain, bad or vendor-specific quality translated to a standard set |
| sequence\_id | Detect duplicate, missing and out-of-order messages |
| calibration\_version | Interpret measurement changes and maintenance events |
| schema\_version | Keep producers and consumers compatible |

## Telemetry quality and topic design

![Figure 95.2 - Context and quality metadata are as important as the sensor value.](data:image/png;base64...)

*Figure 95.2 - Context and quality metadata are as important as the sensor value.*

![Figure 95.3 - An illustrative governed MQTT topic namespace.](data:image/png;base64...)

*Figure 95.3 - An illustrative governed MQTT topic namespace.*

|  |
| --- |
| {  "asset\_id": "L1\_S24",  "tag\_id": "F1844",  "event\_time\_utc": "2026-07-21T08:30:14.125Z",  "ingest\_time\_utc": "2026-07-21T08:30:14.240Z",  "value": 12.47,  "unit": "configured-engineering-unit",  "quality\_code": "GOOD",  "sequence\_id": 912873,  "calibration\_version": "CAL-2026-06",  "schema\_version": "telemetry-v1" } |

## OPC UA and MQTT responsibilities

| **Technology** | **Primary role** | **Future Bosch use** |
| --- | --- | --- |
| OPC UA | Industrial information model, secure client/server and PubSub access | Read approved PLC/device values with asset and type context |
| MQTT | Lightweight broker-based publish/subscribe messaging | Move normalized telemetry and events to edge/cloud consumers |
| Schema registry | Version and validate payload definitions | Prevent producer/consumer drift |
| Time-series store | Efficient event and telemetry retention | Feature windows, trend and drift analysis |
| Stream processor | Window, join, deduplicate and enrich events | Create online feature state and alert metrics |

|  |
| --- |
| **Security boundary:** Do not expose PLCs directly to public networks or analytics clients. Use approved industrial network zones, gateways, certificates, least privilege and ISA/IEC 62443-aligned controls. |

## IoT implementation roadmap

1. Select a small set of stations linked to strong model signals, such as L1\_S24, L1\_S25 or L3\_S32, after engineering review.

2. Create an asset/tag dictionary with units, ranges, quality semantics and owners.

3. Collect in shadow mode through an edge gateway without writing to control devices.

4. Measure clock skew, data loss, duplicate rate and message latency.

5. Build feature windows that reproduce approved historical definitions.

6. Compare live feature distributions with the training reference.

7. Add retention tiers: high-frequency raw, normalized telemetry and long-term aggregates.

8. Expand only after data-quality and cybersecurity gates pass.

## Common mistakes

| **Mistake** | **Risk** | **Correction** |
| --- | --- | --- |
| Sending every PLC tag at maximum frequency | Cost and noise grow without decision value | Collect only approved use-case tags at justified rates. |
| Ignoring quality codes | Bad sensor values look valid | Carry and validate source quality. |
| Using gateway time as device time | Event order and windows become misleading | Preserve both event and ingest time. |
| No offline buffer | Network interruption loses production evidence | Use durable edge store-and-forward. |
| Uncontrolled MQTT wildcards | Clients receive excessive or unauthorized data | Use topic ACLs and explicit subscriptions. |

**CHAPTER 96**

# PLC

*Accessing deterministic control-system context without putting ML in the control loop*

## Learning objectives

* Explain why PLC scan logic and predictive analytics have different timing and assurance requirements.
* Keep model scoring read-only and asynchronous.
* Select approved tags with data types, engineering units and update semantics.
* Use an edge gateway or OPC UA server rather than direct database-style polling.
* Define safe behavior when the analytics service is unavailable.

![Figure 96.1 - PLC logic remains deterministic; the ML score is advisory and consumed through MES/HMI workflow.](data:image/png;base64...)

*Figure 96.1 - PLC logic remains deterministic; the ML score is advisory and consumed through MES/HMI workflow.*

|  |
| --- |
| **Non-negotiable boundary:** The future Bosch failure-risk model must not execute safety interlocks, motion control, emergency stops or automatic product disposition. PLC and safety-system owners retain control authority. |

## PLC tag contract

| **Field** | **Example** | **Why it matters** |
| --- | --- | --- |
| asset\_id | L3\_S32 | Maps the tag to the station and knowledge graph |
| tag\_name | InspectionReady | Human-readable governed identity |
| data\_type | BOOL / DINT / REAL | Prevents unsafe or ambiguous conversion |
| engineering\_unit | mm, bar, degC or none | Supports ranges and comparability |
| access | Read-only | Prevents analytics from changing control state |
| update\_mode | Cyclic / change / event | Determines sampling and message volume |
| quality | Good / uncertain / bad | Prevents invalid control data from entering features |
| owner | Controls Engineering | Defines approval and change process |

## Timing and architecture

![Figure 96.2 - Predictive-quality scoring operates at a slower asynchronous time scale than deterministic control.](data:image/png;base64...)

*Figure 96.2 - Predictive-quality scoring operates at a slower asynchronous time scale than deterministic control.*

| **PLC integration rule** | **Implementation** |
| --- | --- |
| Read-only by default | Analytics clients cannot write control tags. |
| No dependency for safe operation | Production follows the approved manual/control fallback if scoring is unavailable. |
| Edge isolation | Gateway terminates industrial protocol and publishes normalized events. |
| Rate control | Use subscriptions/change events where possible; avoid aggressive polling. |
| Versioned mapping | Tag-to-feature mappings are reviewed and released with the feature contract. |
| Commissioning | Controls engineer validates type, unit, state transition and failover. |
| Change management | PLC logic or tag changes trigger schema and leakage review. |

## Illustrative edge mapping

|  |
| --- |
| tag\_mapping:  - plc\_node: "ns=4;s=Line3.Station32.MeasurementReady"  asset\_id: "L3\_S32"  canonical\_tag: "measurement\_ready"  data\_type: "boolean"  access: "read\_only"  update\_mode: "data\_change"  feature\_use: "availability\_context"  owner: "Controls Engineering" |

## Commissioning checklist

1. Review the tag list with Controls, OT Security, Data Engineering and Quality.

2. Prove read-only permissions in a test environment.

3. Verify values against the HMI or engineering workstation.

4. Test restart, network loss, stale value and bad-quality behavior.

5. Confirm that no prediction delay blocks the PLC or operator process.

6. Document the safe fallback and escalation owner.

# PLC common mistakes and interview questions

| **Mistake** | **Why it is unsafe or unreliable** | **Better design** |
| --- | --- | --- |
| Writing the model class to a PLC control bit | A statistical model enters deterministic control | Show advisory status through MES/HMI with human approval. |
| Polling every scan | Adds network and PLC load | Use justified subscriptions or edge sampling. |
| Ignoring stale values | Old states look current | Track timestamp, age and quality. |
| No tag versioning | Feature meaning changes silently | Release mappings with the model contract. |
| Using the PLC as a long-term historian | Retention and query requirements do not match | Move normalized events to historian/time-series storage. |

## Interview questions

* Why should an ML service not be placed in a PLC safety loop?
* What is the difference between cyclic polling and data-change subscriptions?
* How would you detect a stale PLC value?
* Why are data type, unit and quality code required in a tag contract?
* What is a safe fallback when the edge gateway is offline?

**CHAPTER 97**

# Digital Twins

*Extending process mining and the knowledge graph into synchronized, validated what-if models*

## Learning objectives

* Differentiate a digital twin from a static dashboard, simulation or data lake.
* Connect the existing process graph, product families, risk model and live plant state.
* Define synchronization frequency, fidelity and intended decision.
* Apply verification, validation and uncertainty quantification before use.
* Prioritize a small, credible twin use case rather than modelling the entire factory.

![Figure 97.1 - A future twin combines physical assets, trusted data, semantics, models and decisions.](data:image/png;base64...)

*Figure 97.1 - A future twin combines physical assets, trusted data, semantics, models and decisions.*

The existing Bosch project already contains useful twin foundations: station and line metadata, manufacturing paths, process transitions, bottleneck scores, product families, a knowledge graph, risk predictions and diagnostic explanations. A digital twin would add synchronization with a real plant, validated state estimation and scenario models tied to a specific decision.

| **Existing project asset** | **Digital-twin extension** |
| --- | --- |
| 52-station metadata | Map to real asset hierarchy and engineering metadata |
| 531 process transitions | Validate against actual routes and event sequence |
| Bottleneck scores | Replace timestamp proxies with verified queue and cycle measures |
| Product families | Link to product type, variant and route master |
| Knowledge graph | Become semantic relationship layer for assets, products and events |
| LightGBM risk model | Provide predictive-quality component, not the entire twin |
| SHAP and action plan | Support investigation with uncertainty and physical validation |

## Closed-loop twin workflow

![Figure 97.2 - The loop is complete only when outcomes and model uncertainty return to the system.](data:image/png;base64...)

*Figure 97.2 - The loop is complete only when outcomes and model uncertainty return to the system.*

![Figure 97.3 - Predictive quality has the strongest current project foundation; maintenance and energy need new data.](data:image/png;base64...)

*Figure 97.3 - Predictive quality has the strongest current project foundation; maintenance and energy need new data.*

## Recommended first twin

|  |
| --- |
| **Pilot scope:** Build a line- or station-level predictive-quality and flow twin for one approved product family. Synchronize route state, queue/WIP, selected telemetry, quality-risk score and inspection outcome. Do not begin with a full-plant twin. |

| **Twin requirement** | **Definition for the pilot** |
| --- | --- |
| Physical scope | One line or small station group with clear ownership |
| Decision | Prioritize review and compare queue/risk scenarios |
| Synchronization | Event-driven or approved periodic refresh |
| Fidelity | Enough detail to reproduce route, state and review decision |
| Models | Process-flow logic, risk model and simple what-if simulation |
| Uncertainty | Data quality, model confidence and scenario assumptions displayed |
| Validation | Compare predicted and observed states/outcomes over a defined period |
| Human control | Engineer approves any operational recommendation |

## Twin credibility and governance

| **Discipline** | **Question** |
| --- | --- |
| Verification | Was the digital-twin implementation built correctly according to its design? |
| Validation | Does the twin represent the real process well enough for its intended decision? |
| Uncertainty quantification | How uncertain are data, parameters, model predictions and scenarios? |
| Configuration control | Which asset, route, model and software versions does the twin represent? |
| Interoperability | Can MES, IoT, PLC, models and engineering tools exchange meaning consistently? |
| Cybersecurity | Can the twin be used without creating unsafe paths into operational technology? |
| Change management | What requires revalidation after equipment, product or process change? |

## Common mistakes

| **Mistake** | **Why it fails** |
| --- | --- |
| Calling any dashboard a digital twin | A twin requires synchronization, defined fidelity and an intended use. |
| Starting with the whole plant | Scope, data and validation become unmanageable. |
| Ignoring uncertainty | Simulation output appears more precise than the evidence supports. |
| Using unverified timestamp proxies | The current benchmark does not provide physical cycle or queue truth. |
| Allowing direct control too early | An unvalidated model gains operational authority. |
| No lifecycle ownership | The twin becomes stale after process or equipment change. |

## Interview questions

* What distinguishes a digital twin from a simulation?
* Why are synchronization frequency and fidelity use-case dependent?
* What is VVUQ, and why is it important for manufacturing twins?
* How can the current knowledge graph support a digital twin?
* Why should a small line-level twin be built before a plant-wide twin?

**CHAPTER 98**

# LLMs

*Adding grounded language-model assistance without giving generative AI control authority*

## Learning objectives

* Understand the current deterministic offline Copilot and its limitations.
* Design a retrieval-augmented, tool-using LLM layer over approved evidence.
* Protect manufacturing data, secrets and operational systems.
* Evaluate groundedness, citations, tool accuracy, safety and usefulness.
* Keep all high-impact recommendations human-approved.

The current Manufacturing Copilot is deliberately offline and deterministic. It uses reviewed intent templates, parameterized SQL, TF-IDF retrieval and local reports. A future LLM can improve language flexibility and multi-step reasoning, but it must preserve the current evidence-first and human-authority design.

![Figure 98.1 - The LLM is one governed component between approved tools and a human decision.](data:image/png;base64...)

*Figure 98.1 - The LLM is one governed component between approved tools and a human decision.*

| **Current capability** | **Future LLM extension** | **Control** |
| --- | --- | --- |
| Intent templates | Flexible intent classification | Allowed topic and tool policy |
| Parameterized SQL | Tool calling over approved query functions | No arbitrary database access |
| TF-IDF retrieval | Embedding/hybrid retrieval and reranking | Document ACL and source citation |
| Curated explanations | Context-aware response generation | Grounded answer schema and verifier |
| Offline project files | MES, QMS, maintenance and SOP evidence | Role-based access and data minimization |
| Static question session | Case memory and follow-up | Explicit retention and case boundary |

## Capability and risk roadmap

![Figure 98.2 - Higher-value capabilities also require stronger governance and evaluation.](data:image/png;base64...)

*Figure 98.2 - Higher-value capabilities also require stronger governance and evaluation.*

![Figure 98.3 - Groundedness, citations, tool correctness, safety and privacy are release gates.](data:image/png;base64...)

*Figure 98.3 - Groundedness, citations, tool correctness, safety and privacy are release gates.*

| **Risk** | **Example** | **Required mitigation** |
| --- | --- | --- |
| Hallucination | Invented maintenance action or station fact | Answer only from retrieved/tool evidence; cite sources |
| Prompt injection | A document instructs the model to ignore policy | Treat retrieved text as data; tool and instruction isolation |
| Data leakage | Sensitive production data sent to an unauthorized model | Approved deployment, encryption, minimization and access control |
| Unauthorized action | Model writes to MES/PLC or changes threshold | Read-only tools; human approval and workflow authorization |
| Stale evidence | Old SOP or model version appears current | Version, effective date and freshness checks |
| Over-reliance | Operator accepts a fluent but weak answer | Confidence, evidence display, training and escalation |
| Evaluation blindness | Only example demos are tested | Versioned benchmark questions, adversarial tests and user studies |

|  |
| --- |
| **LLM boundary:** The LLM may summarize, retrieve, compare and propose. It must not make final quality disposition, modify PLC logic, bypass MES approvals or claim physical root cause without engineering evidence. |

## Tool-using RAG design

|  |
| --- |
| allowed\_tools = {  "get\_model\_card": read\_only,  "get\_station\_risk": parameterized\_sql,  "get\_bottlenecks": parameterized\_sql,  "get\_product\_route": authorized\_mes\_query,  "get\_recent\_alarms": authorized\_historian\_query,  "search\_sop": access\_controlled\_retrieval,  "create\_review\_case": human\_confirmation\_required }  response\_schema = {  "answer": "...",  "evidence": [{"source": "...", "version": "...", "quote\_or\_row": "..."}],  "limitations": ["..."],  "recommended\_next\_check": "...",  "action\_requires\_approval": true } |

## Evaluation set

| **Test group** | **Examples** |
| --- | --- |
| Factual project questions | Selected model, threshold, top bottleneck and known limitations |
| Numerical questions | Counts, rates and station rankings from governed tables |
| Evidence questions | SOP, model card, calibration or maintenance record retrieval |
| Adversarial prompts | Instruction override, secret request and unauthorized action |
| Ambiguous questions | Station name collision, date range and product-context clarification |
| No-answer cases | Evidence does not support a conclusion |
| Human factors | Usefulness, cognitive load, trust calibration and escalation behavior |

## Common mistakes

* Connecting a general chatbot directly to production databases.
* Treating citations as correct without verifying that they support the answer.
* Giving the model write access before read-only value is proven.
* Using conversation memory without retention and access rules.
* Evaluating only friendly questions instead of adversarial and no-answer cases.
* Hiding uncertainty because the answer sounds confident.

**Project evidence:** The current `src/copilot/offline\_agent.py` uses local TF-IDF retrieval, curated explanations and reviewed evidence. The future design should extend, not discard, these governance strengths.

**CHAPTER 99**

# Manufacturing Copilot Roadmap

*A staged program from offline project Q&A to a factory engineering workspace*

## Current baseline

| **Delivered capability** | **Current implementation** |
| --- | --- |
| Evidence database | SQLite with 14 reviewed analytical tables |
| Natural-language interaction | Curated intent routing and offline TF-IDF retrieval |
| Deterministic analytics | Parameterized SQL for model, station, route and KPI questions |
| Unified interface | Streamlit dashboard with model, families, process, SHAP and graph views |
| Governance language | Association-versus-causation and proxy warnings |
| Official risk source | Production-safe Phase 6 LightGBM remains primary |

![Figure 99.1 - A 24-month indicative roadmap; actual timing depends on data access and plant approvals.](data:image/png;base64...)

*Figure 99.1 - A 24-month indicative roadmap; actual timing depends on data access and plant approvals.*

## Roadmap phases and release gates

| **Phase** | **Capability** | **Release gate** |
| --- | --- | --- |
| 0-3 months | MES historical extract, master-data mapping and label lifecycle | Data-owner approval and row/event reconciliation |
| 3-6 months | Read-only IoT/PLC shadow collection for selected stations | OT security and telemetry-quality acceptance |
| 6-12 months | Live point-in-time features, scoring and quality feedback | Shadow stability, holdout and operator workflow evidence |
| 12-18 months | Line-level digital-twin scenarios | VVUQ and scenario credibility for one decision |
| 18-24 months | Tool-using grounded LLM Copilot pilot | Safety, privacy, groundedness and user-acceptance tests |
| 24+ months | Multi-line scaling, case management and continuous learning | Realized impact, monitoring and accountable ownership |

![Figure 99.2 - Target architecture connects governed evidence and analytics to an approved engineering case workflow.](data:image/png;base64...)

*Figure 99.2 - Target architecture connects governed evidence and analytics to an approved engineering case workflow.*

## Future Copilot use-case backlog

| **Use case** | **Evidence required** | **Human outcome** |
| --- | --- | --- |
| Shift summary | MES events, alarms, downtime and quality records | Supervisor reviews exceptions and handover |
| Station investigation | Telemetry, maintenance, calibration, SHAP and bottleneck evidence | Engineer creates an investigation case |
| Product review | Route, risk, inspection history and family context | Quality engineer prioritizes inspection |
| SOP assistant | Approved effective SOPs and work instructions | Operator follows controlled document |
| Maintenance preparation | Condition signals, work orders and asset history | Maintenance planner approves work |
| What-if simulation | Validated digital twin and scenario assumptions | Process engineer compares alternatives |
| Change-impact review | Tag, route, model and document versions | Cross-functional release approval |
| Management briefing | Governed KPI and business-impact evidence | Decision owner reviews value and risk |

|  |
| --- |
| **Roadmap priority:** The first high-value Copilot improvement should be better access to trusted MES, quality and maintenance evidence - not a larger language model. |

## Roadmap success measures

![Figure 99.3 - Scaled deployment needs strong data, twin credibility, grounded answers and operator adoption.](data:image/png;base64...)

*Figure 99.3 - Scaled deployment needs strong data, twin credibility, grounded answers and operator adoption.*

| **Dimension** | **Example KPI** |
| --- | --- |
| MES coverage | Share of scoreable products with complete route and outcome records |
| Telemetry quality | Good-quality rate, loss rate, clock skew and stale-message rate |
| Model value | Precision@K, recall@K, calibration and false-negative rate |
| Twin credibility | State accuracy, scenario error and uncertainty coverage |
| Copilot groundedness | Supported answer rate and citation correctness |
| Tool reliability | Successful authorized query/action rate |
| Human adoption | Usefulness, override quality and time saved |
| Business impact | Avoided investigation cost minus review and platform cost |

# Roadmap governance and final checklist

1. Name a Business owner, Quality owner, Controls/OT owner, Data owner, ML owner, Platform owner and Security owner.

2. Keep all system and model versions traceable in every Copilot answer and review case.

3. Use read-only integrations until evidence justifies controlled workflow actions.

4. Require human confirmation for case creation, maintenance requests, threshold changes or any operational action.

5. Retain no-answer and escalation behavior when evidence is insufficient.

6. Measure real workflow outcomes, not only model and language quality.

7. Revalidate after plant, product, route, sensor, PLC, MES or model changes.

8. Maintain a tested manual fallback and rollback path.

|  |
| --- |
| **End-state vision:** The Manufacturing Copilot should become a governed engineering workspace that unifies trusted plant evidence, predictive risk, process context, digital-twin scenarios and human decisions. It should not become an autonomous factory controller. |

# Part XIV Summary

The future of the Bosch project is not a single new algorithm. It is a controlled integration program. MES provides manufacturing context and mature outcomes. IoT and PLC integrations provide trustworthy live state without weakening control-system safety. Digital twins turn synchronized evidence into validated what-if analysis. LLMs make the evidence easier to access, but only through grounded tools, citations, guardrails and human approval. The Manufacturing Copilot roadmap combines these layers into an accountable engineering workflow.

| **Future layer** | **Primary value** | **Main dependency** |
| --- | --- | --- |
| MES | Product, route, operation and quality context | System-of-record access and point-in-time event model |
| IoT | Live sensor and machine-state evidence | Trusted telemetry, edge buffering and security |
| PLC | Deterministic equipment state and operation events | Read-only governed tag contract |
| Digital twin | Synchronized state and what-if scenarios | Validated models, semantics and uncertainty |
| LLM | Flexible evidence access and multi-step assistance | Grounding, evaluation, privacy and tool policy |
| Manufacturing Copilot | Integrated engineering review workspace | All prior layers plus human workflow ownership |

## References and further reading

[1] Project repository: kkr199/Bosch-Production-line-performance - Phase 11 Manufacturing Copilot report and offline agent.

[2] ISA - ISA-95 / IEC 62264 Enterprise-Control System Integration; 2025 Part 1 update.

[3] OPC Foundation - OPC Unified Architecture overview, core specifications and PLC companion specifications.

[4] OASIS Open - MQTT Version 5.0 OASIS Standard.

[5] ISA - ISA/IEC 62443 series for industrial automation and control-system cybersecurity.

[6] NIST - Digital Twins for Advanced Manufacturing and Manufacturing Digital Twin Standards.

[7] NIST - AI Risk Management Framework and Generative AI Profile.

[8] NIST - 2026 Roadmap on Artificial Intelligence and Machine Learning for Smart Manufacturing.

[9] Digital Twin Consortium - digital twin definitions and manufacturing resources.

**Final boundary: This future-scope architecture is a roadmap for a factory-specific POC/MVP. It requires actual asset semantics, live data, quality labels, cybersecurity approval, validation, operator acceptance and accountable ownership before production use.**