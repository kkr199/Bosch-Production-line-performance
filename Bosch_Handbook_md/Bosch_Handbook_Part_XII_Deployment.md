**PART XII**

Deployment

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Chapters 86–89**

Local Deployment • Cloud Deployment • Factory Deployment • Operational Considerations

![](data:image/png;base64...)

*A controlled deployment path for the Bosch manufacturing analytics project.*

**Bosch Production Line Performance — Technical Handbook**

Deployment patterns, operational gates and factory-adoption boundaries

# Contents and Deployment Context

| **Chapter** | **Topic** | **Primary question** |
| --- | --- | --- |
| 86 | Local Deployment | How can the project be run and verified on one controlled machine? |
| 87 | Cloud Deployment | How should images, secrets, storage, gateways and observability be arranged remotely? |
| 88 | Factory Deployment | How can predictions enter a plant workflow without becoming unsafe automatic control? |
| 89 | Operational Considerations | How will the deployed service be monitored, secured, recovered and governed? |

![](data:image/png;base64...)

*Figure XII.1 — Maturity rises with evidence, control and accountable ownership.*

|  |
| --- |
| **Current state:** The repository implements repeatable local serving and a deployable dashboard container. Cloud and factory sections are reference architectures and readiness plans, not claims of a live automotive production deployment. |

## Deployment modes

| **Mode** | **Environment** | **Use** | **Decision impact** |
| --- | --- | --- | --- |
| Local Python | Developer workstation | Development and debugging | None |
| Local Docker | Developer or reviewer workstation | Repeatable demonstration and integration checks | None |
| Cloud demonstration | Managed container or VM | Remote stakeholder review | No factory decision |
| Factory shadow | Approved plant or edge zone | Compatibility and operational observation | No disposition |
| Assisted pilot | Restricted line, shift or family | Human-reviewed priority queue | Advisory only |
| Factory production | Governed plant service | Manual quality-review prioritization | Human final decision |

|  |
| --- |
| **Safety principle:** The model ranks scoreable products for manual quality review. It must not automatically release, reject or control a product, line or safety device. |

**CHAPTER 86**

# Local Deployment

*Running the dashboard and protected prediction service on one workstation*

## Learning objectives

* Prepare a clean environment from the exact dependency lock.
* Run tests before exposing the application.
* Start Streamlit and FastAPI as separate services.
* Verify authentication, the 323-feature contract, the registered model and audit logging.
* Understand the limits of a single-machine demonstration.

![](data:image/png;base64...)

*Figure 86.1 — Source, services, governance artifacts and databases remain on one machine.*

| **Component** | **Role** | **Default location** |
| --- | --- | --- |
| Streamlit dashboard | Project analytics and review interface | Port 8501 |
| FastAPI service | Protected contract and prediction endpoint | Port 8000 |
| Model registry | Selects the single production model | models/registry/registry.json |
| Model artifact | Phase 6 production-safe bundle | models/phase6\_best\_model.joblib |
| Prediction audit | Request-level trace records | data/database/prediction\_audit.db |
| Dashboard database | Reviewed project evidence tables | data/database/manufacturing\_copilot.db |

## Environment setup and verification

|  |
| --- |
| python -m venv .venv .\.venv\Scripts\python.exe -m pip install --upgrade pip .\.venv\Scripts\python.exe -m pip install -r requirements.lock  .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test\_\*.py" |

|  |
| --- |
| **Dependency rule:** Use requirements.lock for serving demonstrations because Docker and CI install the same reviewed package set. |

## Start the services

|  |
| --- |
| # Terminal 1 — protected prediction API $env:BOSCH\_API\_KEY = "replace-with-a-long-random-secret" .\.venv\Scripts\python.exe -m uvicorn src.serving.api:app --host 127.0.0.1 --port 8000  # Terminal 2 — unified dashboard .\.venv\Scripts\streamlit.exe run app\project\_streamlit\_dashboard.py |

![](data:image/png;base64...)

*Figure 86.2 — Tests and the secret are prepared before the services are verified.*

## Local Docker deployment

|  |
| --- |
| docker build -t bosch-production-dashboard . docker run --rm -p 8501:8501 bosch-production-dashboard |

The current Dockerfile starts the Streamlit dashboard. The API can run in a second container by overriding the command, or through a dedicated API image.

|  |
| --- |
| docker run --rm -e BOSCH\_API\_KEY="replace-with-a-secret" -p 8000:8000 bosch-production-dashboard python -m uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 |

## Verification checklist

1. Confirm the dashboard loads the expected project pages.

2. Call GET /health and confirm `status: ok`.

3. Call GET /contract with the key and confirm model `phase6-best-v1` and 323 required features.

4. Score a golden record and confirm its approved probability and class.

5. Confirm the SQLite prediction-audit row contains model version, latency and success status.

6. Send a request with a missing field and confirm the 422 response and `rejected` audit status.

|  |
| --- |
| **Local limitation:** A workstation deployment has no high availability, shared identity, durable centralized storage, plant integration or production SLO. |

**Repository evidence:** `README.md`, `Dockerfile`, `src/serving/api.py`, `src/serving/model\_service.py`, and `docs/production\_operations\_guide.md`.

## Common mistakes

| **Mistake** | **Risk** | **Better practice** |
| --- | --- | --- |
| Skipping tests | Broken contract or registry appears during a demonstration | Run the full suite before startup. |
| Hard-coding the key | Secret enters source, image or Git history | Use environment variables or a secret manager. |
| Binding to 0.0.0.0 locally | Service is reachable beyond the workstation | Use 127.0.0.1 for local-only access. |
| Dropping null features | Sparse input semantics are changed | Supply every key and preserve nulls. |
| Calling the dashboard live monitoring | Static Kaggle outputs are overstated | Label the environment as demonstration or integration testing. |

**CHAPTER 87**

# Cloud Deployment

*Hosting immutable containers through a managed and auditable platform*

## Learning objectives

* Separate source, CI, image registry, runtime, secrets, storage and gateway responsibilities.
* Deploy immutable image digests instead of mutable source folders.
* Protect the API with TLS, identity and request controls.
* Move audit and analytical state away from ephemeral container filesystems.
* Retain release evidence with the model version and feature contract.

![](data:image/png;base64...)

*Figure 87.1 — Provider-neutral cloud architecture for the current containerized project.*

| **Layer** | **Required decision** |
| --- | --- |
| Source control | Protected branch, reviewed pull requests and release tags. |
| CI | Locked install, tests, image build and security checks. |
| Image registry | Immutable digest, retention and least-privilege access. |
| Runtime | Managed container service, VM or Kubernetes based on team maturity. |
| Secrets | Key vault or secret manager; no secret in the image. |
| Storage | Durable audit records and controlled model-artifact access. |
| Ingress | HTTPS/TLS, identity, rate limiting and request-size controls. |
| Observability | Logs, metrics, traces, alerts and deployed version metadata. |

## Separate dashboard and API services

| **Property** | **Dashboard** | **Prediction API** |
| --- | --- | --- |
| Users | Reviewers and engineering stakeholders | Approved systems and service accounts |
| Exposure | Authenticated web access | Private endpoint or API gateway |
| Scaling driver | Interactive sessions | Scoring request or batch volume |
| State | Reads analytical DB and reports | Reads registry; writes audit events |
| Authentication | User identity / application access | Service identity, token, mTLS or scoped key |
| Availability | Review convenience | Approved review-workflow SLO |

![](data:image/png;base64...)

*Figure 87.2 — Cloud deployment is shared work between ML/application and platform teams.*

## Reference container manifest

|  |
| --- |
| apiVersion: apps/v1 kind: Deployment metadata:  name: bosch-prediction-api spec:  replicas: 2  selector:  matchLabels:  app: bosch-prediction-api  template:  metadata:  labels:  app: bosch-prediction-api  spec:  containers:  - name: api  image: registry.example/bosch-api@sha256:<approved-digest>  command: ["python", "-m", "uvicorn", "src.serving.api:app",  "--host", "0.0.0.0", "--port", "8000"]  env:  - name: BOSCH\_API\_KEY  valueFrom:  secretKeyRef:  name: bosch-api-secret  key: api-key  readinessProbe:  httpGet:  path: /health  port: 8000 |

|  |
| --- |
| **Reference only:** The manifest demonstrates concepts. Actual configuration must follow the selected platform, network, storage and organizational standards. |

## Cloud persistence and release controls

| **Artifact** | **Unsafe default** | **Preferred pattern** |
| --- | --- | --- |
| Registry JSON | Mutable file inside an ephemeral container | Read-only release artifact or transactional registry |
| Model joblib | Manual copy to each instance | Versioned artifact store or immutable image |
| Audit SQLite | Local container filesystem | Central log/SQL service or durable mounted storage |
| Dashboard SQLite | Instance-local stale copy | Versioned snapshot or managed database refresh |
| API secret | Dockerfile or repository variable | Secret manager with rotation and access audit |

1. Promote only an image digest that passed the same CI run as the model and contract.

2. Use private networking when only internal systems consume predictions.

3. Set CPU, memory, timeout, concurrency and request-size limits.

4. Retain deployment configuration, image digest, model version and approvals together.

5. Keep development, staging and production environments separate.

6. Stop unused demonstration environments to control cost.

|  |
| --- |
| **Cloud readiness gate:** A container that starts successfully is not yet production-ready. Authentication, persistence, observability, rollback, version traceability and controlled access must also be proven. |

**Repository evidence:** `Dockerfile`, `.github/workflows/ci.yml`, `models/registry/registry.json`, and `src/serving/\*`.

**CHAPTER 88**

# Factory Deployment

*Integrating advisory scoring into a human-controlled quality workflow*

## Learning objectives

* Declare the scoring point and prove every feature is available at that time.
* Keep the model outside safety and automatic product-disposition loops.
* Shadow-score before showing recommendations to operators.
* Connect predictions to a capacity-constrained review queue.
* Return mature outcomes and action records for monitoring and retraining.

![](data:image/png;base64...)

*Figure 88.1 — Predictions enter a manual review workflow and return outcome evidence.*

|  |
| --- |
| **Intended use:** The model prioritizes completed or scoreable records for manual review. It is not a safety interlock, causal diagnosis engine or automatic release/reject system. |

## Point-in-time feature contract

| **Question** | **Factory requirement** |
| --- | --- |
| When is the score created? | After the declared required station and timing data is available. |
| Which events are permitted? | Only information available before or at the scoring point. |
| How are nulls interpreted? | According to approved sensor and route semantics; null is not automatically an error. |
| What if a field is absent? | Reject or stop scoring; do not silently invent a value. |
| How is the record joined? | Approved product identifier with lifecycle and retention controls. |
| When is the label mature? | After the quality process completes; label latency must be documented. |

![](data:image/png;base64...)

*Figure 88.2 — Operational technology, integration and ML service zones use controlled interfaces.*

| **Interface** | **Minimum control** |
| --- | --- |
| MES / quality → feature service | Approved fields, schema version, timestamps and source identity. |
| Feature service → model API | Authenticated service, TLS and request ID. |
| Model API → review queue | Version, score, threshold and expiry time. |
| Review queue → engineer | Ranked workload, context and manual decision controls. |
| Quality system → monitoring | Mature outcome, action taken and label timestamp. |

## Factory rollout plan

![](data:image/png;base64...)

*Figure 88.3 — Decision impact increases only after compatibility, quality and operator evidence improves.*

| **Stage** | **Visibility** | **Required evidence** |
| --- | --- | --- |
| Offline replay | Data/ML team | Feature reproducibility, locked holdout and leakage review. |
| Shadow scoring | Operations dashboard only | Schema success, latency, drift and alert-volume stability. |
| Assisted pilot | Selected quality engineers | Usability, precision@K, recall@K and capacity evidence. |
| Staged scale | Approved lines, shifts or families | Guardrail stability and incident-free operating period. |
| Production | Approved review population | Signed ownership, SLO, cost matrix, safety review and rollback. |

## Factory acceptance criteria

1. Evaluate the frozen pipeline once on a physically isolated labelled holdout.

2. Compare rules-only, logistic-regression and tree-model baselines on the same holdout.

3. Verify feature availability and sensor semantics at the declared scoring point.

4. Obtain Quality approval for false-negative tolerance and daily review capacity.

5. Obtain Business approval for false-positive, missed-failure and review costs.

6. Complete security, privacy, retention and data-owner approval.

7. Train operators to understand, override and report problems with recommendations.

8. Rehearse rollback to the prior verified model before the pilot.

**Repository evidence:** `docs/problem\_definition\_charter.md`, `docs/data\_strategy\_and\_test\_set\_policy.md`, `docs/model\_card.md`, and `docs/monitoring\_and\_retraining\_plan.md`.

# Factory Failure Modes and Human Factors

| **Failure mode** | **Risk** | **Response** |
| --- | --- | --- |
| Upstream schema change | Wrong or rejected predictions | Stop scoring and restore the contract. |
| Sensor/route semantics change | Null patterns mean something new | Revalidate features before resuming. |
| Operator over-trust | Advisory score becomes automatic disposition | Training, visible limitations and manual authority. |
| Alert overload | Queue exceeds inspection capacity | Control threshold and K by approved capacity. |
| Label delay | Performance is unknown for too long | Define provisional and mature-label windows. |
| Service outage | No prediction is available | Use an approved safe manual fallback. |
| Model drift | Ranking quality declines | Pause promotion, retain incumbent or use rules-only baseline. |

|  |
| --- |
| **Human authority:** Quality Engineering retains the final decision. Every recommendation must carry its model version, timestamp, scope and limitations. |

## Interview questions

* Why is shadow deployment safer than immediate operator exposure?
* What is label latency, and how does it affect monitoring?
* Why must the scoring point be declared before feature engineering?
* What is the difference between an advisory model and a safety interlock?
* How would you design a safe fallback when the model service is unavailable?

**CHAPTER 89**

# Operational Considerations

*Operating the service with reliability, security, traceability and model-quality controls*

## Learning objectives

* Define monitoring signals with named owners and first responses.
* Separate infrastructure, data, model, security and workflow incidents.
* Retain enough release evidence to reproduce every score.
* Plan backup, recovery and rollback before go-live.
* Use mature labels and controlled thresholds to maintain value.

![](data:image/png;base64...)

*Figure 89.1 — Factory decision support depends on more than model accuracy.*

| **Area** | **Operational question** |
| --- | --- |
| Reliability | Can the review workflow continue safely when scoring is unavailable? |
| Performance | Does scoring meet the approved batch or request SLO? |
| Security | Who can call the API and access artifacts or logs? |
| Data quality | Are required fields present, typed correctly and fresh? |
| Model quality | Are ranking, calibration and alert workload within approved limits? |
| Traceability | Can a score be tied to code, contract, data window, model and threshold? |
| Recoverability | Can the prior verified service be restored quickly? |
| Human workflow | Can engineers understand, review and override the recommendation? |

## Monitoring contract

| **Signal** | **Threshold or gate** | **First response** |
| --- | --- | --- |
| Input schema / feature availability | Any missing field or unexpected type | Stop scoring and restore the contract |
| Priority-feature drift | PSI > 0.20 | Check source, units and training reference |
| Prediction drift | PSI > 0.20 | Inspect feature drift, queue volume and threshold |
| Operational health | Error rate > 1% or p95 exceeds approved SLO | Rollback and investigate logs |
| Label-lagged performance | PR-AUC or precision@K below release floor | Pause promotion and perform error analysis |

![](data:image/png;base64...)

*Figure 89.2 — Monitoring ownership spans Data, Platform, ML and Quality teams.*

|  |
| --- |
| **Threshold status:** PSI 0.20 and scoring error rate 1% are repository proposals. Latency, review capacity, precision@K, false-negative and cost thresholds remain deployment-specific approvals. |

## Incident response

![](data:image/png;base64...)

*Figure 89.3 — Containment can pause scoring, restore the prior model or activate the manual fallback.*

| **Incident class** | **Examples** | **Owner** |
| --- | --- | --- |
| Application | Crash, dependency failure, malformed response | Platform |
| Data contract | Missing field, unexpected type, stale feed | Data Engineering |
| Artifact / registry | Hash mismatch, missing model, ambiguous production status | ML owner |
| Security | Exposed key, unauthorized request, suspicious traffic | Security / Platform |
| Model behavior | Prediction drift, alert-volume or calibration shift | ML owner |
| Quality outcome | Precision@K or false-negative guardrail breach | ML + Quality |
| Workflow | Review backlog, operator confusion or misuse | Quality + Business |

## Recoverability and traceability

![](data:image/png;base64...)

*Figure 89.4 — Source, image, model, contract, logs and runbook must be restored as one release system.*

| **Recoverable item** | **Minimum evidence** |
| --- | --- |
| Source | Git commit, release tag and approved pull request. |
| Dependencies | Lock file and container image digest. |
| Model | Artifact, SHA-256, registry version and status history. |
| Contract | Ordered features, types, null policy and scoring point. |
| Configuration | Secret references, thresholds, routes and environment. |
| Audit | Prediction metadata, status, latency and model version. |
| Monitoring | Reference distributions, alerts and mature-label performance. |
| Runbook | Owners, escalation, rollback, fallback and verification. |

## Retraining and promotion

1. Retrain monthly when mature labels are available and also after confirmed drift.

2. Keep the incumbent live when a challenger fails data, metric or operational gates.

3. Compare the challenger with the incumbent and approved baselines on a recent locked holdout.

4. Shadow-score before staged rollout.

5. Retain the previous production artifact as the tested rollback model.

**Repository evidence:** `docs/monitoring\_and\_retraining\_plan.md`, `src/serving/model\_service.py`, and `data/database/prediction\_audit.db`.

# Operational Readiness Checklist

| **Gate** | **Required evidence** | **Owner** |
| --- | --- | --- |
| Business | Approved review capacity, cost matrix and scope | Business owner |
| Quality | False-negative tolerance, workflow and fallback | Quality Engineering |
| Data | Source inventory, freshness, schema and label latency | Data owner / Engineering |
| Model | Locked holdout, baselines, calibration and error analysis | ML owner |
| Platform | SLO, scaling, monitoring, backups and rollback | Platform owner |
| Security | Identity, secrets, network, retention and access | Security owner |
| Human factors | Training, understandable output, override and feedback | Quality + Operations |
| Release | Source, image, model, contract, tests and approvals linked | Release authority |

|  |
| --- |
| **Go-live rule:** Deployment is ready only when every required owner accepts the documented residual risk. A technically working API or dashboard is not a production approval. |

## Practice exercises

1. Design a safe fallback when the prediction API is unavailable for two hours.

2. Define p95 latency, review-capacity and alert-rate guardrails for a factory pilot.

3. Write a rollback drill that restores the prior model and verifies golden predictions.

4. Create a retention policy for audit records that balances traceability and privacy.

5. Explain how to distinguish feature drift from a legitimate new product-family launch.

# Part XII Summary

Deployment is the controlled movement of a reviewed model from one machine into an operational decision workflow. The Bosch repository already provides local reproducibility, containerization, protected serving, registry integrity, audit logging, rollback and golden regression checks. Cloud and factory adoption require additional infrastructure, live data validation, durable identity and observability, named owners and human-workflow approval.

| **Level** | **Supported now** | **Still required** |
| --- | --- | --- |
| Local | Dashboard container, FastAPI, tests, registry, audit and golden checks | High availability, centralized identity and production SLO |
| Cloud | Container-ready reference design | Platform implementation, durable storage, gateway, secrets and monitoring |
| Factory pilot | Advisory workflow and monitoring plan | Live feature validation, shadow evidence and operator acceptance |
| Factory production | Governance and rollback framework | Locked holdout, safety/business sign-off and accountable owners |

## Primary repository references

| **Artifact** | **Deployment relevance** |
| --- | --- |
| README.md | Local setup and production-serving commands. |
| Dockerfile | Dashboard container build and runtime. |
| .github/workflows/ci.yml | Automated verification gate. |
| src/serving/api.py | Authenticated health, contract and prediction routes. |
| src/serving/model\_service.py | Registry, validation, audit and rollback. |
| models/registry/registry.json | Current production model and artifact hash. |
| docs/production\_operations\_guide.md | Local run, promotion and rollback instructions. |
| docs/problem\_definition\_charter.md | Decision scope, release gates and approvals. |
| docs/data\_strategy\_and\_test\_set\_policy.md | Factory data, holdout and leakage rules. |
| docs/model\_card.md | Intended use, limitations and ownership. |
| docs/monitoring\_and\_retraining\_plan.md | Monitoring, staged rollout, retraining and rollback. |

|  |
| --- |
| **Final boundary:** The public Bosch Kaggle dataset can demonstrate a disciplined deployment architecture, but it cannot prove live-factory performance, operational value, privacy compliance, operator acceptance or safety suitability. |