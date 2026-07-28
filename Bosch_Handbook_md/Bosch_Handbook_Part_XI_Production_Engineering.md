**PART XI**

Production Engineering

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Chapters 76–85**

Docker • FastAPI • GitHub Actions • CI/CD • Testing • Model Registry • Rollback • Audit Logging • Authentication • Golden Tests

![](data:image/png;base64...)

*Production-serving controls implemented around the Phase 6 production-safe model.*

**Bosch Production Line Performance — Technical Handbook**

A repository-grounded guide to repeatable local serving and controlled production readiness

# Contents and Production-Readiness Boundary

| **Chapter** | **Topic** | **Primary control** |
| --- | --- | --- |
| 76 | Docker | Reproducible build and runtime environment. |
| 77 | FastAPI | Authenticated, schema-validated prediction service. |
| 78 | GitHub Actions | Automated compilation and test checks. |
| 79 | CI/CD | Controlled promotion from code change to registered release. |
| 80 | Testing | Unit, integration, governance and regression assurance. |
| 81 | Model Registry | Version, artifact hash, status and lineage. |
| 82 | Rollback | Return to the last approved artifact after a guardrail breach. |
| 83 | Audit Logging | Trace each request without persisting raw feature payloads. |
| 84 | Authentication | Protect model contract and prediction routes. |
| 85 | Golden Tests | Detect unexpected scoring changes before release. |

![](data:image/png;base64...)

*Figure XI.1 — Local controls are implemented; live-factory validation remains outside the public benchmark.*

|  |
| --- |
| **Scope:** The repository now supports repeatable local demonstration, integration testing and controlled internal use. It is not evidence that the Kaggle model is validated for live factory decisions. |

# Implemented Serving Architecture

| **Control** | **Repository implementation** | **Key limitation** |
| --- | --- | --- |
| Build | `Dockerfile` + `requirements.lock` | Current image starts the Streamlit dashboard; API can be run separately. |
| Continuous integration | GitHub Actions on pull requests and pushes to `main` | No automatic production deployment. |
| API | FastAPI `/health`, `/contract`, `/predict` | Local API-key control, not enterprise identity integration. |
| Input contract | Exact 323 features; numeric or null; unknown fields rejected | Feature acquisition is outside the API. |
| Registry | JSON file with version, status, path and SHA-256 | Single-file local registry; not a distributed registry service. |
| Audit | SQLite request-level log | No centralized log aggregation or retention policy. |
| Regression | Three-record golden set | Small fixture; future factory cases must be added. |
| Rollback | Prior production entry becomes `rollback` and can be promoted | Requires operational owner and deployment runbook. |

![](data:image/png;base64...)

*Figure XI.2 — A production release requires technical, quality, security and business ownership.*

**CHAPTER 76**

# Docker

*Packaging the dashboard and exact Python environment into a repeatable image*

## Why containerization matters

A machine-learning application can behave differently when Python, libraries, system packages or working directories change. The repository addresses this by installing an exact dependency lock into a small Python image and starting the unified Streamlit dashboard with fixed network settings.

![](data:image/png;base64...)

*Figure 76.1 — The lock file is copied before the application to improve dependency-layer reuse.*

|  |
| --- |
| FROM python:3.14-slim  WORKDIR /workspace  COPY requirements.lock . RUN pip install --no-cache-dir --requirement requirements.lock  COPY . .  EXPOSE 8501 ENV STREAMLIT\_SERVER\_ADDRESS=0.0.0.0 ENV STREAMLIT\_SERVER\_PORT=8501  CMD ["streamlit", "run", "app/project\_streamlit\_dashboard.py"] |

## Build and run

|  |
| --- |
| docker build -t bosch-production-dashboard . docker run --rm -p 8501:8501 bosch-production-dashboard |

![](data:image/png;base64...)

*Figure 76.2 — The Docker context excludes raw/processed data, virtual environments and other non-runtime paths.*

| **Practice** | **Reason** |
| --- | --- |
| Exact lock file | CI and Docker install the same reviewed dependency versions. |
| Slim base image | Reduces unnecessary packages and attack surface. |
| No cache during pip install | Avoids retaining package download caches in the image layer. |
| `.dockerignore` | Avoids copying datasets, notebooks, `.git`, `.venv` and generated caches. |
| Fixed working directory | Makes relative paths predictable for models, reports and the SQLite database. |
| Port declaration | Documents Streamlit's expected container port. |

|  |
| --- |
| **Production hardening:** For a real deployment, add a non-root runtime user, image vulnerability scanning, a read-only filesystem where feasible, mounted writable volumes for databases, health checks and a separate API image or multi-service composition. |

**Repository evidence:** `Dockerfile`, `.dockerignore`, `requirements.lock`, and `docs/production\_operations\_guide.md`.

**CHAPTER 77**

# FastAPI

*Serving the registered model behind authentication and an exact input contract*

## API surface

![](data:image/png;base64...)

*Figure 77.1 — Only liveness is public; contract and prediction require the API key.*

| **Endpoint** | **Authentication** | **Response purpose** |
| --- | --- | --- |
| GET /health | No API key | Process liveness only |
| GET /contract | X-API-Key | Returns versioned 323-feature input contract |
| POST /predict | X-API-Key | Validates input, scores, audits, returns probability |

|  |
| --- |
| class PredictionRequest(BaseModel):  features: dict[str, Any]  request\_id: str | None = Field(default=None, max\_length=100)  @app.get("/health") def health() -> dict[str, str]:  return {"status": "ok"}  @app.get("/contract", dependencies=[Depends(require\_api\_key)]) def contract(service: ModelService = Depends(get\_service)):  return service.contract()  @app.post("/predict", dependencies=[Depends(require\_api\_key)]) def predict(request: PredictionRequest, service: ModelService = Depends(get\_service)):  return service.predict(request.features, request.request\_id) |

## Prediction request lifecycle

![](data:image/png;base64...)

*Figure 77.2 — Validation, registry integrity, inference and auditing occur in one governed service path.*

| **Validation rule** | **Current behavior** |
| --- | --- |
| Required fields | All 323 model features must be supplied. |
| Unknown fields | Rejected to prevent silent schema drift. |
| Value types | Numeric values or null only. |
| Infinity | Rejected; finite values required when supplied. |
| Null semantics | Preserved as model missing values because sparse measurements are intentional. |
| Model selection | Only the single registry entry marked `production` is served. |
| Artifact integrity | SHA-256 must match the registry entry before loading. |
| Response | Request ID, model version, probability, class, threshold and latency. |

|  |
| --- |
| **Architecture boundary:** The API expects fully engineered model features. Raw Bosch CSV transformation and feature construction remain upstream responsibilities; they must be versioned with the same input contract. |

**Repository evidence:** `src/serving/api.py` and `src/serving/model\_service.py`.

**CHAPTER 78**

# GitHub Actions

*Running the same locked installation, compile checks and automated tests on every change*

## Current CI workflow

![](data:image/png;base64...)

*Figure 78.1 — The workflow runs for pull requests and pushes to `main`.*

|  |
| --- |
| name: CI  on:  pull\_request:  push:  branches: [main]  jobs:  test:  runs-on: ubuntu-latest  steps:  - uses: actions/checkout@v4  - uses: actions/setup-python@v5  with:  python-version: "3.14"  - name: Install locked dependencies  run: pip install -r requirements.lock  - name: Compile application and serving modules  run: python -m py\_compile app/project\_streamlit\_dashboard.py src/serving/model\_service.py src/serving/api.py  - name: Run automated tests  run: python -m unittest discover -s tests -p "test\_\*.py" |

## What CI prevents

| **Failure mode** | **CI gate** |
| --- | --- |
| Dependency drift | Install the exact `requirements.lock` set. |
| Syntax/import breakage | Compile dashboard and serving modules. |
| Schema contract regression | Serving tests reject missing and unknown fields. |
| Audit logging regression | Integration test verifies SQLite audit row. |
| Model output drift | Golden set checks exact class and probability tolerance. |
| Governance utility regression | Split-manifest and PSI tests. |

|  |
| --- |
| **Branch protection:** The CI workflow becomes an actual release gate only when repository settings require the test job to pass before merging to `main` and restrict direct pushes. |

## Recommended next checks

* Build the Docker image in CI and run a container smoke test.
* Add FastAPI endpoint tests for 200, 401 and 422 behavior.
* Run dependency and container vulnerability scans.
* Generate and retain test reports and the image digest as workflow artifacts.
* Pin reusable actions to reviewed commit SHAs for stricter supply-chain control.

**Repository evidence:** `.github/workflows/ci.yml` and pull request #4, “Add local production readiness controls.”

**CHAPTER 79**

# CI/CD

*Separating automated verification from controlled deployment and model promotion*

## CI is implemented; CD must remain governed

The repository automatically verifies changes, but it does not automatically deploy a model to a factory. That distinction is appropriate. Continuous delivery should create an immutable, reviewed release candidate; production promotion should still require quality, platform and business approval.

![](data:image/png;base64...)

*Figure 79.1 — A safe delivery flow includes staging, shadow validation and explicit approval.*

| **Stage** | **Automated evidence** | **Human approval** |
| --- | --- | --- |
| Pull request | Compile, unit, integration and golden checks | Code review |
| Build | Image digest, dependency lock, source commit | Platform review if base image changes |
| Staging registry | Artifact hash, metrics, contract version | ML owner |
| Shadow mode | Latency, error rate, score distribution, alert volume | Quality + operations |
| Controlled pilot | Business and quality guardrails | Business owner |
| Production | Monitoring and rollback readiness | Named release authority |

## Release artifact manifest

| **Required field** | **Example** |
| --- | --- |
| Git commit | Immutable source revision |
| Container digest | SHA-256 image identity |
| Model version | `phase6-best-v1` |
| Model artifact hash | Registry `artifact\_sha256` |
| Feature contract version | Feature count and ordered feature list |
| Dependency lock hash | Reviewed `requirements.lock` digest |
| Test evidence | CI run and golden-set result |
| Approval record | ML, Quality, Platform and Business decisions |

|  |
| --- |
| **Deployment claim:** The current public-data project has a local production-serving layer, not a live automotive CD pipeline. Factory deployment still requires infrastructure, live data, security approval, SLOs and operational ownership. |

**Repository evidence:** `docs/production\_operations\_guide.md`, `docs/monitoring\_and\_retraining\_plan.md`, and `.github/workflows/ci.yml`.

**CHAPTER 80**

# Testing

*Verifying governance utilities, input contracts, service behavior and approved model outputs*

## Current automated checks

![](data:image/png;base64...)

*Figure 80.1 — The repository has unit, integration and golden regression checks; an API/container smoke layer is the next step.*

![](data:image/png;base64...)

*Figure 80.2 — Five automated checks were validated when the production-readiness controls were merged.*

| **Check** | **Type** | **What it protects** |
| --- | --- | --- |
| Split manifest | Unit | Counts, rates, seed and source SHA-256 |
| PSI identical distribution | Unit | Governance drift utility |
| Serving success + audit | Integration | Schema, model result and SQLite write |
| Missing/unknown feature rejection | Unit | Strict contract enforcement |
| Golden predictions | Regression | 3 approved records; probability to 9 decimals |

## Serving test design

|  |
| --- |
| class FakeProbabilityModel:  def predict\_proba(self, frame):  probability = float(frame["f1"].fillna(0).iloc[0]) / 10  return np.array([[1 - probability, probability]])  def test\_prediction\_enforces\_schema\_and\_writes\_audit\_record():  result = service.predict({"f1": 8.0, "f2": None}, request\_id="request-1")  assert result.model\_version == "fake-v1"  assert result.predicted\_failure == 1  # Then query SQLite and confirm status="success", feature\_count=2. |

| **Testing principle** | **Application** |
| --- | --- |
| Use a fake model for service logic | Makes schema, registry and audit tests fast and deterministic. |
| Use the real registered model for regression | Golden tests detect artifact or dependency output changes. |
| Use temporary directories and databases | Prevents tests from changing project state. |
| Test invalid inputs deliberately | Missing and unknown fields must fail closed. |
| Test governance utilities separately | Split manifests and PSI do not require the full model stack. |

|  |
| --- |
| **Coverage gap:** The current suite does not yet exercise the FastAPI HTTP layer, Docker startup, concurrent requests, audit retention, large payload limits or performance SLOs. |

**Repository evidence:** `tests/test\_ml\_governance.py`, `tests/test\_model\_serving.py`, and `tests/test\_golden\_predictions.py`.

**CHAPTER 81**

# Model Registry

*Selecting exactly one production model and verifying the artifact before use*

## Current registry entry

|  |
| --- |
| {  "registry\_version": 1,  "models": [  {  "version": "phase6-best-v1",  "status": "production",  "artifact\_path": "models/phase6\_best\_model.joblib",  "artifact\_sha256": "e46733d6...048688b0",  "training\_run": "phase6\_predictive\_failure\_modeling",  "validation\_status": "experimental: labelled holdout required before factory use"  }  ] } |

![](data:image/png;base64...)

*Figure 81.1 — A candidate is registered with evidence before it can replace the production entry.*

## Registry enforcement

![](data:image/png;base64...)

*Figure 81.2 — ModelService refuses to load a model when registry or artifact integrity checks fail.*

| **Registry field or rule** | **Purpose** |
| --- | --- |
| `version` | Stable identity returned with every prediction. |
| `status` | Controls production, staging and rollback eligibility. |
| `artifact\_path` | Locates the joblib prediction bundle. |
| `artifact\_sha256` | Detects artifact replacement or corruption. |
| `training\_run` | Links the artifact to the model-building phase. |
| `validation\_status` | Preserves the benchmark/factory-readiness limitation. |
| Exactly one production entry | Prevents ambiguous model selection. |
| Atomic temporary-file replace | Reduces partial registry writes. |

|  |
| --- |
| **Scale boundary:** A JSON registry is appropriate for local serving and demonstrations. Multiple replicas, concurrent promotions, remote artifacts, approvals and lineage should move to a transactional registry such as MLflow or an equivalent governed platform. |

**Repository evidence:** `models/registry/registry.json` and `ModelRegistry` in `src/serving/model\_service.py`.

**CHAPTER 82**

# Rollback

*Restoring the prior registered model when operational or quality guardrails fail*

## How the current rollback works

|  |
| --- |
| registry = ModelRegistry() registry.promote("candidate-version")  # If release checks or runtime guardrails fail: registry.rollback() |

![](data:image/png;base64...)

*Figure 82.1 — Rollback changes the registry state, reloads the prior artifact and immediately re-runs verification.*

| **Promotion behavior** | **Result** |
| --- | --- |
| Promote a validated staging model | Current production entry becomes `rollback`; candidate becomes `production`. |
| Promote the current production model | State remains logically production. |
| Promote an unknown version | Raises `KeyError`. |
| Promote an unapproved status | Raises `ValueError`. |
| Rollback with no rollback candidate | Fails explicitly rather than selecting an arbitrary model. |
| Registry write | Writes JSON to a temporary file and atomically replaces the registry. |

## When to roll back

![](data:image/png;base64...)

*Figure 82.2 — Example urgency prioritization; actual thresholds require deployment-specific approval.*

| **Trigger class** | **Example response** |
| --- | --- |
| Integrity | Stop serving immediately on artifact hash mismatch. |
| Schema | Pause scoring when required fields disappear or rejection rate spikes. |
| Reliability | Rollback when error rate or latency exceeds the approved SLO. |
| Prediction behavior | Pause rollout when alert rate leaves approved control limits. |
| Quality | Rollback after mature labels show a release-floor breach. |
| Security | Rotate credentials and disable the endpoint after suspected key exposure. |

|  |
| --- |
| **Runbook rule:** Rollback is not complete until the golden set and full test suite pass, the audit log is reviewed and the incident owner documents the cause and corrective action. |

**Repository evidence:** `ModelRegistry.promote`, `ModelRegistry.rollback`, `docs/production\_operations\_guide.md`, and `docs/monitoring\_and\_retraining\_plan.md`.

**CHAPTER 83**

# Audit Logging

*Creating a traceable prediction record without storing the raw feature payload*

## Audit database design

![](data:image/png;base64...)

*Figure 83.1 — Nine request-level fields are stored in `prediction\_audit.db`.*

|  |
| --- |
| CREATE TABLE IF NOT EXISTS prediction\_audit (  request\_id TEXT PRIMARY KEY,  timestamp\_utc TEXT NOT NULL,  model\_version TEXT NOT NULL,  input\_feature\_count INTEGER NOT NULL,  failure\_probability REAL,  predicted\_failure INTEGER,  latency\_ms REAL NOT NULL,  status TEXT NOT NULL,  error\_code TEXT ) |

## Status semantics

![](data:image/png;base64...)

*Figure 83.2 — Success, rejected input and runtime error requests all receive a terminal record.*

| **Status** | **Meaning** | **Recorded fields** |
| --- | --- | --- |
| `success` | Contract passed and model returned a result. | Probability, class, latency and version. |
| `rejected` | Input contract failed. | Latency, feature count and exception class. |
| `error` | Unexpected model, registry, artifact or runtime failure. | Latency, version and exception class. |

| **Privacy choice** | **Reason** |
| --- | --- |
| No raw feature values | Reduces exposure of potentially confidential manufacturing measurements. |
| UTC timestamp | Supports cross-system ordering and incident review. |
| Request ID primary key | Enables idempotent update of a repeated request identifier. |
| Model version | Makes each score attributable to a registered artifact. |
| Latency and status | Supports operational error and performance analysis. |

|  |
| --- |
| **Future production requirement:** Define retention, access controls, encryption, centralized aggregation, tamper resistance, alert rules and a correlation ID shared with upstream and downstream systems. |

**Repository evidence:** `PredictionAuditLog` in `src/serving/model\_service.py` and `data/database/prediction\_audit.db`.

**CHAPTER 84**

# Authentication

*Protecting model metadata and prediction access with a secret kept outside source control*

## Current API-key control

|  |
| --- |
| def require\_api\_key(x\_api\_key: str = Header(...)) -> None:  expected\_key = os.environ.get("BOSCH\_API\_KEY")  if not expected\_key or x\_api\_key != expected\_key:  raise HTTPException(status\_code=401, detail="Invalid API key.") |

![](data:image/png;base64...)

*Figure 84.1 — The secret is read from the environment and compared for protected routes.*

| **Route** | **Current access** |
| --- | --- |
| `/health` | Unauthenticated liveness response. |
| `/contract` | Requires `X-API-Key`. |
| `/predict` | Requires `X-API-Key`. |

## Secure operation

![](data:image/png;base64...)

*Figure 84.2 — The current key is a local control; real deployment requires identity, transport and authorization layers.*

1. Generate a long random secret and keep it in a secret manager or protected environment variable.

2. Never write the key into source code, registry files, Docker images, notebooks or audit logs.

3. Use HTTPS/TLS so the header is not exposed in transit.

4. Rotate keys and immediately revoke them after suspected exposure.

5. Add rate limiting, request-size limits and network controls at an API gateway.

6. Use OAuth2/JWT, mTLS or plant identity infrastructure when user/service identity and roles are required.

7. Separate permission to read the contract from permission to create predictions when risk requires it.

|  |
| --- |
| **Authorization gap:** The current implementation authenticates one shared secret but does not identify individual users, assign roles, record actor identity or provide fine-grained authorization. |

**Repository evidence:** `require\_api\_key` and protected endpoint dependencies in `src/serving/api.py`.

**CHAPTER 85**

# Golden Tests

*Freezing approved model outputs so dependency or artifact changes cannot pass silently*

## What the golden fixture contains

| **Fixture element** | **Current value** |
| --- | --- |
| Model version | `phase6-best-v1` |
| Approved samples | 3 product records |
| Feature payload | Complete 323-feature request for each record |
| Null handling | Sparse values retained as null |
| Expected outputs | Failure probability and binary class |
| Probability comparison | Nine decimal places |

![](data:image/png;base64...)

*Figure 85.1 — The test uses the real registered model through the same ModelService path as production requests.*

## Approved predictions

![](data:image/png;base64...)

*Figure 85.2 — Approved probabilities stored in `phase6\_golden\_set.json`.*

|  |
| --- |
| for sample in golden\_set["samples"]:  result = service.predict(  sample["features"],  request\_id=f"golden-{sample['id']}",  )  self.assertEqual(result.model\_version, golden\_set["model\_version"])  self.assertEqual(result.predicted\_failure, sample["expected\_class"])  self.assertAlmostEqual(  result.failure\_probability,  sample["expected\_probability"],  places=9,  ) |

| **Sample Id** | **Expected probability** | **Expected class** |
| --- | --- | --- |
| 590939 | 0.182670476505 | 0 |
| 11606 | 0.524306831284 | 0 |
| 1824571 | 0.325604714883 | 0 |

|  |
| --- |
| **Change management:** Do not update golden outputs merely to make CI green. A changed probability requires an explained dependency, feature, preprocessing, model or numerical change plus review and re-approval. |

## How to expand the golden set

* Add both predicted-positive and predicted-negative records.
* Include sparse, dense, boundary-threshold and high-risk product-family cases.
* Add invalid contract fixtures for missing, unknown, text and infinite values.
* Store a reason for each approved sample.
* Review tolerance when changing model libraries or hardware; keep class behavior and business thresholds explicit.

**Repository evidence:** `tests/fixtures/phase6\_golden\_set.json` and `tests/test\_golden\_predictions.py`.

# Part XI Summary and Release Checklist

The repository now has a meaningful local production-serving layer: exact dependencies, a Docker dashboard image, protected FastAPI scoring, strict schema validation, a versioned and hashed model registry, SQLite audit logging, rollback support, automated tests and a golden prediction fixture. These controls improve repeatability and traceability without changing the model's benchmark limitations.

## Release checklist

1. Confirm the source commit and exact dependency lock.

2. Run compile checks and the full automated test suite.

3. Build an immutable container and record its digest.

4. Verify the registry has exactly one production model and a matching artifact SHA-256.

5. Call `/contract` and confirm the expected model version and feature count.

6. Run golden predictions through the registered ModelService.

7. Verify audit records for success, rejected input and simulated error paths.

8. Store the API secret outside code and require TLS in any networked environment.

9. Retain the prior production model as the tested rollback candidate.

10. Complete live-factory, security, SLO, quality and business approvals before production use.

## Primary repository artifacts

| **Artifact** | **Purpose** |
| --- | --- |
| Dockerfile | Reproducible Streamlit application image. |
| .dockerignore | Restricts the Docker build context. |
| requirements.lock | Exact dependency set used by CI and Docker. |
| .github/workflows/ci.yml | Compile and automated-test gate. |
| src/serving/api.py | Authenticated FastAPI endpoint. |
| src/serving/model\_service.py | Contract validation, registry, inference, audit and rollback. |
| models/registry/registry.json | Production model identity and artifact hash. |
| tests/test\_model\_serving.py | Service and audit behavior. |
| tests/test\_golden\_predictions.py | Approved model-output regression. |
| tests/fixtures/phase6\_golden\_set.json | Versioned golden request/response records. |
| docs/production\_operations\_guide.md | Run, promote and rollback instructions. |

|  |
| --- |
| **Final production boundary:** Local production engineering controls do not replace a locked labelled holdout, factory-specific data validation, shadow deployment, live monitoring, operator acceptance, realized-impact measurement or accountable release ownership. |