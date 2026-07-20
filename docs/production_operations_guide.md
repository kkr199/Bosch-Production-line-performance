# Local Production Operations Guide

This project now has a repeatable local serving layer. It is suitable for
demonstration, integration testing, and controlled internal use. It is not a
claim that the public Kaggle data has been validated in a live factory.

## What is implemented

- Dockerfile and requirements.lock for reproducible application builds.
- GitHub Actions CI for compilation and automated tests.
- A strict, versioned model-input contract exposed at GET /contract.
- File-backed model registry at models/registry/registry.json.
- SQLite prediction audit log at data/database/prediction_audit.db.
- Golden-set regression test at tests/fixtures/phase6_golden_set.json.
- API-key protected FastAPI prediction endpoint.

## Run the dashboard in Docker

~~~powershell
docker build -t bosch-production-dashboard .
docker run --rm -p 8501:8501 bosch-production-dashboard
~~~

Open http://localhost:8501.

## Run the protected prediction API

Choose and retain a secret outside source control:

~~~powershell
$env:BOSCH_API_KEY = "replace-with-a-long-random-secret"
.\.venv\Scripts\python.exe -m uvicorn src.serving.api:app --host 127.0.0.1 --port 8000
~~~

Call GET /contract first. It returns the exact 323 required model feature
names. Each feature key must be present in /predict; values must be numeric
or null. Nulls are accepted because sparse raw Bosch measurements are part of
the model's intended input semantics.

~~~powershell
Invoke-RestMethod http://127.0.0.1:8000/contract -Headers @{"X-API-Key"=$env:BOSCH_API_KEY}
~~~

The API writes timestamp, model version, feature count, probability, predicted
class, latency, and status to the audit SQLite database. It intentionally does
not persist raw feature values.

## Model promotion and rollback

The production entry in models/registry/registry.json must point to an
artifact whose SHA-256 matches artifact_sha256. To introduce a candidate,
add a validated staging entry to the registry with an artifact path and hash.
Then use the registry API:

~~~powershell
@'
from src.serving.model_service import ModelRegistry

registry = ModelRegistry()
registry.promote("candidate-version")
# If the candidate fails checks:
registry.rollback()
'@ | .\.venv\Scripts\python.exe -
~~~

Run the golden-set and full test suite before promotion and immediately after
rollback:

~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
~~~

## Still requiring real factory access

The project cannot validate current-factory performance, live traffic latency,
drift, business impact, operator acceptance, or retraining outcomes without
real production data, systems, labels, and stakeholder approval.
