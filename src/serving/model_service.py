"""Schema-validated model serving, registry, rollback, and audit logging."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "models" / "registry" / "registry.json"
DEFAULT_AUDIT_PATH = PROJECT_ROOT / "data" / "database" / "prediction_audit.db"


class InputContractError(ValueError):
    """Raised when a prediction request does not meet the model input contract."""


@dataclass(frozen=True)
class PredictionResult:
    request_id: str
    model_version: str
    failure_probability: float
    predicted_failure: int
    threshold: float
    latency_ms: float


class ModelRegistry:
    """File-backed registry for locally versioned, validated model artifacts."""

    def __init__(self, registry_path: Path = DEFAULT_REGISTRY_PATH) -> None:
        self.registry_path = Path(registry_path)

    def load(self) -> dict[str, Any]:
        with self.registry_path.open(encoding="utf-8") as file:
            registry = json.load(file)
        if "models" not in registry or not isinstance(registry["models"], list):
            raise ValueError("Registry must contain a models list.")
        return registry

    def production_model(self) -> dict[str, Any]:
        models = [item for item in self.load()["models"] if item.get("status") == "production"]
        if len(models) != 1:
            raise ValueError("Registry must contain exactly one production model.")
        return models[0]

    def promote(self, version: str) -> dict[str, Any]:
        registry = self.load()
        candidate = next((item for item in registry["models"] if item["version"] == version), None)
        if candidate is None:
            raise KeyError(f"Unknown model version: {version}")
        if candidate.get("status") not in {"staging", "rollback", "production"}:
            raise ValueError("Only validated staging, rollback, or production models may be promoted.")
        for item in registry["models"]:
            if item.get("status") == "production":
                item["status"] = "rollback"
        candidate["status"] = "production"
        self._write(registry)
        return candidate

    def rollback(self) -> dict[str, Any]:
        rollback_candidates = [
            item for item in self.load()["models"] if item.get("status") == "rollback"
        ]
        if not rollback_candidates:
            raise ValueError("No rollback model is registered.")
        return self.promote(rollback_candidates[-1]["version"])

    def _write(self, registry: dict[str, Any]) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.registry_path.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(self.registry_path)


class PredictionAuditLog:
    """SQLite audit store with no raw feature values or sensitive payloads."""

    def __init__(self, database_path: Path = DEFAULT_AUDIT_PATH) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_audit (
                    request_id TEXT PRIMARY KEY,
                    timestamp_utc TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    input_feature_count INTEGER NOT NULL,
                    failure_probability REAL,
                    predicted_failure INTEGER,
                    latency_ms REAL NOT NULL,
                    status TEXT NOT NULL,
                    error_code TEXT
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def record(
        self,
        *,
        request_id: str,
        model_version: str,
        input_feature_count: int,
        latency_ms: float,
        status: str,
        failure_probability: float | None = None,
        predicted_failure: int | None = None,
        error_code: str | None = None,
    ) -> None:
        connection = sqlite3.connect(self.database_path)
        try:
            connection.execute(
                """
                INSERT INTO prediction_audit (
                    request_id, timestamp_utc, model_version, input_feature_count,
                    failure_probability, predicted_failure, latency_ms, status, error_code
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(request_id) DO UPDATE SET
                    timestamp_utc=excluded.timestamp_utc,
                    model_version=excluded.model_version,
                    input_feature_count=excluded.input_feature_count,
                    failure_probability=excluded.failure_probability,
                    predicted_failure=excluded.predicted_failure,
                    latency_ms=excluded.latency_ms,
                    status=excluded.status,
                    error_code=excluded.error_code
                """,
                (
                    request_id,
                    datetime.now(timezone.utc).isoformat(),
                    model_version,
                    input_feature_count,
                    failure_probability,
                    predicted_failure,
                    latency_ms,
                    status,
                    error_code,
                ),
            )
            connection.commit()
        finally:
            connection.close()


class ModelService:
    """Serve the registered model only after enforcing its exact feature contract."""

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        audit_log: PredictionAuditLog | None = None,
        project_root: Path = PROJECT_ROOT,
    ) -> None:
        self.registry = registry or ModelRegistry()
        self.audit_log = audit_log or PredictionAuditLog()
        self.project_root = Path(project_root)
        self._loaded_version: str | None = None
        self._bundle: dict[str, Any] | None = None

    def contract(self) -> dict[str, Any]:
        model = self.registry.production_model()
        bundle = self._load_bundle(model)
        return {
            "model_version": model["version"],
            "feature_count": len(bundle["feature_cols"]),
            "required_features": bundle["feature_cols"],
            "value_type": "finite number",
            "missing_values": "permitted as null and preserved as model missing values",
            "unknown_features": "not permitted",
        }

    def predict(self, features: dict[str, Any], request_id: str | None = None) -> PredictionResult:
        request_id = request_id or str(uuid4())
        started = time.perf_counter()
        model = self.registry.production_model()
        try:
            bundle = self._load_bundle(model)
            frame = self._validate_features(features, bundle["feature_cols"])
            probability = float(bundle["model"].predict_proba(frame)[:, 1][0])
            threshold = float(bundle["best_threshold"])
            result = PredictionResult(
                request_id=request_id,
                model_version=model["version"],
                failure_probability=probability,
                predicted_failure=int(probability >= threshold),
                threshold=threshold,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            self.audit_log.record(
                request_id=result.request_id,
                model_version=result.model_version,
                input_feature_count=len(features),
                failure_probability=result.failure_probability,
                predicted_failure=result.predicted_failure,
                latency_ms=result.latency_ms,
                status="success",
            )
            return result
        except Exception as error:
            self.audit_log.record(
                request_id=request_id,
                model_version=model["version"],
                input_feature_count=len(features),
                latency_ms=(time.perf_counter() - started) * 1000,
                status="rejected" if isinstance(error, InputContractError) else "error",
                error_code=type(error).__name__,
            )
            raise

    def _load_bundle(self, model: dict[str, Any]) -> dict[str, Any]:
        if self._bundle is not None and self._loaded_version == model["version"]:
            return self._bundle
        artifact_path = self.project_root / model["artifact_path"]
        actual_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if actual_sha256 != model["artifact_sha256"]:
            raise ValueError("Model artifact hash does not match the registry entry.")
        bundle = joblib.load(artifact_path)
        required = {"model", "feature_cols", "best_threshold"}
        if not required.issubset(bundle):
            raise ValueError("Model artifact is not a supported prediction bundle.")
        self._bundle = bundle
        self._loaded_version = model["version"]
        return bundle

    @staticmethod
    def _validate_features(features: dict[str, Any], required_features: list[str]) -> pd.DataFrame:
        supplied = set(features)
        required = set(required_features)
        missing = sorted(required - supplied)
        unknown = sorted(supplied - required)
        if missing or unknown:
            parts = []
            if missing:
                parts.append(f"missing required features: {', '.join(missing[:10])}")
            if unknown:
                parts.append(f"unknown features: {', '.join(unknown[:10])}")
            raise InputContractError("; ".join(parts))
        frame = pd.DataFrame([{name: features[name] for name in required_features}])
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        non_numeric = frame.notna() & numeric.isna()
        if non_numeric.any().any():
            raise InputContractError("feature values must be numeric or null")
        finite_values = numeric.to_numpy()[~np.isnan(numeric.to_numpy())]
        if not np.isfinite(finite_values).all():
            raise InputContractError("feature values must be finite when supplied")
        return numeric
