import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from src.serving.model_service import (
    InputContractError,
    ModelRegistry,
    ModelService,
    PredictionAuditLog,
)


class FakeProbabilityModel:
    def predict_proba(self, frame):
        probability = float(frame["f1"].fillna(0).iloc[0]) / 10
        return np.array([[1 - probability, probability]])


class ModelServingTests(unittest.TestCase):
    def _service(self, directory: Path) -> ModelService:
        artifact = directory / "models" / "fake.joblib"
        artifact.parent.mkdir()
        joblib.dump(
            {
                "model": FakeProbabilityModel(),
                "feature_cols": ["f1", "f2"],
                "best_threshold": 0.5,
            },
            artifact,
        )
        registry = {
            "models": [
                {
                    "version": "fake-v1",
                    "status": "production",
                    "artifact_path": "models/fake.joblib",
                    "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                }
            ]
        }
        registry_path = directory / "registry.json"
        registry_path.write_text(json.dumps(registry), encoding="utf-8")
        return ModelService(
            registry=ModelRegistry(registry_path),
            audit_log=PredictionAuditLog(directory / "prediction_audit.db"),
            project_root=directory,
        )

    def test_prediction_enforces_schema_and_writes_audit_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            service = self._service(directory)
            result = service.predict({"f1": 8.0, "f2": None}, request_id="request-1")

            self.assertEqual(result.model_version, "fake-v1")
            self.assertEqual(result.predicted_failure, 1)
            connection = sqlite3.connect(directory / "prediction_audit.db")
            try:
                row = connection.execute(
                    "SELECT status, input_feature_count FROM prediction_audit WHERE request_id='request-1'"
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(row, ("success", 2))

    def test_schema_rejects_missing_or_unknown_features(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            service = self._service(Path(temporary_directory))
            with self.assertRaises(InputContractError):
                service.predict({"f1": 1.0})
            with self.assertRaises(InputContractError):
                service.predict({"f1": 1.0, "f2": 2.0, "unexpected": 3.0})


if __name__ == "__main__":
    unittest.main()
