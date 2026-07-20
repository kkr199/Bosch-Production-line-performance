import json
import unittest
from pathlib import Path

from src.serving.model_service import ModelService


class GoldenPredictionTests(unittest.TestCase):
    def test_registered_model_matches_approved_golden_set(self) -> None:
        fixture_path = Path(__file__).parent / "fixtures" / "phase6_golden_set.json"
        golden_set = json.loads(fixture_path.read_text(encoding="utf-8"))
        service = ModelService()

        for sample in golden_set["samples"]:
            result = service.predict(sample["features"], request_id=f"golden-{sample['id']}")
            self.assertEqual(result.model_version, golden_set["model_version"])
            self.assertEqual(result.predicted_failure, sample["expected_class"])
            self.assertAlmostEqual(
                result.failure_probability, sample["expected_probability"], places=9
            )


if __name__ == "__main__":
    unittest.main()
