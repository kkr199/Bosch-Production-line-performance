from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.utils.ml_governance import build_split_manifest, population_stability_index


class MlGovernanceTests(unittest.TestCase):
    def test_split_manifest_records_counts_and_rates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "source.csv"
            source.write_text("Id,Response\n1,0\n2,1\n", encoding="utf-8")
            train = pd.DataFrame({"Id": [1, 2], "Response": [0, 1]})
            validation = pd.DataFrame({"Id": [3, 4], "Response": [0, 1]})
            holdout = pd.DataFrame({"Id": [5, 6], "Response": [0, 1]})

            manifest = build_split_manifest(train, validation, holdout, source, seed=42)

        self.assertEqual(manifest.train_rows, 2)
        self.assertEqual(manifest.seed, 42)
        self.assertEqual(manifest.train_positive_rate, 0.5)
        self.assertEqual(len(manifest.source_sha256), 64)

    def test_psi_is_zero_for_identical_distributions(self) -> None:
        values = pd.Series([1, 2, 3, 4, 5, 6])
        self.assertAlmostEqual(population_stability_index(values, values), 0.0, places=8)


if __name__ == "__main__":
    unittest.main()
