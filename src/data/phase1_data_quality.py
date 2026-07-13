"""Phase 1 data quality checks for the Bosch Production Line Performance data."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"

DATASET_FILES = {
    "train_numeric": "train_numeric.csv",
    "train_categorical": "train_categorical.csv",
    "train_date": "train_date.csv",
    "test_numeric": "test_numeric.csv",
    "test_categorical": "test_categorical.csv",
    "test_date": "test_date.csv",
}

IGNORED_REFERENCE_FILES = {
    "sample_submission": "sample_submission.csv",
}


@dataclass(frozen=True)
class DatasetProfile:
    dataset: str
    path: Path
    file_size_mb: float
    rows: int
    columns: int
    feature_columns: int
    id_present: bool
    target_present: bool
    total_missing_values: int
    missing_value_pct: float


def find_dataset_path(filename: str) -> Path:
    """Find a dataset file in the current Kaggle download layout."""
    candidates = [
        PROJECT_ROOT / filename,
        PROJECT_ROOT / "data" / "raw" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {filename} in project root or data/raw/")


def read_header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return next(csv.reader(file))


def profile_csv(dataset: str, path: Path, chunksize: int) -> tuple[DatasetProfile, pd.DataFrame]:
    columns = read_header(path)
    missing_counts = pd.Series(0, index=columns, dtype="int64")
    rows = 0

    reader = pd.read_csv(path, chunksize=chunksize, low_memory=False)
    for chunk in tqdm(reader, desc=f"Profiling {dataset}", unit="chunk"):
        rows += len(chunk)
        missing_counts = missing_counts.add(chunk.isna().sum(), fill_value=0).astype("int64")

    total_cells = rows * len(columns)
    total_missing = int(missing_counts.sum())
    id_present = "Id" in columns
    target_present = "Response" in columns
    feature_columns = len(columns) - int(id_present) - int(target_present)

    profile = DatasetProfile(
        dataset=dataset,
        path=path,
        file_size_mb=round(path.stat().st_size / (1024**2), 2),
        rows=rows,
        columns=len(columns),
        feature_columns=feature_columns,
        id_present=id_present,
        target_present=target_present,
        total_missing_values=total_missing,
        missing_value_pct=round((total_missing / total_cells) * 100, 4) if total_cells else 0.0,
    )

    missing_df = (
        missing_counts.rename("missing_values")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    missing_df.insert(0, "dataset", dataset)
    missing_df["rows"] = rows
    missing_df["missing_pct"] = (missing_df["missing_values"] / rows * 100).round(4)
    return profile, missing_df


def load_dataset_samples(nrows: int = 5) -> dict[str, pd.DataFrame]:
    """Lightweight loader proving numeric, categorical, date, and target files are readable."""
    samples: dict[str, pd.DataFrame] = {}
    for dataset, filename in DATASET_FILES.items():
        path = find_dataset_path(filename)
        samples[dataset] = pd.read_csv(path, nrows=nrows)
    return samples


def write_report(profiles: Iterable[DatasetProfile], missing_by_column: pd.DataFrame) -> None:
    profiles_df = pd.DataFrame([profile.__dict__ for profile in profiles])
    profiles_df["path"] = profiles_df["path"].astype(str)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    profiles_df.to_csv(REPORTS_DIR / "phase1_file_summary.csv", index=False)
    missing_by_column.to_csv(REPORTS_DIR / "phase1_missing_values_by_column.csv", index=False)

    train_numeric_missing = missing_by_column[
        missing_by_column["dataset"].eq("train_numeric")
        & missing_by_column["column"].eq("Response")
    ]
    response_missing = (
        int(train_numeric_missing["missing_values"].iloc[0])
        if not train_numeric_missing.empty
        else "not found"
    )

    top_missing = (
        missing_by_column.sort_values(["dataset", "missing_values"], ascending=[True, False])
        .groupby("dataset")
        .head(10)
        .copy()
    )

    report_path = REPORTS_DIR / "phase1_data_quality_report.md"
    with report_path.open("w", encoding="utf-8") as report:
        report.write("# Phase 1 Data Quality Report\n\n")
        report.write(
            "`sample_submission.csv` is intentionally excluded because it is only a Kaggle "
            "submission-format reference file, not a modeling dataset.\n\n"
        )
        report.write("## Dataset Inventory\n\n")
        report.write(profiles_df.to_markdown(index=False))
        report.write("\n\n")
        report.write("## Target Check\n\n")
        report.write(
            "- `Response` is expected in `train_numeric.csv` and should be absent from test files.\n"
        )
        report.write(f"- Missing values in `Response`: {response_missing}\n\n")
        report.write("## Top Missing Columns Per Dataset\n\n")
        report.write(
            top_missing[
                ["dataset", "column", "missing_values", "missing_pct"]
            ].to_markdown(index=False)
        )
        report.write("\n\n")
        report.write("## Notes\n\n")
        report.write(
            "- Full per-column missing-value counts are saved in "
            "`reports/phase1_missing_values_by_column.csv`.\n"
        )
        report.write(
            "- The original Kaggle CSV files are ignored by git because they are large raw data assets.\n"
        )
        report.write(
            "- `sample_submission.csv` is ignored for Phase 1 profiling and downstream modeling.\n"
        )


def main() -> None:
    samples = load_dataset_samples(nrows=5)
    print("Loaded dataset samples:")
    for dataset, sample in samples.items():
        print(f"- {dataset}: {sample.shape[0]} sample rows x {sample.shape[1]} columns")

    profiles: list[DatasetProfile] = []
    missing_frames: list[pd.DataFrame] = []
    for dataset, filename in DATASET_FILES.items():
        path = find_dataset_path(filename)
        profile, missing_df = profile_csv(dataset, path, chunksize=20_000)
        profiles.append(profile)
        missing_frames.append(missing_df)

    missing_by_column = pd.concat(missing_frames, ignore_index=True)
    write_report(profiles, missing_by_column)
    print(f"Phase 1 report written to {REPORTS_DIR / 'phase1_data_quality_report.md'}")


if __name__ == "__main__":
    main()
