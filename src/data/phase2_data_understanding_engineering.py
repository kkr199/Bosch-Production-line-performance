"""Phase 2 data understanding and manufacturing-flow engineering."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATASET_FILES = {
    "train_numeric": "train_numeric.csv",
    "train_categorical": "train_categorical.csv",
    "train_date": "train_date.csv",
    "test_numeric": "test_numeric.csv",
    "test_categorical": "test_categorical.csv",
    "test_date": "test_date.csv",
}

DATASET_TYPES = {
    "train_numeric": ("train", "numeric"),
    "train_categorical": ("train", "categorical"),
    "train_date": ("train", "date"),
    "test_numeric": ("test", "numeric"),
    "test_categorical": ("test", "categorical"),
    "test_date": ("test", "date"),
}

FEATURE_RE = re.compile(r"^L(?P<line>\d+)_S(?P<station>\d+)_(?P<kind>[FD])(?P<feature>\d+)$")


def find_dataset_path(filename: str) -> Path:
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


def parse_feature_name(column: str) -> dict[str, object] | None:
    match = FEATURE_RE.match(column)
    if not match:
        return None
    line = int(match.group("line"))
    station = int(match.group("station"))
    feature_kind = match.group("kind")
    feature_id = int(match.group("feature"))
    return {
        "line": line,
        "station": station,
        "station_key": f"L{line}_S{station}",
        "feature_kind": feature_kind,
        "feature_id": feature_id,
    }


def build_feature_metadata() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for dataset, filename in DATASET_FILES.items():
        split, data_type = DATASET_TYPES[dataset]
        for column in read_header(find_dataset_path(filename)):
            parsed = parse_feature_name(column)
            if parsed is None:
                continue
            records.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "data_type": data_type,
                    "column": column,
                    **parsed,
                }
            )
    return pd.DataFrame(records)


def build_metadata_tables(feature_metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_features = feature_metadata.drop_duplicates(["data_type", "column"]).copy()

    station_metadata = (
        unique_features.pivot_table(
            index=["line", "station", "station_key"],
            columns="data_type",
            values="column",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for column in ["numeric", "categorical", "date"]:
        if column not in station_metadata.columns:
            station_metadata[column] = 0
    station_metadata = station_metadata.rename(
        columns={
            "numeric": "numeric_feature_count",
            "categorical": "categorical_feature_count",
            "date": "date_feature_count",
        }
    )
    station_metadata["total_feature_count"] = station_metadata[
        ["numeric_feature_count", "categorical_feature_count", "date_feature_count"]
    ].sum(axis=1)
    station_metadata["has_numeric"] = station_metadata["numeric_feature_count"].gt(0)
    station_metadata["has_categorical"] = station_metadata["categorical_feature_count"].gt(0)
    station_metadata["has_date"] = station_metadata["date_feature_count"].gt(0)
    station_metadata = station_metadata.sort_values(["line", "station"]).reset_index(drop=True)

    line_metadata = (
        station_metadata.groupby("line", as_index=False)
        .agg(
            station_count=("station", "nunique"),
            numeric_feature_count=("numeric_feature_count", "sum"),
            categorical_feature_count=("categorical_feature_count", "sum"),
            date_feature_count=("date_feature_count", "sum"),
            total_feature_count=("total_feature_count", "sum"),
        )
        .sort_values("line")
        .reset_index(drop=True)
    )
    return station_metadata, line_metadata


def build_completeness_metrics(feature_metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_path = REPORTS_DIR / "phase1_missing_values_by_column.csv"
    if not missing_path.exists():
        raise FileNotFoundError("Run Phase 1 first so reports/phase1_missing_values_by_column.csv exists.")

    missing = pd.read_csv(missing_path)
    missing = missing.merge(
        feature_metadata[
            ["dataset", "split", "data_type", "column", "line", "station", "station_key"]
        ],
        on=["dataset", "column"],
        how="inner",
    )
    missing["observed_values"] = missing["rows"] - missing["missing_values"]
    missing["completeness_pct"] = (100 - missing["missing_pct"]).round(4)

    station_completeness = (
        missing.groupby(["split", "data_type", "line", "station", "station_key"], as_index=False)
        .agg(
            feature_count=("column", "nunique"),
            rows=("rows", "max"),
            missing_values=("missing_values", "sum"),
            observed_values=("observed_values", "sum"),
        )
        .sort_values(["split", "data_type", "line", "station"])
        .reset_index(drop=True)
    )
    station_completeness["possible_values"] = (
        station_completeness["rows"] * station_completeness["feature_count"]
    )
    station_completeness["completeness_pct"] = (
        station_completeness["observed_values"] / station_completeness["possible_values"] * 100
    ).round(4)
    station_completeness["missing_pct"] = (100 - station_completeness["completeness_pct"]).round(4)
    return missing, station_completeness


def station_date_columns(feature_metadata: pd.DataFrame) -> dict[str, list[str]]:
    date_features = (
        feature_metadata[
            (feature_metadata["split"] == "train") & (feature_metadata["data_type"] == "date")
        ]
        .sort_values(["line", "station", "feature_id"])
        .copy()
    )
    station_columns: dict[str, list[str]] = {}
    for row in date_features.itertuples(index=False):
        station_columns.setdefault(row.station_key, []).append(row.column)
    return station_columns


def build_manufacturing_flow_dataset(
    split: str,
    feature_metadata: pd.DataFrame,
    chunksize: int = 20_000,
) -> Path:
    date_path = find_dataset_path(f"{split}_date.csv")
    output_path = PROCESSED_DIR / f"manufacturing_flow_{split}.parquet"
    station_columns = station_date_columns(feature_metadata)
    station_keys = list(station_columns)
    line_ids = sorted({int(key.split("_")[0][1:]) for key in station_keys})
    date_feature_count = sum(len(columns) for columns in station_columns.values())

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None

    date_reader = pd.read_csv(date_path, chunksize=chunksize, low_memory=False)
    if split == "train":
        target_reader = pd.read_csv(
            find_dataset_path("train_numeric.csv"),
            usecols=["Id", "Response"],
            chunksize=chunksize,
            low_memory=False,
        )
        iterator = zip(date_reader, target_reader)
    else:
        iterator = ((date_chunk, None) for date_chunk in date_reader)

    try:
        for date_chunk, target_chunk in tqdm(iterator, desc=f"Building {split} flow", unit="chunk"):
            flow = pd.DataFrame({"Id": date_chunk["Id"].astype("int64")})
            if target_chunk is not None:
                if not flow["Id"].equals(target_chunk["Id"].astype("int64")):
                    raise ValueError("Id order mismatch between train_date.csv and train_numeric.csv")
                flow["Response"] = target_chunk["Response"].astype("int8")

            station_matrix = pd.DataFrame(index=date_chunk.index)
            for station_key, columns in station_columns.items():
                station_matrix[f"present_{station_key}"] = (
                    date_chunk[columns].notna().any(axis=1).astype("uint8")
                )

            line_matrix = pd.DataFrame(index=date_chunk.index)
            for line_id in line_ids:
                line_station_cols = [
                    f"present_{key}" for key in station_keys if key.startswith(f"L{line_id}_")
                ]
                line_matrix[f"line_{line_id}_present"] = (
                    station_matrix[line_station_cols].any(axis=1).astype("uint8")
                )

            observed_date_values = date_chunk.drop(columns=["Id"]).notna().sum(axis=1).astype("int16")
            station_count = station_matrix.sum(axis=1).astype("int16")
            line_count = line_matrix.sum(axis=1).astype("int8")

            station_numbers = np.array([int(key.split("_S")[1]) for key in station_keys], dtype=np.int16)
            station_values = station_matrix.to_numpy(dtype=np.uint8)
            any_station = station_values.any(axis=1)
            first_station = np.where(any_station, np.where(station_values, station_numbers, 999).min(axis=1), -1)
            last_station = np.where(any_station, np.where(station_values, station_numbers, -1).max(axis=1), -1)

            flow = pd.concat([flow, station_matrix, line_matrix], axis=1)
            flow["station_count"] = station_count
            flow["line_count"] = line_count
            flow["first_station"] = first_station.astype("int16")
            flow["last_station"] = last_station.astype("int16")
            flow["observed_date_values"] = observed_date_values
            flow["possible_date_values"] = np.int16(date_feature_count)
            flow["date_completeness_pct"] = (
                observed_date_values / date_feature_count * 100
            ).astype("float32")

            table = pa.Table.from_pandas(flow, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    return output_path


def write_report(
    feature_metadata: pd.DataFrame,
    station_metadata: pd.DataFrame,
    line_metadata: pd.DataFrame,
    station_completeness: pd.DataFrame,
    flow_paths: dict[str, Path],
) -> None:
    report_path = REPORTS_DIR / "phase2_data_understanding_engineering_report.md"
    sparsest_stations = station_completeness.sort_values("completeness_pct").head(15)
    densest_stations = station_completeness.sort_values("completeness_pct", ascending=False).head(15)

    with report_path.open("w", encoding="utf-8") as report:
        report.write("# Phase 2 Data Understanding and Engineering Report\n\n")
        report.write("## Outputs Created\n\n")
        report.write("- `reports/phase2_feature_metadata.csv`\n")
        report.write("- `reports/phase2_station_metadata.csv`\n")
        report.write("- `reports/phase2_line_metadata.csv`\n")
        report.write("- `reports/phase2_feature_completeness_metrics.csv`\n")
        report.write("- `reports/phase2_station_completeness_metrics.csv`\n")
        report.write("- `data/processed/manufacturing_flow_train.parquet`\n")
        report.write("- `data/processed/manufacturing_flow_test.parquet`\n\n")
        report.write("## Feature Metadata Summary\n\n")
        report.write(f"- Parsed feature columns: {feature_metadata['column'].nunique():,}\n")
        report.write(f"- Production lines: {station_metadata['line'].nunique():,}\n")
        report.write(f"- Stations: {station_metadata['station_key'].nunique():,}\n\n")
        report.write("## Line Metadata\n\n")
        report.write(line_metadata.to_markdown(index=False))
        report.write("\n\n## Sparsest Station/Data-Type Groups\n\n")
        report.write(sparsest_stations.to_markdown(index=False))
        report.write("\n\n## Most Complete Station/Data-Type Groups\n\n")
        report.write(densest_stations.to_markdown(index=False))
        report.write("\n\n## Manufacturing Flow Datasets\n\n")
        for split, path in flow_paths.items():
            report.write(f"- {split}: `{path}`\n")
        report.write("\n")
        report.write(
            "Station presence indicators are derived from date features because date values "
            "capture whether a part appears to have passed through a station.\n"
        )


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    feature_metadata = build_feature_metadata()
    station_metadata, line_metadata = build_metadata_tables(feature_metadata)
    feature_completeness, station_completeness = build_completeness_metrics(feature_metadata)

    feature_metadata.to_csv(REPORTS_DIR / "phase2_feature_metadata.csv", index=False)
    station_metadata.to_csv(REPORTS_DIR / "phase2_station_metadata.csv", index=False)
    line_metadata.to_csv(REPORTS_DIR / "phase2_line_metadata.csv", index=False)
    feature_completeness.to_csv(REPORTS_DIR / "phase2_feature_completeness_metrics.csv", index=False)
    station_completeness.to_csv(REPORTS_DIR / "phase2_station_completeness_metrics.csv", index=False)

    flow_paths = {
        "train": build_manufacturing_flow_dataset("train", feature_metadata),
        "test": build_manufacturing_flow_dataset("test", feature_metadata),
    }
    write_report(feature_metadata, station_metadata, line_metadata, station_completeness, flow_paths)

    print("Phase 2 outputs written:")
    print(f"- {REPORTS_DIR / 'phase2_data_understanding_engineering_report.md'}")
    for split, path in flow_paths.items():
        print(f"- {split}: {path}")


if __name__ == "__main__":
    main()
