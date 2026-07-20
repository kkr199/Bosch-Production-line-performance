"""Phase 4 feature engineering from raw Bosch date datasets."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FEATURE_RE = re.compile(r"^L(?P<line>\d+)_S(?P<station>\d+)_[FD](?P<feature>\d+)$")
CHUNKSIZE = 20_000


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


def parse_feature_name(column: str) -> dict[str, int | str] | None:
    match = FEATURE_RE.match(column)
    if not match:
        return None
    line = int(match.group("line"))
    station = int(match.group("station"))
    return {
        "line": line,
        "station": station,
        "station_key": f"L{line}_S{station}",
        "feature_id": int(match.group("feature")),
    }


def split_station_key(station_key: str) -> tuple[int, int]:
    line_part, station_part = station_key.split("_")
    return int(line_part[1:]), int(station_part[1:])


def build_date_column_groups() -> tuple[dict[str, list[str]], dict[int, list[str]]]:
    station_columns: dict[str, list[str]] = {}
    line_columns: dict[int, list[str]] = {}
    for column in read_header(find_dataset_path("train_date.csv")):
        if column == "Id":
            continue
        parsed = parse_feature_name(column)
        if parsed is None:
            continue
        station_key = str(parsed["station_key"])
        line = int(parsed["line"])
        station_columns.setdefault(station_key, []).append(column)
        line_columns.setdefault(line, []).append(column)

    station_columns = dict(sorted(station_columns.items(), key=lambda item: split_station_key(item[0])))
    line_columns = dict(sorted(line_columns.items()))
    return station_columns, line_columns


def rowwise_line_switch_count(station_presence: np.ndarray, line_ids: np.ndarray) -> np.ndarray:
    switches = np.zeros(station_presence.shape[0], dtype=np.int16)
    for row_idx, row in enumerate(station_presence):
        active_lines = line_ids[row.astype(bool)]
        if len(active_lines) > 1:
            switches[row_idx] = int(np.sum(active_lines[1:] != active_lines[:-1]))
    return switches


def rowwise_waiting_metrics(
    station_start: np.ndarray,
    station_end: np.ndarray,
    index: pd.Index,
) -> pd.DataFrame:
    rows = station_start.shape[0]
    total_waiting = np.zeros(rows, dtype=np.float32)
    mean_waiting = np.zeros(rows, dtype=np.float32)
    max_waiting = np.zeros(rows, dtype=np.float32)
    wait_event_count = np.zeros(rows, dtype=np.int16)

    for row_idx in range(rows):
        present = ~np.isnan(station_start[row_idx])
        if present.sum() < 2:
            continue
        starts = station_start[row_idx, present]
        ends = station_end[row_idx, present]
        gaps = starts[1:] - ends[:-1]
        gaps = gaps[gaps > 0]
        if len(gaps) == 0:
            continue
        total_waiting[row_idx] = float(gaps.sum())
        mean_waiting[row_idx] = float(gaps.mean())
        max_waiting[row_idx] = float(gaps.max())
        wait_event_count[row_idx] = len(gaps)

    return pd.DataFrame(
        {
            "waiting_time": total_waiting,
            "mean_waiting_time": mean_waiting,
            "max_waiting_time": max_waiting,
            "wait_event_count": wait_event_count,
        },
        index=index,
    )


def engineer_chunk(
    date_chunk: pd.DataFrame,
    station_columns: dict[str, list[str]],
    line_columns: dict[int, list[str]],
    target_chunk: pd.DataFrame | None = None,
) -> pd.DataFrame:
    station_keys = list(station_columns)
    station_numbers = np.array([split_station_key(key)[1] for key in station_keys], dtype=np.int16)
    station_line_ids = np.array([split_station_key(key)[0] for key in station_keys], dtype=np.int8)

    features = pd.DataFrame({"Id": date_chunk["Id"].astype("int64")})
    if target_chunk is not None:
        if not features["Id"].equals(target_chunk["Id"].astype("int64")):
            raise ValueError("Id order mismatch between date and numeric target files.")
        features["Response"] = target_chunk["Response"].astype("int8")

    date_values = date_chunk.drop(columns=["Id"])
    start_time = date_values.min(axis=1, skipna=True).astype("float32")
    end_time = date_values.max(axis=1, skipna=True).astype("float32")
    cycle_time = (end_time - start_time).astype("float32")
    observed_date_values = date_values.notna().sum(axis=1).astype("int16")

    station_start_frames: list[pd.Series] = []
    station_end_frames: list[pd.Series] = []
    station_presence_frames: list[pd.Series] = []
    station_processing_frames: list[pd.Series] = []
    for station_key, columns in station_columns.items():
        station_values = date_chunk[columns]
        station_start = station_values.min(axis=1, skipna=True).astype("float32")
        station_end = station_values.max(axis=1, skipna=True).astype("float32")
        station_present = station_values.notna().any(axis=1).astype("uint8")
        station_processing = (station_end - station_start).fillna(0).clip(lower=0).astype("float32")
        station_start_frames.append(station_start)
        station_end_frames.append(station_end)
        station_presence_frames.append(station_present)
        station_processing_frames.append(station_processing)

    station_start_matrix = pd.concat(station_start_frames, axis=1).to_numpy(dtype=np.float32)
    station_end_matrix = pd.concat(station_end_frames, axis=1).to_numpy(dtype=np.float32)
    station_presence_matrix = pd.concat(station_presence_frames, axis=1).to_numpy(dtype=np.uint8)
    station_processing_matrix = pd.concat(station_processing_frames, axis=1).to_numpy(dtype=np.float32)

    station_count = station_presence_matrix.sum(axis=1).astype(np.int16)
    line_count = np.zeros(len(date_chunk), dtype=np.int8)
    line_presence_frames: list[pd.Series] = []
    for line_id, columns in line_columns.items():
        line_present = date_chunk[columns].notna().any(axis=1).astype("uint8")
        line_presence_frames.append(line_present.rename(f"line_{line_id}_present"))
        line_count += line_present.to_numpy(dtype=np.int8)

    any_station = station_presence_matrix.any(axis=1)
    first_station = np.where(
        any_station,
        np.where(station_presence_matrix, station_numbers, 999).min(axis=1),
        -1,
    ).astype(np.int16)
    last_station = np.where(
        any_station,
        np.where(station_presence_matrix, station_numbers, -1).max(axis=1),
        -1,
    ).astype(np.int16)
    station_span = np.where(any_station, last_station - first_station + 1, 0).astype(np.int16)
    path_density = np.divide(
        station_count,
        station_span,
        out=np.zeros_like(station_count, dtype=np.float32),
        where=station_span > 0,
    ).astype(np.float32)
    line_switch_count = rowwise_line_switch_count(station_presence_matrix, station_line_ids)

    processing_duration = np.nan_to_num(station_processing_matrix, nan=0).sum(axis=1).astype(np.float32)
    waiting_metrics = rowwise_waiting_metrics(station_start_matrix, station_end_matrix, date_chunk.index)
    delay_ratio = np.divide(
        waiting_metrics["waiting_time"].to_numpy(dtype=np.float32),
        cycle_time.to_numpy(dtype=np.float32),
        out=np.zeros(len(date_chunk), dtype=np.float32),
        where=cycle_time.to_numpy(dtype=np.float32) > 0,
    )

    features["start_time"] = start_time
    features["end_time"] = end_time
    features["cycle_time"] = cycle_time
    features["processing_duration"] = processing_duration
    features["waiting_time"] = waiting_metrics["waiting_time"].astype("float32")
    features["mean_waiting_time"] = waiting_metrics["mean_waiting_time"].astype("float32")
    features["max_waiting_time"] = waiting_metrics["max_waiting_time"].astype("float32")
    features["wait_event_count"] = waiting_metrics["wait_event_count"].astype("int16")
    features["delay_ratio"] = delay_ratio.astype("float32")
    features["observed_date_values"] = observed_date_values
    features["station_count"] = station_count
    features["line_count"] = line_count
    features["first_station"] = first_station
    features["last_station"] = last_station
    features["station_span"] = station_span
    features["path_density"] = path_density
    features["line_switch_count"] = line_switch_count
    features["path_complexity_score"] = (
        station_count + (2 * line_count) + line_switch_count + (1 - path_density)
    ).astype("float32")

    features = pd.concat([features, *line_presence_frames], axis=1)

    total_date_feature_count = date_values.shape[1]
    features["date_completeness_pct"] = (
        observed_date_values / total_date_feature_count * 100
    ).astype("float32")

    for line_id, columns in line_columns.items():
        line_values = date_chunk[columns]
        line_start = line_values.min(axis=1, skipna=True).astype("float32")
        line_end = line_values.max(axis=1, skipna=True).astype("float32")
        line_observed = line_values.notna().sum(axis=1).astype("int16")
        line_station_cols = [
            idx for idx, key in enumerate(station_keys) if split_station_key(key)[0] == line_id
        ]
        line_station_count = station_presence_matrix[:, line_station_cols].sum(axis=1).astype(np.int16)

        features[f"line_{line_id}_start_time"] = line_start
        features[f"line_{line_id}_end_time"] = line_end
        features[f"line_{line_id}_processing_duration"] = (line_end - line_start).fillna(0).clip(lower=0).astype("float32")
        features[f"line_{line_id}_observed_date_values"] = line_observed
        features[f"line_{line_id}_station_count"] = line_station_count
        features[f"line_{line_id}_date_completeness_pct"] = (
            line_observed / len(columns) * 100
        ).astype("float32")

    return features


def engineer_split(split: str, chunksize: int = CHUNKSIZE) -> Path:
    station_columns, line_columns = build_date_column_groups()
    output_path = PROCESSED_DIR / f"phase4_{split}_engineered_features.csv"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    date_reader = pd.read_csv(find_dataset_path(f"{split}_date.csv"), chunksize=chunksize, low_memory=False)
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

    first_write = True
    for date_chunk, target_chunk in tqdm(iterator, desc=f"Engineering Phase 4 {split}", unit="chunk"):
        features = engineer_chunk(date_chunk, station_columns, line_columns, target_chunk)
        features.to_csv(output_path, mode="w" if first_write else "a", header=first_write, index=False)
        first_write = False

    return output_path


def create_feature_dictionary() -> pd.DataFrame:
    rows = [
        ("start_time", "Earliest measurement timestamp (relative/anonymized; not an official production start time)."),
        ("end_time", "Latest measurement timestamp (relative/anonymized; not an official production end time)."),
        ("cycle_time", "Observed measurement time span between earliest and latest timestamps; not a verified cycle time."),
        ("processing_duration", "Observed within-station measurement span; not a verified processing duration."),
        ("waiting_time", "Sum of positive inter-station timestamp gaps; a temporal proxy, not confirmed queue waiting time."),
        ("mean_waiting_time", "Mean positive inter-station timestamp gap; a temporal proxy, not confirmed queue waiting time."),
        ("max_waiting_time", "Maximum positive inter-station timestamp gap; a temporal proxy, not confirmed queue waiting time."),
        ("wait_event_count", "Number of positive inter-station timestamp gaps; not confirmed queue events."),
        ("delay_ratio", "Relative inter-station timestamp-gap ratio; not a verified physical-delay measure."),
        ("station_count", "Number of stations with at least one observed date value."),
        ("line_count", "Number of production lines with at least one observed date value."),
        ("station_span", "Distance from first observed station to last observed station."),
        ("path_density", "station_count divided by station_span."),
        ("line_switch_count", "Number of production-line changes along the observed station path."),
        ("path_complexity_score", "Composite path score using station_count, line_count, switches, and density."),
        ("date_completeness_pct", "Share of raw date features observed for a product."),
        ("line_{n}_present", "Whether production line n has at least one observed date value."),
        ("line_{n}_start_time", "Earliest measurement timestamp within production line n (relative/anonymized)."),
        ("line_{n}_end_time", "Latest measurement timestamp within production line n (relative/anonymized)."),
        ("line_{n}_processing_duration", "Observed measurement time span within production line n; not a verified processing duration."),
        ("line_{n}_observed_date_values", "Observed raw date values within production line n."),
        ("line_{n}_station_count", "Stations visited within production line n."),
        ("line_{n}_date_completeness_pct", "Share of line n date features observed."),
    ]
    return pd.DataFrame(rows, columns=["feature", "description"])


def summarize_engineered_features(train_path: Path, test_path: Path) -> pd.DataFrame:
    summaries: list[dict[str, object]] = []
    for split, path in [("train", train_path), ("test", test_path)]:
        preview = pd.read_csv(path, nrows=5)
        row_count = sum(1 for _ in path.open("r", encoding="utf-8")) - 1
        summaries.append(
            {
                "split": split,
                "rows": row_count,
                "columns": preview.shape[1],
                "file_size_mb": round(path.stat().st_size / (1024**2), 2),
                "has_response": "Response" in preview.columns,
            }
        )
    return pd.DataFrame(summaries)


def write_report(summary: pd.DataFrame, feature_dictionary: pd.DataFrame) -> None:
    report_path = REPORTS_DIR / "phase4_feature_engineering_report.md"
    with report_path.open("w", encoding="utf-8") as report:
        report.write("# Phase 4 Feature Engineering Report\n\n")
        report.write("All Phase 4 features are engineered from the raw date CSV files. `Response` is joined from `train_numeric.csv` for the train output only.\n\n")
        report.write("## Outputs Created\n\n")
        report.write("- `data/processed/phase4_train_engineered_features.csv`\n")
        report.write("- `data/processed/phase4_test_engineered_features.csv`\n")
        report.write("- `reports/phase4_feature_dictionary.csv`\n")
        report.write("- `reports/phase4_engineered_feature_summary.csv`\n\n")
        report.write("## Engineered Output Summary\n\n")
        report.write(summary.to_markdown(index=False))
        report.write("\n\n## Feature Dictionary\n\n")
        report.write(feature_dictionary.to_markdown(index=False))
        report.write("\n")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_path = engineer_split("train")
    test_path = engineer_split("test")
    feature_dictionary = create_feature_dictionary()
    summary = summarize_engineered_features(train_path, test_path)

    feature_dictionary.to_csv(REPORTS_DIR / "phase4_feature_dictionary.csv", index=False)
    summary.to_csv(REPORTS_DIR / "phase4_engineered_feature_summary.csv", index=False)
    write_report(summary, feature_dictionary)

    print("Phase 4 outputs written:")
    print(f"- {train_path}")
    print(f"- {test_path}")
    print(f"- {REPORTS_DIR / 'phase4_feature_engineering_report.md'}")


if __name__ == "__main__":
    main()
