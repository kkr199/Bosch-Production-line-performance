"""Phase 3 exploratory data analysis built directly from raw Bosch CSV files."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TIME_FEATURES_CSV_PATH = PROCESSED_DIR / "phase3_train_time_features.csv"
FEATURE_RE = re.compile(r"^L(?P<line>\d+)_S(?P<station>\d+)_[FD](?P<feature>\d+)$")
MAX_CATEGORICAL_COLUMNS_FOR_OHE = 30
MIN_CATEGORICAL_OBSERVATIONS = 1_000
CATEGORICAL_VARIABILITY_SAMPLE_ROWS = 50_000
CATEGORICAL_SAMPLE_MIN_OBSERVED = 100


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


def build_station_columns_from_raw_date() -> dict[str, list[str]]:
    date_columns = [column for column in read_header(find_dataset_path("train_date.csv")) if column != "Id"]
    station_columns: dict[str, list[str]] = {}
    for column in date_columns:
        parsed = parse_feature_name(column)
        if parsed is None:
            continue
        station_columns.setdefault(str(parsed["station_key"]), []).append(column)
    return dict(sorted(station_columns.items(), key=lambda item: split_station_key(item[0])))


def build_station_metadata_from_raw_headers() -> pd.DataFrame:
    dataset_files = {
        "numeric": "train_numeric.csv",
        "categorical": "train_categorical.csv",
        "date": "train_date.csv",
    }
    records: list[dict[str, int | str]] = []
    for data_type, filename in dataset_files.items():
        for column in read_header(find_dataset_path(filename)):
            parsed = parse_feature_name(column)
            if parsed is None:
                continue
            records.append(
                {
                    "data_type": data_type,
                    "column": column,
                    **parsed,
                }
            )
    metadata = pd.DataFrame(records)
    station_metadata = (
        metadata.pivot_table(
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
    return station_metadata.sort_values(["line", "station"]).reset_index(drop=True)


def select_categorical_columns_for_ohe(
    max_columns: int = MAX_CATEGORICAL_COLUMNS_FOR_OHE,
    priority_station_keys: list[str] | None = None,
) -> list[str]:
    """Select manageable raw categorical columns for one-hot EDA using completeness."""
    categorical_path = find_dataset_path("train_categorical.csv")
    sample = pd.read_csv(categorical_path, nrows=CATEGORICAL_VARIABILITY_SAMPLE_ROWS, low_memory=False)
    variable_columns: list[str] = []
    for column in sample.columns:
        if column in ["Id", "Response"]:
            continue
        observed = int(sample[column].notna().sum())
        unique_values = int(sample[column].nunique(dropna=True))
        if observed >= CATEGORICAL_SAMPLE_MIN_OBSERVED and unique_values > 1:
            variable_columns.append(column)

    missing_path = REPORTS_DIR / "phase1_missing_values_by_column.csv"
    if missing_path.exists():
        missing = pd.read_csv(missing_path)
        candidates = (
            missing[
                missing["dataset"].eq("train_categorical")
                & ~missing["column"].isin(["Id", "Response"])
                & missing["column"].isin(variable_columns)
            ]
            .assign(observed_values=lambda frame: frame["rows"] - frame["missing_values"])
            .query("observed_values >= @MIN_CATEGORICAL_OBSERVATIONS")
            .copy()
        )
        parsed = candidates["column"].map(parse_feature_name)
        candidates["station_key"] = parsed.map(lambda value: value["station_key"] if value else None)

        selected: list[str] = []
        if priority_station_keys:
            priority = (
                candidates[candidates["station_key"].isin(priority_station_keys)]
                .sort_values(["observed_values", "missing_pct"], ascending=[False, True])
                ["column"]
                .tolist()
            )
            selected.extend(priority[: max_columns // 2])

        fill = (
            candidates[~candidates["column"].isin(selected)]
            .sort_values(["observed_values", "missing_pct"], ascending=[False, True])
            ["column"]
            .tolist()
        )
        selected.extend(fill[: max_columns - len(selected)])
        if selected:
            return selected

    return variable_columns[:max_columns]


def analyze_categorical_one_hot_patterns(
    max_columns: int = MAX_CATEGORICAL_COLUMNS_FOR_OHE,
    chunksize: int = 20_000,
    priority_station_keys: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One-hot selected raw categorical columns in chunks and aggregate EDA metrics."""
    selected_columns = select_categorical_columns_for_ohe(
        max_columns=max_columns,
        priority_station_keys=priority_station_keys,
    )
    categorical_path = find_dataset_path("train_categorical.csv")
    target_path = find_dataset_path("train_numeric.csv")
    target_response = pd.read_csv(target_path, usecols=["Response"])["Response"]
    n = len(target_response)
    response_sum = int(target_response.sum())
    overall_failure_rate = response_sum / n

    count_totals: pd.Series | None = None
    failure_totals: pd.Series | None = None
    selected_summary: dict[str, dict[str, int | str]] = {
        column: {"column": column, "observed_values": 0, "unique_values": set()} for column in selected_columns
    }

    categorical_reader = pd.read_csv(
        categorical_path,
        usecols=["Id", *selected_columns],
        chunksize=chunksize,
        low_memory=False,
    )
    target_reader = pd.read_csv(
        target_path,
        usecols=["Id", "Response"],
        chunksize=chunksize,
        low_memory=False,
    )

    for cat_chunk, target_chunk in tqdm(
        zip(categorical_reader, target_reader),
        desc="One-hot encoding selected categorical raw columns",
        unit="chunk",
    ):
        if not cat_chunk["Id"].astype("int64").equals(target_chunk["Id"].astype("int64")):
            raise ValueError("Id order mismatch between train_categorical.csv and train_numeric.csv")

        for column in selected_columns:
            non_null = cat_chunk[column].dropna()
            selected_summary[column]["observed_values"] = int(selected_summary[column]["observed_values"]) + len(non_null)
            selected_summary[column]["unique_values"].update(non_null.astype(str).unique().tolist())

        encoded = pd.get_dummies(
            cat_chunk[selected_columns].astype("string"),
            prefix=selected_columns,
            dummy_na=False,
            dtype="uint8",
        )
        counts = encoded.sum(axis=0).astype("int64")
        failures = encoded.mul(target_chunk["Response"].to_numpy(), axis=0).sum(axis=0).astype("int64")

        count_totals = counts if count_totals is None else count_totals.add(counts, fill_value=0).astype("int64")
        failure_totals = (
            failures
            if failure_totals is None
            else failure_totals.add(failures, fill_value=0).astype("int64")
        )

    if count_totals is None or failure_totals is None:
        raise ValueError("No categorical chunks were processed.")

    encoded_summary = (
        pd.DataFrame(
            {
                "encoded_feature": count_totals.index,
                "part_count": count_totals.values,
                "failure_count": failure_totals.reindex(count_totals.index).fillna(0).astype("int64").values,
            }
        )
        .query("part_count >= @MIN_CATEGORICAL_OBSERVATIONS")
        .copy()
    )
    encoded_summary["failure_rate"] = encoded_summary["failure_count"] / encoded_summary["part_count"]
    encoded_summary["failure_rate_pct"] = encoded_summary["failure_rate"] * 100
    encoded_summary["overall_failure_rate_pct"] = overall_failure_rate * 100
    encoded_summary["failure_rate_lift"] = encoded_summary["failure_rate"] / overall_failure_rate

    # For binary one-hot indicators, this is the phi correlation with Response.
    p_x = encoded_summary["part_count"] / n
    p_y = response_sum / n
    p_xy = encoded_summary["failure_count"] / n
    denom = np.sqrt(p_x * (1 - p_x) * p_y * (1 - p_y))
    encoded_summary["correlation_with_response"] = np.where(
        denom > 0,
        (p_xy - p_x * p_y) / denom,
        np.nan,
    )
    encoded_summary["absolute_correlation"] = encoded_summary["correlation_with_response"].abs()
    encoded_summary = encoded_summary.sort_values(
        ["failure_rate_lift", "part_count"], ascending=[False, False]
    ).reset_index(drop=True)

    selected_columns_summary = pd.DataFrame(
        [
            {
                "column": column,
                "observed_values": values["observed_values"],
                "unique_values": len(values["unique_values"]),
                "encoded_levels_after_min_count_filter": int(
                    encoded_summary["encoded_feature"].str.startswith(f"{column}_").sum()
                ),
            }
            for column, values in selected_summary.items()
        ]
    ).sort_values("observed_values", ascending=False)

    categorical_correlations = encoded_summary.sort_values(
        "absolute_correlation", ascending=False
    ).reset_index(drop=True)
    return selected_columns_summary, encoded_summary, categorical_correlations


def build_train_flow_from_raw(chunksize: int = 20_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build compact EDA flow features directly from raw train_date/train_numeric CSVs."""
    date_path = find_dataset_path("train_date.csv")
    target_path = find_dataset_path("train_numeric.csv")
    station_columns = build_station_columns_from_raw_date()
    station_keys = list(station_columns)
    line_ids = sorted({split_station_key(key)[0] for key in station_keys})
    date_feature_count = sum(len(columns) for columns in station_columns.values())

    flow_frames: list[pd.DataFrame] = []
    time_frames: list[pd.DataFrame] = []

    date_reader = pd.read_csv(date_path, chunksize=chunksize, low_memory=False)
    target_reader = pd.read_csv(
        target_path,
        usecols=["Id", "Response"],
        chunksize=chunksize,
        low_memory=False,
    )

    for date_chunk, target_chunk in tqdm(
        zip(date_reader, target_reader), desc="Building raw-sourced Phase 3 features", unit="chunk"
    ):
        ids = date_chunk["Id"].astype("int64")
        target_ids = target_chunk["Id"].astype("int64")
        if not ids.equals(target_ids):
            raise ValueError("Id order mismatch between train_date.csv and train_numeric.csv")

        flow = pd.DataFrame(
            {
                "Id": ids,
                "Response": target_chunk["Response"].astype("int8"),
            }
        )
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

        date_values = date_chunk.drop(columns=["Id"])
        first_event_time = date_values.min(axis=1, skipna=True).astype("float32")
        last_event_time = date_values.max(axis=1, skipna=True).astype("float32")
        observed_date_values = date_values.notna().sum(axis=1).astype("int16")

        station_numbers = np.array([split_station_key(key)[1] for key in station_keys], dtype=np.int16)
        station_values = station_matrix.to_numpy(dtype=np.uint8)
        any_station = station_values.any(axis=1)
        first_station = np.where(any_station, np.where(station_values, station_numbers, 999).min(axis=1), -1)
        last_station = np.where(any_station, np.where(station_values, station_numbers, -1).max(axis=1), -1)

        flow = pd.concat([flow, station_matrix, line_matrix], axis=1)
        flow["station_count"] = station_matrix.sum(axis=1).astype("int16")
        flow["line_count"] = line_matrix.sum(axis=1).astype("int8")
        flow["first_station"] = first_station.astype("int16")
        flow["last_station"] = last_station.astype("int16")
        flow["observed_date_values"] = observed_date_values
        flow["possible_date_values"] = np.int16(date_feature_count)
        flow["date_completeness_pct"] = (
            observed_date_values / date_feature_count * 100
        ).astype("float32")
        flow_frames.append(flow)

        time_frames.append(
            pd.DataFrame(
                {
                    "Id": ids,
                    "Response": target_chunk["Response"].astype("int8"),
                    "first_event_time": first_event_time,
                    "last_event_time": last_event_time,
                    "process_duration": (last_event_time - first_event_time).astype("float32"),
                    "observed_date_values": observed_date_values,
                }
            )
        )

    train_flow = pd.concat(flow_frames, ignore_index=True)
    time_features = pd.concat(time_frames, ignore_index=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    time_features.to_csv(TIME_FEATURES_CSV_PATH, index=False)
    return train_flow, time_features


def calculate_line_failure_rates(train_flow: pd.DataFrame) -> pd.DataFrame:
    overall_rate = train_flow["Response"].mean()
    records: list[dict[str, float | int]] = []
    for column in sorted(
        [column for column in train_flow.columns if column.startswith("line_") and column.endswith("_present")],
        key=lambda value: int(value.split("_")[1]),
    ):
        line = int(column.split("_")[1])
        present = train_flow[column].eq(1)
        part_count = int(present.sum())
        failure_count = int(train_flow.loc[present, "Response"].sum())
        failure_rate = failure_count / part_count if part_count else 0.0
        records.append(
            {
                "line": line,
                "part_count": part_count,
                "failure_count": failure_count,
                "failure_rate": failure_rate,
                "failure_rate_pct": failure_rate * 100,
                "overall_failure_rate_pct": overall_rate * 100,
                "failure_rate_lift": failure_rate / overall_rate if overall_rate else np.nan,
            }
        )
    return pd.DataFrame(records).sort_values("failure_rate", ascending=False).reset_index(drop=True)


def calculate_station_failure_rates(train_flow: pd.DataFrame) -> pd.DataFrame:
    overall_rate = train_flow["Response"].mean()
    station_metadata = build_station_metadata_from_raw_headers()
    records: list[dict[str, float | int | str]] = []

    station_columns = sorted(
        [column for column in train_flow.columns if column.startswith("present_L")],
        key=lambda column: split_station_key(column.replace("present_", "", 1)),
    )
    for column in station_columns:
        station_key = column.replace("present_", "", 1)
        line, station = split_station_key(station_key)
        present = train_flow[column].eq(1)
        part_count = int(present.sum())
        failure_count = int(train_flow.loc[present, "Response"].sum())
        failure_rate = failure_count / part_count if part_count else 0.0
        records.append(
            {
                "line": line,
                "station": station,
                "station_key": station_key,
                "presence_column": column,
                "part_count": part_count,
                "failure_count": failure_count,
                "failure_rate": failure_rate,
                "failure_rate_pct": failure_rate * 100,
                "overall_failure_rate_pct": overall_rate * 100,
                "failure_rate_lift": failure_rate / overall_rate if overall_rate else np.nan,
            }
        )

    station_rates = pd.DataFrame(records)
    station_rates = station_rates.merge(station_metadata, on=["line", "station", "station_key"], how="left")
    return station_rates.sort_values("failure_rate", ascending=False).reset_index(drop=True)


def identify_high_risk_stations(station_rates: pd.DataFrame, min_parts: int = 1_000) -> pd.DataFrame:
    eligible = station_rates[station_rates["part_count"].ge(min_parts)].copy()
    eligible["excess_failure_rate_pct"] = eligible["failure_rate_pct"] - eligible["overall_failure_rate_pct"]
    eligible["risk_score"] = eligible["excess_failure_rate_pct"] * np.log10(eligible["part_count"])
    return (
        eligible[eligible["excess_failure_rate_pct"].gt(0)]
        .sort_values(["risk_score", "failure_rate_pct"], ascending=False)
        .reset_index(drop=True)
    )


def create_flow_path_summary(train_flow: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    station_columns = sorted(
        [column for column in train_flow.columns if column.startswith("present_L")],
        key=lambda column: split_station_key(column.replace("present_", "", 1)),
    )
    station_keys = [column.replace("present_", "", 1) for column in station_columns]

    path_signature = np.empty(len(train_flow), dtype=object)
    station_values = train_flow[station_columns].to_numpy(dtype=np.uint8)
    for idx, row in enumerate(station_values):
        active = [station_key for station_key, value in zip(station_keys, row) if value == 1]
        path_signature[idx] = " > ".join(active) if active else "no_station_detected"

    grouped = (
        pd.DataFrame({"path_signature": path_signature, "Response": train_flow["Response"].to_numpy()})
        .groupby("path_signature", observed=True)["Response"]
        .agg(part_count="count", failure_count="sum", failure_rate="mean")
        .reset_index()
        .sort_values("part_count", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    grouped.insert(0, "path_rank", np.arange(1, len(grouped) + 1))
    grouped["failure_rate_pct"] = grouped["failure_rate"] * 100
    return grouped[["path_rank", "path_signature", "part_count", "failure_count", "failure_rate_pct"]]


def analyze_time_patterns(time_features: pd.DataFrame, bins: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = time_features.dropna(subset=["first_event_time", "last_event_time"]).copy()
    valid["first_time_bin"] = pd.qcut(valid["first_event_time"], q=bins, duplicates="drop")
    valid["duration_bin"] = pd.qcut(valid["process_duration"], q=bins, duplicates="drop")

    first_time_patterns = (
        valid.groupby("first_time_bin", observed=True)
        .agg(
            part_count=("Id", "count"),
            failure_count=("Response", "sum"),
            failure_rate=("Response", "mean"),
            avg_duration=("process_duration", "mean"),
            avg_observed_date_values=("observed_date_values", "mean"),
        )
        .reset_index()
    )
    first_time_patterns["first_time_bin"] = first_time_patterns["first_time_bin"].astype(str)
    first_time_patterns["failure_rate_pct"] = first_time_patterns["failure_rate"] * 100

    duration_patterns = (
        valid.groupby("duration_bin", observed=True)
        .agg(
            part_count=("Id", "count"),
            failure_count=("Response", "sum"),
            failure_rate=("Response", "mean"),
            avg_first_event_time=("first_event_time", "mean"),
            avg_observed_date_values=("observed_date_values", "mean"),
        )
        .reset_index()
    )
    duration_patterns["duration_bin"] = duration_patterns["duration_bin"].astype(str)
    duration_patterns["failure_rate_pct"] = duration_patterns["failure_rate"] * 100
    return first_time_patterns, duration_patterns


def create_correlation_report(train_flow: pd.DataFrame) -> pd.DataFrame:
    candidate_columns = [
        column
        for column in train_flow.columns
        if column != "Id" and pd.api.types.is_numeric_dtype(train_flow[column])
    ]
    correlations = (
        train_flow[candidate_columns]
        .corr(numeric_only=True)["Response"]
        .drop(index="Response")
        .rename("correlation_with_response")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    correlations["absolute_correlation"] = correlations["correlation_with_response"].abs()
    return correlations.sort_values("absolute_correlation", ascending=False).reset_index(drop=True)


def create_distribution_report(train_flow: pd.DataFrame, time_features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    metrics = [
        ("flow", "station_count", train_flow),
        ("flow", "line_count", train_flow),
        ("flow", "observed_date_values", train_flow),
        ("flow", "date_completeness_pct", train_flow),
        ("time", "first_event_time", time_features),
        ("time", "last_event_time", time_features),
        ("time", "process_duration", time_features),
    ]
    for source, metric, frame in metrics:
        for response, group in frame.groupby("Response"):
            rows.append(
                {
                    "source": source,
                    "metric": metric,
                    "response": int(response),
                    "count": int(group[metric].count()),
                    "mean": group[metric].mean(),
                    "median": group[metric].median(),
                    "std": group[metric].std(),
                    "p10": group[metric].quantile(0.10),
                    "p90": group[metric].quantile(0.90),
                }
            )
    return pd.DataFrame(rows)


def save_plots(
    line_rates: pd.DataFrame,
    station_rates: pd.DataFrame,
    high_risk: pd.DataFrame,
    flow_paths: pd.DataFrame,
    first_time_patterns: pd.DataFrame,
    duration_patterns: pd.DataFrame,
    correlations: pd.DataFrame,
    categorical_encoded_summary: pd.DataFrame,
    train_flow: pd.DataFrame,
) -> dict[str, Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    outputs: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=line_rates.sort_values("line"), x="line", y="failure_rate_pct", color="#4C78A8", ax=ax)
    ax.axhline(line_rates["overall_failure_rate_pct"].iloc[0], color="#E45756", linestyle="--", label="Overall")
    ax.set_title("Failure Rate by Production Line")
    ax.set_xlabel("Production line")
    ax.set_ylabel("Failure rate (%)")
    ax.legend()
    outputs["line_failure_rates"] = FIGURES_DIR / "phase3_line_failure_rates.png"
    fig.tight_layout()
    fig.savefig(outputs["line_failure_rates"], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.scatterplot(
        data=station_rates.sort_values(["line", "station"]),
        x="station",
        y="failure_rate_pct",
        hue="line",
        size="part_count",
        sizes=(30, 300),
        palette="tab10",
        ax=ax,
    )
    ax.axhline(station_rates["overall_failure_rate_pct"].iloc[0], color="#E45756", linestyle="--")
    ax.set_title("Failure Rate by Station")
    ax.set_xlabel("Station")
    ax.set_ylabel("Failure rate (%)")
    outputs["station_failure_rates"] = FIGURES_DIR / "phase3_station_failure_rates.png"
    fig.tight_layout()
    fig.savefig(outputs["station_failure_rates"], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=high_risk.head(15).sort_values("risk_score"), x="risk_score", y="station_key", color="#F58518", ax=ax)
    ax.set_title("Top High-Risk Stations")
    ax.set_xlabel("Risk score")
    ax.set_ylabel("Station")
    outputs["high_risk_stations"] = FIGURES_DIR / "phase3_high_risk_stations.png"
    fig.tight_layout()
    fig.savefig(outputs["high_risk_stations"], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=flow_paths.head(15).sort_values("part_count"), x="part_count", y="path_rank", orient="h", color="#54A24B", ax=ax)
    ax.set_title("Top Product Flow Paths by Volume")
    ax.set_xlabel("Part count")
    ax.set_ylabel("Path rank")
    outputs["flow_paths"] = FIGURES_DIR / "phase3_top_flow_paths.png"
    fig.tight_layout()
    fig.savefig(outputs["flow_paths"], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=first_time_patterns.reset_index(), x="index", y="failure_rate_pct", marker="o", color="#B279A2", ax=ax)
    ax.set_title("Failure Rate by Relative First-Event Time Bin")
    ax.set_xlabel("First-event time bin index")
    ax.set_ylabel("Failure rate (%)")
    outputs["first_time_patterns"] = FIGURES_DIR / "phase3_first_time_failure_pattern.png"
    fig.tight_layout()
    fig.savefig(outputs["first_time_patterns"], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=duration_patterns.reset_index(), x="index", y="failure_rate_pct", marker="o", color="#72B7B2", ax=ax)
    ax.set_title("Failure Rate by Process-Duration Bin")
    ax.set_xlabel("Duration bin index")
    ax.set_ylabel("Failure rate (%)")
    outputs["duration_patterns"] = FIGURES_DIR / "phase3_duration_failure_pattern.png"
    fig.tight_layout()
    fig.savefig(outputs["duration_patterns"], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=correlations.head(20).sort_values("absolute_correlation"), x="correlation_with_response", y="feature", color="#9D755D", ax=ax)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_title("Top Raw-Sourced Flow-Feature Correlations with Response")
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Feature")
    outputs["correlations"] = FIGURES_DIR / "phase3_top_correlations.png"
    fig.tight_layout()
    fig.savefig(outputs["correlations"], dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric in zip(axes, ["station_count", "line_count", "date_completeness_pct"]):
        sns.boxplot(data=train_flow, x="Response", y=metric, ax=ax, color="#A0CBE8")
        ax.set_title(metric)
        ax.set_xlabel("Response")
    outputs["distributions"] = FIGURES_DIR / "phase3_flow_metric_distributions.png"
    fig.tight_layout()
    fig.savefig(outputs["distributions"], dpi=150)
    plt.close(fig)

    if not categorical_encoded_summary.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_data = (
            categorical_encoded_summary.head(20)
            .sort_values("failure_rate_lift")
            .copy()
        )
        sns.barplot(data=plot_data, x="failure_rate_lift", y="encoded_feature", color="#ECA82C", ax=ax)
        ax.axvline(1, color="#333333", linewidth=1, linestyle="--")
        ax.set_title("Top One-Hot Categorical Levels by Failure-Rate Lift")
        ax.set_xlabel("Failure-rate lift vs overall")
        ax.set_ylabel("Encoded categorical level")
        outputs["categorical_ohe_lift"] = FIGURES_DIR / "phase3_categorical_ohe_lift.png"
        fig.tight_layout()
        fig.savefig(outputs["categorical_ohe_lift"], dpi=150)
        plt.close(fig)

    return outputs


def write_report(
    line_rates: pd.DataFrame,
    high_risk: pd.DataFrame,
    flow_paths: pd.DataFrame,
    first_time_patterns: pd.DataFrame,
    duration_patterns: pd.DataFrame,
    correlations: pd.DataFrame,
    distributions: pd.DataFrame,
    categorical_selected_columns: pd.DataFrame,
    categorical_encoded_summary: pd.DataFrame,
    categorical_correlations: pd.DataFrame,
    figures: dict[str, Path],
) -> None:
    report_path = REPORTS_DIR / "phase3_exploratory_data_analysis_report.md"
    overall_failure_rate = line_rates["overall_failure_rate_pct"].iloc[0]
    with report_path.open("w", encoding="utf-8") as report:
        report.write("# Phase 3 Exploratory Data Analysis Report\n\n")
        report.write("All Phase 3 calculations are sourced directly from the raw CSV datasets: `train_date.csv`, `train_numeric.csv`, `train_categorical.csv`, and raw headers. No Phase 2 Parquet flow dataset is used.\n\n")
        report.write("## Summary\n\n")
        report.write(f"- Overall training failure rate: {overall_failure_rate:.4f}%\n")
        report.write(f"- Highest-risk line: L{int(line_rates.iloc[0]['line'])}\n")
        report.write(f"- Highest-risk station: {high_risk.iloc[0]['station_key'] if not high_risk.empty else 'none'}\n")
        report.write("- Date values are anonymized relative production times, so seasonality is analyzed as relative-time bins rather than calendar dates.\n\n")

        report.write("## Failure Rate by Production Line\n\n")
        report.write(line_rates.to_markdown(index=False))
        report.write("\n\n## Top High-Risk Stations\n\n")
        report.write(high_risk.head(20).to_markdown(index=False))
        report.write("\n\n## Top Product Flow Paths\n\n")
        report.write(flow_paths.head(15).to_markdown(index=False))
        report.write("\n\n## First-Event Time Failure Pattern\n\n")
        report.write(first_time_patterns.to_markdown(index=False))
        report.write("\n\n## Process-Duration Failure Pattern\n\n")
        report.write(duration_patterns.to_markdown(index=False))
        report.write("\n\n## Top Correlations with Response\n\n")
        report.write(correlations.head(30).to_markdown(index=False))
        report.write("\n\n## Selected Raw Categorical Columns for One-Hot EDA\n\n")
        report.write(categorical_selected_columns.to_markdown(index=False))
        report.write("\n\n## Top One-Hot Categorical Levels by Failure-Rate Lift\n\n")
        report.write(categorical_encoded_summary.head(30).to_markdown(index=False))
        report.write("\n\n## Top One-Hot Categorical Correlations with Response\n\n")
        report.write(categorical_correlations.head(30).to_markdown(index=False))
        report.write("\n\n## Distribution Summary by Response\n\n")
        report.write(distributions.to_markdown(index=False))
        report.write("\n\n## Figures\n\n")
        for name, path in figures.items():
            report.write(f"- {name}: `{path}`\n")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_flow, time_features = build_train_flow_from_raw()
    line_rates = calculate_line_failure_rates(train_flow)
    station_rates = calculate_station_failure_rates(train_flow)
    high_risk = identify_high_risk_stations(station_rates)
    flow_paths = create_flow_path_summary(train_flow)
    first_time_patterns, duration_patterns = analyze_time_patterns(time_features)
    correlations = create_correlation_report(train_flow)
    distributions = create_distribution_report(train_flow, time_features)
    priority_station_keys = high_risk["station_key"].head(10).tolist()
    categorical_selected_columns, categorical_encoded_summary, categorical_correlations = (
        analyze_categorical_one_hot_patterns(priority_station_keys=priority_station_keys)
    )

    line_rates.to_csv(REPORTS_DIR / "phase3_line_failure_rates.csv", index=False)
    station_rates.to_csv(REPORTS_DIR / "phase3_station_failure_rates.csv", index=False)
    high_risk.to_csv(REPORTS_DIR / "phase3_high_risk_stations.csv", index=False)
    flow_paths.to_csv(REPORTS_DIR / "phase3_flow_path_summary.csv", index=False)
    first_time_patterns.to_csv(REPORTS_DIR / "phase3_first_time_failure_patterns.csv", index=False)
    duration_patterns.to_csv(REPORTS_DIR / "phase3_duration_failure_patterns.csv", index=False)
    correlations.to_csv(REPORTS_DIR / "phase3_correlation_report.csv", index=False)
    distributions.to_csv(REPORTS_DIR / "phase3_distribution_report.csv", index=False)
    categorical_selected_columns.to_csv(
        REPORTS_DIR / "phase3_categorical_ohe_selected_columns.csv", index=False
    )
    categorical_encoded_summary.to_csv(
        REPORTS_DIR / "phase3_categorical_ohe_failure_rates.csv", index=False
    )
    categorical_correlations.to_csv(
        REPORTS_DIR / "phase3_categorical_ohe_correlation_report.csv", index=False
    )

    figures = save_plots(
        line_rates=line_rates,
        station_rates=station_rates,
        high_risk=high_risk,
        flow_paths=flow_paths,
        first_time_patterns=first_time_patterns,
        duration_patterns=duration_patterns,
        correlations=correlations,
        categorical_encoded_summary=categorical_encoded_summary,
        train_flow=train_flow,
    )
    write_report(
        line_rates=line_rates,
        high_risk=high_risk,
        flow_paths=flow_paths,
        first_time_patterns=first_time_patterns,
        duration_patterns=duration_patterns,
        correlations=correlations,
        distributions=distributions,
        categorical_selected_columns=categorical_selected_columns,
        categorical_encoded_summary=categorical_encoded_summary,
        categorical_correlations=categorical_correlations,
        figures=figures,
    )

    print("Phase 3 raw-sourced outputs written:")
    print(f"- {REPORTS_DIR / 'phase3_exploratory_data_analysis_report.md'}")
    print(f"- {TIME_FEATURES_CSV_PATH}")
    for path in figures.values():
        print(f"- {path}")


if __name__ == "__main__":
    main()
