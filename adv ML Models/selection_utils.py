"""Leakage-safe feature selection helpers shared by the advanced ML runners."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RAW_NUMERIC = ROOT / "data" / "raw" / "train_numeric.csv"
CHUNKSIZE = 20_000
TOP_NUMERIC_FEATURES = 80
TOP_CATEGORICAL_COLUMNS = 20
TOP_CATEGORICAL_LEVELS = 12


def read_header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return next(csv.reader(file))


def profile_training_fold(training_ids: set[int], expected_rows: int) -> pd.DataFrame:
    """Calculate numeric availability and response signal on training IDs only."""
    features = [name for name in read_header(RAW_NUMERIC) if name not in {"Id", "Response"}]
    stats = {
        name: {"n": 0, "missing": 0, "sum_x": 0.0, "sum_x2": 0.0, "sum_y": 0.0,
               "sum_y2": 0.0, "sum_xy": 0.0, "missing_y": 0.0}
        for name in features
    }
    observed_rows = 0
    for chunk in pd.read_csv(RAW_NUMERIC, chunksize=CHUNKSIZE):
        fold = chunk.loc[chunk["Id"].isin(training_ids)]
        if fold.empty:
            continue
        observed_rows += len(fold)
        y = fold["Response"].to_numpy(dtype=np.float64)
        for name in features:
            values = fold[name].to_numpy(dtype=np.float64, na_value=np.nan)
            present = ~np.isnan(values)
            stat = stats[name]
            stat["missing"] += len(values) - int(present.sum())
            stat["missing_y"] += float(y[~present].sum())
            if not present.any():
                continue
            x = values[present]
            y_present = y[present]
            stat["n"] += len(x)
            stat["sum_x"] += float(x.sum())
            stat["sum_x2"] += float(np.dot(x, x))
            stat["sum_y"] += float(y_present.sum())
            stat["sum_y2"] += float(np.dot(y_present, y_present))
            stat["sum_xy"] += float(np.dot(x, y_present))
    if observed_rows != expected_rows:
        raise RuntimeError(f"Expected {expected_rows:,} training rows but profiled {observed_rows:,}.")
    rows = []
    for feature, stat in stats.items():
        n = stat["n"]
        numerator = n * stat["sum_xy"] - stat["sum_x"] * stat["sum_y"]
        denom_x = n * stat["sum_x2"] - stat["sum_x"] ** 2
        denom_y = n * stat["sum_y2"] - stat["sum_y"] ** 2
        corr = numerator / np.sqrt(denom_x * denom_y) if denom_x > 0 and denom_y > 0 else 0.0
        present_failure = stat["sum_y"] / n if n else 0.0
        missing_failure = stat["missing_y"] / stat["missing"] if stat["missing"] else 0.0
        rows.append({"feature": feature, "present_rate": n / observed_rows,
                     "selection_score": max(abs(corr), abs(missing_failure - present_failure))})
    return pd.DataFrame(rows)


def selected_features(profile: pd.DataFrame, threshold: float) -> tuple[list[str], int]:
    candidates = profile.loc[profile["present_rate"] >= threshold]
    return candidates.nlargest(TOP_NUMERIC_FEATURES, "selection_score")["feature"].tolist(), len(candidates)


def load_fold_features(ids: set[int], columns: list[str]) -> pd.DataFrame:
    """Load selected numeric feature columns for the supplied IDs."""
    parts = []
    for chunk in pd.read_csv(
        RAW_NUMERIC,
        usecols=["Id", *columns],
        dtype={column: "float32" for column in columns},
        chunksize=CHUNKSIZE,
    ):
        subset = chunk.loc[chunk["Id"].isin(ids)]
        if not subset.empty:
            parts.append(subset)
    if not parts:
        raise RuntimeError("No numeric rows were loaded for the requested IDs.")
    return pd.concat(parts, ignore_index=True).set_index("Id")


def validation_threshold(train_scores: np.ndarray, y_train: np.ndarray) -> float:
    """Choose the alert rate from training prevalence only."""
    return float(np.quantile(train_scores, 1.0 - float(y_train.mean())))


def select_categories_on_train(train: pd.DataFrame) -> tuple[list[str], dict[str, list[str]]]:
    """Select categorical columns and one-hot levels using training IDs only."""
    categorical_path = ROOT / "data" / "raw" / "train_categorical.csv"
    columns = [column for column in read_header(categorical_path) if column != "Id"]
    train_ids = set(train["Id"].astype("int64"))
    response_by_id = train.set_index("Id")["Response"]
    stats = {column: [0, 0.0, 0, 0.0] for column in columns}
    for chunk in pd.read_csv(categorical_path, chunksize=CHUNKSIZE, dtype=str):
        chunk = chunk.loc[chunk["Id"].astype("int64").isin(train_ids)]
        if chunk.empty:
            continue
        y = chunk["Id"].astype("int64").map(response_by_id).to_numpy(dtype=float)
        for column in columns:
            present = chunk[column].notna().to_numpy()
            stat = stats[column]
            stat[0] += int(present.sum()); stat[1] += float(y[present].sum())
            stat[2] += int((~present).sum()); stat[3] += float(y[~present].sum())
    ranked = []
    for column, (present_count, present_y, missing_count, missing_y) in stats.items():
        if present_count >= 200 and missing_count:
            ranked.append((abs(present_y / present_count - missing_y / missing_count), column))
    selected = [column for _, column in sorted(ranked, reverse=True)[:TOP_CATEGORICAL_COLUMNS]]
    counts = {column: {} for column in selected}
    for chunk in pd.read_csv(categorical_path, usecols=["Id", *selected], chunksize=CHUNKSIZE, dtype=str):
        chunk = chunk.loc[chunk["Id"].astype("int64").isin(train_ids)]
        if chunk.empty:
            continue
        y = chunk["Id"].astype("int64").map(response_by_id)
        for column in selected:
            grouped = pd.DataFrame({"value": chunk[column].fillna("__MISSING__"), "y": y}).groupby("value")["y"].agg(["size", "sum"])
            for value, row in grouped.iterrows():
                old_count, old_y = counts[column].get(str(value), (0, 0.0))
                counts[column][str(value)] = (old_count + int(row["size"]), old_y + float(row["sum"]))
    return selected, {
        column: [value for value, (count, _) in sorted(values.items(), key=lambda item: (item[1][1] / item[1][0], item[1][0]), reverse=True)[:TOP_CATEGORICAL_LEVELS] if count >= 50]
        for column, values in counts.items()
    }
