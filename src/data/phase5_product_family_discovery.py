"""Phase 5 product family discovery from raw Bosch date datasets."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

FEATURE_RE = re.compile(r"^L(?P<line>\d+)_S(?P<station>\d+)_[FD](?P<feature>\d+)$")
CHUNKSIZE = 20_000
RANDOM_STATE = 42
KMEANS_FAMILIES = 8
MODEL_MAX_ROWS_PER_FAMILY = 150_000


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


def build_date_column_groups() -> dict[str, list[str]]:
    station_columns: dict[str, list[str]] = {}
    for column in read_header(find_dataset_path("train_date.csv")):
        if column == "Id":
            continue
        parsed = parse_feature_name(column)
        if parsed is None:
            continue
        station_key = str(parsed["station_key"])
        station_columns.setdefault(station_key, []).append(column)
    return dict(sorted(station_columns.items(), key=lambda item: split_station_key(item[0])))


def get_target_response() -> pd.DataFrame:
    numeric_path = find_dataset_path("train_numeric.csv")
    return pd.read_csv(numeric_path, usecols=["Id", "Response"])


def station_presence_columns(station_columns: dict[str, list[str]]) -> list[str]:
    return [f"present_{station_key}" for station_key in station_columns]


def build_station_presence_matrix(
    split: str,
    station_columns: dict[str, list[str]],
    target: pd.DataFrame | None = None,
    overwrite: bool = False,
) -> Path:
    date_path = find_dataset_path(f"{split}_date.csv")
    output_path = PROCESSED_DIR / f"phase5_{split}_station_presence_matrix.csv"
    station_presence = station_presence_columns(station_columns)
    first_chunk = True

    if output_path.exists() and not overwrite:
        print(f"Reusing existing station presence matrix: {output_path}")
        return output_path

    if output_path.exists() and overwrite:
        output_path.unlink()

    target_lookup = None
    if target is not None:
        target_lookup = target.set_index("Id")["Response"]

    reader = pd.read_csv(date_path, chunksize=CHUNKSIZE)
    for chunk in tqdm(reader, desc=f"Building {split} station presence", unit="chunk"):
        matrix = pd.DataFrame({"Id": chunk["Id"].astype(np.int64)})
        if target_lookup is not None:
            matrix["Response"] = matrix["Id"].map(target_lookup).astype(np.int8)

        station_values: dict[str, np.ndarray] = {}
        for station_key, columns in station_columns.items():
            present_col = f"present_{station_key}"
            station_values[present_col] = chunk[columns].notna().any(axis=1).astype(np.uint8).to_numpy()

        station_frame = pd.DataFrame(station_values, index=chunk.index)
        matrix = pd.concat([matrix, station_frame], axis=1)
        matrix["station_count"] = matrix[station_presence].sum(axis=1).astype(np.int16)
        matrix["line_count"] = (
            pd.DataFrame(
                {
                    f"L{line}": matrix[
                        [column for column in station_presence if column.startswith(f"present_L{line}_")]
                    ].any(axis=1)
                    for line in sorted({split_station_key(key)[0] for key in station_columns})
                }
            )
            .sum(axis=1)
            .astype(np.int8)
        )

        matrix.to_csv(output_path, index=False, mode="w" if first_chunk else "a", header=first_chunk)
        first_chunk = False

    return output_path


def path_signature_from_matrix(values: np.ndarray) -> list[str]:
    packed = np.packbits(values.astype(np.uint8), axis=1)
    return [bytes(row).hex() for row in packed]


def add_path_signature(frame: pd.DataFrame, station_cols: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    frame["path_signature"] = path_signature_from_matrix(frame[station_cols].to_numpy())
    return frame


def cluster_product_paths(
    train_presence: pd.DataFrame,
    test_presence: pd.DataFrame,
    station_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_presence = add_path_signature(train_presence, station_cols)
    test_presence = add_path_signature(test_presence, station_cols)

    x_train = train_presence[station_cols].to_numpy(dtype=np.float32)
    x_test = test_presence[station_cols].to_numpy(dtype=np.float32)

    kmeans = MiniBatchKMeans(
        n_clusters=KMEANS_FAMILIES,
        random_state=RANDOM_STATE,
        batch_size=20_000,
        n_init=20,
    )
    train_presence["kmeans_family"] = kmeans.fit_predict(x_train).astype(np.int16)
    test_presence["kmeans_family"] = kmeans.predict(x_test).astype(np.int16)

    pattern_base = train_presence[["path_signature", "Response", *station_cols]].copy()
    aggregations = {"Response": ["size", "sum"]}
    aggregations.update({column: "max" for column in station_cols})
    unique_paths = pattern_base.groupby("path_signature", as_index=False).agg(aggregations)
    unique_paths.columns = [
        "path_signature",
        "part_count",
        "failure_count",
        *station_cols,
    ]
    unique_paths["failure_rate_pct"] = unique_paths["failure_count"] / unique_paths["part_count"] * 100
    unique_x = unique_paths[station_cols].to_numpy(dtype=np.float32)

    dbscan = DBSCAN(eps=1.5, min_samples=50, metric="euclidean")
    dbscan.fit(unique_x, sample_weight=unique_paths["part_count"].to_numpy())
    unique_paths["dbscan_family"] = dbscan.labels_.astype(np.int16)

    if len(unique_paths) <= 5_000:
        hierarchical = AgglomerativeClustering(n_clusters=KMEANS_FAMILIES, linkage="ward")
        unique_paths["hierarchical_family"] = hierarchical.fit_predict(unique_x).astype(np.int16)
    else:
        # Hierarchical clustering is quadratic. Cluster the most common paths, then assign rarer paths to the
        # nearest discovered family profile so every product still receives a usable label.
        common_paths = unique_paths.nlargest(5_000, "part_count").copy()
        common_x = common_paths[station_cols].to_numpy(dtype=np.float32)
        hierarchical = AgglomerativeClustering(n_clusters=KMEANS_FAMILIES, linkage="ward")
        common_paths["hierarchical_family"] = hierarchical.fit_predict(common_x).astype(np.int16)
        centroids = common_paths.groupby("hierarchical_family")[station_cols].mean().sort_index()
        distances = ((unique_x[:, None, :] - centroids.to_numpy()[None, :, :]) ** 2).sum(axis=2)
        unique_paths["hierarchical_family"] = centroids.index[np.argmin(distances, axis=1)].to_numpy().astype(np.int16)

    label_map = unique_paths.set_index("path_signature")[["dbscan_family", "hierarchical_family"]]
    train_presence = train_presence.join(label_map, on="path_signature")
    test_presence = test_presence.join(label_map, on="path_signature")
    test_presence[["dbscan_family", "hierarchical_family"]] = (
        test_presence[["dbscan_family", "hierarchical_family"]].fillna(-2).astype(np.int16)
    )

    train_presence["final_product_family"] = train_presence["kmeans_family"].astype(np.int16)
    test_presence["final_product_family"] = test_presence["kmeans_family"].astype(np.int16)

    diagnostics = build_cluster_diagnostics(train_presence, unique_paths, station_cols)
    return train_presence, test_presence, unique_paths, diagnostics


def build_cluster_diagnostics(
    train_presence: pd.DataFrame,
    unique_paths: pd.DataFrame,
    station_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    sample = train_presence.sample(
        n=min(25_000, len(train_presence)),
        random_state=RANDOM_STATE,
    )
    sample_x = sample[station_cols].to_numpy(dtype=np.float32)

    for method, label_col, source in [
        ("KMeans", "kmeans_family", train_presence),
        ("DBSCAN", "dbscan_family", train_presence),
        ("Hierarchical", "hierarchical_family", train_presence),
    ]:
        labels = source[label_col]
        cluster_count = int(labels[labels >= 0].nunique())
        noise_count = int((labels == -1).sum())
        metric = np.nan
        try:
            sample_labels = sample[label_col].to_numpy()
            if len(np.unique(sample_labels)) > 1:
                metric = float(silhouette_score(sample_x, sample_labels, metric="euclidean"))
        except Exception:
            metric = np.nan
        rows.append(
            {
                "method": method,
                "cluster_count": cluster_count,
                "noise_count": noise_count,
                "rows_labeled": int(len(labels)),
                "unique_path_count": int(len(unique_paths)),
                "sample_silhouette": metric,
            }
        )

    return pd.DataFrame(rows)


def build_family_profiles(
    train_labels: pd.DataFrame,
    unique_paths: pd.DataFrame,
    station_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_failure_rate = train_labels["Response"].mean()
    failure_rates = (
        train_labels.groupby("final_product_family")
        .agg(
            part_count=("Id", "size"),
            failure_count=("Response", "sum"),
            avg_station_count=("station_count", "mean"),
            avg_line_count=("line_count", "mean"),
        )
        .reset_index()
    )
    failure_rates["failure_rate_pct"] = failure_rates["failure_count"] / failure_rates["part_count"] * 100
    failure_rates["failure_lift_vs_overall"] = (
        failure_rates["failure_count"] / failure_rates["part_count"] / overall_failure_rate
    )
    failure_rates = failure_rates.sort_values("failure_rate_pct", ascending=False)

    family_station_rates = (
        train_labels.groupby("final_product_family")[station_cols]
        .mean()
        .reset_index()
        .rename(columns={column: column.replace("present_", "presence_rate_") for column in station_cols})
    )
    profiles = failure_rates.merge(family_station_rates, on="final_product_family", how="left")

    top_station_rows: list[dict[str, str | int | float]] = []
    profile_station_cols = [column.replace("present_", "presence_rate_") for column in station_cols]
    for _, row in profiles.iterrows():
        top_stations = (
            row[profile_station_cols]
            .sort_values(ascending=False)
            .head(8)
            .rename(lambda value: str(value).replace("presence_rate_", ""))
        )
        top_station_rows.append(
            {
                "final_product_family": int(row["final_product_family"]),
                "top_stations": ", ".join(f"{station}:{rate:.1%}" for station, rate in top_stations.items()),
            }
        )
    top_station_summary = pd.DataFrame(top_station_rows)
    profiles = profiles.merge(top_station_summary, on="final_product_family", how="left")

    path_map = unique_paths.sort_values("part_count", ascending=False)
    return profiles, path_map


def train_family_models(labels_path: Path) -> pd.DataFrame:
    feature_path = PROCESSED_DIR / "phase4_train_engineered_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError("Phase 4 engineered features are required before Phase 5 family models.")

    labels = pd.read_csv(labels_path, usecols=["Id", "final_product_family"])
    features = pd.read_csv(feature_path)
    data = features.merge(labels, on="Id", how="inner")

    feature_cols = [
        column
        for column in data.columns
        if column not in {"Id", "Response", "final_product_family"}
        and pd.api.types.is_numeric_dtype(data[column])
    ]

    metrics: list[dict[str, float | int | str]] = []
    for family_id, family_frame in tqdm(
        data.groupby("final_product_family"),
        desc="Training family models",
        unit="family",
    ):
        family_frame = family_frame.copy()
        positives = int(family_frame["Response"].sum())
        negatives = int(len(family_frame) - positives)
        if len(family_frame) < 5_000 or positives < 20 or negatives < 20:
            metrics.append(
                {
                    "final_product_family": int(family_id),
                    "status": "skipped_insufficient_class_balance",
                    "rows_available": int(len(family_frame)),
                    "training_rows": 0,
                    "validation_rows": 0,
                    "failure_count": positives,
                    "failure_rate_pct": positives / len(family_frame) * 100,
                    "roc_auc": np.nan,
                    "average_precision": np.nan,
                    "model_path": "",
                }
            )
            continue

        if len(family_frame) > MODEL_MAX_ROWS_PER_FAMILY:
            family_frame, _ = train_test_split(
                family_frame,
                train_size=MODEL_MAX_ROWS_PER_FAMILY,
                random_state=RANDOM_STATE,
                stratify=family_frame["Response"],
            )
            family_frame = family_frame.reset_index(drop=True)

        x = family_frame[feature_cols].replace([np.inf, -np.inf], np.nan)
        y = family_frame["Response"].astype(np.int8)
        x_train, x_valid, y_train, y_valid = train_test_split(
            x,
            y,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        model = HistGradientBoostingClassifier(
            max_iter=50,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=RANDOM_STATE,
        )
        model.fit(x_train, y_train, sample_weight=sample_weight)
        scores = model.predict_proba(x_valid)[:, 1]

        model_path = MODELS_DIR / f"phase5_family_{int(family_id)}_hgb.joblib"
        joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)
        metrics.append(
            {
                "final_product_family": int(family_id),
                "status": "trained",
                "rows_available": int(len(family_frame)),
                "training_rows": int(len(x_train)),
                "validation_rows": int(len(x_valid)),
                "failure_count": int(y.sum()),
                "failure_rate_pct": float(y.mean() * 100),
                "roc_auc": float(roc_auc_score(y_valid, scores)),
                "average_precision": float(average_precision_score(y_valid, scores)),
                "model_path": str(model_path.relative_to(PROJECT_ROOT)),
            }
        )

    return pd.DataFrame(metrics).sort_values(["status", "final_product_family"])


def write_report(
    train_labels: pd.DataFrame,
    test_labels: pd.DataFrame,
    profiles: pd.DataFrame,
    diagnostics: pd.DataFrame,
    model_metrics: pd.DataFrame,
) -> Path:
    report_path = REPORTS_DIR / "phase5_product_family_discovery_report.md"
    trained_count = int((model_metrics["status"] == "trained").sum())
    highest_risk = profiles.sort_values("failure_rate_pct", ascending=False).iloc[0]

    lines = [
        "# Phase 5: Product Family Discovery",
        "",
        "## Source Data",
        "",
        "- Station presence matrix was built from raw `train_date.csv` and `test_date.csv`.",
        "- The target `Response` was loaded from raw `train_numeric.csv`.",
        "- Family-level models used `data/processed/phase4_train_engineered_features.csv` because Phase 4 created the timing and path features required for modeling.",
        "",
        "## Dataset Size",
        "",
        f"- Train products clustered: {len(train_labels):,}",
        f"- Test products clustered: {len(test_labels):,}",
        f"- Product families selected for final labels: {profiles['final_product_family'].nunique():,}",
        "",
        "## Clustering Diagnostics",
        "",
        diagnostics.to_markdown(index=False),
        "",
        "## Highest-Risk Product Family",
        "",
        (
            f"Family {int(highest_risk['final_product_family'])} has the highest observed failure rate "
            f"at {highest_risk['failure_rate_pct']:.3f}% across {int(highest_risk['part_count']):,} products "
            f"({highest_risk['failure_lift_vs_overall']:.2f}x the overall train failure rate)."
        ),
        "",
        "## Family Model Training",
        "",
        f"- Trained family-specific models: {trained_count}",
        f"- Skipped families: {len(model_metrics) - trained_count}",
        "- Model type: `HistGradientBoostingClassifier` with balanced sample weights.",
        "- Validation metric files include ROC AUC and average precision for each family.",
        "",
        "## Output Files",
        "",
        "- `data/processed/phase5_train_station_presence_matrix.csv`",
        "- `data/processed/phase5_test_station_presence_matrix.csv`",
        "- `data/processed/phase5_train_product_families.csv`",
        "- `data/processed/phase5_test_product_families.csv`",
        "- `reports/phase5_unique_path_cluster_map.csv`",
        "- `reports/phase5_cluster_diagnostics.csv`",
        "- `reports/phase5_product_family_profiles.csv`",
        "- `reports/phase5_product_family_failure_rates.csv`",
        "- `reports/phase5_family_model_metrics.csv`",
        "- `models/phase5_family_*_hgb.joblib`",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    station_columns = build_date_column_groups()
    station_cols = station_presence_columns(station_columns)
    target = get_target_response()

    train_matrix_path = build_station_presence_matrix("train", station_columns, target)
    test_matrix_path = build_station_presence_matrix("test", station_columns)

    train_presence = pd.read_csv(train_matrix_path)
    test_presence = pd.read_csv(test_matrix_path)

    train_labels, test_labels, unique_paths, diagnostics = cluster_product_paths(
        train_presence=train_presence,
        test_presence=test_presence,
        station_cols=station_cols,
    )

    train_labels_path = PROCESSED_DIR / "phase5_train_product_families.csv"
    test_labels_path = PROCESSED_DIR / "phase5_test_product_families.csv"
    train_labels[
        [
            "Id",
            "Response",
            "station_count",
            "line_count",
            "path_signature",
            "kmeans_family",
            "dbscan_family",
            "hierarchical_family",
            "final_product_family",
        ]
    ].to_csv(train_labels_path, index=False)
    test_labels[
        [
            "Id",
            "station_count",
            "line_count",
            "path_signature",
            "kmeans_family",
            "dbscan_family",
            "hierarchical_family",
            "final_product_family",
        ]
    ].to_csv(test_labels_path, index=False)

    profiles, path_map = build_family_profiles(train_labels, unique_paths, station_cols)
    diagnostics.to_csv(REPORTS_DIR / "phase5_cluster_diagnostics.csv", index=False)
    profiles.to_csv(REPORTS_DIR / "phase5_product_family_profiles.csv", index=False)
    profiles[
        [
            "final_product_family",
            "part_count",
            "failure_count",
            "failure_rate_pct",
            "failure_lift_vs_overall",
            "avg_station_count",
            "avg_line_count",
            "top_stations",
        ]
    ].to_csv(REPORTS_DIR / "phase5_product_family_failure_rates.csv", index=False)
    path_map.to_csv(REPORTS_DIR / "phase5_unique_path_cluster_map.csv", index=False)

    model_metrics = train_family_models(train_labels_path)
    model_metrics.to_csv(REPORTS_DIR / "phase5_family_model_metrics.csv", index=False)

    report_path = write_report(train_labels, test_labels, profiles, diagnostics, model_metrics)
    print(f"Phase 5 complete. Report written to {report_path}")


if __name__ == "__main__":
    main()
