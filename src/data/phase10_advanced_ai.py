"""Phase 10: Advanced AI experiments for Bosch production failure analysis.

This phase keeps the Phase 6 LightGBM model as the official production-safe
classifier, then adds advanced diagnostic experiments:

1. Isolation Forest anomaly detection.
2. MLP reconstruction-error anomaly detection.
3. Graph message-passing risk features from the Phase 9 station graph.
4. Failure trajectory prediction using ordered station risk exposure.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"
NOTEBOOKS = ROOT / "notebooks"

RANDOM_STATE = 42
MAX_IFOREST_NORMALS = 80_000
MAX_AUTOENCODER_NORMALS = 45_000
MAX_AUTOENCODER_FEATURES = 120

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
    "blue": "#A3BEFA",
    "blue_dark": "#2E4780",
    "gold": "#FFE15B",
    "gold_dark": "#736422",
    "orange": "#F0986E",
    "orange_dark": "#804126",
    "olive": "#A3D576",
    "olive_dark": "#386411",
    "pink": "#F390CA",
    "pink_dark": "#8A3A6F",
}


def setup_plotting() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        },
    )


def add_chart_header(fig, ax, title: str, subtitle: str) -> None:
    ax.set_title("")
    fig.subplots_adjust(top=0.82)
    left = ax.get_position().x0
    fig.text(left, 0.975, title, ha="left", va="top", fontsize=14, fontweight="semibold", color=TOKENS["ink"])
    fig.text(left, 0.925, subtitle, ha="left", va="top", fontsize=9, color=TOKENS["muted"])
    sns.despine(ax=ax)


def require_inputs() -> None:
    required = [
        PROCESSED / "phase6_train_dataset.csv",
        PROCESSED / "phase6_validation_dataset.csv",
        PROCESSED / "phase6_test_dataset_preview.csv",
        PROCESSED / "phase5_train_station_presence_matrix.csv",
        PROCESSED / "phase5_test_station_presence_matrix.csv",
        REPORTS / "phase9_critical_nodes.csv",
        REPORTS / "phase9_knowledge_graph_edges.csv",
        REPORTS / "phase8_process_map_edges.csv",
        REPORTS / "phase6_model_comparison_metrics.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Phase 10 needs prior phase outputs. Missing: " + ", ".join(missing))


def load_phase6_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(PROCESSED / "phase6_train_dataset.csv")
    valid = pd.read_csv(PROCESSED / "phase6_validation_dataset.csv")
    test_preview = pd.read_csv(PROCESSED / "phase6_test_dataset_preview.csv")
    return train, valid, test_preview


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in {"Id", "Response"}]


def select_autoencoder_features(train: pd.DataFrame) -> list[str]:
    feature_cols = feature_columns(train)
    corr_path = REPORTS / "phase6_final_feature_correlation_report.csv"
    if corr_path.exists():
        corr = pd.read_csv(corr_path)
        ordered = [f for f in corr["feature"].tolist() if f in feature_cols]
        if len(ordered) >= MAX_AUTOENCODER_FEATURES:
            return ordered[:MAX_AUTOENCODER_FEATURES]
    variances = train[feature_cols].replace([np.inf, -np.inf], np.nan).var(numeric_only=True).sort_values(ascending=False)
    return variances.head(MAX_AUTOENCODER_FEATURES).index.tolist()


def best_threshold_metrics(y_true: pd.Series, scores: np.ndarray) -> dict[str, float]:
    y = y_true.astype(int).to_numpy()
    finite_scores = np.asarray(scores, dtype=float)
    finite_scores = np.nan_to_num(finite_scores, nan=np.nanmedian(finite_scores), posinf=np.nanmax(finite_scores), neginf=np.nanmin(finite_scores))
    thresholds = np.unique(np.quantile(finite_scores, np.linspace(0.50, 0.995, 120)))
    best: dict[str, float] | None = None
    for threshold in thresholds:
        pred = (finite_scores >= threshold).astype(int)
        metrics = {
            "threshold": float(threshold),
            "mcc": float(matthews_corrcoef(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
        }
        if best is None or metrics["mcc"] > best["mcc"]:
            best = metrics
    assert best is not None
    best["pr_auc"] = float(average_precision_score(y, finite_scores))
    return best


def metric_row(model_name: str, y_true: pd.Series, scores: np.ndarray, notes: str) -> dict[str, float | str]:
    metrics = best_threshold_metrics(y_true, scores)
    return {"model": model_name, **metrics, "notes": notes}


def sample_normals(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    normals = frame[frame["Response"].eq(0)]
    if len(normals) <= max_rows:
        return normals
    return normals.sample(max_rows, random_state=RANDOM_STATE)


def fit_isolation_forest(train: pd.DataFrame, valid: pd.DataFrame, test_preview: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, Pipeline]:
    cols = feature_columns(train)
    normal_train = sample_normals(train, MAX_IFOREST_NORMALS)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                IsolationForest(
                    n_estimators=260,
                    contamination=0.03,
                    max_samples=min(20_000, len(normal_train)),
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(normal_train[cols].replace([np.inf, -np.inf], np.nan))
    valid_score = -pipeline.decision_function(valid[cols].replace([np.inf, -np.inf], np.nan))
    test_score = -pipeline.decision_function(test_preview[cols].replace([np.inf, -np.inf], np.nan))
    joblib.dump(pipeline, MODELS / "phase10_isolation_forest.joblib")
    return valid_score, test_score, pipeline


def fit_reconstruction_model(train: pd.DataFrame, valid: pd.DataFrame, test_preview: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, Pipeline, list[str]]:
    cols = select_autoencoder_features(train)
    normal_train = sample_normals(train, MAX_AUTOENCODER_NORMALS)
    x_fit = normal_train[cols].replace([np.inf, -np.inf], np.nan)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(64, 16, 64),
                    activation="relu",
                    solver="adam",
                    alpha=0.0005,
                    learning_rate_init=0.001,
                    max_iter=45,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=6,
                    random_state=RANDOM_STATE,
                    verbose=False,
                ),
            ),
        ]
    )
    transformed = pipeline[:-1].fit_transform(x_fit)
    pipeline[-1].fit(transformed, transformed)

    def reconstruction_error(frame: pd.DataFrame) -> np.ndarray:
        x = pipeline[:-1].transform(frame[cols].replace([np.inf, -np.inf], np.nan))
        reconstructed = pipeline[-1].predict(x)
        return np.mean(np.square(x - reconstructed), axis=1)

    valid_score = reconstruction_error(valid)
    test_score = reconstruction_error(test_preview)
    joblib.dump({"pipeline": pipeline, "features": cols}, MODELS / "phase10_mlp_reconstruction_anomaly.joblib")
    return valid_score, test_score, pipeline, cols


def station_sort_key(station: str) -> tuple[int, int]:
    station = str(station).replace("present_", "")
    match = re.match(r"L(\d+)_S(\d+)", str(station))
    if not match:
        return (999, 999)
    return int(match.group(1)), int(match.group(2))


def load_presence_for_ids(ids: pd.Series, split: str) -> pd.DataFrame:
    path = PROCESSED / f"phase5_{split}_station_presence_matrix.csv"
    id_set = set(ids.astype(int).tolist())
    chunks = []
    for chunk in pd.read_csv(path, chunksize=70_000):
        chunk = chunk[chunk["Id"].isin(id_set)]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        return pd.DataFrame({"Id": ids})
    frame = pd.concat(chunks, ignore_index=True)
    return pd.DataFrame({"Id": ids}).merge(frame, on="Id", how="left").fillna(0)


def station_columns(presence: pd.DataFrame) -> list[str]:
    return sorted(
        [c for c in presence.columns if re.match(r"(present_)?L\d+_S\d+", str(c))],
        key=station_sort_key,
    )


def load_station_scores() -> pd.DataFrame:
    scores = pd.read_csv(REPORTS / "phase9_critical_nodes.csv")
    cols = [
        "station",
        "critical_node_score",
        "bottleneck_score",
        "failure_lift",
        "station_shap_importance",
        "pagerank",
        "betweenness_centrality",
    ]
    scores = scores[[c for c in cols if c in scores.columns]].copy()
    for column in scores.columns:
        if column != "station":
            scores[column] = pd.to_numeric(scores[column], errors="coerce").fillna(0.0)
    return scores


def normalize_vector(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values.astype(float), nan=0.0)
    if values.size == 0:
        return np.array([], dtype=float)
    lo, hi = values.min(), values.max()
    if hi <= lo:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def build_message_passing_station_scores(stations: list[str], station_scores: pd.DataFrame) -> pd.DataFrame:
    edges = pd.read_csv(REPORTS / "phase8_process_map_edges.csv")
    graph = nx.DiGraph()
    for row in edges.itertuples(index=False):
        graph.add_edge(str(row.from_station), str(row.to_station), weight=float(row.transition_count))
    graph.add_nodes_from(stations)

    base = station_scores.set_index("station").reindex(stations).fillna(0.0)
    risk = normalize_vector(
        0.45 * base["critical_node_score"].to_numpy()
        + 0.25 * base["bottleneck_score"].to_numpy()
        + 0.20 * base["failure_lift"].to_numpy()
        + 0.10 * base["station_shap_importance"].to_numpy()
    )
    station_index = {station: idx for idx, station in enumerate(stations)}
    adjacency = np.zeros((len(stations), len(stations)), dtype=np.float32)
    for u, v, data in graph.edges(data=True):
        if u in station_index and v in station_index:
            adjacency[station_index[u], station_index[v]] += math.log1p(float(data.get("weight", 1.0)))
    row_sums = adjacency.sum(axis=1, keepdims=True)
    adjacency = np.divide(adjacency, row_sums, out=np.zeros_like(adjacency), where=row_sums > 0)
    first_hop = adjacency @ risk
    second_hop = adjacency @ first_hop
    propagated = normalize_vector(0.55 * risk + 0.30 * first_hop + 0.15 * second_hop)
    return pd.DataFrame(
        {
            "station": stations,
            "base_station_risk": risk,
            "one_hop_station_risk": first_hop,
            "two_hop_station_risk": second_hop,
            "message_passing_station_risk": propagated,
        }
    )


def build_graph_and_trajectory_features(presence: pd.DataFrame, station_risk: pd.DataFrame) -> pd.DataFrame:
    presence_columns = station_columns(presence)
    stations = [column.replace("present_", "") for column in presence_columns]
    station_risk = station_risk.set_index("station").reindex(stations).fillna(0.0)
    matrix = presence[presence_columns].astype(np.float32).to_numpy()
    present_counts = np.maximum(matrix.sum(axis=1), 1.0)
    risk = station_risk["message_passing_station_risk"].to_numpy(dtype=np.float32)
    base = station_risk["base_station_risk"].to_numpy(dtype=np.float32)

    weighted_risk = matrix @ risk
    weighted_base = matrix @ base
    max_risk = np.where(matrix.sum(axis=1) > 0, (matrix * risk).max(axis=1), 0.0)
    mean_risk = weighted_risk / present_counts
    risk_density = weighted_risk / np.maximum(np.sqrt(present_counts), 1.0)

    high_risk_threshold = float(np.quantile(risk, 0.80))
    high_risk_station_count = matrix[:, risk >= high_risk_threshold].sum(axis=1)
    early_weight = np.linspace(1.0, 0.2, len(stations), dtype=np.float32)
    late_weight = np.linspace(0.2, 1.0, len(stations), dtype=np.float32)
    early_exposure = matrix @ (risk * early_weight)
    late_exposure = matrix @ (risk * late_weight)
    trajectory_slope = late_exposure - early_exposure

    first_high_risk_position = []
    last_station_position = []
    for row in matrix:
        present_idx = np.flatnonzero(row > 0)
        last_station_position.append(float(present_idx.max() / max(len(stations) - 1, 1)) if len(present_idx) else 0.0)
        high_idx = np.flatnonzero((row > 0) & (risk >= high_risk_threshold))
        first_high_risk_position.append(float(high_idx.min() / max(len(stations) - 1, 1)) if len(high_idx) else 1.0)

    return pd.DataFrame(
        {
            "Id": presence["Id"].astype(int).to_numpy(),
            "graph_weighted_risk": weighted_risk,
            "graph_mean_risk": mean_risk,
            "graph_max_risk": max_risk,
            "graph_base_weighted_risk": weighted_base,
            "graph_risk_density": risk_density,
            "graph_high_risk_station_count": high_risk_station_count,
            "trajectory_early_exposure": early_exposure,
            "trajectory_late_exposure": late_exposure,
            "trajectory_slope": trajectory_slope,
            "trajectory_first_high_risk_position": first_high_risk_position,
            "trajectory_last_station_position": last_station_position,
        }
    )


def build_graph_feature_sets(train: pd.DataFrame, valid: pd.DataFrame, test_preview: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_presence = load_presence_for_ids(train["Id"], "train")
    valid_presence = load_presence_for_ids(valid["Id"], "train")
    test_presence = load_presence_for_ids(test_preview["Id"], "test")
    stations = [column.replace("present_", "") for column in station_columns(train_presence)]
    station_scores = load_station_scores()
    station_risk = build_message_passing_station_scores(stations, station_scores)
    station_risk.to_csv(REPORTS / "phase10_station_message_passing_risk.csv", index=False)
    train_graph = build_graph_and_trajectory_features(train_presence, station_risk)
    valid_graph = build_graph_and_trajectory_features(valid_presence, station_risk)
    test_graph = build_graph_and_trajectory_features(test_presence, station_risk)
    train_graph.to_csv(PROCESSED / "phase10_train_graph_trajectory_features.csv", index=False)
    valid_graph.to_csv(PROCESSED / "phase10_validation_graph_trajectory_features.csv", index=False)
    test_graph.to_csv(PROCESSED / "phase10_test_preview_graph_trajectory_features.csv", index=False)
    return train_graph, valid_graph, test_graph, station_risk


def fit_graph_message_model(train: pd.DataFrame, valid: pd.DataFrame, test_preview: pd.DataFrame, train_graph: pd.DataFrame, valid_graph: pd.DataFrame, test_graph: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, Pipeline]:
    graph_cols = [c for c in train_graph.columns if c != "Id" and c.startswith("graph_")]
    x_train = train[["Id", "Response"]].merge(train_graph, on="Id", how="left")
    x_valid = valid[["Id", "Response"]].merge(valid_graph, on="Id", how="left")
    x_test = test_preview[["Id"]].merge(test_graph, on="Id", how="left")
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=600, random_state=RANDOM_STATE)),
        ]
    )
    pipeline.fit(x_train[graph_cols], x_train["Response"].astype(int))
    valid_score = pipeline.predict_proba(x_valid[graph_cols])[:, 1]
    test_score = pipeline.predict_proba(x_test[graph_cols])[:, 1]
    joblib.dump({"pipeline": pipeline, "features": graph_cols}, MODELS / "phase10_graph_message_passing_model.joblib")
    return valid_score, test_score, pipeline


def fit_failure_trajectory_model(train: pd.DataFrame, valid: pd.DataFrame, test_preview: pd.DataFrame, train_graph: pd.DataFrame, valid_graph: pd.DataFrame, test_graph: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, LGBMClassifier, list[str]]:
    trajectory_cols = [c for c in train_graph.columns if c != "Id"]
    phase6_context = [
        "start_time",
        "end_time",
        "cycle_time",
        "processing_duration",
        "waiting_time",
        "mean_waiting_time",
        "max_waiting_time",
        "station_count",
        "line_count",
        "path_complexity_score",
        "final_product_family",
    ]
    context_cols = [c for c in phase6_context if c in train.columns]
    train_model = train[["Id", "Response", *context_cols]].merge(train_graph, on="Id", how="left")
    valid_model = valid[["Id", "Response", *context_cols]].merge(valid_graph, on="Id", how="left")
    test_model = test_preview[["Id", *context_cols]].merge(test_graph, on="Id", how="left")
    model_cols = context_cols + trajectory_cols
    pos = int(train_model["Response"].sum())
    neg = int(len(train_model) - pos)
    model = LGBMClassifier(
        n_estimators=220,
        learning_rate=0.045,
        num_leaves=32,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
        scale_pos_weight=neg / max(1, pos),
    )
    x_train = train_model[model_cols].replace([np.inf, -np.inf], np.nan)
    x_valid = valid_model[model_cols].replace([np.inf, -np.inf], np.nan)
    x_test = test_model[model_cols].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    model.fit(imputer.fit_transform(x_train), train_model["Response"].astype(int))
    valid_score = model.predict_proba(imputer.transform(x_valid))[:, 1]
    test_score = model.predict_proba(imputer.transform(x_test))[:, 1]
    joblib.dump({"imputer": imputer, "model": model, "features": model_cols}, MODELS / "phase10_failure_trajectory_model.joblib")
    return valid_score, test_score, model, model_cols


def build_score_outputs(
    valid: pd.DataFrame,
    test_preview: pd.DataFrame,
    scores: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_scores = valid[["Id", "Response"]].copy()
    test_scores = test_preview[["Id"]].copy()
    for name, (valid_score, test_score) in scores.items():
        valid_scores[name] = valid_score
        test_scores[name] = test_score
    valid_scores.to_csv(PROCESSED / "phase10_validation_advanced_ai_scores.csv", index=False)
    test_scores.to_csv(PROCESSED / "phase10_test_preview_advanced_ai_scores.csv", index=False)
    return valid_scores, test_scores


def create_figures(metrics: pd.DataFrame, valid_scores: pd.DataFrame) -> None:
    setup_plotting()
    FIGURES.mkdir(parents=True, exist_ok=True)

    metric_order = metrics.sort_values("mcc", ascending=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    sns.barplot(data=metric_order, y="model", x="mcc", color=TOKENS["blue_dark"], ax=ax)
    ax.set_xlabel("Best validation MCC")
    ax.set_ylabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=4, fontsize=8, color=TOKENS["ink"])
    add_chart_header(
        fig,
        ax,
        "Phase 10 Advanced AI Validation MCC",
        "Thresholds chosen on validation score quantiles; production-safe Phase 6 LightGBM is shown as reference.",
    )
    fig.savefig(FIGURES / "phase10_advanced_ai_mcc_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    long_scores = valid_scores.melt(id_vars=["Id", "Response"], var_name="score_name", value_name="score")
    plot_scores = long_scores[
        long_scores["score_name"].isin(
            ["isolation_forest_anomaly_score", "mlp_reconstruction_anomaly_score", "trajectory_failure_risk"]
        )
    ].copy()
    plot_scores["plot_score"] = plot_scores.groupby("score_name")["score"].transform(
        lambda s: s.clip(lower=s.quantile(0.01), upper=s.quantile(0.99))
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=False)
    for ax, (score_name, frame) in zip(axes, plot_scores.groupby("score_name")):
        sns.kdeplot(data=frame, x="plot_score", hue="Response", common_norm=False, fill=True, alpha=0.35, ax=ax, palette=[TOKENS["muted"], TOKENS["orange_dark"]])
        ax.set_title(score_name.replace("_", " "), fontsize=10, color=TOKENS["ink"])
        ax.set_xlabel("Display score, p1-p99 clipped")
        ax.set_ylabel("Density")
    fig.subplots_adjust(top=0.76)
    fig.text(0.06, 0.98, "Advanced AI Score Distributions", ha="left", va="top", fontsize=14, fontweight="semibold", color=TOKENS["ink"])
    fig.text(0.06, 0.92, "Failures should shift toward higher scores when the method separates risk effectively.", ha="left", va="top", fontsize=9, color=TOKENS["muted"])
    fig.savefig(FIGURES / "phase10_advanced_ai_score_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_report(metrics: pd.DataFrame, station_risk: pd.DataFrame, trajectory_features: list[str]) -> None:
    phase6_metrics = pd.read_csv(REPORTS / "phase6_model_comparison_metrics.csv")
    phase6_best = phase6_metrics.sort_values("mcc", ascending=False).iloc[0]
    best_phase10 = metrics[metrics["model"].ne("Phase 6 LightGBM reference")].sort_values("mcc", ascending=False).iloc[0]
    top_station_risk = station_risk.sort_values("message_passing_station_risk", ascending=False).head(8)

    report = f"""# Phase 10: Advanced AI

## Executive Summary

Phase 10 adds advanced diagnostic AI around the production-safe Phase 6 model. The best Phase 10 experiment is **{best_phase10['model']}** with validation MCC **{best_phase10['mcc']:.4f}** and PR-AUC **{best_phase10['pr_auc']:.4f}**. The official production-safe benchmark remains **{phase6_best['model']}** from Phase 6 with MCC **{phase6_best['mcc']:.4f}**.

These experiments are useful as engineering intelligence layers:

- **Isolation Forest** flags products whose numeric/categorical/date pattern looks unusual compared with normal products.
- **MLP reconstruction anomaly detection** finds products that a normal-pattern reconstruction model cannot reproduce well.
- **Graph message passing** spreads station risk through the production flow graph to estimate route-level risk exposure.
- **Failure trajectory prediction** combines ordered station exposure, timing, waiting, path, and product-family context to predict risk along the manufacturing path.

## Validation Results

{metrics.to_markdown(index=False, floatfmt='.4f')}

## Top Propagated Station Risks

{top_station_risk.to_markdown(index=False, floatfmt='.4f')}

## Interpretation For Manufacturing Teams

The anomaly models should be treated as early-warning and triage tools. A high anomaly score means the product path or measurements look different from normal production history; it does not automatically prove a defect cause. The graph and trajectory models are closer to process intelligence: they show whether a product crossed stations that previous phases identified as central, bottlenecked, failure-associated, or downstream of risky stations.

## Recommended Use

1. Keep **Phase 6 LightGBM** as the main production-safe failure classifier.
2. Use **failure trajectory risk** as a second operational score for routing products to extra inspection.
3. Use **graph message-passing station risk** to explain whether a product's path crossed critical process nodes.
4. Use **Isolation Forest** and **MLP reconstruction error** for anomaly monitoring, alerting, and investigation queues.
5. Do not use these methods as causal proof without engineer review, sensor validation, and controlled process evidence.

## Outputs

- Validation score file: `data/processed/phase10_validation_advanced_ai_scores.csv`
- Test preview score file: `data/processed/phase10_test_preview_advanced_ai_scores.csv`
- Model comparison: `reports/phase10_advanced_ai_model_comparison.csv`
- Station message-passing risk: `reports/phase10_station_message_passing_risk.csv`
- Graph/trajectory feature sets: `data/processed/phase10_*_graph_trajectory_features.csv`
- Notebook: `notebooks/phase10_advanced_ai.ipynb`

## Caveats

- PyTorch/TensorFlow are not installed in the project environment, so the autoencoder is implemented as an MLP reconstruction-error model with scikit-learn.
- The graph neural network step is a lightweight graph message-passing experiment, not a full PyTorch Geometric GNN. It is appropriate for a portfolio-grade process graph prototype without adding a heavy deep-learning dependency.
- Test advanced-AI scores are produced for the existing Phase 6 test preview feature file. The validation metrics are the reliable comparison point because the Kaggle test labels are not available.
- The score levels are diagnostic and observational. They should be calibrated with real factory feedback before being used as production decision thresholds.
"""
    (REPORTS / "phase10_advanced_ai_report.md").write_text(report, encoding="utf-8")


def create_notebook() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Phase 10: Advanced AI\n",
                "\n",
                "This notebook runs the advanced AI phase for the Bosch production-line project. It covers Isolation Forest anomaly detection, MLP reconstruction-error anomaly detection, graph message-passing risk, and failure trajectory prediction.\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Why This Phase Exists\n",
                "\n",
                "Phase 6 remains the production-safe failure model. Phase 10 adds complementary AI layers for anomaly monitoring, process graph intelligence, and trajectory-style risk scoring. These methods help engineers prioritize investigation, but they should not be treated as causal proof by themselves.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "\n",
                "ROOT = Path.cwd()\n",
                "REPORTS = ROOT / 'reports'\n",
                "PROCESSED = ROOT / 'data' / 'processed'\n",
                "FIGURES = REPORTS / 'figures'\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Run Phase 10 Pipeline\n",
                "\n",
                "The script reads the Phase 6 validation feature matrix, Phase 5 station presence matrix, and Phase 9 knowledge graph outputs. It then trains the advanced AI experiments and writes reports, charts, models, and score files.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%run src/data/phase10_advanced_ai.py\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Compare Advanced AI Models\n",
                "\n",
                "MCC is emphasized because the Bosch target is highly imbalanced. PR-AUC is also useful because it measures whether failures are ranked above normal products.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "metrics = pd.read_csv(REPORTS / 'phase10_advanced_ai_model_comparison.csv')\n",
                "metrics.sort_values('mcc', ascending=False)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from IPython.display import Image, display\n",
                "display(Image(filename=str(FIGURES / 'phase10_advanced_ai_mcc_comparison.png')))\n",
                "display(Image(filename=str(FIGURES / 'phase10_advanced_ai_score_distributions.png')))\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Inspect Station Message-Passing Risk\n",
                "\n",
                "The graph step starts from Phase 9 critical-node scores and propagates station risk through the process map. This provides a graph-aware station risk score without requiring heavy GNN dependencies.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "station_risk = pd.read_csv(REPORTS / 'phase10_station_message_passing_risk.csv')\n",
                "station_risk.sort_values('message_passing_station_risk', ascending=False).head(15)\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Review Product-Level Scores\n",
                "\n",
                "These validation scores can be joined back to product IDs for root-cause review, process-mining overlays, or dashboarding in the next phases.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "scores = pd.read_csv(PROCESSED / 'phase10_validation_advanced_ai_scores.csv')\n",
                "scores.head()\n",
            ],
        },
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NOTEBOOKS / "phase10_advanced_ai.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def update_readme() -> None:
    readme_path = ROOT / "README.md"
    if not readme_path.exists():
        return
    readme = readme_path.read_text(encoding="utf-8")
    marker = "## Phase 10 - Advanced AI"
    section = """## Phase 10 - Advanced AI

Phase 10 adds advanced diagnostic AI layers around the production-safe Phase 6 classifier:

- Isolation Forest anomaly detection.
- MLP reconstruction-error anomaly detection.
- Graph message-passing station risk using the Phase 9 knowledge graph.
- Failure trajectory prediction using station, timing, waiting, path, and family features.

Key files:

- `src/data/phase10_advanced_ai.py`
- `notebooks/phase10_advanced_ai.ipynb`
- `reports/phase10_advanced_ai_report.md`
- `reports/phase10_advanced_ai_model_comparison.csv`
- `reports/phase10_station_message_passing_risk.csv`
- `data/processed/phase10_validation_advanced_ai_scores.csv`
- `data/processed/phase10_test_preview_advanced_ai_scores.csv`

Important interpretation: Phase 10 scores are diagnostic add-ons. The production-safe Phase 6 LightGBM model remains the main failure prediction model unless future factory validation proves one of these advanced methods is more reliable.

"""
    if marker in readme:
        before = readme.split(marker)[0].rstrip()
        readme = before + "\n\n" + section
    else:
        readme = readme.rstrip() + "\n\n" + section
    readme_path.write_text(readme, encoding="utf-8")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    require_inputs()

    train, valid, test_preview = load_phase6_frames()
    y_valid = valid["Response"].astype(int)

    isolation_valid, isolation_test, _ = fit_isolation_forest(train, valid, test_preview)
    reconstruction_valid, reconstruction_test, _, autoencoder_features = fit_reconstruction_model(train, valid, test_preview)

    train_graph, valid_graph, test_graph, station_risk = build_graph_feature_sets(train, valid, test_preview)
    graph_valid, graph_test, _ = fit_graph_message_model(train, valid, test_preview, train_graph, valid_graph, test_graph)
    trajectory_valid, trajectory_test, _, trajectory_features = fit_failure_trajectory_model(
        train, valid, test_preview, train_graph, valid_graph, test_graph
    )

    phase6 = pd.read_csv(REPORTS / "phase6_model_comparison_metrics.csv").sort_values("mcc", ascending=False).iloc[0]
    rows = [
        metric_row("Isolation Forest anomaly detection", y_valid, isolation_valid, "Unsupervised model trained on normal products only."),
        metric_row("MLP reconstruction anomaly detection", y_valid, reconstruction_valid, f"Scikit-learn reconstruction model using {len(autoencoder_features)} selected features."),
        metric_row("Graph message-passing risk model", y_valid, graph_valid, "Logistic model using graph-propagated station risk exposure."),
        metric_row("Failure trajectory prediction", y_valid, trajectory_valid, "LightGBM model using timing, path, family, and ordered station-risk exposure."),
        {
            "model": "Phase 6 LightGBM reference",
            "threshold": float(phase6.get("threshold", np.nan)),
            "mcc": float(phase6["mcc"]),
            "precision": float(phase6["precision"]),
            "recall": float(phase6["recall"]),
            "f1": float(phase6["f1"]),
            "pr_auc": float(phase6["pr_auc"]),
            "notes": "Production-safe supervised benchmark from Phase 6.",
        },
    ]
    metrics = pd.DataFrame(rows).sort_values("mcc", ascending=False)
    metrics.to_csv(REPORTS / "phase10_advanced_ai_model_comparison.csv", index=False)

    valid_scores, test_scores = build_score_outputs(
        valid,
        test_preview,
        {
            "isolation_forest_anomaly_score": (isolation_valid, isolation_test),
            "mlp_reconstruction_anomaly_score": (reconstruction_valid, reconstruction_test),
            "graph_message_passing_failure_risk": (graph_valid, graph_test),
            "trajectory_failure_risk": (trajectory_valid, trajectory_test),
        },
    )
    create_figures(metrics, valid_scores)
    create_report(metrics, station_risk, trajectory_features)
    create_notebook()
    update_readme()

    print("Phase 10 complete.")
    print(metrics.to_string(index=False))
    print(f"Wrote {REPORTS / 'phase10_advanced_ai_report.md'}")
    print(f"Wrote {NOTEBOOKS / 'phase10_advanced_ai.ipynb'}")


if __name__ == "__main__":
    main()
