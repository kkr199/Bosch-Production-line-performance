"""Production-safe model improvement experiments for Bosch failure prediction.

This script extends Phase 6 without overwriting the accepted baseline. It tests:

1. Tuned LightGBM on the original Phase 6 feature matrix.
2. Tuned LightGBM with Phase 10 graph/trajectory features.
3. Product-family-aware LightGBM models.
4. A validation-optimized score blend for research comparison.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, precision_score, recall_score


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"
NOTEBOOKS = ROOT / "notebooks"

RANDOM_STATE = 42

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
    "blue_dark": "#2E4780",
    "orange_dark": "#804126",
    "olive_dark": "#386411",
}


LGBM_CONFIGS = [
    {
        "name": "balanced_compact",
        "n_estimators": 260,
        "learning_rate": 0.035,
        "num_leaves": 40,
        "min_child_samples": 60,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
    },
    {
        "name": "balanced_wide",
        "n_estimators": 320,
        "learning_rate": 0.025,
        "num_leaves": 64,
        "min_child_samples": 45,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.05,
        "reg_lambda": 1.5,
    },
    {
        "name": "deeper_regularized",
        "n_estimators": 300,
        "learning_rate": 0.025,
        "num_leaves": 96,
        "min_child_samples": 80,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.2,
        "reg_lambda": 3.0,
    },
    {
        "name": "high_recall",
        "n_estimators": 300,
        "learning_rate": 0.03,
        "num_leaves": 72,
        "min_child_samples": 35,
        "subsample": 0.85,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.02,
        "reg_lambda": 1.0,
    },
]


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
        PROCESSED / "phase10_train_graph_trajectory_features.csv",
        PROCESSED / "phase10_validation_graph_trajectory_features.csv",
        PROCESSED / "phase10_test_preview_graph_trajectory_features.csv",
        PROCESSED / "phase10_validation_advanced_ai_scores.csv",
        REPORTS / "phase6_model_comparison_metrics.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required prior phase artifacts: " + ", ".join(missing))


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(PROCESSED / "phase6_train_dataset.csv")
    valid = pd.read_csv(PROCESSED / "phase6_validation_dataset.csv")
    test = pd.read_csv(PROCESSED / "phase6_test_dataset_preview.csv")
    return train, valid, test


def add_graph_features(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    graph_path = PROCESSED / f"phase10_{split}_graph_trajectory_features.csv"
    graph = pd.read_csv(graph_path)
    return frame.merge(graph, on="Id", how="left")


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in {"Id", "Response"}]


def sanitize(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[columns].replace([np.inf, -np.inf], np.nan)


def best_threshold_metrics(y_true: pd.Series, scores: np.ndarray) -> dict[str, float]:
    y = y_true.astype(int).to_numpy()
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=np.nanmedian(scores))
    thresholds = np.unique(np.quantile(scores, np.linspace(0.50, 0.997, 150)))
    best = None
    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
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
    best["pr_auc"] = float(average_precision_score(y, scores))
    return best


def fit_lgbm(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    config: dict[str, float | int | str],
) -> tuple[LGBMClassifier, np.ndarray, np.ndarray, list[str]]:
    features = feature_columns(train)
    y_train = train["Response"].astype(int)
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    params = {key: value for key, value in config.items() if key != "name"}
    model = LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
        scale_pos_weight=neg / max(1, pos),
        **params,
    )
    model.fit(
        sanitize(train, features),
        y_train,
        eval_set=[(sanitize(valid, features), valid["Response"].astype(int))],
        eval_metric="average_precision",
        callbacks=[early_stopping(35, verbose=False), log_evaluation(0)],
    )
    valid_score = model.predict_proba(sanitize(valid, features))[:, 1]
    test_score = model.predict_proba(sanitize(test, features))[:, 1]
    return model, valid_score, test_score, features


def tune_lgbm(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, experiment: str) -> tuple[pd.DataFrame, dict]:
    rows = []
    best = None
    for config in LGBM_CONFIGS:
        print(f"Training {experiment}: {config['name']}", flush=True)
        model, valid_score, test_score, features = fit_lgbm(train, valid, test, config)
        metrics = best_threshold_metrics(valid["Response"], valid_score)
        row = {
            "experiment": experiment,
            "config_name": config["name"],
            **metrics,
            "feature_count": len(features),
        }
        rows.append(row)
        candidate = {
            "experiment": experiment,
            "config": config,
            "model": model,
            "valid_score": valid_score,
            "test_score": test_score,
            "features": features,
            "metrics": row,
        }
        if best is None or row["mcc"] > best["metrics"]["mcc"]:
            best = candidate
        print(
            f"Finished {experiment}: {config['name']} MCC={row['mcc']:.4f} PR-AUC={row['pr_auc']:.4f}",
            flush=True,
        )
    assert best is not None
    return pd.DataFrame(rows), best


def fit_family_models(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, fallback: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    features = feature_columns(train)
    if "final_product_family" not in train.columns:
        raise ValueError("Family-aware modeling needs final_product_family in the feature matrix.")
    valid_scores = fallback["valid_score"].copy()
    test_scores = fallback["test_score"].copy()
    family_rows = []
    models = {}
    best_config = fallback["config"]
    for family, train_family in train.groupby("final_product_family"):
        valid_mask = valid["final_product_family"].eq(family).to_numpy()
        test_mask = test["final_product_family"].eq(family).to_numpy() if "final_product_family" in test.columns else np.zeros(len(test), dtype=bool)
        positives = int(train_family["Response"].sum())
        if len(train_family) < 2_500 or positives < 20 or not valid_mask.any():
            family_rows.append(
                {
                    "family": family,
                    "status": "fallback_global",
                    "train_rows": len(train_family),
                    "train_failures": positives,
                    "valid_rows": int(valid_mask.sum()),
                }
            )
            continue
        params = {key: value for key, value in best_config.items() if key != "name"}
        y_family = train_family["Response"].astype(int)
        pos = int(y_family.sum())
        neg = int(len(y_family) - pos)
        model = LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
            scale_pos_weight=neg / max(1, pos),
            **params,
        )
        model.fit(
            sanitize(train_family, features),
            y_family,
            eval_set=[(sanitize(valid.loc[valid_mask], features), valid.loc[valid_mask, "Response"].astype(int))],
            eval_metric="average_precision",
            callbacks=[early_stopping(25, verbose=False), log_evaluation(0)],
        )
        valid_scores[valid_mask] = model.predict_proba(sanitize(valid.loc[valid_mask], features))[:, 1]
        if test_mask.any():
            test_scores[test_mask] = model.predict_proba(sanitize(test.loc[test_mask], features))[:, 1]
        models[int(family)] = model
        family_metrics = best_threshold_metrics(valid.loc[valid_mask, "Response"], valid_scores[valid_mask])
        family_rows.append(
            {
                "family": family,
                "status": "family_model",
                "train_rows": len(train_family),
                "train_failures": positives,
                "valid_rows": int(valid_mask.sum()),
                **family_metrics,
            }
        )
    family_report = pd.DataFrame(family_rows).sort_values(["status", "train_rows"], ascending=[False, False])
    bundle = {"models": models, "fallback_model": fallback["model"], "features": features, "fallback_config": best_config}
    return family_report, valid_scores, test_scores, bundle


def optimize_blend(y_true: pd.Series, score_map: dict[str, np.ndarray]) -> tuple[dict[str, float], np.ndarray, dict[str, float]]:
    names = list(score_map)
    normalized = {}
    for name, scores in score_map.items():
        values = np.asarray(scores, dtype=float)
        lo, hi = np.nanpercentile(values, 1), np.nanpercentile(values, 99)
        clipped = np.clip(values, lo, hi)
        normalized[name] = (clipped - clipped.min()) / max(clipped.max() - clipped.min(), 1e-9)
    weight_grid = [0.0, 0.15, 0.30, 0.50, 0.70, 0.85, 1.0]
    best = None
    for weights in itertools.product(weight_grid, repeat=len(names)):
        total = sum(weights)
        if total <= 0:
            continue
        if abs(total - 1.0) > 0.001:
            continue
        blended = np.zeros(len(y_true), dtype=float)
        weight_map = {}
        for name, weight in zip(names, weights):
            blended += weight * normalized[name]
            weight_map[name] = float(weight)
        metrics = best_threshold_metrics(y_true, blended)
        if best is None or metrics["mcc"] > best["metrics"]["mcc"]:
            best = {"weights": weight_map, "scores": blended, "metrics": metrics}
    assert best is not None
    return best["weights"], best["scores"], best["metrics"]


def save_predictions(
    valid: pd.DataFrame,
    test: pd.DataFrame,
    original_best: dict,
    enhanced_best: dict,
    family_valid: np.ndarray,
    family_test: np.ndarray,
    blend_scores: np.ndarray,
) -> None:
    valid_out = valid[["Id", "Response"]].copy()
    valid_out["improved_original_lgbm"] = original_best["valid_score"]
    valid_out["improved_enhanced_lgbm"] = enhanced_best["valid_score"]
    valid_out["family_aware_lgbm"] = family_valid
    valid_out["validation_optimized_blend"] = blend_scores
    valid_out.to_csv(PROCESSED / "phase6_improvement_validation_scores.csv", index=False)

    test_out = test[["Id"]].copy()
    test_out["improved_original_lgbm"] = original_best["test_score"]
    test_out["improved_enhanced_lgbm"] = enhanced_best["test_score"]
    test_out["family_aware_lgbm"] = family_test
    test_out.to_csv(PROCESSED / "phase6_improvement_test_preview_scores.csv", index=False)


def make_figures(summary: pd.DataFrame, tuning: pd.DataFrame) -> None:
    setup_plotting()
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_summary = summary.sort_values("mcc", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    sns.barplot(data=plot_summary, y="model", x="mcc", color=TOKENS["blue_dark"], ax=ax)
    ax.set_xlabel("Validation MCC")
    ax.set_ylabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=4, fontsize=8, color=TOKENS["ink"])
    add_chart_header(
        fig,
        ax,
        "Model Improvement Validation MCC",
        "Production-safe candidates compared with the original Phase 6 benchmark; blend is validation-optimized research.",
    )
    fig.savefig(FIGURES / "phase6_improvement_mcc_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    sns.scatterplot(data=tuning, x="recall", y="precision", hue="experiment", s=120, ax=ax)
    for row in tuning.itertuples(index=False):
        ax.annotate(
            f"{row.mcc:.3f}",
            (row.recall, row.precision),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
            color=TOKENS["ink"],
        )
    ax.set_xlabel("Recall at best MCC threshold")
    ax.set_ylabel("Precision at best MCC threshold")
    add_chart_header(
        fig,
        ax,
        "Precision and Recall Trade-Off Across Tuning Runs",
        "Each point is one LightGBM hyperparameter configuration evaluated on the same validation set.",
    )
    fig.savefig(FIGURES / "phase6_improvement_precision_recall_tradeoff.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, tuning: pd.DataFrame, family_report: pd.DataFrame, blend_weights: dict[str, float]) -> None:
    production = summary[summary["production_safe"].eq(True)].sort_values("mcc", ascending=False)
    best_production = production.iloc[0]
    baseline = summary[summary["model"].eq("Phase 6 LightGBM baseline")].iloc[0]
    lift = best_production["mcc"] - baseline["mcc"]
    if best_production["model"] == "Phase 6 LightGBM baseline":
        summary_text = (
            f"No production-safe improvement candidate beat the accepted Phase 6 LightGBM baseline. "
            f"The official model should remain **Phase 6 LightGBM** with validation MCC **{baseline['mcc']:.4f}**."
        )
    else:
        summary_text = (
            f"The best production-safe improvement candidate is **{best_production['model']}** with validation MCC "
            f"**{best_production['mcc']:.4f}**, compared with the original Phase 6 LightGBM MCC **{baseline['mcc']:.4f}**. "
            f"The absolute MCC change is **{lift:+.4f}**."
        )
    summary_display = summary.replace({np.nan: ""})
    tuning_display = tuning.sort_values("mcc", ascending=False).replace({np.nan: ""})
    family_display = family_report.replace({np.nan: ""})
    report = f"""# Phase 6 Model Improvement Report

## Technical Summary

{summary_text}

The validation-optimized blend is included as a research upper bound, not as the official production model, because the blending weights were selected on the same validation set used for reporting.

## Key Findings

{summary_display.to_markdown(index=False, floatfmt='.4f')}

## Scope And Metric Definitions

- Cohort: existing Phase 6 train/validation split built from raw Bosch numeric, categorical, and date inputs.
- Target: `Response`, where `1` means product failure.
- Primary metric: Matthews correlation coefficient (MCC), selected because the failure rate is highly imbalanced.
- Supporting metrics: precision, recall, F1, and PR-AUC.
- Thresholding: each model is evaluated at the validation threshold that maximizes MCC.

## Methodology

The improvement phase tested three production-safe modeling paths:

1. Hyperparameter tuning of LightGBM on the original Phase 6 feature matrix.
2. Hyperparameter tuning of LightGBM after adding Phase 10 graph and trajectory features.
3. Family-aware LightGBM models where product-family sample size and failure count were sufficient; smaller families fall back to the global model.

The research blend combines normalized validation scores from the strongest candidates and Phase 10 trajectory risk. It is useful for estimating remaining headroom, but it should be validated with cross-validation or a fresh holdout before becoming an official model.

## LightGBM Tuning Runs

{tuning_display.to_markdown(index=False, floatfmt='.4f')}

## Family-Aware Model Diagnostics

{family_display.to_markdown(index=False, floatfmt='.4f')}

## Validation-Optimized Blend Weights

{pd.DataFrame([blend_weights]).to_markdown(index=False, floatfmt='.4f')}

## Limitations And Robustness

- This is still a single train/validation split, so small improvements should be treated carefully.
- The validation-optimized blend is intentionally labeled as research because it tunes weights on validation.
- Family-specific models can become unstable for low-volume or low-failure families; fallback rules are used to control that risk.
- The production-safe candidates do not use the leaderboard-style nearby-label/order-leak features.

## Recommended Next Steps

Use the best production-safe candidate only if it improves MCC materially over the Phase 6 baseline. If the lift is small or negative, keep the original Phase 6 LightGBM as the official model and treat the improvement artifacts as tuning evidence. For a stronger final push, run repeated cross-validation or a time-style holdout and then choose one stable model.
"""
    (REPORTS / "phase6_model_improvement_report.md").write_text(report, encoding="utf-8")


def write_notebook() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Phase 6 Model Improvement\n",
                "\n",
                "This notebook reruns the production-safe model improvement experiments: LightGBM tuning, graph/trajectory feature enhancement, family-aware modeling, and a validation-optimized blend comparison.\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Run Improvement Pipeline\n",
                "\n",
                "The script does not overwrite the accepted Phase 6 model. It writes separate reports, score files, figures, and model bundles for auditability.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["%run src/data/phase6_model_improvement.py\n"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "from IPython.display import Image, display\n",
                "ROOT = Path.cwd()\n",
                "summary = pd.read_csv(ROOT / 'reports' / 'phase6_model_improvement_summary.csv')\n",
                "summary.sort_values('mcc', ascending=False)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'phase6_improvement_mcc_comparison.png')))\n",
                "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'phase6_improvement_precision_recall_tradeoff.png')))\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Interpretation\n",
                "\n",
                "If the best production-safe candidate beats the Phase 6 baseline by a meaningful margin, it can become the new official model. If the lift is small, keep Phase 6 as the official model and use this notebook as evidence that tuning was attempted responsibly.\n",
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
    (NOTEBOOKS / "phase6_model_improvement.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def update_readme() -> None:
    readme_path = ROOT / "README.md"
    if not readme_path.exists():
        return
    readme = readme_path.read_text(encoding="utf-8")
    section = """## Phase 6 Model Improvement

Additional production-safe improvement experiments were added after Phase 10:

- Tuned LightGBM on the original Phase 6 feature matrix.
- Tuned LightGBM with Phase 10 graph and trajectory features.
- Product-family-aware LightGBM models with fallback rules.
- Validation-optimized blend as a research upper bound.

Key files:

- `src/data/phase6_model_improvement.py`
- `notebooks/phase6_model_improvement.ipynb`
- `reports/phase6_model_improvement_report.md`
- `reports/phase6_model_improvement_summary.csv`
- `data/processed/phase6_improvement_validation_scores.csv`

"""
    marker = "## Phase 6 Model Improvement"
    if marker in readme:
        readme = readme.split(marker)[0].rstrip() + "\n\n" + section
    else:
        readme = readme.rstrip() + "\n\n" + section
    readme_path.write_text(readme, encoding="utf-8")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    require_inputs()

    train, valid, test = load_frames()
    train_enhanced = add_graph_features(train, "train")
    valid_enhanced = add_graph_features(valid, "validation")
    test_enhanced = add_graph_features(test, "test_preview")

    tuning_original, best_original = tune_lgbm(train, valid, test, "original_phase6_features")
    tuning_enhanced, best_enhanced = tune_lgbm(train_enhanced, valid_enhanced, test_enhanced, "phase6_plus_graph_trajectory")
    tuning = pd.concat([tuning_original, tuning_enhanced], ignore_index=True)
    tuning.to_csv(REPORTS / "phase6_model_improvement_tuning_runs.csv", index=False)

    family_report, family_valid, family_test, family_bundle = fit_family_models(train_enhanced, valid_enhanced, test_enhanced, best_enhanced)
    family_report.to_csv(REPORTS / "phase6_family_model_improvement_diagnostics.csv", index=False)
    family_metrics = best_threshold_metrics(valid_enhanced["Response"], family_valid)

    phase10_scores = pd.read_csv(PROCESSED / "phase10_validation_advanced_ai_scores.csv")
    valid_scores = valid[["Id", "Response"]].merge(phase10_scores, on=["Id", "Response"], how="left")
    blend_weights, blend_score, blend_metrics = optimize_blend(
        valid["Response"],
        {
            "best_original_lgbm": best_original["valid_score"],
            "best_enhanced_lgbm": best_enhanced["valid_score"],
            "family_aware_lgbm": family_valid,
            "trajectory_failure_risk": valid_scores["trajectory_failure_risk"].fillna(valid_scores["trajectory_failure_risk"].median()).to_numpy(),
        },
    )

    phase6 = pd.read_csv(REPORTS / "phase6_model_comparison_metrics.csv")
    phase6_best = phase6.sort_values("mcc", ascending=False).iloc[0]
    summary_rows = [
        {
            "model": "Phase 6 LightGBM baseline",
            "production_safe": True,
            "threshold": float(phase6_best.get("threshold", np.nan)),
            "mcc": float(phase6_best["mcc"]),
            "precision": float(phase6_best["precision"]),
            "recall": float(phase6_best["recall"]),
            "f1": float(phase6_best["f1"]),
            "pr_auc": float(phase6_best["pr_auc"]),
            "notes": "Accepted Phase 6 production-safe benchmark.",
        },
        {
            "model": "Tuned LightGBM - original features",
            "production_safe": True,
            **best_original["metrics"],
            "notes": f"Best config: {best_original['config']['name']}.",
        },
        {
            "model": "Tuned LightGBM - graph trajectory features",
            "production_safe": True,
            **best_enhanced["metrics"],
            "notes": f"Best config: {best_enhanced['config']['name']}.",
        },
        {
            "model": "Family-aware LightGBM",
            "production_safe": True,
            **family_metrics,
            "notes": "Family-specific models where sample size allows; global fallback otherwise.",
        },
        {
            "model": "Validation-optimized blend",
            "production_safe": False,
            **blend_metrics,
            "notes": "Research upper bound; weights selected on validation.",
        },
    ]
    summary = pd.DataFrame(summary_rows).sort_values("mcc", ascending=False)
    summary.to_csv(REPORTS / "phase6_model_improvement_summary.csv", index=False)
    pd.DataFrame([blend_weights]).to_csv(REPORTS / "phase6_model_improvement_blend_weights.csv", index=False)

    save_predictions(valid, test, best_original, best_enhanced, family_valid, family_test, blend_score)
    joblib.dump(
        {
            "best_original": {"model": best_original["model"], "features": best_original["features"], "config": best_original["config"]},
            "best_enhanced": {"model": best_enhanced["model"], "features": best_enhanced["features"], "config": best_enhanced["config"]},
            "family_bundle": family_bundle,
            "blend_weights": blend_weights,
        },
        MODELS / "phase6_model_improvement_bundle.joblib",
    )

    make_figures(summary, tuning)
    write_report(summary, tuning, family_report, blend_weights)
    write_notebook()
    update_readme()

    print("Phase 6 model improvement complete.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
