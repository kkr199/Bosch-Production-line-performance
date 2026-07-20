"""Small, dependency-light controls for reproducible ML release checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitManifest:
    """Audit record for a split made before target-informed feature work."""

    protocol_version: str
    seed: int
    train_rows: int
    validation_rows: int
    holdout_rows: int
    train_positive_rate: float
    validation_positive_rate: float
    holdout_positive_rate: float
    source_sha256: str

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading a dataset into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_split_manifest(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    holdout: pd.DataFrame,
    source_path: Path,
    seed: int,
    target_column: str = "Response",
) -> SplitManifest:
    """Create an auditable manifest for the three isolated labelled partitions."""
    partitions: Iterable[pd.DataFrame] = (train, validation, holdout)
    if any(target_column not in frame.columns for frame in partitions):
        raise ValueError(f"All partitions must contain the target column {target_column!r}.")
    if train.empty or validation.empty or holdout.empty:
        raise ValueError("Train, validation, and holdout partitions must all contain rows.")

    return SplitManifest(
        protocol_version="industry-grade-ml-v1",
        seed=seed,
        train_rows=len(train),
        validation_rows=len(validation),
        holdout_rows=len(holdout),
        train_positive_rate=float(train[target_column].mean()),
        validation_positive_rate=float(validation[target_column].mean()),
        holdout_positive_rate=float(holdout[target_column].mean()),
        source_sha256=sha256_file(source_path),
    )


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Compute PSI using reference quantiles; values above 0.20 require investigation."""
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    observed = pd.to_numeric(current, errors="coerce").dropna()
    if ref.empty or observed.empty:
        raise ValueError("PSI needs at least one numeric value in each population.")
    if bins < 2:
        raise ValueError("PSI needs at least two bins.")

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    obs_counts, _ = np.histogram(observed, bins=edges)
    epsilon = 1e-6
    ref_rate = np.maximum(ref_counts / ref_counts.sum(), epsilon)
    obs_rate = np.maximum(obs_counts / obs_counts.sum(), epsilon)
    return float(np.sum((obs_rate - ref_rate) * np.log(obs_rate / ref_rate)))
