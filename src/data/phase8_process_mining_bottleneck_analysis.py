"""Phase 8 process mining and bottleneck analysis from raw Bosch date data."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FEATURE_RE = re.compile(r"^L(?P<line>\d+)_S(?P<station>\d+)_D(?P<feature>\d+)$")
CHUNKSIZE = 20_000


def find_dataset_path(filename: str) -> Path:
    for candidate in [PROJECT_ROOT / filename, PROJECT_ROOT / "data" / "raw" / filename]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {filename}")


def read_header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return next(csv.reader(file))


def split_station_key(key: str) -> tuple[int, int]:
    line, station = key.split("_")
    return int(line[1:]), int(station[1:])


def build_station_groups() -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for column in read_header(find_dataset_path("train_date.csv")):
        match = FEATURE_RE.match(column)
        if not match:
            continue
        key = f"L{match.group('line')}_S{match.group('station')}"
        groups.setdefault(key, []).append(column)
    return dict(sorted(groups.items(), key=lambda item: split_station_key(item[0])))


def weighted_average(group: pd.DataFrame, value: str, weight: str = "transition_count") -> float:
    weights = group[weight].to_numpy(dtype=float)
    values = group[value].to_numpy(dtype=float)
    return float(np.average(values, weights=weights)) if weights.sum() else np.nan


def aggregate_chunk_summaries(frame: pd.DataFrame, keys: list[str], count_col: str) -> pd.DataFrame:
    rows = []
    for key_values, group in frame.groupby(keys, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = dict(zip(keys, key_values))
        row[count_col] = int(group[count_col].sum())
        for column in group.columns:
            if column in keys or column == count_col:
                continue
            if column.startswith("avg_") or column.startswith("median_") or column.startswith("p90_"):
                row[column] = weighted_average(group, column, count_col)
            elif column.endswith("_sum") or column.endswith("_count"):
                row[column] = float(group[column].sum())
            elif pd.api.types.is_numeric_dtype(group[column]):
                row[column] = weighted_average(group, column, count_col)
        rows.append(row)
    return pd.DataFrame(rows)


def process_split(
    split: str,
    station_groups: dict[str, list[str]],
    target_lookup: pd.Series | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    station_keys = list(station_groups)
    transition_chunks: list[pd.DataFrame] = []
    station_chunks: list[pd.DataFrame] = []
    path_chunks: list[pd.DataFrame] = []
    efficiency_chunks: list[pd.DataFrame] = []

    reader = pd.read_csv(find_dataset_path(f"{split}_date.csv"), chunksize=CHUNKSIZE)
    for chunk in tqdm(reader, desc=f"Mining {split} process", unit="chunk"):
        rows = len(chunk)
        response = (
            chunk["Id"].map(target_lookup).to_numpy(dtype=float)
            if target_lookup is not None
            else np.full(rows, np.nan)
        )

        starts = np.empty((rows, len(station_keys)), dtype=np.float32)
        ends = np.empty((rows, len(station_keys)), dtype=np.float32)
        for index, station in enumerate(station_keys):
            station_data = chunk[station_groups[station]]
            starts[:, index] = station_data.min(axis=1, skipna=True).to_numpy(dtype=np.float32)
            ends[:, index] = station_data.max(axis=1, skipna=True).to_numpy(dtype=np.float32)

        present = np.isfinite(starts)
        counts = present.sum(axis=1)
        safe_starts = np.where(present, starts, np.inf)
        order = np.argsort(safe_starts, axis=1)
        sorted_starts = np.take_along_axis(starts, order, axis=1)
        sorted_ends = np.take_along_axis(ends, order, axis=1)

        product_wait = np.zeros(rows, dtype=np.float32)
        wait_events = np.zeros(rows, dtype=np.int16)
        for position in range(len(station_keys) - 1):
            mask = counts > position + 1
            if not mask.any():
                continue
            row_index = np.flatnonzero(mask)
            from_index = order[mask, position]
            to_index = order[mask, position + 1]
            raw_gap = sorted_starts[mask, position + 1] - sorted_ends[mask, position]
            wait = np.maximum(raw_gap, 0).astype(np.float32)
            product_wait[mask] += wait
            wait_events[mask] += (wait > 0).astype(np.int16)

            transitions = pd.DataFrame(
                {
                    "split": split,
                    "from_station": np.asarray(station_keys, dtype=object)[from_index],
                    "to_station": np.asarray(station_keys, dtype=object)[to_index],
                    "wait": wait,
                    "positive_wait": (wait > 0).astype(np.int8),
                    "response": response[row_index],
                }
            )
            grouped = transitions.groupby(["split", "from_station", "to_station"], as_index=False).agg(
                transition_count=("wait", "size"),
                wait_sum=("wait", "sum"),
                avg_wait=("wait", "mean"),
                median_wait=("wait", "median"),
                p90_wait=("wait", lambda values: values.quantile(0.90)),
                max_wait=("wait", "max"),
                positive_wait_count=("positive_wait", "sum"),
                failure_count=("response", "sum"),
                labeled_count=("response", "count"),
            )
            transition_chunks.append(grouped)

        dwell = np.maximum(ends - starts, 0)
        for index, station in enumerate(station_keys):
            mask = present[:, index]
            if not mask.any():
                continue
            station_chunks.append(
                pd.DataFrame(
                    {
                        "split": [split],
                        "station": [station],
                        "product_count": [int(mask.sum())],
                        "dwell_sum": [float(np.nansum(dwell[mask, index]))],
                        "avg_dwell": [float(np.nanmean(dwell[mask, index]))],
                        "median_dwell": [float(np.nanmedian(dwell[mask, index]))],
                        "p90_dwell": [float(np.nanquantile(dwell[mask, index], 0.90))],
                        "failure_count": [float(np.nansum(response[mask]))],
                        "labeled_count": [int(np.isfinite(response[mask]).sum())],
                    }
                )
            )

        productive_time = np.nansum(dwell, axis=1).astype(np.float32)
        total_accounted_time = productive_time + product_wait
        efficiency = np.divide(
            productive_time,
            total_accounted_time,
            out=np.zeros_like(productive_time),
            where=total_accounted_time > 0,
        )
        cycle_time = np.full(rows, np.nan, dtype=np.float32)
        has_timestamps = counts > 0
        cycle_time[has_timestamps] = (
            np.nanmax(ends[has_timestamps], axis=1) - np.nanmin(starts[has_timestamps], axis=1)
        )
        efficiency_chunks.append(
            pd.DataFrame(
                {
                    "split": split,
                    "product_count": 1,
                    "productive_time": productive_time,
                    "waiting_time": product_wait,
                    "cycle_time": cycle_time,
                    "throughput_efficiency": efficiency,
                    "wait_event_count": wait_events,
                    "response": response,
                }
            )
        )

        path_values = []
        for row in range(rows):
            ordered_stations = [station_keys[index] for index in order[row, : counts[row]]]
            path_values.append(">".join(ordered_stations) if ordered_stations else "NO_RECORDED_PATH")
        path_frame = pd.DataFrame(
            {
                "split": split,
                "process_path": path_values,
                "response": response,
                "waiting_time": product_wait,
                "productive_time": productive_time,
                "cycle_time": cycle_time,
                "throughput_efficiency": efficiency,
                "station_count": counts,
            }
        )
        path_chunks.append(
            path_frame.groupby(["split", "process_path"], as_index=False).agg(
                product_count=("process_path", "size"),
                failure_count=("response", "sum"),
                labeled_count=("response", "count"),
                avg_waiting_time=("waiting_time", "mean"),
                median_waiting_time=("waiting_time", "median"),
                p90_waiting_time=("waiting_time", lambda values: values.quantile(0.90)),
                avg_productive_time=("productive_time", "mean"),
                avg_cycle_time=("cycle_time", "mean"),
                avg_throughput_efficiency=("throughput_efficiency", "mean"),
                avg_station_count=("station_count", "mean"),
            )
        )

    transitions = aggregate_chunk_summaries(
        pd.concat(transition_chunks, ignore_index=True),
        ["split", "from_station", "to_station"],
        "transition_count",
    )
    stations = aggregate_chunk_summaries(
        pd.concat(station_chunks, ignore_index=True),
        ["split", "station"],
        "product_count",
    )
    paths = aggregate_chunk_summaries(
        pd.concat(path_chunks, ignore_index=True),
        ["split", "process_path"],
        "product_count",
    )
    efficiency = pd.concat(efficiency_chunks, ignore_index=True)
    return transitions, stations, paths, efficiency


def minmax(series: pd.Series) -> pd.Series:
    low, high = series.min(), series.max()
    if pd.isna(low) or high == low:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - low) / (high - low)


def create_process_map(transitions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = transitions.groupby(["from_station", "to_station"], as_index=False).agg(
        transition_count=("transition_count", "sum"),
        wait_sum=("wait_sum", "sum"),
        positive_wait_count=("positive_wait_count", "sum"),
        failure_count=("failure_count", "sum"),
        labeled_count=("labeled_count", "sum"),
    )
    combined["avg_waiting_time"] = combined["wait_sum"] / combined["transition_count"]
    combined["positive_wait_rate"] = combined["positive_wait_count"] / combined["transition_count"]
    combined["failure_rate_pct"] = combined["failure_count"] / combined["labeled_count"].replace(0, np.nan) * 100
    combined = combined.sort_values("transition_count", ascending=False)

    graph = nx.DiGraph()
    for row in combined.itertuples(index=False):
        graph.add_edge(row.from_station, row.to_station, weight=row.transition_count, wait=row.avg_waiting_time)
    nodes = pd.DataFrame(
        {
            "station": list(graph.nodes),
            "in_degree": [graph.in_degree(node) for node in graph.nodes],
            "out_degree": [graph.out_degree(node) for node in graph.nodes],
            "weighted_inflow": [graph.in_degree(node, weight="weight") for node in graph.nodes],
            "weighted_outflow": [graph.out_degree(node, weight="weight") for node in graph.nodes],
        }
    )
    combined.to_csv(REPORTS_DIR / "phase8_process_map_edges.csv", index=False)
    nodes.to_csv(REPORTS_DIR / "phase8_process_map_nodes.csv", index=False)
    return combined, nodes


def create_station_wait_and_bottleneck_reports(
    stations: pd.DataFrame,
    transitions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inbound = transitions.groupby(["split", "to_station"], as_index=False).agg(
        inbound_transition_count=("transition_count", "sum"),
        inbound_wait_sum=("wait_sum", "sum"),
        positive_wait_count=("positive_wait_count", "sum"),
        avg_waiting_time=("avg_wait", "mean"),
        median_waiting_time=("median_wait", "mean"),
        p90_waiting_time=("p90_wait", "mean"),
        max_waiting_time=("max_wait", "max"),
    ).rename(columns={"to_station": "station"})
    report = stations.merge(inbound, on=["split", "station"], how="left").fillna(0)
    report["positive_wait_rate"] = report["positive_wait_count"] / report["inbound_transition_count"].replace(0, np.nan)
    report["failure_rate_pct"] = report["failure_count"] / report["labeled_count"].replace(0, np.nan) * 100
    report.to_csv(REPORTS_DIR / "phase8_station_waiting_times.csv", index=False)

    bottlenecks = report.groupby("station", as_index=False).agg(
        product_count=("product_count", "sum"),
        avg_dwell=("avg_dwell", "mean"),
        p90_dwell=("p90_dwell", "mean"),
        avg_waiting_time=("avg_waiting_time", "mean"),
        p90_waiting_time=("p90_waiting_time", "mean"),
        positive_wait_rate=("positive_wait_rate", "mean"),
        failure_count=("failure_count", "sum"),
        labeled_count=("labeled_count", "sum"),
    )
    bottlenecks["failure_rate_pct"] = bottlenecks["failure_count"] / bottlenecks["labeled_count"].replace(0, np.nan) * 100
    overall_failure = bottlenecks["failure_count"].sum() / bottlenecks["labeled_count"].sum() * 100
    bottlenecks["failure_lift"] = bottlenecks["failure_rate_pct"] / overall_failure
    bottlenecks["bottleneck_score"] = 100 * (
        0.30 * minmax(np.log1p(bottlenecks["p90_waiting_time"]))
        + 0.20 * minmax(np.log1p(bottlenecks["avg_waiting_time"]))
        + 0.15 * minmax(bottlenecks["positive_wait_rate"].fillna(0))
        + 0.15 * minmax(np.log1p(bottlenecks["p90_dwell"]))
        + 0.10 * minmax(np.log1p(bottlenecks["product_count"]))
        + 0.10 * minmax(bottlenecks["failure_lift"].fillna(0))
    )
    bottlenecks["bottleneck_rank"] = bottlenecks["bottleneck_score"].rank(ascending=False, method="dense").astype(int)
    bottlenecks = bottlenecks.sort_values("bottleneck_score", ascending=False)
    bottlenecks.to_csv(REPORTS_DIR / "phase8_bottleneck_scores.csv", index=False)
    return report, bottlenecks


def create_path_and_efficiency_reports(
    paths: pd.DataFrame,
    efficiency: pd.DataFrame,
    bottlenecks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path_report = paths.groupby("process_path", as_index=False).agg(
        product_count=("product_count", "sum"),
        failure_count=("failure_count", "sum"),
        labeled_count=("labeled_count", "sum"),
        avg_waiting_time=("avg_waiting_time", "mean"),
        p90_waiting_time=("p90_waiting_time", "mean"),
        avg_productive_time=("avg_productive_time", "mean"),
        avg_cycle_time=("avg_cycle_time", "mean"),
        avg_throughput_efficiency=("avg_throughput_efficiency", "mean"),
        avg_station_count=("avg_station_count", "mean"),
    )
    path_report["avg_station_count"] = path_report["process_path"].map(
        lambda path: 0 if path == "NO_RECORDED_PATH" else len(path.split(">"))
    )
    path_report["failure_rate_pct"] = path_report["failure_count"] / path_report["labeled_count"].replace(0, np.nan) * 100
    overall_failure = path_report["failure_count"].sum() / path_report["labeled_count"].sum() * 100
    prior_strength = 500
    path_report["smoothed_failure_rate_pct"] = (
        path_report["failure_count"] + (overall_failure / 100) * prior_strength
    ) / (path_report["labeled_count"] + prior_strength) * 100
    path_report["failure_lift"] = path_report["smoothed_failure_rate_pct"] / overall_failure
    path_report["priority_eligible"] = (path_report["product_count"] >= 500) & (path_report["labeled_count"] >= 100)
    bottleneck_map = bottlenecks.set_index("station")["bottleneck_score"].to_dict()
    path_report["max_station_bottleneck_score"] = path_report["process_path"].map(
        lambda path: max((bottleneck_map.get(station, 0) for station in path.split(">")), default=0)
    )
    raw_score = 100 * (
        0.25 * minmax(np.log1p(path_report["product_count"]))
        + 0.20 * minmax(np.log1p(path_report["p90_waiting_time"]))
        + 0.20 * minmax(path_report["failure_lift"].fillna(0))
        + 0.20 * minmax(1 - path_report["avg_throughput_efficiency"].fillna(0))
        + 0.15 * minmax(path_report["max_station_bottleneck_score"])
    )
    path_report["critical_path_score"] = np.where(path_report["priority_eligible"], raw_score, raw_score * 0.35)
    path_report["critical_path_rank"] = path_report["critical_path_score"].rank(ascending=False, method="dense").astype(int)
    path_report = path_report.sort_values("critical_path_score", ascending=False)
    path_report.to_csv(REPORTS_DIR / "phase8_critical_process_paths.csv", index=False)

    throughput = efficiency.groupby("split", as_index=False).agg(
        product_count=("product_count", "sum"),
        avg_productive_time=("productive_time", "mean"),
        avg_waiting_time=("waiting_time", "mean"),
        median_waiting_time=("waiting_time", "median"),
        p90_waiting_time=("waiting_time", lambda values: values.quantile(0.90)),
        avg_cycle_time=("cycle_time", "mean"),
        avg_throughput_efficiency=("throughput_efficiency", "mean"),
        median_throughput_efficiency=("throughput_efficiency", "median"),
        avg_wait_events=("wait_event_count", "mean"),
        failure_count=("response", "sum"),
        labeled_count=("response", "count"),
    )
    throughput["failure_rate_pct"] = throughput["failure_count"] / throughput["labeled_count"].replace(0, np.nan) * 100
    throughput.to_csv(REPORTS_DIR / "phase8_throughput_efficiency.csv", index=False)
    return path_report, throughput


def create_figures(edges: pd.DataFrame, bottlenecks: pd.DataFrame, paths: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    top_bottlenecks = bottlenecks.head(15).sort_values("bottleneck_score")
    plt.figure(figsize=(10, 7))
    sns.barplot(data=top_bottlenecks, x="bottleneck_score", y="station", color="#c94c4c")
    plt.title("Top Station Bottleneck Scores")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase8_bottleneck_scores.png", dpi=160, bbox_inches="tight")
    plt.close()

    top_paths = paths.head(12).copy()
    top_paths["path_label"] = top_paths["process_path"].str.slice(0, 70)
    plt.figure(figsize=(11, 7))
    sns.barplot(data=top_paths.sort_values("critical_path_score"), x="critical_path_score", y="path_label", color="#3a7ca5")
    plt.title("Top Critical Production Paths")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase8_critical_paths.png", dpi=160, bbox_inches="tight")
    plt.close()

    top_edges = edges.head(80)
    graph = nx.from_pandas_edgelist(
        top_edges,
        source="from_station",
        target="to_station",
        edge_attr="transition_count",
        create_using=nx.DiGraph,
    )
    plt.figure(figsize=(16, 11))
    pos = nx.spring_layout(graph, seed=42, k=0.8)
    widths = [0.5 + 4 * graph[u][v]["transition_count"] / top_edges["transition_count"].max() for u, v in graph.edges]
    nx.draw_networkx_nodes(graph, pos, node_size=650, node_color="#d9e6f2", edgecolors="#315b7d")
    nx.draw_networkx_edges(graph, pos, width=widths, alpha=0.45, arrows=True, arrowsize=12)
    nx.draw_networkx_labels(graph, pos, font_size=7)
    plt.title("Bosch Production Process Map (Top Transitions)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase8_production_process_map.png", dpi=160, bbox_inches="tight")
    plt.close()


def write_report(
    edges: pd.DataFrame,
    bottlenecks: pd.DataFrame,
    paths: pd.DataFrame,
    throughput: pd.DataFrame,
) -> Path:
    report_path = REPORTS_DIR / "phase8_process_mining_bottleneck_report.md"
    top_station = bottlenecks.iloc[0]
    top_path = paths.iloc[0]
    lines = [
        "# Phase 8: Process Mining & Bottleneck Analysis",
        "",
        "## Data Source",
        "",
        "Production routes were reconstructed from raw `train_date.csv` and `test_date.csv` station timestamps. Train failure labels were used only for failure-rate and failure-lift diagnostics.",
        "",
        "## Process Map",
        "",
        f"- Unique station transitions: {len(edges):,}",
        f"- Stations represented: {len(set(edges['from_station']).union(edges['to_station'])):,}",
        "",
        "## Highest Bottleneck Station",
        "",
        f"`{top_station['station']}` ranks first with bottleneck score {top_station['bottleneck_score']:.2f}. Its average waiting time is {top_station['avg_waiting_time']:.4f} and p90 waiting time is {top_station['p90_waiting_time']:.4f}.",
        "",
        "## Highest Critical Process Path",
        "",
        f"The top path has critical-path score {top_path['critical_path_score']:.2f}, throughput efficiency {top_path['avg_throughput_efficiency']:.2%}, and failure rate {top_path['failure_rate_pct']:.3f}%.",
        "",
        "## Throughput Efficiency",
        "",
        throughput.to_markdown(index=False),
        "",
        "## Top 15 Bottlenecks",
        "",
        bottlenecks.head(15)[["bottleneck_rank", "station", "bottleneck_score", "avg_waiting_time", "p90_waiting_time", "avg_dwell", "failure_rate_pct", "failure_lift"]].to_markdown(index=False),
        "",
        "## Recommended Operations Focus",
        "",
        "- Investigate the top-ranked stations for queue buildup, uneven staffing, tooling delays, or maintenance-related slowdowns.",
        "- Compare high-wait transitions with low-wait alternatives to identify routing or scheduling improvements.",
        "- Prioritize paths that combine high volume, low throughput efficiency, high waiting time, and elevated failure lift.",
        "- Validate whether long waits are true queues or overlapping/parallel production timestamps before operational changes.",
        "",
        "## Output Files",
        "",
        "- `reports/phase8_process_map_edges.csv`",
        "- `reports/phase8_process_map_nodes.csv`",
        "- `reports/phase8_station_waiting_times.csv`",
        "- `reports/phase8_bottleneck_scores.csv`",
        "- `reports/phase8_critical_process_paths.csv`",
        "- `reports/phase8_throughput_efficiency.csv`",
        "- `reports/figures/phase8_production_process_map.png`",
        "- `reports/figures/phase8_bottleneck_scores.png`",
        "- `reports/figures/phase8_critical_paths.png`",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    station_groups = build_station_groups()
    target = pd.read_csv(find_dataset_path("train_numeric.csv"), usecols=["Id", "Response"])
    target_lookup = target.set_index("Id")["Response"]

    train_outputs = process_split("train", station_groups, target_lookup)
    test_outputs = process_split("test", station_groups, None)
    transitions = pd.concat([train_outputs[0], test_outputs[0]], ignore_index=True)
    stations = pd.concat([train_outputs[1], test_outputs[1]], ignore_index=True)
    paths = pd.concat([train_outputs[2], test_outputs[2]], ignore_index=True)
    efficiency = pd.concat([train_outputs[3], test_outputs[3]], ignore_index=True)

    edges, _ = create_process_map(transitions)
    _, bottlenecks = create_station_wait_and_bottleneck_reports(stations, transitions)
    critical_paths, throughput = create_path_and_efficiency_reports(paths, efficiency, bottlenecks)
    create_figures(edges, bottlenecks, critical_paths)
    report_path = write_report(edges, bottlenecks, critical_paths, throughput)
    print(f"Phase 8 complete. Report written to {report_path}")


if __name__ == "__main__":
    main()
