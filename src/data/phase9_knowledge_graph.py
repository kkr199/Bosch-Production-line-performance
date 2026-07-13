"""Phase 9: build a manufacturing knowledge graph and rank critical nodes."""

from __future__ import annotations

import html
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

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


def minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    lo, hi = values.min(), values.max()
    if not np.isfinite(lo) or hi <= lo:
        return pd.Series(0.0, index=series.index)
    return (values - lo) / (hi - lo)


def add_chart_header(fig, ax, title: str, subtitle: str) -> None:
    ax.set_title("")
    fig.subplots_adjust(top=0.82)
    left = ax.get_position().x0
    fig.text(left, 0.975, title, ha="left", va="top", fontsize=14,
             fontweight="semibold", color=TOKENS["ink"])
    fig.text(left, 0.925, subtitle, ha="left", va="top", fontsize=9,
             color=TOKENS["muted"])
    sns.despine(ax=ax)


def setup_plotting() -> None:
    sns.set_theme(style="whitegrid", rc={
        "figure.facecolor": TOKENS["surface"],
        "axes.facecolor": TOKENS["panel"],
        "axes.edgecolor": TOKENS["axis"],
        "axes.labelcolor": TOKENS["ink"],
        "grid.color": TOKENS["grid"],
        "grid.linewidth": 0.8,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    })


def load_sources() -> dict[str, pd.DataFrame]:
    paths = {
        "features": REPORTS / "phase2_feature_metadata.csv",
        "transitions": REPORTS / "phase8_process_map_edges.csv",
        "process_nodes": REPORTS / "phase8_process_map_nodes.csv",
        "bottlenecks": REPORTS / "phase8_bottleneck_scores.csv",
        "shap": REPORTS / "phase7_shap_global_importance.csv",
        "root_causes": REPORTS / "phase7_station_root_cause_report.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Required Phase 2/7/8 outputs are missing: " + ", ".join(missing))
    return {name: pd.read_csv(path) for name, path in paths.items()}


def prepare_feature_evidence(features: pd.DataFrame, shap: pd.DataFrame) -> pd.DataFrame:
    feature_meta = (
        features.loc[features["column"].ne("Response")]
        .sort_values(["column", "split"])
        .drop_duplicates("column")
        .copy()
    )
    shap_evidence = shap.copy()
    shap_evidence["base_feature"] = (
        shap_evidence["feature"].astype(str)
        .str.replace(r"__is_missing$", "", regex=True)
        .str.replace(r"__eq_.*$", "", regex=True)
    )
    raw_shap = (
        shap_evidence.groupby("base_feature", as_index=False)
        .agg(mean_abs_shap=("mean_abs_shap", "sum"), mean_signed_shap=("mean_signed_shap", "sum"))
        .rename(columns={"base_feature": "column"})
    )
    feature_meta = feature_meta.merge(raw_shap, on="column", how="left")
    feature_meta[["mean_abs_shap", "mean_signed_shap"]] = feature_meta[
        ["mean_abs_shap", "mean_signed_shap"]
    ].fillna(0.0)
    feature_meta["failure_evidence"] = feature_meta["mean_abs_shap"].gt(0)
    return feature_meta


def calculate_station_metrics(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    nodes = sources["process_nodes"].copy()
    bottlenecks = sources["bottlenecks"].copy()
    transitions = sources["transitions"].copy()

    graph = nx.DiGraph()
    for row in transitions.itertuples(index=False):
        graph.add_edge(
            row.from_station,
            row.to_station,
            transition_count=float(row.transition_count),
            distance=1.0 / math.log1p(max(float(row.transition_count), 1.0)),
        )
    all_stations = sorted(set(nodes["station"]) | set(bottlenecks["station"]) | set(graph.nodes))
    graph.add_nodes_from(all_stations)

    pagerank = nx.pagerank(graph, weight="transition_count")
    betweenness = nx.betweenness_centrality(graph, weight="distance", normalized=True)
    closeness = nx.closeness_centrality(graph, distance="distance")
    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())
    weighted_in = dict(graph.in_degree(weight="transition_count"))
    weighted_out = dict(graph.out_degree(weight="transition_count"))

    centrality = pd.DataFrame({
        "station": all_stations,
        "in_degree_centrality": [in_degree.get(s, 0) / max(len(all_stations) - 1, 1) for s in all_stations],
        "out_degree_centrality": [out_degree.get(s, 0) / max(len(all_stations) - 1, 1) for s in all_stations],
        "weighted_inflow": [weighted_in.get(s, 0.0) for s in all_stations],
        "weighted_outflow": [weighted_out.get(s, 0.0) for s in all_stations],
        "pagerank": [pagerank.get(s, 0.0) for s in all_stations],
        "betweenness_centrality": [betweenness.get(s, 0.0) for s in all_stations],
        "closeness_centrality": [closeness.get(s, 0.0) for s in all_stations],
    })

    root_station = sources["root_causes"].copy()
    root_station = root_station[root_station["station"].astype(str).str.match(r"L\d+_S\d+")]
    station_shap = (
        root_station.groupby("station", as_index=False)["total_mean_abs_shap"].sum()
        .rename(columns={"total_mean_abs_shap": "station_shap_importance"})
    )
    metrics = centrality.merge(bottlenecks, on="station", how="left").merge(
        station_shap, on="station", how="left"
    )
    fill_columns = [
        "product_count", "avg_waiting_time", "p90_waiting_time", "positive_wait_rate",
        "failure_count", "labeled_count", "failure_rate_pct", "failure_lift",
        "bottleneck_score", "station_shap_importance",
    ]
    for column in fill_columns:
        metrics[column] = pd.to_numeric(metrics.get(column), errors="coerce").fillna(0.0)

    metrics["centrality_score"] = (
        0.35 * minmax(metrics["pagerank"])
        + 0.35 * minmax(metrics["betweenness_centrality"])
        + 0.15 * minmax(metrics["closeness_centrality"])
        + 0.15 * minmax(metrics["weighted_outflow"])
    )
    metrics["critical_node_score"] = 100 * (
        0.25 * minmax(metrics["bottleneck_score"])
        + 0.20 * minmax(metrics["failure_lift"])
        + 0.20 * minmax(metrics["station_shap_importance"])
        + 0.20 * minmax(metrics["centrality_score"])
        + 0.15 * minmax(np.log1p(metrics["product_count"]))
    )
    metrics["critical_rank"] = metrics["critical_node_score"].rank(
        method="dense", ascending=False
    ).astype(int)
    metrics["line"] = metrics["station"].str.extract(r"(L\d+)", expand=False)
    return metrics.sort_values("critical_rank")


def build_graph_tables(
    feature_evidence: pd.DataFrame,
    transitions: pd.DataFrame,
    station_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, nx.MultiDiGraph]:
    nodes: list[dict] = []
    edges: list[dict] = []

    failure_id = "failure:Response_1"
    nodes.append({"node_id": failure_id, "node_type": "failure", "label": "Failure (Response=1)",
                  "line": "", "station": "", "critical_score": 100.0, "evidence_value": 1.0})

    station_lookup = station_metrics.set_index("station")
    for line in sorted(feature_evidence["line"].dropna().astype(int).unique()):
        line_key = f"L{line}"
        line_id = f"line:{line_key}"
        line_stations = station_metrics.loc[station_metrics["line"].eq(line_key)]
        line_score = float(line_stations["critical_node_score"].max()) if len(line_stations) else 0.0
        nodes.append({"node_id": line_id, "node_type": "line", "label": line_key,
                      "line": line_key, "station": "", "critical_score": line_score,
                      "evidence_value": float(line_stations["product_count"].sum()) if len(line_stations) else 0.0})

    for station in sorted(feature_evidence["station_key"].dropna().unique()):
        line_key = station.split("_")[0]
        row = station_lookup.loc[station] if station in station_lookup.index else None
        score = float(row["critical_node_score"]) if row is not None else 0.0
        volume = float(row["product_count"]) if row is not None else 0.0
        station_id = f"station:{station}"
        nodes.append({"node_id": station_id, "node_type": "station", "label": station,
                      "line": line_key, "station": station, "critical_score": score,
                      "evidence_value": volume})
        edges.append({"source": f"line:{line_key}", "target": station_id,
                      "relationship": "CONTAINS_STATION", "weight": 1.0,
                      "transition_count": 0.0, "failure_evidence": 0.0})

    for row in feature_evidence.itertuples(index=False):
        feature_id = f"feature:{row.column}"
        line_key = f"L{int(row.line)}"
        station = str(row.station_key)
        nodes.append({"node_id": feature_id, "node_type": "feature", "label": row.column,
                      "line": line_key, "station": station, "critical_score": 0.0,
                      "evidence_value": float(row.mean_abs_shap)})
        edges.append({"source": f"station:{station}", "target": feature_id,
                      "relationship": "HAS_FEATURE", "weight": 1.0,
                      "transition_count": 0.0, "failure_evidence": 0.0})
        if row.failure_evidence:
            edges.append({"source": feature_id, "target": failure_id,
                          "relationship": "MODEL_ASSOCIATED_WITH_FAILURE",
                          "weight": float(row.mean_abs_shap), "transition_count": 0.0,
                          "failure_evidence": float(row.mean_abs_shap)})

    for row in transitions.itertuples(index=False):
        edges.append({"source": f"station:{row.from_station}", "target": f"station:{row.to_station}",
                      "relationship": "TRANSITIONS_TO", "weight": float(row.transition_count),
                      "transition_count": float(row.transition_count),
                      "failure_evidence": float(row.failure_rate_pct)})

    node_df = pd.DataFrame(nodes).drop_duplicates("node_id")
    feature_mask = node_df["node_type"].eq("feature")
    node_df.loc[feature_mask, "critical_score"] = 100 * minmax(
        node_df.loc[feature_mask, "evidence_value"]
    )
    edge_df = pd.DataFrame(edges)

    graph = nx.MultiDiGraph()
    for row in node_df.itertuples(index=False):
        graph.add_node(row.node_id, node_type=row.node_type, label=row.label,
                       line=str(row.line), station=str(row.station),
                       critical_score=float(row.critical_score), evidence_value=float(row.evidence_value))
    for row in edge_df.itertuples(index=False):
        graph.add_edge(row.source, row.target, relationship=row.relationship,
                       weight=float(row.weight), transition_count=float(row.transition_count),
                       failure_evidence=float(row.failure_evidence))
    return node_df, edge_df, graph


def find_candidate_routes(
    transitions: pd.DataFrame,
    station_metrics: pd.DataFrame,
    max_depth: int = 6,
    beam_width: int = 160,
) -> pd.DataFrame:
    metrics = station_metrics.set_index("station")
    edge_df = transitions.loc[transitions["transition_count"].ge(500)].copy()
    edge_df["volume_score"] = minmax(np.log1p(edge_df["transition_count"]))
    edge_df["edge_risk"] = (
        0.45 * edge_df["volume_score"]
        + 0.30 * minmax(edge_df["failure_rate_pct"])
        + 0.25 * edge_df["to_station"].map(metrics["critical_node_score"]).fillna(0) / 100
    )
    adjacency = {
        station: part.sort_values("edge_risk", ascending=False).head(8).to_dict("records")
        for station, part in edge_df.groupby("from_station")
    }
    starts = station_metrics.head(15)["station"].tolist()
    beam = [([station], [], 0.0) for station in starts]
    completed: list[tuple[list[str], list[dict], float]] = []
    for _ in range(max_depth - 1):
        expanded = []
        for path, used_edges, score in beam:
            for edge in adjacency.get(path[-1], []):
                nxt = edge["to_station"]
                if nxt in path:
                    continue
                new_edges = used_edges + [edge]
                station_scores = [metrics.loc[s, "critical_node_score"] / 100 for s in path + [nxt] if s in metrics.index]
                route_score = 100 * (
                    0.55 * np.mean([e["edge_risk"] for e in new_edges])
                    + 0.45 * np.mean(station_scores)
                )
                expanded.append((path + [nxt], new_edges, route_score))
        expanded.sort(key=lambda item: item[2], reverse=True)
        beam = expanded[:beam_width]
        completed.extend(item for item in beam if len(item[0]) >= 4)

    records = []
    seen = set()
    selected_station_sets: list[set[str]] = []
    for path, route_edges, route_score in sorted(completed, key=lambda item: item[2], reverse=True):
        key = ">".join(path)
        if key in seen:
            continue
        station_set = set(path)
        if any(
            len(station_set & prior) / len(station_set | prior) >= 0.75
            for prior in selected_station_sets
        ):
            continue
        seen.add(key)
        selected_station_sets.append(station_set)
        records.append({
            "candidate_route": key,
            "station_count": len(path),
            "route_score": route_score,
            "minimum_transition_count": min(e["transition_count"] for e in route_edges),
            "mean_transition_count": np.mean([e["transition_count"] for e in route_edges]),
            "mean_transition_failure_rate_pct": np.mean([e["failure_rate_pct"] for e in route_edges]),
            "mean_station_critical_score": np.mean([metrics.loc[s, "critical_node_score"] for s in path if s in metrics.index]),
            "start_station": path[0],
            "end_station": path[-1],
        })
        if len(records) >= 50:
            break
    routes = pd.DataFrame(records)
    if not routes.empty:
        routes["propagation_rank"] = np.arange(1, len(routes) + 1)
    return routes


def plot_critical_nodes(station_metrics: pd.DataFrame) -> Path:
    plot_df = station_metrics.head(15).sort_values("critical_node_score")
    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(plot_df["station"], plot_df["critical_node_score"],
                   color=TOKENS["orange"], edgecolor=TOKENS["orange_dark"], linewidth=1)
    ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=8, color=TOKENS["ink"])
    ax.set_xlabel("Composite critical-node score (0-100)")
    ax.set_ylabel("")
    ax.set_xlim(0, max(100, plot_df["critical_node_score"].max() * 1.15))
    add_chart_header(fig, ax, "Critical manufacturing stations",
                     "Structural centrality, bottleneck severity, failure lift, model attribution, and product volume")
    path = FIGURES / "phase9_critical_nodes.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_centrality_map(transitions: pd.DataFrame, station_metrics: pd.DataFrame) -> Path:
    top = set(station_metrics.head(22)["station"])
    edge_plot = transitions[
        transitions["from_station"].isin(top) & transitions["to_station"].isin(top)
    ].nlargest(45, "transition_count")
    graph = nx.DiGraph()
    for row in edge_plot.itertuples(index=False):
        graph.add_edge(row.from_station, row.to_station, weight=row.transition_count)
    pos = {}
    lines = sorted({node.split("_")[0] for node in graph.nodes})
    for line_index, line in enumerate(lines):
        line_nodes = sorted(
            [node for node in graph.nodes if node.startswith(f"{line}_")],
            key=lambda value: int(value.split("_S")[1]),
        )
        y_values = np.linspace(0.05, 0.95, len(line_nodes)) if len(line_nodes) > 1 else [0.5]
        for node, y_value in zip(line_nodes, y_values):
            pos[node] = (line_index, float(y_value))
    score = station_metrics.set_index("station")["critical_node_score"]
    node_sizes = [450 + 20 * score.get(node, 0) for node in graph.nodes]
    node_colors = [score.get(node, 0) for node in graph.nodes]
    widths = [0.5 + 3 * math.log1p(data["weight"]) / math.log1p(edge_plot["transition_count"].max())
              for _, _, data in graph.edges(data=True)]
    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_edges(graph, pos, ax=ax, width=widths, alpha=0.28,
                           edge_color="#7A828F", arrows=True, arrowsize=12)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=node_sizes, node_color=node_colors,
                           cmap=sns.blend_palette(["#EAF1FE", TOKENS["blue"], TOKENS["orange"]], as_cmap=True),
                           edgecolors=TOKENS["ink"], linewidths=0.8)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8, font_color=TOKENS["ink"])
    for line_index, line in enumerate(lines):
        ax.text(line_index, 1.04, line, ha="center", va="bottom", fontsize=10,
                fontweight="semibold", color=TOKENS["muted"])
    ax.set_xlim(-0.35, max(len(lines) - 1, 0) + 0.35)
    ax.set_ylim(-0.03, 1.09)
    ax.set_axis_off()
    add_chart_header(fig, ax, "Station relationship network",
                     "Top 22 critical stations; arrows show the 45 highest-volume observed transitions")
    path = FIGURES / "phase9_station_relationship_network.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_routes(routes: pd.DataFrame) -> Path:
    plot_df = routes.head(10).copy()
    plot_df["route_label"] = plot_df["candidate_route"].str.replace(">", " → ", regex=False)
    plot_df = plot_df.sort_values("route_score")
    fig, ax = plt.subplots(figsize=(13, 8))
    bars = ax.barh(plot_df["route_label"], plot_df["route_score"],
                   color=TOKENS["gold"], edgecolor=TOKENS["gold_dark"], linewidth=1)
    ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=8, color=TOKENS["ink"])
    ax.set_xlabel("Candidate propagation-route score (0-100)")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=8)
    add_chart_header(fig, ax, "Candidate failure-propagation routes",
                     "Ranked observational routes; high score is a diagnostic priority, not proof of causal transmission")
    path = FIGURES / "phase9_candidate_propagation_routes.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def dataframe_html(df: pd.DataFrame, columns: list[str], rows: int = 10) -> str:
    show = df.head(rows)[columns].copy()
    for column in show.select_dtypes(include=["number"]).columns:
        show[column] = show[column].map(lambda value: f"{value:,.3f}")
    return show.to_html(index=False, border=0, classes="data-table", escape=True)


def write_html_report(
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    station_metrics: pd.DataFrame,
    routes: pd.DataFrame,
) -> Path:
    top = station_metrics.iloc[0]
    route = routes.iloc[0]
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phase 9 Manufacturing Knowledge Graph</title>
<style>
body{{margin:0;background:#FCFCFD;color:#1F2430;font-family:'Segoe UI',Arial,sans-serif;line-height:1.55}}
main{{max-width:1120px;margin:auto;padding:38px 28px 60px}} h1{{font-size:30px;margin:0 0 20px}} h2{{margin-top:42px;font-size:21px}}
.summary{{border-left:4px solid #5477C4;padding:6px 18px;background:#fff}} .metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}}
.metric{{background:#fff;border:1px solid #E6E8F0;padding:16px}} .metric strong{{display:block;font-size:23px}} .metric span{{color:#6F768A;font-size:13px}}
img{{width:100%;height:auto;background:#fff;margin:10px 0}} .note{{background:#FFF4C2;border-left:4px solid #B8A037;padding:12px 16px}}
.data-table{{width:100%;border-collapse:collapse;background:#fff;font-size:13px}} th,td{{padding:9px;border-bottom:1px solid #E6E8F0;text-align:left}} th{{background:#F4F5F7}}
code{{font-family:Consolas,monospace}} @media(max-width:760px){{.metrics{{grid-template-columns:1fr 1fr}}main{{padding:24px 16px}}}}
</style></head><body><main>
<h1>Phase 9: Manufacturing Knowledge Graph</h1>
<section data-contract-section="technical-summary"><h2>Technical summary</h2><div class="summary">
<p>The graph connects production lines, stations, raw numeric/categorical/date features, observed station transitions, and the production-safe model's failure evidence. <strong>{html.escape(str(top.station))}</strong> is the highest-ranked critical station with a composite score of <strong>{top.critical_node_score:.1f}</strong>.</p>
<p>The top candidate propagation route is <strong>{html.escape(str(route.candidate_route).replace('>', ' → '))}</strong>. It is a prioritized investigation path, not proof that one station physically caused a later failure.</p></div></section>
<div class="metrics"><div class="metric"><strong>{len(node_df):,}</strong><span>Graph nodes</span></div><div class="metric"><strong>{len(edge_df):,}</strong><span>Graph relationships</span></div><div class="metric"><strong>{station_metrics.shape[0]}</strong><span>Stations scored</span></div><div class="metric"><strong>{len(routes)}</strong><span>Candidate routes</span></div></div>
<section data-contract-section="key-findings"><h2>Critical nodes combine operational and model evidence</h2><p>The score balances bottlenecks, observed failure lift, station-level SHAP attribution, network centrality, and product volume. This makes the ranking suitable for investigation queues without treating any one signal as a root-cause verdict.</p><img src="figures/phase9_critical_nodes.png" alt="Ranked critical stations"></section>
<section><h2>Station relationships reveal influential transfer points</h2><p>PageRank emphasizes stations receiving flow from other influential stations, while betweenness highlights bridges that sit on many high-volume paths. The network view limits itself to the most critical nodes so labels and direction remain readable.</p><img src="figures/phase9_station_relationship_network.png" alt="Station transition network"></section>
<section><h2>Candidate propagation routes focus engineering investigation</h2><p>Routes are ranked from transition volume, transition-level failure rate, and critical-node evidence. Engineers should validate the route against sensor history, maintenance logs, product family, and timestamps before taking action.</p><img src="figures/phase9_candidate_propagation_routes.png" alt="Candidate propagation routes"></section>
<section data-contract-section="scope-data-metrics"><h2>Scope and definitions</h2><p>The hierarchy covers deduplicated raw feature metadata from numeric, categorical, and date datasets. Failure associations use absolute SHAP values from the production-safe Phase 6 LightGBM model. Station relationships use Phase 8 transitions reconstructed from raw train and test date files; failure rates use labeled training products only.</p></section>
<section data-contract-section="methodology"><h2>Methodology and audit tables</h2>{dataframe_html(station_metrics, ['critical_rank','station','critical_node_score','centrality_score','bottleneck_score','failure_lift','station_shap_importance'], 12)}<br>{dataframe_html(routes, ['propagation_rank','candidate_route','route_score','minimum_transition_count','mean_transition_failure_rate_pct'], 10)}</section>
<section data-contract-section="limitations"><h2>Limitations and uncertainty</h2><div class="note"><strong>Association is not causation.</strong> SHAP explains model behavior, station transitions describe observed routing, and timestamps are sparse relative-time markers. The graph cannot establish that an upstream station physically caused a downstream defect without intervention, maintenance, sensor, and quality-control evidence.</div></section>
<section data-contract-section="recommended-next-steps"><h2>Recommended next steps</h2><ol><li>Inspect the top critical stations and routes by product family and time window.</li><li>Join future maintenance, alarm, calibration, and operator-event data to graph nodes.</li><li>Track whether corrective actions reduce failure lift and bottleneck scores.</li></ol></section>
<section data-contract-section="further-questions"><h2>Further questions</h2><p>Do the same routes remain critical after controlling for product family? Which nodes align with known machine ownership and maintenance events? Can future timestamp data provide true event durations rather than relative-time proxies?</p></section>
</main></body></html>"""
    path = REPORTS / "phase9_knowledge_graph_report.html"
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    setup_plotting()
    sources = load_sources()
    feature_evidence = prepare_feature_evidence(sources["features"], sources["shap"])
    station_metrics = calculate_station_metrics(sources)
    node_df, edge_df, graph = build_graph_tables(
        feature_evidence, sources["transitions"], station_metrics
    )
    routes = find_candidate_routes(sources["transitions"], station_metrics)

    node_df.to_csv(REPORTS / "phase9_knowledge_graph_nodes.csv", index=False)
    edge_df.to_csv(REPORTS / "phase9_knowledge_graph_edges.csv", index=False)
    station_metrics.to_csv(REPORTS / "phase9_station_centrality_metrics.csv", index=False)
    station_metrics.to_csv(REPORTS / "phase9_critical_nodes.csv", index=False)
    routes.to_csv(REPORTS / "phase9_failure_propagation_routes.csv", index=False)
    nx.write_graphml(graph, REPORTS / "phase9_manufacturing_knowledge_graph.graphml")

    plot_critical_nodes(station_metrics)
    plot_centrality_map(sources["transitions"], station_metrics)
    plot_routes(routes)
    report_path = write_html_report(node_df, edge_df, station_metrics, routes)

    print(f"Created {len(node_df):,} nodes and {len(edge_df):,} relationships.")
    print(f"Top critical station: {station_metrics.iloc[0]['station']}")
    print(f"HTML report: {report_path}")


if __name__ == "__main__":
    main()
