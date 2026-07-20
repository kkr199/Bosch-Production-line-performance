"""Offline project Q&A agent for non-technical manufacturing stakeholders."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs" / "final_deliverables"
DATABASE_PATH = PROJECT_ROOT / "data" / "database" / "manufacturing_copilot.db"


@dataclass
class EvidenceSnippet:
    source: str
    text: str
    score: float


@dataclass
class AgentResponse:
    answer: str
    evidence: list[EvidenceSnippet]
    topic: str


REPORT_FILES = [
    "phase1_data_quality_report.md",
    "phase2_data_understanding_engineering_report.md",
    "phase3_exploratory_data_analysis_report.md",
    "phase4_feature_engineering_report.md",
    "phase5_product_family_discovery_report.md",
    "phase6_predictive_failure_modeling_report.md",
    "phase6_model_improvement_report.md",
    "phase6_leaderboard_research_notes.md",
    "phase7_root_cause_analysis_report.md",
    "phase8_process_mining_bottleneck_report.md",
    "phase10_advanced_ai_report.md",
    "phase11_manufacturing_copilot_report.md",
    "phase12_executive_dashboard_report.md",
]


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _chunk_text(text: str, source: str, words_per_chunk: int = 110) -> list[dict[str, str]]:
    words = text.split()
    chunks = []
    for start in range(0, len(words), words_per_chunk):
        chunk = " ".join(words[start : start + words_per_chunk])
        if len(chunk.split()) >= 25:
            chunks.append({"source": source, "text": chunk})
    return chunks


def _safe_frame(path: Path, rows: int = 8) -> str:
    if not path.exists():
        return ""
    frame = pd.read_csv(path).head(rows)
    return frame.to_string(index=False)


def build_knowledge_chunks() -> list[dict[str, str]]:
    chunks: list[dict[str, str]] = []
    for name in REPORT_FILES:
        chunks.extend(_chunk_text(_read_text(REPORTS_DIR / name), f"reports/{name}"))

    full_paper = DOCS_DIR / "Bosch_Production_Line_Performance_Full_Research_Paper_20plus.md"
    chunks.extend(_chunk_text(_read_text(full_paper), f"docs/final_deliverables/{full_paper.name}"))

    table_sources = {
        "reports/phase5_product_family_failure_rates.csv": "Product family failure rates",
        "reports/phase6_model_comparison_metrics.csv": "Model comparison metrics",
        "reports/phase7_top_failure_drivers.csv": "Top failure drivers",
        "reports/phase8_bottleneck_scores.csv": "Bottleneck scores",
        "reports/phase9_critical_nodes.csv": "Critical graph nodes",
        "reports/phase10_advanced_ai_model_comparison.csv": "Advanced AI results",
    }
    for relative, label in table_sources.items():
        text = f"{label}\n{_safe_frame(PROJECT_ROOT / relative)}"
        chunks.append({"source": relative, "text": text})

    return chunks


def retrieve_evidence(question: str, top_k: int = 5) -> list[EvidenceSnippet]:
    chunks = build_knowledge_chunks()
    if not chunks:
        return []
    documents = [chunk["text"] for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=12000)
    matrix = vectorizer.fit_transform(documents + [question])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    top_indices = scores.argsort()[::-1][:top_k]
    return [
        EvidenceSnippet(
            source=chunks[index]["source"],
            text=chunks[index]["text"],
            score=float(scores[index]),
        )
        for index in top_indices
        if scores[index] > 0
    ]


def _project_metrics() -> dict[str, object]:
    with sqlite3.connect(DATABASE_PATH) as connection:
        summary = pd.read_sql_query("SELECT * FROM production_summary", connection)
        models = pd.read_sql_query("SELECT * FROM model_metrics ORDER BY rank", connection)
        stations = pd.read_sql_query(
            "SELECT * FROM station_failure_rates ORDER BY failure_rate_pct DESC LIMIT 5",
            connection,
        )
        bottlenecks = pd.read_sql_query(
            "SELECT * FROM bottlenecks ORDER BY bottleneck_rank LIMIT 5",
            connection,
        )
    families = pd.read_csv(REPORTS_DIR / "phase5_product_family_failure_rates.csv")
    return {
        "summary": summary,
        "models": models,
        "stations": stations,
        "bottlenecks": bottlenecks,
        "families": families,
    }


def _family_answer() -> str:
    metrics = _project_metrics()
    families = metrics["families"].copy()
    top = families.sort_values("failure_rate_pct", ascending=False).iloc[0]
    count = families["final_product_family"].nunique()
    family_lines = []
    for _, row in families.sort_values("final_product_family").iterrows():
        family_lines.append(
            f"- Family {int(row['final_product_family'])}: {int(row['part_count']):,} products, "
            f"{row['failure_rate_pct']:.3f}% failure rate, usually follows stations such as {row['top_stations'].split(',')[0]}."
        )
    return (
        f"We found {count} product families because products do not all travel through the same stations. "
        "Think of a product family as a group of products that followed a similar route through the factory. "
        "The clustering method grouped products by station-presence patterns, then we reviewed the family profiles. "
        "Eight groups gave a practical balance: enough groups to separate different routing patterns, but not so many "
        "that each group became too small to explain.\n\n"
        f"The highest-risk group is Family {int(top['final_product_family'])}, with a {top['failure_rate_pct']:.3f}% "
        f"failure rate. That does not mean the family causes failures by itself; it means its route and product mix "
        "deserve closer monitoring.\n\n"
        "Family summary:\n" + "\n".join(family_lines)
    )


def _model_answer() -> str:
    models = _project_metrics()["models"]
    best = models.iloc[0]
    return (
        f"The official prediction model is {best['model']}. In simple terms, the model gives each product a risk score "
        "so engineers can decide which products or routes deserve closer review.\n\n"
        f"The validation MCC is {best['mcc']:.3f}. MCC is useful here because failures are rare, so ordinary accuracy "
        "would be misleading. Precision is "
        f"{best['precision']:.3f}, meaning that when the model raises an alert, a meaningful share of those alerts are real failures "
        "in validation. Recall is "
        f"{best['recall']:.3f}, meaning it catches some failures but not all of them.\n\n"
        "Operationally, this model should be used for prioritization, not automatic accept/reject decisions."
    )


def _root_cause_answer() -> str:
    drivers = pd.read_csv(REPORTS_DIR / "phase7_top_failure_drivers.csv").head(8)
    lines = [
        f"- {row.feature}: {row.driver_type}"
        for row in drivers.itertuples(index=False)
    ]
    return (
        "The main failure drivers are mostly timing, waiting, line-level timing, station measurements, and missing measurement signals. "
        "In non-technical terms, the model is telling us that failures are linked to when and how products move through the factory, "
        "which stations they visit, and whether some expected measurements are missing.\n\n"
        "Important: this is not final proof of physical cause. It is a ranked investigation list for engineers.\n\n"
        "Top signals:\n" + "\n".join(lines)
    )


def _bottleneck_answer() -> str:
    bottlenecks = _project_metrics()["bottlenecks"]
    top = bottlenecks.iloc[0]
    return (
        f"The biggest bottleneck is {top['station']}. A bottleneck is a station or part of the process where products tend to wait, "
        "which can slow the overall flow.\n\n"
        f"{top['station']} has a bottleneck score of {top['bottleneck_score']:.2f}, average waiting time of "
        f"{top['avg_waiting_time']:.2f}, and p90 waiting time of {top['p90_waiting_time']:.2f}. "
        "This means most products pass through faster than the high-wait cases, but the slow cases are large enough to matter.\n\n"
        "The recommendation is to check queue buildup, staffing, tooling availability, maintenance windows, and routing rules."
    )


def _leaderboard_answer() -> str:
    return (
        "The high Kaggle-style MCC result came from competition-style features that use order or nearby known failures. "
        "That can work in a closed competition file, but it is not safe for live production because future product outcomes are not known "
        "when a product is being scored.\n\n"
        "For this project, the high-score leaderboard model is kept as research evidence only. The production-safe LightGBM model is used "
        "for dashboards, root-cause analysis, and the copilot."
    )


def _dashboard_answer() -> str:
    return (
        "The project has three Streamlit apps. The Manufacturing Copilot focuses on question answering and operational review. "
        "The Executive Dashboard focuses on KPIs, heatmaps, bottlenecks, SHAP drivers, and business impact. "
        "The unified Project Dashboard combines the whole project: KPIs, model comparison, product families, process mining, model explainability, "
        "knowledge graph, copilot, business impact, and deliverables.\n\n"
        "For a non-technical reviewer, the unified dashboard is the best starting point."
    )


def _advanced_ai_answer() -> str:
    advanced = pd.read_csv(REPORTS_DIR / "phase10_advanced_ai_model_comparison.csv")
    best = advanced.iloc[1] if len(advanced) > 1 else advanced.iloc[0]
    return (
        "We tested advanced AI methods such as anomaly detection, reconstruction error, graph message-passing risk, and failure trajectory prediction. "
        "They were useful for diagnostics, but they did not beat the official LightGBM model.\n\n"
        f"The strongest advanced diagnostic was {best['model']} with MCC {best['mcc']:.3f}. "
        "Because it was weaker than the Phase 6 LightGBM reference, we kept it as supporting evidence rather than the official model."
    )


def curated_answer(question: str) -> tuple[str | None, str]:
    text = question.lower()
    if any(term in text for term in ["leaderboard", "kaggle", "leak", "nearby", "high mcc"]):
        return _leaderboard_answer(), "leaderboard_boundary_explainer"
    if any(term in text for term in ["family", "families", "cluster", "segmentation", "8 families", "eight families"]):
        return _family_answer(), "product_family_explainer"
    if any(term in text for term in ["model", "lightgbm", "mcc", "precision", "recall", "prediction", "score"]):
        return _model_answer(), "model_explainer"
    if any(term in text for term in ["root cause", "shap", "driver", "why fail", "reason"]):
        return _root_cause_answer(), "root_cause_explainer"
    if any(term in text for term in ["bottleneck", "waiting", "queue", "slow"]):
        return _bottleneck_answer(), "bottleneck_explainer"
    if any(term in text for term in ["dashboard", "streamlit", "copilot", "app"]):
        return _dashboard_answer(), "dashboard_explainer"
    if any(term in text for term in ["advanced ai", "anomaly", "autoencoder", "isolation forest", "gnn", "trajectory"]):
        return _advanced_ai_answer(), "advanced_ai_explainer"
    return None, "retrieval"


def retrieval_answer(question: str, evidence: list[EvidenceSnippet]) -> str:
    if not evidence:
        return (
            "I could not find enough matching evidence in the local project files. "
            "Try asking about product families, the prediction model, failures, bottlenecks, SHAP drivers, the knowledge graph, dashboards, or deliverables."
        )

    bullets = []
    for snippet in evidence[:3]:
        text = snippet.text
        if len(text) > 520:
            text = text[:520].rsplit(" ", 1)[0] + "..."
        bullets.append(f"- {text}")
    return (
        "Based on the local project files, here is the simplest explanation I can support:\n\n"
        + "\n".join(bullets)
        + "\n\nIn short, the answer above is based on reviewed project artifacts, not a live external AI call. "
        "If the question needs a decision, use the linked evidence below and the relevant dashboard page."
    )


def answer_project_question(question: str) -> AgentResponse:
    answer, topic = curated_answer(question)
    evidence = retrieve_evidence(question, top_k=5)
    if answer is None:
        answer = retrieval_answer(question, evidence)
    else:
        if evidence:
            answer += "\n\nSupporting project evidence is listed below."
    return AgentResponse(answer=answer, evidence=evidence, topic=topic)
