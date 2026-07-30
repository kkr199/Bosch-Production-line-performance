"""LangChain RAG Copilot grounded in the Bosch Markdown handbook."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HANDBOOK_DIR_NAMES = ("Bosch_Handbook_md", "Bosch_Handbook_MD")
LOCAL_CHAT_MODEL = "llama3.2:1b"
LOCAL_EMBEDDING_MODEL = "nomic-embed-text"


class HandbookCopilotError(RuntimeError):
    """A safe, user-facing error for an unavailable local Copilot request."""


@dataclass(frozen=True)
class HandbookAnswer:
    answer: str
    sources: list[Document]
    model: str


def _handbook_dir() -> Path:
    for name in HANDBOOK_DIR_NAMES:
        candidate = PROJECT_ROOT / name
        if candidate.exists():
            return candidate
    raise HandbookCopilotError("The Bosch handbook Markdown folder is not available in this deployment.")


def _clean_markdown(markdown: str) -> str:
    """Remove embedded images and presentation markup before embedding text."""
    markdown = re.sub(r"!\[[^\]]*\]\(data:image/[^)]*\)", " ", markdown, flags=re.DOTALL)
    markdown = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", markdown)
    markdown = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", markdown)
    return re.sub(r"\n{3,}", "\n\n", markdown).strip()


@lru_cache(maxsize=1)
def handbook_documents() -> tuple[Document, ...]:
    """Load every handbook Markdown file; API-key files are never read."""
    handbook_dir = _handbook_dir()
    source_documents: list[Document] = []
    excluded_name_tokens = {"api_key", "apikey", "secret", "credential", "token", "password"}
    for path in sorted(handbook_dir.glob("*.md")):
        normalized_name = path.name.lower().replace("-", "_").replace(" ", "_")
        if any(token in normalized_name for token in excluded_name_tokens):
            continue
        text = _clean_markdown(path.read_text(encoding="utf-8", errors="ignore"))
        if text:
            source_documents.append(Document(page_content=text, metadata={"source": path.name}))

    if not source_documents:
        raise HandbookCopilotError("No handbook Markdown documents were found for the Copilot.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1_600, chunk_overlap=220)
    return tuple(splitter.split_documents(source_documents))


def _candidate_chunks(question: str, *, limit: int = 12) -> list[Document]:
    """Use a fast lexical pass before embedding the most relevant handbook chunks."""
    terms = set(re.findall(r"[a-z0-9_]{3,}", question.lower()))
    documents = list(handbook_documents())
    if not terms:
        return documents[:limit]

    def score(document: Document) -> int:
        text = document.page_content.lower()
        relevance = sum(text.count(term) for term in terms)
        # This is the current controlled benchmark record. Prefer it to legacy
        # examples when the question concerns model performance.
        is_current_model_question = bool(
            terms & {"lightgbm", "mcc", "pr_auc", "precision", "recall", "threshold", "present"}
        )
        if (
            is_current_model_question
            and document.metadata.get("source") == "Bosch_Full_Data_Model_Performance_Deep_Dive.md"
        ):
            relevance += 50
        return relevance

    return sorted(documents, key=score, reverse=True)[:limit]


def build_retriever(question: str):
    """Create a local LangChain vector index for the current question."""
    try:
        embeddings = OllamaEmbeddings(model=LOCAL_EMBEDDING_MODEL)
        store = InMemoryVectorStore(embedding=embeddings)
        store.add_documents(_candidate_chunks(question))
        return store.as_retriever(search_kwargs={"k": 3})
    except Exception as error:
        raise HandbookCopilotError(
            "Ollama is not ready. Install Ollama, then run `ollama pull nomic-embed-text` and try again."
        ) from error


def _content_as_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    return str(content).strip()


def answer_handbook_question(question: str, *, retriever) -> HandbookAnswer:
    """Retrieve handbook evidence with LangChain and answer using local Ollama."""
    if not question.strip():
        raise HandbookCopilotError("Enter a project question before asking the Copilot.")
    try:
        sources = retriever.invoke(question)
        if not sources:
            raise HandbookCopilotError("No relevant handbook material was found for that question.")

        context = "\n\n".join(
            f"[{index}] {document.metadata.get('source', 'Bosch handbook')}\n{document.page_content}"
            for index, document in enumerate(sources, start=1)
        )
        llm = ChatOllama(
            model=LOCAL_CHAT_MODEL,
            temperature=0.2,
            # A small local model is most useful here for a short handbook answer.
            # Limiting the completion keeps the dashboard responsive on a laptop.
            num_predict=64,
            num_ctx=2_048,
            keep_alive="10m",
            client_kwargs={"timeout": 300},
        )
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are the Bosch Production Line Performance Copilot. Answer only from the supplied handbook "
                        "excerpts. Use clear, human language for a mixed plant and analytics audience. MCC means "
                        "Matthews correlation coefficient. Never invent metric definitions or values; say when the "
                        "provided excerpts do not state an answer. Answer the user's exact question in at most two "
                        "short sentences; do not introduce additional questions or extra metric definitions unless "
                        "asked. Cite claims with [1], [2], and so on."
                    )
                ),
                HumanMessage(content=f"Question: {question}\n\nHandbook excerpts:\n{context}"),
            ]
        )
        answer = _content_as_text(response.content)
        if not answer:
            raise HandbookCopilotError("Ollama returned an empty answer. Please try again.")
        return HandbookAnswer(answer=answer, sources=list(sources), model=LOCAL_CHAT_MODEL)
    except HandbookCopilotError:
        raise
    except Exception as error:
        raise HandbookCopilotError(
            "Ollama could not finish the response. Keep Ollama running and retry; the first answer can take a minute on a CPU-only computer."
        ) from error
