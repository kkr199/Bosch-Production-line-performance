"""LangChain RAG Copilot grounded in the Bosch Markdown handbook."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HANDBOOK_DIR_NAMES = ("Bosch_Handbook_md", "Bosch_Handbook_MD")
EMBEDDING_MODEL = "models/gemini-embedding-001"


class HandbookCopilotError(RuntimeError):
    """A safe, user-facing error for an unavailable LangChain/Gemini request."""


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
    for path in sorted(handbook_dir.glob("Bosch_Handbook_*.md")):
        text = _clean_markdown(path.read_text(encoding="utf-8", errors="ignore"))
        if text:
            source_documents.append(Document(page_content=text, metadata={"source": path.name}))

    if not source_documents:
        raise HandbookCopilotError("No handbook Markdown documents were found for the Copilot.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1_600, chunk_overlap=220)
    return tuple(splitter.split_documents(source_documents))


def build_retriever(api_key: str):
    """Create an in-memory, per-process LangChain vector index.

    Handbook chunks are embedded for retrieval. The index is held only in the
    running application process; it is not stored in the API key or committed
    to the repository.
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
        store = InMemoryVectorStore(embedding=embeddings)
        store.add_documents(list(handbook_documents()))
        return store.as_retriever(search_kwargs={"k": 6})
    except HandbookCopilotError:
        raise
    except Exception as error:
        raise HandbookCopilotError(
            "Gemini could not build the handbook search index. Check the API key, Gemini API access, and quota."
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


def answer_handbook_question(question: str, *, api_key: str, model: str, retriever) -> HandbookAnswer:
    """Retrieve handbook evidence with LangChain and answer with Gemini only."""
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
        llm = ChatGoogleGenerativeAI(
            model=model,
            api_key=api_key,
            temperature=0.2,
            max_tokens=900,
            retries=2,
        )
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are the Bosch Production Line Performance Copilot. Answer only from the supplied handbook "
                        "excerpts. Use clear, human language for a mixed plant and analytics audience. Do not invent "
                        "facts or claim that a recommendation is already deployed. Cite claims with [1], [2], and so on."
                    )
                ),
                HumanMessage(content=f"Question: {question}\n\nHandbook excerpts:\n{context}"),
            ]
        )
        answer = _content_as_text(response.content)
        if not answer:
            raise HandbookCopilotError("Gemini returned an empty answer. Please try again.")
        return HandbookAnswer(answer=answer, sources=list(sources), model=model)
    except HandbookCopilotError:
        raise
    except Exception as error:
        raise HandbookCopilotError(
            "Gemini could not answer this question. Check the API key, selected model, and Gemini quota."
        ) from error
