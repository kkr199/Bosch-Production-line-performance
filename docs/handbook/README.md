# Bosch Production Line Performance Handbook

This folder contains the searchable corpus used by the Streamlit Offline Project Q&A Agent.

- The original handbook documents are stored in [`Bosch_Handbook/`](../../Bosch_Handbook/).
- `bosch_handbook_corpus.json` is generated from those Word documents for fast, local retrieval in the dashboard.
- Rebuild the corpus after editing the handbook with:

  ```powershell
  python scripts/build_handbook_corpus.py
  ```

The Copilot uses the handbook as its primary source for explanatory questions and complements it with project reports, dashboard tables, and curated metric answers when a question needs project-specific evidence.

## Local Ollama answers

The unified Streamlit dashboard first retrieves relevant excerpts from every
`Bosch_Handbook_*.md` file in `Bosch_Handbook_md/` (or the equivalent
`Bosch_Handbook_MD/` folder on Windows). It uses Ollama's local embedding
model to retrieve relevant excerpts, then sends them to a local Ollama language
model for a natural-language answer with numbered citations.
The visible **References used** table lets a reader inspect the exact sources.

Install Ollama on the machine that runs Streamlit, then download the local
models once:

```powershell
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

No API key or cloud AI service is used. This local-only Copilot cannot run on
Streamlit Community Cloud, because that service cannot host the Ollama runtime.
