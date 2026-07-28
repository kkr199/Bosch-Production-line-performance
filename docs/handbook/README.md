# Bosch Production Line Performance Handbook

This folder contains the searchable corpus used by the Streamlit Offline Project Q&A Agent.

- The original handbook documents are stored in [`Bosch_Handbook/`](../../Bosch_Handbook/).
- `bosch_handbook_corpus.json` is generated from those Word documents for fast, local retrieval in the dashboard.
- Rebuild the corpus after editing the handbook with:

  ```powershell
  python scripts/build_handbook_corpus.py
  ```

The Copilot uses the handbook as its primary source for explanatory questions and complements it with project reports, dashboard tables, and curated metric answers when a question needs project-specific evidence.

## Gemini-backed answers

The unified Streamlit dashboard first retrieves relevant excerpts from every
`Bosch_Handbook_*.md` file in `Bosch_Handbook_md/` (or the equivalent
`Bosch_Handbook_MD/` folder on Windows). It then sends only those retrieved
excerpts to Gemini to write a natural-language answer with numbered citations.
The visible **References used** table lets a reader inspect the exact sources.

For Streamlit Community Cloud, configure these application secrets; do not put
the key in a repository file:

```toml
GEMINI_API_KEY = "your-key"
GEMINI_MODEL = "gemini-2.0-flash"
```

If no key is configured, or Gemini is temporarily unavailable, the Copilot
falls back to the local handbook retrieval answer instead of failing.
