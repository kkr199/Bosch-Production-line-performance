# Bosch Production Line Performance Handbook

This folder contains the searchable corpus used by the Streamlit Offline Project Q&A Agent.

- The original handbook documents are stored in [`Bosch_Handbook/`](../../Bosch_Handbook/).
- `bosch_handbook_corpus.json` is generated from those Word documents for fast, local retrieval in the dashboard.
- Rebuild the corpus after editing the handbook with:

  ```powershell
  python scripts/build_handbook_corpus.py
  ```

The Copilot uses the handbook as its primary source for explanatory questions and complements it with project reports, dashboard tables, and curated metric answers when a question needs project-specific evidence.
