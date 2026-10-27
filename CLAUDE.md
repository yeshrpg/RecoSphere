# CLAUDE.md — context anchor for this project

Read PRD.md, schema.md, architecture.md, tasks.md before doing any work here. Spec is locked — don't re-decide domains/stack/architecture mid-build; only tasks.md checkboxes move.

## Locked decisions
- Domains: Movies (MovieLens 100K) + Books (Goodbooks-10k)
- Method: Hybrid (CF-SVD + content embeddings) per domain, plus a shared sentence-transformer embedding space for cross-domain cold-start recs
- App: Streamlit, hosted on Hugging Face Spaces (Streamlit SDK)
- No auth, no real-time retraining, no mobile app (out of scope — see PRD §7)

## Conventions
- Conventional commits: `feat:` / `fix:` / `chore:` / `docs:` — commit + push at every safe checkpoint
- Model artifacts go in `models/` (check `.gitignore` — large files may need Git LFS, decide at Phase 5)
- All offline training/eval happens in `src/` scripts, never inside the Streamlit app at request time

## Anchor Checkpoint (update at end of each session)
**2026-10-27**: Spec files created (PRD, schema, architecture, tasks, CLAUDE). Nothing built yet. Next: requirements.txt + dataset download (Phase 0 remainder), then src/data_prep.py (Phase 1).
