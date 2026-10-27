# tasks.md

Legend: [ ] todo  [~] in progress  [x] done — update as we go, this is the anchor source.

## Phase 0 — Setup
- [x] Repo + folder structure + spec files (PRD/schema/architecture/tasks/CLAUDE)
- [ ] `requirements.txt` (surprise, sentence-transformers, scikit-learn, pandas, streamlit, pyarrow)
- [ ] Download datasets → `data/raw/` (MovieLens 100K zip, Goodbooks-10k csvs)

## Phase 1 — Data
- [ ] `src/data_prep.py`: clean both domains → `data/processed/*.parquet`
- [ ] Sanity check: row counts, no-null text_blob, genre/tag distributions

## Phase 2 — Models
- [ ] `src/train_cf.py`: SVD per domain → `models/cf_movie_svd.pkl`, `models/cf_book_svd.pkl`
- [ ] `src/train_content.py`: content sim matrices per domain
- [ ] `src/train_shared_embed.py`: shared embedding space across both domains' genre/tag text

## Phase 3 — Recommend + Evaluate
- [ ] `src/recommend.py`: `Recommender` class (in-domain + cross-domain)
- [ ] `src/evaluate.py`: compute Precision@10/Recall@10/NDCG@10/RMSE/MAE for CF-only, content-only, hybrid; cross-domain vs popularity baseline
- [ ] Write `reports/metrics_*.json` + `reports/results.md` (numbers + 1-2 charts)

## Phase 4 — App
- [ ] `app/streamlit_app.py`: domain picker, rate-some-items or pick-taste-tags flow, recommendation list with poster/cover placeholder + score
- [ ] Local run + smoke test (`streamlit run app/streamlit_app.py`)

## Phase 5 — Deploy
- [ ] Push to GitHub
- [ ] Decide model-artifact strategy (commit small ones directly; if >100MB use Git LFS or a `scripts/build_artifacts.sh` first-run step — check sizes after Phase 2)
- [ ] Deploy on Streamlit Community Cloud, verify live URL

## Phase 6 — Report (4 required sections)
- [ ] §1 Datasets — MovieLens 100K + Goodbooks-10k description, stats, preprocessing
- [ ] §2 Existing Methodologies — CF (user/item-based, MF/SVD), content-based, hybrid systems, deep learning (NCF/autoencoders), cross-domain transfer approaches (cite general literature, no fabricated quotes)
- [ ] §3 Proposed Methodology — the unified hybrid + shared-embedding architecture (from architecture.md), with a diagram
- [ ] §4 Expected Results — pull from reports/results.md, present as tables/charts + discussion
- [ ] Export as PDF/Doc

## Delegation notes (per YESH's workflow)
- Phases 1–3 (data/model scripts) and Phase 4 (Streamlit app) are best run in **Cursor/OpenCode** locally — I'll hand over full file contents / exact prompts per phase, not drip-fed one function at a time.
- I handle: spec files (done), report writing, review/debugging guidance, deploy troubleshooting, and evaluation-results interpretation.
