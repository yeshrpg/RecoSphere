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

**2026-10-28 (latest)**: Project renamed RecoSphere, repo live at github.com/yeshrpg/RecoSphere, hosting locked to Hugging Face Spaces (Streamlit SDK). Phases 0–2 complete and pushed:
- Phase 0: venv + requirements.txt + both datasets downloaded (MovieLens 100K from files.grouplens.org/datasets/movielens/ml-100k.zip — grouplens.org itself was flaky; Goodbooks-10k from GitHub zygmuntz/goodbooks-10k).
- Phase 1: `src/data_prep.py` run clean → 1682 movies/100k ratings, 10000 books/5.97M ratings in `data/processed/*.parquet`.
- Phase 2: `src/train_cf.py` run clean → movie SVD (RMSE=0.9348, MAE=0.7377), book SVD (RMSE=0.8312, MAE=0.6424), saved to `models/cf_*_svd.pkl`. `src/train_shared_embed.py` run clean → shared cross-domain space, 11682 items, dim=384, in `models/shared_embeddings.npy` + `shared_item_index.json`.
- **Known issue fixed in code, not yet rerun**: `src/train_content.py` originally saved full NxN cosine similarity matrices; book domain's was ~381MB (too big for GitHub/HF Spaces free tier). Rewrote it to save only raw embeddings (`models/raw_emb_movie.npy`, `models/raw_emb_book.npy`, ~15MB for books) and compute cosine similarity on-the-fly at inference time instead. **schema.md already updated to reflect this. The new train_content.py has been handed to the user but not yet rerun on their machine — the old 381MB content_sim_book.npy / content_sim_movie.npy files may still be sitting in their local models/ folder and should be deleted, then `python src\train_content.py` rerun, before Phase 3 scripts are written** (recommend.py's content-based scoring depends on raw_emb_*.npy existing, not content_sim_*.npy).
- User is running behind their original schedule but explicitly wants **no scope cuts, no quality compromise** — compress by working faster/more in parallel, not by dropping methodology, evaluation rigor, or report depth.

### Next up (Phase 3)
1. User reruns `src\train_content.py` (new version) to produce `raw_emb_movie.npy` / `raw_emb_book.npy` and replace the oversized matrix files.
2. Write `src/recommend.py` — `Recommender` class: in-domain hybrid (CF score + on-the-fly content cosine sim) and cross-domain (shared embedding space lookup).
3. Write `src/evaluate.py` — Precision@10/Recall@10/NDCG@10/RMSE/MAE for CF-only vs content-only vs hybrid, cross-domain vs popularity baseline, outputs to `reports/metrics_*.json` + `reports/results.md`.
4. Then Phase 4 (Streamlit app), Phase 5 (deploy to HF Spaces — remember two git remotes, GitHub + HF Space), Phase 6 (report, 4 sections, can draft §1/§2 in parallel any time since they don't depend on results).
