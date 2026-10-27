# architecture.md

## Stack
- **Language**: Python 3.11
- **CF**: `scikit-surprise` (SVD) — simple, well-documented, fast enough for MovieLens 100K / Goodbooks 10k scale
- **Content embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`) — small, CPU-friendly, good enough for genre/tag text
- **Similarity/vector ops**: numpy, scikit-learn (cosine_similarity)
- **App/UI**: Streamlit
- **Hosting**: Streamlit Community Cloud (free tier, connects directly to GitHub repo, zero infra setup — best fit given deadline + "most suitable" hosting requirement)
- **Packaging**: requirements.txt (no heavy deep learning framework needed — keeps Streamlit Cloud build fast/within free resource limits)

## Why this stack over alternatives (for report §2 "existing methodologies" contrast + justification)
- Full Neural CF (PyTorch) would need GPU-ish training time and a heavier deploy — too slow for deadline, marginal gain over SVD+content hybrid for datasets this size.
- React+FastAPI would need two hosts (e.g. Vercel + Render) and CORS/auth wiring — more moving parts, more failure points, no UX benefit for a recommender demo.
- Streamlit gives: file upload/sliders for "build your taste profile", tables + charts out of the box, single repo → single deploy.

## Pipeline (offline, run once via scripts, not in Streamlit at runtime)
```
data/raw → clean/preprocess → data/processed
  → train CF (per domain)        → models/cf_*_svd.pkl
  → build content sim matrix     → models/content_sim_*.npy
  → build shared embedding space → models/shared_embeddings.npy
  → evaluate all                 → reports/metrics_*.json
```
Streamlit app loads only the pre-trained artifacts — no training at request time, so the hosted demo is fast and never breaks on cold start.

## Module layout
```
src/
  data_prep.py        # clean raw CSVs -> data/processed/*.parquet
  train_cf.py          # train + save SVD models per domain
  train_content.py     # build TF-IDF/embedding sim matrices
  train_shared_embed.py # cross-domain shared embedding space
  recommend.py          # Recommender class: in-domain + cross-domain inference
  evaluate.py            # Precision@K/Recall@K/NDCG@K/RMSE/MAE, writes reports/metrics_*.json
app/
  streamlit_app.py      # UI: pick domain, rate items or pick taste tags, show recs
```

## `Recommender` class contract (src/recommend.py)
```python
class Recommender:
    def __init__(self, domain: str): ...
    def recommend_in_domain(self, user_ratings: dict[int, float], k=10) -> list[dict]: ...
    def recommend_cross_domain(self, source_domain: str, source_ratings: dict,
                                target_domain: str, k=10) -> list[dict]: ...
```
Hybrid score = `alpha * cf_score + beta * content_score` (+ cross-domain embedding score when applicable); alpha/beta tuned in evaluate.py and logged in reports/results.md.

## Deployment steps (Tier — do this last, after app works locally)
1. Push repo to GitHub (public, since Streamlit Cloud free tier needs that — or a connected private repo if available).
2. Commit trained model artifacts under `models/` via Git LFS *or* regenerate them in a `setup.sh`/first-run script if repo size is a concern (decide once artifact sizes are known — flagged in tasks.md).
3. streamlit.io/cloud → New app → point at repo/branch/`app/streamlit_app.py` → deploy.
4. Verify live URL, add to README + report.
