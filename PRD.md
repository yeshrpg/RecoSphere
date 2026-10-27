# PRD — RecoSphere (AI-Based Multi-Domain Recommendation System)

## 1. Problem
Build an ML-based recommender that works across two domains (Movies, Books) under one unified system, demo'd as a hosted web app, with an accompanying academic report covering datasets, existing methodologies, proposed methodology, and expected results.

## 2. Why "multi-domain" (the real challenge)
MovieLens (movies) and Goodbooks-10k (books) have **no overlapping users** — this is the core research problem, not a detail to hide. We solve it with a shared semantic embedding space so a user's taste expressed in one domain can drive recommendations in the other (cross-domain cold-start transfer), rather than training two unrelated models side by side.

## 3. Domains & Datasets (locked)
- **Movies**: MovieLens 100K (ratings.csv, movies.csv) — small enough to ship with repo, well documented, has genres.
- **Books**: Goodbooks-10k (ratings.csv, books.csv, tags.csv, book_tags.csv) — clean, has tags/genres analogous to movie genres.
- Both are public, CSV-based, no API keys needed, CC/open-ish licenses — safe for a hosted demo.

## 4. Proposed Methodology (novelty for report)
**Unified Hybrid Cross-Domain Recommender**, three layers per domain + one shared layer:
1. **Collaborative Filtering** (per domain): matrix factorization (SVD via `surprise` or `implicit`) on domain ratings → in-domain personalized recs.
2. **Content-Based Filtering** (per domain): TF-IDF / sentence-transformer embeddings over genres+tags+title → item-item similarity, handles item cold-start.
3. **Shared Semantic Layer** (cross-domain): embed both domains' genre/tag vocab into one sentence-transformer latent space (e.g., "Action/Adventure" movie genre ↔ "adventure" book tag) → lets a synthetic/explicit user taste profile ("I like sci-fi, mystery") produce recommendations in *either* domain, solving cross-domain user cold-start.
4. **Hybrid reranker**: weighted blend of CF score + content score (+ cross-domain score when applicable), weights tunable, evaluated offline.

## 5. Expected Results (report section 4 — to fill in after running)
- In-domain: hybrid should beat pure-CF and pure-content on Precision@10 / Recall@10 / NDCG@10, and beat pure-CF on RMSE/MAE where content fills cold-start gaps.
- Cross-domain cold-start: shared-embedding recs should beat a random/popularity baseline by a measurable margin (reported via qualitative + Precision@K on held-out tag overlap, since no ground-truth cross-domain ratings exist).
- All results go into `reports/results.md` + chart images, then into the final report doc.

## 6. Deliverables
1. Working hybrid recommender (movies + books + cross-domain layer) as Python package.
2. Hosted web app — Streamlit on Hugging Face Spaces (Streamlit SDK) (free, fastest path for an ML demo UI, no separate frontend/backend to wire).
3. Academic report (PDF/Doc) with exactly 4 required sections: Datasets, Existing Methodologies, Proposed Methodology, Expected Results.
4. GitHub repo, spec-driven file set, committed at every safe checkpoint.

## 7. Out of scope (to hit the deadline)
- No user auth / accounts — demo uses pre-set sample users + a "build your taste profile" mode.
- No real-time retraining — models trained offline, loaded as artifacts in the app.
- No mobile app — responsive web only.
