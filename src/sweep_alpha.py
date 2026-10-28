"""
src/sweep_alpha.py

Sweeps the hybrid alpha weight (CF vs content) across both domains and
reports P@10/R@10/NDCG@10 for each value, so the report can cite an
empirically chosen alpha instead of a hardcoded 0.5.

Reuses evaluate.py's split/train/score functions directly rather than
reimplementing — run after evaluate.py has been run at least once
(not a hard dependency, just avoids duplicate logic).
"""

import json
from pathlib import Path

import numpy as np

from evaluate import (
    K, REPORTS_DIR, RELEVANCE_THRESHOLD, RANDOM_SEED,
    load_ratings, per_user_split, train_svd,
    score_all_items_cf, score_all_items_content, score_all_items_hybrid,
    precision_recall_ndcg_at_k,
)
from recommend import Recommender

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def sweep_domain(domain: str, rec: Recommender):
    print(f"\n=== alpha sweep: {domain} ===")
    full_df = load_ratings(domain)
    train_df, test_df = per_user_split(full_df)
    model = train_svd(train_df)

    content_index = rec.content_index_movie if domain == "movie" else rec.content_index_book
    content_idx_rev = rec._content_idx_rev_movie if domain == "movie" else rec._content_idx_rev_book
    raw_emb = rec.raw_emb_movie if domain == "movie" else rec.raw_emb_book
    all_item_ids = list(content_index.keys())

    relevant_by_user = (
        test_df[test_df.rating >= RELEVANCE_THRESHOLD]
        .groupby("user_id")["item_id"]
        .apply(lambda s: set(s.astype(str)))
        .to_dict()
    )
    eval_users = [u for u in relevant_by_user if relevant_by_user[u]]
    if len(eval_users) > 200:  # smaller sample than evaluate.py — 5 alphas x this many users
        rng = np.random.default_rng(RANDOM_SEED)
        eval_users = list(rng.choice(eval_users, size=200, replace=False))

    # precompute per-user cf_ranked / content_ranked once, reuse across all alphas
    per_user_rankings = {}
    for user_id in eval_users:
        user_train = train_df[train_df.user_id == user_id]
        if user_train.empty:
            continue
        seed_item = str(user_train.iloc[user_train.rating.values.argmax()].item_id)
        if seed_item not in content_index:
            continue
        cf_ranked = score_all_items_cf(model, user_id, all_item_ids)
        content_ranked = score_all_items_content(raw_emb, content_index, content_idx_rev, seed_item)
        per_user_rankings[user_id] = (cf_ranked, content_ranked)

    results = {}
    for alpha in ALPHAS:
        vals = []
        for user_id, (cf_ranked, content_ranked) in per_user_rankings.items():
            relevant = relevant_by_user[user_id]
            hybrid_ranked = score_all_items_hybrid(cf_ranked, content_ranked, all_item_ids, alpha=alpha)
            p, r, n = precision_recall_ndcg_at_k(hybrid_ranked, relevant, k=K)
            vals.append((p, r, n))
        arr = np.array(vals)
        results[alpha] = {
            "precision@10": float(arr[:, 0].mean()),
            "recall@10": float(arr[:, 1].mean()),
            "ndcg@10": float(arr[:, 2].mean()),
        }
        print(f"  alpha={alpha:.2f}  P@10={results[alpha]['precision@10']:.4f}  "
              f"R@10={results[alpha]['recall@10']:.4f}  NDCG@10={results[alpha]['ndcg@10']:.4f}")

    best_alpha = max(results, key=lambda a: results[a]["ndcg@10"])
    print(f"  best alpha by NDCG@10: {best_alpha}")

    with open(REPORTS_DIR / f"alpha_sweep_{domain}.json", "w") as f:
        json.dump({"results": results, "best_alpha": best_alpha}, f, indent=2)

    return results, best_alpha


if __name__ == "__main__":
    rec = Recommender()
    movie_sweep, movie_best = sweep_domain("movie", rec)
    book_sweep, book_best = sweep_domain("book", rec)
    print(f"\nRecommended alpha — movie: {movie_best}, book: {book_best}")
    print("Wrote reports/alpha_sweep_movie.json, reports/alpha_sweep_book.json")
