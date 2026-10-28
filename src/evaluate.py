"""
src/evaluate.py

Evaluation for RecoSphere.

Metrics:
  - RMSE / MAE               : rating-prediction accuracy (CF only — content/hybrid
                                don't predict ratings, they rank items)
  - Precision@10 / Recall@10 / NDCG@10 : ranking quality, for CF-only, content-only,
                                and hybrid, per domain (movie, book)
  - Cross-domain proxy eval  : for users with ratings in BOTH domains, check whether
                                cross-domain recs (from their movie ratings) surface
                                books they rated >= 4 (and vice versa), vs a
                                popularity baseline

Does its own 80/20 per-user train/test split (not reusing train_cf.py's split,
so CF-only / content-only / hybrid are all judged on an identical held-out set).
Retrains a fresh SVD on the 80% train split for fair comparison — the saved
models/cf_*_svd.pkl (trained on 100% of data) are NOT used for scoring here,
only the saved content embeddings are (content embeddings don't leak test-set
rating info, so reusing them is fine).

Outputs:
  reports/metrics_movie.json
  reports/metrics_book.json
  reports/metrics_cross_domain.json
  reports/results.md
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader

from recommend import Recommender, _cosine_sim_matrix

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

K = 10
RELEVANCE_THRESHOLD = 4.0  # rating >= this counts as "relevant" for P/R/NDCG
RANDOM_SEED = 42


# ---------- data loading + split ----------

def load_ratings(domain: str) -> pd.DataFrame:
    path = DATA_DIR / f"{domain}_ratings.parquet"
    df = pd.read_parquet(path)
    # Expect columns: user_id, item_id, rating (adjust here if yours differ)
    return df[["user_id", "item_id", "rating"]]


def per_user_split(df: pd.DataFrame, test_frac: float = 0.2, seed: int = RANDOM_SEED):
    """80/20 split per user so every user with >=2 ratings has >=1 test rating."""
    rng = np.random.default_rng(seed)
    test_rows = []
    train_rows = []
    for user_id, group in df.groupby("user_id"):
        group = group.sample(frac=1, random_state=seed)  # shuffle
        n_test = max(1, int(len(group) * test_frac)) if len(group) > 1 else 0
        test_rows.append(group.iloc[:n_test])
        train_rows.append(group.iloc[n_test:])
    return pd.concat(train_rows, ignore_index=True), pd.concat(test_rows, ignore_index=True)


# ---------- CF: retrain on train split only ----------

def train_svd(train_df: pd.DataFrame):
    reader = Reader(rating_scale=(train_df.rating.min(), train_df.rating.max()))
    data = Dataset.load_from_df(train_df[["user_id", "item_id", "rating"]], reader)
    trainset = data.build_full_trainset()
    model = SVD(random_state=RANDOM_SEED)
    model.fit(trainset)
    return model


def rmse_mae(model, test_df: pd.DataFrame):
    errors = []
    for _, row in test_df.iterrows():
        pred = model.predict(str(row.user_id), str(row.item_id)).est
        errors.append(pred - row.rating)
    errors = np.array(errors)
    rmse = float(np.sqrt((errors ** 2).mean()))
    mae = float(np.abs(errors).mean())
    return rmse, mae


# ---------- ranking metrics ----------

def precision_recall_ndcg_at_k(ranked_item_ids, relevant_item_ids, k=K):
    """ranked_item_ids: list, best first. relevant_item_ids: set."""
    top_k = ranked_item_ids[:k]
    hits = [1 if iid in relevant_item_ids else 0 for iid in top_k]

    precision = sum(hits) / k
    recall = sum(hits) / len(relevant_item_ids) if relevant_item_ids else 0.0

    dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
    ideal_hits = [1] * min(len(relevant_item_ids), k)
    idcg = sum(h / np.log2(i + 2) for i, h in enumerate(ideal_hits))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return precision, recall, ndcg


def score_all_items_content(raw_emb, content_index, content_idx_rev, query_item_id):
    query_row = content_index[str(query_item_id)]
    sims = _cosine_sim_matrix(raw_emb[query_row], raw_emb)
    order = np.argsort(sims)[::-1]
    return [content_idx_rev[int(r)] for r in order if content_idx_rev[int(r)] != str(query_item_id)]


def score_all_items_cf(model, user_id, all_item_ids):
    scored = [(iid, model.predict(str(user_id), str(iid)).est) for iid in all_item_ids]
    scored.sort(key=lambda x: -x[1])
    return [iid for iid, _ in scored]


def score_all_items_hybrid(cf_ranked, content_ranked, all_item_ids, alpha=0.5):
    """Combine two rank-orderings by normalized rank position (not raw score,
    since CF predict() and cosine sim aren't on the same scale)."""
    n = len(all_item_ids)
    cf_rank_pos = {iid: i for i, iid in enumerate(cf_ranked)}
    content_rank_pos = {iid: i for i, iid in enumerate(content_ranked)}
    combined = []
    for iid in all_item_ids:
        cf_score = 1 - (cf_rank_pos.get(iid, n) / n)
        content_score = 1 - (content_rank_pos.get(iid, n) / n)
        combined.append((iid, alpha * cf_score + (1 - alpha) * content_score))
    combined.sort(key=lambda x: -x[1])
    return [iid for iid, _ in combined]


def evaluate_domain(domain: str, rec: Recommender):
    print(f"\n=== {domain} ===")
    full_df = load_ratings(domain)
    train_df, test_df = per_user_split(full_df)

    print(f"  train: {len(train_df)} ratings, test: {len(test_df)} ratings")
    model = train_svd(train_df)

    rmse, mae = rmse_mae(model, test_df)
    print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}")

    content_index = rec.content_index_movie if domain == "movie" else rec.content_index_book
    content_idx_rev = rec._content_idx_rev_movie if domain == "movie" else rec._content_idx_rev_book
    raw_emb = rec.raw_emb_movie if domain == "movie" else rec.raw_emb_book
    all_item_ids = list(content_index.keys())

    # relevant items per test user: rating >= threshold, held out
    relevant_by_user = (
        test_df[test_df.rating >= RELEVANCE_THRESHOLD]
        .groupby("user_id")["item_id"]
        .apply(lambda s: set(s.astype(str)))
        .to_dict()
    )

    methods = {"cf_only": [], "content_only": [], "hybrid": []}
    eval_users = [u for u in relevant_by_user if relevant_by_user[u]]
    # sample for speed if huge; book domain especially — full item scoring per user is O(n_items)
    if len(eval_users) > 300:
        rng = np.random.default_rng(RANDOM_SEED)
        eval_users = list(rng.choice(eval_users, size=300, replace=False))

    for user_id in eval_users:
        relevant = relevant_by_user[user_id]

        # need a "seed item" for content/hybrid: use the user's highest-rated train item
        user_train = train_df[train_df.user_id == user_id]
        if user_train.empty or str(user_train.iloc[user_train.rating.values.argmax()].item_id) not in content_index:
            continue
        seed_item = str(user_train.iloc[user_train.rating.values.argmax()].item_id)

        cf_ranked = score_all_items_cf(model, user_id, all_item_ids)
        content_ranked = score_all_items_content(raw_emb, content_index, content_idx_rev, seed_item)
        hybrid_ranked = score_all_items_hybrid(cf_ranked, content_ranked, all_item_ids)

        for name, ranked in [("cf_only", cf_ranked), ("content_only", content_ranked), ("hybrid", hybrid_ranked)]:
            p, r, n = precision_recall_ndcg_at_k(ranked, relevant, k=K)
            methods[name].append((p, r, n))

    results = {"rmse": rmse, "mae": mae, "n_eval_users": len(eval_users)}
    for name, vals in methods.items():
        if vals:
            arr = np.array(vals)
            results[name] = {
                "precision@10": float(arr[:, 0].mean()),
                "recall@10": float(arr[:, 1].mean()),
                "ndcg@10": float(arr[:, 2].mean()),
            }
        else:
            results[name] = None

    print(f"  evaluated on {len(eval_users)} users")
    for name in methods:
        if results[name]:
            m = results[name]
            print(f"  {name:14s} P@10={m['precision@10']:.4f}  R@10={m['recall@10']:.4f}  NDCG@10={m['ndcg@10']:.4f}")

    with open(REPORTS_DIR / f"metrics_{domain}.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ---------- cross-domain proxy eval ----------

def evaluate_cross_domain(rec: Recommender):
    print("\n=== cross-domain (movie -> book) ===")
    movie_df = load_ratings("movie")
    book_df = load_ratings("book")

    overlap_users = sorted(set(movie_df.user_id) & set(book_df.user_id))
    eval_users = overlap_users
    if len(eval_users) > 300:
        rng = np.random.default_rng(RANDOM_SEED)
        eval_users = list(rng.choice(eval_users, size=300, replace=False))

    book_popularity = book_df.groupby("item_id").size().sort_values(ascending=False)
    popular_books = [str(i) for i in book_popularity.index.tolist()]

    xdomain_hits, xdomain_total = 0, 0
    popularity_hits, popularity_total = 0, 0

    for user_id in eval_users:
        user_movies = movie_df[movie_df.user_id == user_id]
        user_books_liked = set(
            book_df[(book_df.user_id == user_id) & (book_df.rating >= RELEVANCE_THRESHOLD)]
            .item_id.astype(str)
        )
        if not user_books_liked or user_movies.empty:
            continue

        seed_movie = str(user_movies.iloc[user_movies.rating.values.argmax()].item_id)
        if f"movie_{seed_movie}" not in rec.shared_index:
            continue

        try:
            recs = rec.recommend_cross_domain(seed_movie, "movie", "book", k=K)
        except KeyError:
            continue
        rec_ids = {r["item_id"] for r in recs}

        xdomain_hits += len(rec_ids & user_books_liked)
        xdomain_total += K

        pop_ids = set(popular_books[:K])
        popularity_hits += len(pop_ids & user_books_liked)
        popularity_total += K

    results = {
        "n_eval_users": len(eval_users),
        "cross_domain_precision@10": xdomain_hits / xdomain_total if xdomain_total else 0.0,
        "popularity_baseline_precision@10": popularity_hits / popularity_total if popularity_total else 0.0,
    }
    print(f"  evaluated on {results['n_eval_users']} users")
    print(f"  cross-domain P@10:  {results['cross_domain_precision@10']:.4f}")
    print(f"  popularity P@10:    {results['popularity_baseline_precision@10']:.4f}")

    with open(REPORTS_DIR / "metrics_cross_domain.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ---------- report ----------

def write_results_md(movie_results, book_results, cross_results):
    lines = ["# RecoSphere Evaluation Results\n"]

    lines.append("## Rating prediction (CF)\n")
    lines.append("| Domain | RMSE | MAE |")
    lines.append("|---|---|---|")
    lines.append(f"| Movie | {movie_results['rmse']:.4f} | {movie_results['mae']:.4f} |")
    lines.append(f"| Book | {book_results['rmse']:.4f} | {book_results['mae']:.4f} |\n")

    for domain, results in [("Movie", movie_results), ("Book", book_results)]:
        lines.append(f"## {domain} — ranking quality (@{K})\n")
        lines.append("| Method | Precision@10 | Recall@10 | NDCG@10 |")
        lines.append("|---|---|---|---|")
        for name, label in [("cf_only", "CF only"), ("content_only", "Content only"), ("hybrid", "Hybrid")]:
            m = results[name]
            if m:
                lines.append(f"| {label} | {m['precision@10']:.4f} | {m['recall@10']:.4f} | {m['ndcg@10']:.4f} |")
            else:
                lines.append(f"| {label} | n/a | n/a | n/a |")
        lines.append(f"\n(evaluated on {results['n_eval_users']} users)\n")

    lines.append("## Cross-domain cold-start (movie -> book) vs popularity baseline\n")
    lines.append("| Method | Precision@10 |")
    lines.append("|---|---|")
    lines.append(f"| Cross-domain (shared embedding) | {cross_results['cross_domain_precision@10']:.4f} |")
    lines.append(f"| Popularity baseline | {cross_results['popularity_baseline_precision@10']:.4f} |")
    lines.append(f"\n(evaluated on {cross_results['n_eval_users']} users with ratings in both domains)\n")

    (REPORTS_DIR / "results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    rec = Recommender()
    movie_results = evaluate_domain("movie", rec)
    book_results = evaluate_domain("book", rec)
    cross_results = evaluate_cross_domain(rec)
    write_results_md(movie_results, book_results, cross_results)
    print("\nWrote reports/metrics_movie.json, metrics_book.json, metrics_cross_domain.json, results.md")
