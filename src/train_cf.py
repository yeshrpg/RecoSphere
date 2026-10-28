"""
Phase 2a — train collaborative filtering (SVD) models per domain.
Run from repo root: python src\\train_cf.py  (or python src/train_cf.py)
"""
import pickle
from pathlib import Path
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

PROC = Path("data/processed")
MODELS = Path("models")
MODELS.mkdir(exist_ok=True)


def train_domain_cf(domain: str, ratings_path: str, out_path: str):
    df = pd.read_parquet(PROC / ratings_path)
    print(f"[{domain}] {len(df)} ratings, {df['user_id'].nunique()} users, {df['item_id'].nunique()} items")

    reader = Reader(rating_scale=(df["rating"].min(), df["rating"].max()))
    data = Dataset.load_from_df(df[["user_id", "item_id", "rating"]], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    algo = SVD(n_factors=50, n_epochs=20, random_state=42)
    algo.fit(trainset)

    preds = algo.test(testset)
    rmse = accuracy.rmse(preds, verbose=False)
    mae = accuracy.mae(preds, verbose=False)
    print(f"[{domain}] RMSE={rmse:.4f}  MAE={mae:.4f}")

    # refit on FULL data before saving — test split was only for the metric above
    full_trainset = data.build_full_trainset()
    algo.fit(full_trainset)

    with open(MODELS / out_path, "wb") as f:
        pickle.dump(algo, f)
    print(f"[{domain}] saved -> models/{out_path}")

    return {"rmse": rmse, "mae": mae}


if __name__ == "__main__":
    import json
    results = {}
    results["movie"] = train_domain_cf("movie", "movie_ratings.parquet", "cf_movie_svd.pkl")
    results["book"] = train_domain_cf("book", "book_ratings.parquet", "cf_book_svd.pkl")

    Path("reports").mkdir(exist_ok=True)
    with open("reports/cf_train_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Done. CF models in models/, metrics in reports/cf_train_metrics.json")
