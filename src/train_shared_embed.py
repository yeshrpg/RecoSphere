"""
Phase 2c — build the shared cross-domain embedding space.
Both domains were already embedded with the SAME sentence-transformer model
in train_content.py (that's what makes them comparable) — this script just
merges them into one unified index + matrix so recommend.py can do
cross-domain lookups (e.g. "user likes these movies -> recommend books").
Run from repo root: python src\\train_shared_embed.py
Requires train_content.py to have been run first (reads its raw_emb_*.npy).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

MODELS = Path("models")
PROC = Path("data/processed")


def main():
    movie_emb = np.load(MODELS / "raw_emb_movie.npy")
    book_emb = np.load(MODELS / "raw_emb_book.npy")

    movies = pd.read_parquet(PROC / "movies_clean.parquet")
    books = pd.read_parquet(PROC / "books_clean.parquet")

    assert len(movie_emb) == len(movies)
    assert len(book_emb) == len(books)

    # stack into one shared space: rows 0..N-1 = movies, N..N+M-1 = books
    shared_matrix = np.vstack([movie_emb, book_emb]).astype("float32")

    shared_index = []
    for item_id in movies["item_id"].tolist():
        shared_index.append({"domain": "movie", "item_id": int(item_id)})
    for item_id in books["item_id"].tolist():
        shared_index.append({"domain": "book", "item_id": int(item_id)})

    np.save(MODELS / "shared_embeddings.npy", shared_matrix)
    with open(MODELS / "shared_item_index.json", "w") as f:
        json.dump(shared_index, f)

    print(f"Shared embedding space: {shared_matrix.shape[0]} items "
          f"({len(movies)} movies + {len(books)} books), dim={shared_matrix.shape[1]}")
    print("Saved -> models/shared_embeddings.npy, models/shared_item_index.json")


if __name__ == "__main__":
    main()
