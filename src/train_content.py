"""
Phase 2b — build content-based item embeddings per domain (sentence-transformers
over each item's text_blob). We save the raw embeddings + an item_id index, NOT a
full NxN similarity matrix — for books (10000 items) that matrix would be ~381MB
(too big for GitHub/HF Spaces free tier). Cosine similarity is instead computed
on-the-fly per query in recommend.py (cheap: one item's embedding vs all others,
O(n*d), not O(n^2) storage). Same recommendation quality, no storage problem.

Run from repo root: python src\\train_content.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

PROC = Path("data/processed")
MODELS = Path("models")
MODELS.mkdir(exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"


def build_embeddings(domain: str, clean_path: str, emb_out: str, index_out: str, model):
    df = pd.read_parquet(PROC / clean_path)
    df["text_blob"] = df["text_blob"].fillna("")
    print(f"[{domain}] embedding {len(df)} items...")

    embeddings = model.encode(df["text_blob"].tolist(), batch_size=64,
                               show_progress_bar=True, normalize_embeddings=True)
    embeddings = embeddings.astype("float32")

    np.save(MODELS / emb_out, embeddings)

    index_map = {str(item_id): i for i, item_id in enumerate(df["item_id"].tolist())}
    with open(MODELS / index_out, "w") as f:
        json.dump(index_map, f)

    print(f"[{domain}] saved embeddings {embeddings.shape} -> models/{emb_out}, index -> models/{index_out}")
    return embeddings, df


if __name__ == "__main__":
    model = SentenceTransformer(MODEL_NAME)

    build_embeddings("movie", "movies_clean.parquet", "raw_emb_movie.npy", "content_index_movie.json", model)
    build_embeddings("book", "books_clean.parquet", "raw_emb_book.npy", "content_index_book.json", model)

    print("Done. Embeddings + indexes saved in models/. "
          "(No full similarity matrices saved — computed on-the-fly at inference time.)")
