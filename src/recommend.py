"""
src/recommend.py

Recommender class for RecoSphere.
Two modes:
  - recommend_in_domain: hybrid CF + content, within movie or book domain
  - recommend_cross_domain: shared embedding space lookup, movie<->book cold-start

All artifacts are loaded once in __init__.
"""

import json
import pickle
from pathlib import Path

import numpy as np

MODELS_DIR = Path("models")


def _cosine_sim_matrix(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between one vector and every row of matrix."""
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return matrix_norm @ query_norm


class Recommender:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir

        # CF models (scikit-surprise SVD, pickled)
        with open(self.models_dir / "cf_movie_svd.pkl", "rb") as f:
            self.cf_movie = pickle.load(f)
        with open(self.models_dir / "cf_book_svd.pkl", "rb") as f:
            self.cf_book = pickle.load(f)

        # Raw content embeddings + their item-id indexes
        self.raw_emb_movie = np.load(self.models_dir / "raw_emb_movie.npy")
        self.raw_emb_book = np.load(self.models_dir / "raw_emb_book.npy")

        with open(self.models_dir / "content_index_movie.json") as f:
            self.content_index_movie = json.load(f)  # item_id -> row index
        with open(self.models_dir / "content_index_book.json") as f:
            self.content_index_book = json.load(f)

        self._content_idx_rev_movie = {v: k for k, v in self.content_index_movie.items()}
        self._content_idx_rev_book = {v: k for k, v in self.content_index_book.items()}

        # Shared cross-domain embedding space.
        # shared_item_index.json is a list of {"domain": "movie"/"book", "item_id": N},
        # position in the list == row index into shared_embeddings.npy.
        self.shared_emb = np.load(self.models_dir / "shared_embeddings.npy")
        with open(self.models_dir / "shared_item_index.json") as f:
            shared_raw = json.load(f)

        # self.shared_index: "domain_item_id" string key -> row index
        # self._shared_idx_rev: row index -> "domain_item_id" string key
        self.shared_index = {
            f"{entry['domain']}_{entry['item_id']}": i
            for i, entry in enumerate(shared_raw)
        }
        self._shared_idx_rev = {i: key for key, i in self.shared_index.items()}

    # ---------- helpers ----------

    def _domain_assets(self, domain: str):
        if domain == "movie":
            return self.cf_movie, self.raw_emb_movie, self.content_index_movie, self._content_idx_rev_movie
        elif domain == "book":
            return self.cf_book, self.raw_emb_book, self.content_index_book, self._content_idx_rev_book
        raise ValueError(f"Unknown domain: {domain}")

    def _cf_score(self, cf_model, user_id, item_id) -> float:
        return cf_model.predict(str(user_id), str(item_id)).est

    # ---------- public API ----------

    def recommend_in_domain(self, item_id, domain: str, k: int = 10, alpha: float = 0.5,
                             user_id=None):
        """
        Recommend items similar to item_id within the same domain, using a
        weighted hybrid of CF score and content cosine similarity.

        alpha=1.0 -> pure CF, alpha=0.0 -> pure content.
        If user_id is None, CF score falls back to item-item neighborhood
        via the CF model's latent item factors instead of a user prediction.
        """
        cf_model, raw_emb, content_index, content_idx_rev = self._domain_assets(domain)

        if str(item_id) not in content_index:
            raise KeyError(f"item_id {item_id} not found in {domain} content index")

        query_row = content_index[str(item_id)]
        query_vec = raw_emb[query_row]
        content_scores = _cosine_sim_matrix(query_vec, raw_emb)

        if user_id is not None:
            cf_scores = np.array([
                self._cf_score(cf_model, user_id, content_idx_rev[i])
                for i in range(raw_emb.shape[0])
            ])
            # normalize CF scores to 0-1 to combine fairly with cosine sim
            cf_min, cf_max = cf_scores.min(), cf_scores.max()
            cf_scores = (cf_scores - cf_min) / (cf_max - cf_min + 1e-10)
        else:
            # item-item CF similarity via latent factors (surprise SVD: qi matrix)
            try:
                inner_id = cf_model.trainset.to_inner_iid(str(item_id))
                item_factors = cf_model.qi
                query_factor = item_factors[inner_id]
                cf_scores = _cosine_sim_matrix(query_factor, item_factors)
                # map trainset inner ids back to raw-index order
                aligned = np.zeros(raw_emb.shape[0])
                for inner in range(item_factors.shape[0]):
                    raw_iid = cf_model.trainset.to_raw_iid(inner)
                    if raw_iid in content_index:
                        aligned[content_index[raw_iid]] = cf_scores[inner]
                cf_scores = aligned
            except ValueError:
                # item unseen in CF trainset — fall back to content-only
                cf_scores = content_scores.copy()

        hybrid = alpha * cf_scores + (1 - alpha) * content_scores
        hybrid[query_row] = -np.inf  # exclude the query item itself

        top_rows = np.argsort(hybrid)[::-1][:k]
        return [
            {"item_id": content_idx_rev[int(r)], "score": float(hybrid[r])}
            for r in top_rows
        ]

    def recommend_cross_domain(self, item_id, source_domain: str, target_domain: str, k: int = 10):
        """
        Cold-start cross-domain recs via the shared embedding space.
        e.g. source_domain="movie", target_domain="book"
        """
        source_key = f"{source_domain}_{item_id}"
        if source_key not in self.shared_index:
            raise KeyError(f"{source_key} not found in shared embedding index")

        query_row = self.shared_index[source_key]
        query_vec = self.shared_emb[query_row]
        scores = _cosine_sim_matrix(query_vec, self.shared_emb)

        # mask out everything not in target domain, and the query itself
        masked_scores = scores.copy()
        for key, row in self.shared_index.items():
            if not key.startswith(f"{target_domain}_"):
                masked_scores[row] = -np.inf
        masked_scores[query_row] = -np.inf

        top_rows = np.argsort(masked_scores)[::-1][:k]
        return [
            {"item_id": self._shared_idx_rev[int(r)].split("_", 1)[1], "score": float(masked_scores[r])}
            for r in top_rows
        ]


if __name__ == "__main__":
    rec = Recommender()
    print("In-domain (movie, content+CF item-item, no user):")
    print(rec.recommend_in_domain(item_id=1, domain="movie", k=5))
    print("\nCross-domain (movie -> book):")
    print(rec.recommend_cross_domain(item_id=1, source_domain="movie", target_domain="book", k=5))
