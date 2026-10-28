# schema.md — Data & Model Artifacts

## Raw data (data/raw/)
**Movies — MovieLens 100K**
- `movies.csv`: movieId, title, genres (pipe-separated)
- `ratings.csv`: userId, movieId, rating, timestamp

**Books — Goodbooks-10k**
- `books.csv`: book_id, goodreads_book_id, title, authors, average_rating, ...
- `ratings.csv`: user_id, book_id, rating
- `tags.csv`: tag_id, tag_name
- `book_tags.csv`: goodreads_book_id, tag_id, count

## Processed data (data/processed/)
- `movies_clean.parquet`: movie_id, title, genres[list], text_blob (for embedding)
- `books_clean.parquet`: book_id, title, authors, tags[list], text_blob
- `movie_ratings.parquet`: user_id, item_id, rating (domain='movie')
- `book_ratings.parquet`: user_id, item_id, rating (domain='book')
- `shared_embeddings.parquet`: domain, item_id, embedding_vector (384-dim, sentence-transformers all-MiniLM-L6-v2)

## Model artifacts (models/, gitignored, loaded by app at runtime)
- `cf_movie_svd.pkl` — trained SVD model (movies)
- `cf_book_svd.pkl` — trained SVD model (books)
- `raw_emb_movie.npy` — item embeddings, 384-dim (movies); similarity computed on-the-fly, not stored as a full matrix (book domain's NxN matrix would be ~381MB)
- `raw_emb_book.npy` — item embeddings, 384-dim (books); same approach
- `shared_embeddings.npy` + `item_index.json` — cross-domain embedding space + id lookup

## App-level user object (in-session, not persisted)
```
{
  "user_id": "synthetic_001",
  "domain_ratings": {"movie": {item_id: rating, ...}, "book": {...}},
  "taste_tags": ["sci-fi", "mystery", "adventure"]   # free-text/checkbox input
}
```

## Evaluation output (reports/results.md references these)
- `reports/metrics_movie.json`, `reports/metrics_book.json`: {model_name: {precision@10, recall@10, ndcg@10, rmse, mae}}
- `reports/metrics_crossdomain.json`: {baseline_popularity: {...}, shared_embedding: {...}}
