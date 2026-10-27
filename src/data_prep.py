"""
Phase 1 — clean raw MovieLens + Goodbooks-10k data into data/processed/*.parquet
Run from repo root: python src/data_prep.py
"""
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)


def prep_movies():
    ml_dir = RAW / "ml-100k"
    # u.item is pipe-separated, latin-1 encoded, no header
    item_cols = ["movie_id", "title", "release_date", "video_release_date", "imdb_url"] + \
        ["genre_" + str(i) for i in range(19)]
    movies = pd.read_csv(ml_dir / "u.item", sep="|", names=item_cols, encoding="latin-1")

    genre_names = ["unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
                   "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
                   "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

    def row_genres(row):
        return [genre_names[i] for i in range(19) if row[f"genre_{i}"] == 1]

    movies["genres"] = movies.apply(row_genres, axis=1)
    movies["text_blob"] = movies["title"] + " " + movies["genres"].apply(lambda g: " ".join(g))
    movies_clean = movies[["movie_id", "title", "genres", "text_blob"]].rename(
        columns={"movie_id": "item_id"})
    movies_clean.to_parquet(PROC / "movies_clean.parquet", index=False)

    ratings = pd.read_csv(ml_dir / "u.data", sep="\t",
                           names=["user_id", "item_id", "rating", "timestamp"])
    ratings["domain"] = "movie"
    ratings[["user_id", "item_id", "rating", "domain"]].to_parquet(
        PROC / "movie_ratings.parquet", index=False)

    print(f"movies: {len(movies_clean)} items, {len(ratings)} ratings")
    return movies_clean


def prep_books():
    gb_dir = RAW / "goodbooks-10k"
    books = pd.read_csv(gb_dir / "books.csv")
    tags = pd.read_csv(gb_dir / "tags.csv")
    book_tags = pd.read_csv(gb_dir / "book_tags.csv")

    # keep top 5 tags per book by count, join tag names
    book_tags = book_tags.merge(tags, on="tag_id")
    top_tags = (book_tags.sort_values("count", ascending=False)
                .groupby("goodreads_book_id")
                .head(5)
                .groupby("goodreads_book_id")["tag_name"]
                .apply(list)
                .reset_index())

    books = books.merge(top_tags, left_on="goodreads_book_id",
                         right_on="goodreads_book_id", how="left")
    books["tag_name"] = books["tag_name"].apply(lambda x: x if isinstance(x, list) else [])
    books["text_blob"] = books["title"].fillna("") + " " + books["authors"].fillna("") + \
        " " + books["tag_name"].apply(lambda g: " ".join(g))

    books_clean = books[["book_id", "title", "authors", "tag_name", "text_blob"]].rename(
        columns={"book_id": "item_id", "tag_name": "genres"})
    books_clean.to_parquet(PROC / "books_clean.parquet", index=False)

    ratings = pd.read_csv(gb_dir / "ratings.csv").rename(
        columns={"user_id": "user_id", "book_id": "item_id"})
    ratings["domain"] = "book"
    ratings[["user_id", "item_id", "rating", "domain"]].to_parquet(
        PROC / "book_ratings.parquet", index=False)

    print(f"books: {len(books_clean)} items, {len(ratings)} ratings")
    return books_clean


if __name__ == "__main__":
    prep_movies()
    prep_books()
    print("Done. Check data/processed/ for the 4 parquet files.")
