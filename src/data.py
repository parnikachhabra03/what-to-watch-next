"""Load viewing data in MovieLens format, or generate a small synthetic set.

Real data: download "ml-latest-small" (about 1 MB) from
https://grouplens.org/datasets/movielens/ and copy movies.csv and ratings.csv
into the data/ folder. Every other part of the project works on either source.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

GENRES = ["Drama", "Comedy", "Documentary", "Crime", "Sci-Fi", "Children",
          "Romance", "Thriller", "History", "Nature", "Music", "Sport"]


def load_movielens(data_dir: Path = DATA_DIR, min_item_ratings: int = 10):
    """Return (items, ratings) from MovieLens CSVs, or None if they're missing.

    Titles with fewer than `min_item_ratings` ratings are dropped: there's too
    little signal to recommend them, and it keeps the similarity matrices small
    enough for a laptop (about 2,300 x 2,300 instead of 9,700 x 9,700).
    """
    movies, ratings = data_dir / "movies.csv", data_dir / "ratings.csv"
    if not (movies.exists() and ratings.exists()):
        return None
    items = pd.read_csv(movies).rename(columns={"movieId": "item_id"})
    events = pd.read_csv(ratings).rename(columns={"userId": "user_id", "movieId": "item_id"})
    items["genres"] = items["genres"].str.replace("|", " ", regex=False)
    counts = events["item_id"].value_counts()
    keep = counts[counts >= min_item_ratings].index
    items = items[items["item_id"].isin(keep)]
    events = events[events["item_id"].isin(keep)]
    return items[["item_id", "title", "genres"]], events[["user_id", "item_id", "rating", "timestamp"]]


def make_synthetic(n_users: int = 600, n_items: int = 400, seed: int = 7):
    """Simulate programmes with genres and viewers with genre tastes.

    Popularity follows a long tail (a few hits, many niche titles), which is
    what makes popularity bias worth measuring.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        k = rng.integers(1, 4)
        g = rng.choice(GENRES, size=k, replace=False)
        rows.append({"item_id": i, "title": f"Programme {i:03d}", "genres": " ".join(g)})
    items = pd.DataFrame(rows)

    popularity = rng.pareto(1.2, n_items) + 1
    popularity /= popularity.sum()
    genre_of = [set(g.split()) for g in items["genres"]]

    events = []
    for u in range(n_users):
        liked = set(rng.choice(GENRES, size=rng.integers(2, 4), replace=False))
        affinity = np.array([1 + 4 * len(liked & gs) for gs in genre_of], dtype=float)
        p = popularity * affinity
        p /= p.sum()
        n = int(rng.integers(15, 60))
        watched = rng.choice(n_items, size=n, replace=False, p=p)
        for t, item in enumerate(watched):
            match = len(liked & genre_of[item])
            rating = float(np.clip(round(2.5 + match + rng.normal(0, 0.8)), 1, 5))
            events.append({"user_id": u, "item_id": int(item), "rating": rating,
                           "timestamp": 1_700_000_000 + t * 3600 + u})
    return items, pd.DataFrame(events)


def load(source: str = "auto"):
    """source: 'auto' (MovieLens if present), 'movielens' or 'synthetic'."""
    if source in ("auto", "movielens"):
        data = load_movielens()
        if data is not None:
            return data, "MovieLens"
        if source == "movielens":
            raise FileNotFoundError("Put movies.csv and ratings.csv in data/ first.")
    return make_synthetic(), "synthetic"
