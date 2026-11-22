"""Offline evaluation: hold out each viewer's most recent watches and check
whether the recommender would have suggested them.

Accuracy alone isn't enough for a public service broadcaster, so the report
also measures coverage (how much of the catalogue ever gets recommended),
diversity (how different the items on one list are) and novelty (how far
from the obvious hits the list goes).
"""
import numpy as np
import pandas as pd


def split_last_n(events: pd.DataFrame, n_test: int = 5, min_history: int = 10):
    events = events.sort_values(["user_id", "timestamp"])
    counts = events.groupby("user_id")["item_id"].transform("size")
    events = events[counts >= min_history]
    rank = events.groupby("user_id").cumcount(ascending=False)
    return events[rank >= n_test], events[rank < n_test]


def evaluate(rec, test: pd.DataFrame, k: int = 10, method: str = "hybrid",
             diversity: float = 0.0, max_users: int = 500):
    truth = test.groupby("user_id")["item_id"].apply(set)
    users = [u for u in truth.index if u in rec.users][:max_users]
    hits, precisions, recalls, ild, novelty, shown = 0, [], [], [], [], set()
    pop_rank = pd.Series(rec.popularity).rank(ascending=False).to_numpy()
    for u in users:
        recs = rec.recommend(u, k=k, method=method, diversity=diversity)
        idx = [rec.idx[i] for i in recs]
        h = len(set(recs) & truth[u])
        hits += h > 0
        precisions.append(h / k)
        recalls.append(h / len(truth[u]))
        shown.update(recs)
        if len(idx) > 1:
            sub = rec.content_sim[np.ix_(idx, idx)]
            ild.append(1 - sub[np.triu_indices(len(idx), 1)].mean())
        novelty.append(np.mean(pop_rank[idx]) / len(pop_rank))
    return {
        "method": method,
        "diversity": diversity,
        f"precision@{k}": np.mean(precisions),
        f"recall@{k}": np.mean(recalls),
        "hit_rate": hits / len(users),
        "coverage": len(shown) / len(rec.items),
        "list_diversity": float(np.mean(ild)),
        "novelty": float(np.mean(novelty)),
        "users": len(users),
    }
