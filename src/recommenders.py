"""Three recommenders and a diversity re-ranker.

- Popularity: what most people watched. The baseline every model must beat.
- Item-item collaborative filtering: "people who watched X also watched Y".
- Content-based: TF-IDF on genre (and any text) describing each programme.
- Hybrid: a weighted blend of the two scores.
- MMR re-ranking: trades a little accuracy for more variety on the list.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Recommender:
    def __init__(self, items: pd.DataFrame, train: pd.DataFrame):
        self.items = items.reset_index(drop=True)
        self.idx = {iid: k for k, iid in enumerate(self.items["item_id"])}
        self.users = {uid: k for k, uid in enumerate(sorted(train["user_id"].unique()))}
        rows = train["user_id"].map(self.users).to_numpy()
        cols = train["item_id"].map(self.idx).to_numpy()
        # Implicit feedback: a watch counts as 1 whatever the rating.
        self.ui = sparse.csr_matrix((np.ones(len(train)), (rows, cols)),
                                    shape=(len(self.users), len(self.items)))
        self.popularity = np.asarray(self.ui.sum(axis=0)).ravel()

        self.item_sim = cosine_similarity(self.ui.T, dense_output=False).toarray().astype(np.float32)
        np.fill_diagonal(self.item_sim, 0)

        tfidf = TfidfVectorizer(token_pattern=r"[^\s]+")
        self.content = tfidf.fit_transform(self.items["genres"].fillna(""))
        self.content_sim = cosine_similarity(self.content).astype(np.float32)
        np.fill_diagonal(self.content_sim, 0)

    def _seen(self, user_id):
        return self.ui[self.users[user_id]].indices

    def scores(self, user_id, method: str = "hybrid", alpha: float = 0.7):
        seen = self._seen(user_id)
        if method == "popularity":
            s = self.popularity.astype(float).copy()
        else:
            cf = self.item_sim[seen].sum(axis=0)
            cb = self.content_sim[seen].sum(axis=0)
            if method == "cf":
                s = cf
            elif method == "content":
                s = cb
            elif method == "hybrid":
                s = alpha * _norm(cf) + (1 - alpha) * _norm(cb)
            else:
                raise ValueError(f"unknown method {method}")
        s = s.astype(float)
        s[seen] = -np.inf
        return s

    def recommend(self, user_id, k: int = 10, method: str = "hybrid",
                  diversity: float = 0.0):
        """Top-k item ids. diversity in [0, 1]: 0 = pure relevance (no MMR)."""
        s = self.scores(user_id, method)
        if diversity <= 0:
            top = np.argsort(-s)[:k]
        else:
            top = mmr(s, self.content_sim, k, diversity)
        return [self.items["item_id"].iat[i] for i in top]


def _norm(x):
    x = np.asarray(x, dtype=float)
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 0 else np.zeros_like(x)


def mmr(scores, sim, k, diversity, pool: int = 100):
    """Maximal Marginal Relevance: pick items that are relevant AND unlike
    what's already on the list."""
    candidates = list(np.argsort(-scores)[:pool])
    candidates = [c for c in candidates if np.isfinite(scores[c])]
    rel = _norm(scores[candidates])
    chosen = []
    while candidates and len(chosen) < k:
        best, best_val = None, -np.inf
        for j, c in enumerate(candidates):
            redundancy = max((sim[c, p] for p in chosen), default=0.0)
            val = (1 - diversity) * rel[j] - diversity * redundancy
            if val > best_val:
                best, best_val = j, val
        chosen.append(candidates.pop(best))
        rel = np.delete(rel, best)
    return chosen
