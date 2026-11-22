import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data import make_synthetic  # noqa: E402
from evaluate import evaluate, split_last_n  # noqa: E402
from recommenders import Recommender, mmr  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    items, events = make_synthetic(n_users=150, n_items=120, seed=1)
    train, test = split_last_n(events)
    return items, train, test, Recommender(items, train)


def test_split_holds_out_latest(setup):
    _, train, test, _ = setup
    for u in test["user_id"].unique()[:20]:
        assert train[train.user_id == u].timestamp.max() < test[test.user_id == u].timestamp.min()
    assert test.groupby("user_id").size().eq(5).all()


def test_never_recommends_already_watched(setup):
    _, train, _, rec = setup
    for method in ["popularity", "cf", "content", "hybrid"]:
        for u in list(rec.users)[:20]:
            seen = set(train[train.user_id == u].item_id)
            assert not seen & set(rec.recommend(u, method=method))


def test_returns_k_unique_items(setup):
    rec = setup[3]
    u = next(iter(rec.users))
    for d in (0.0, 0.5):
        recs = rec.recommend(u, k=10, diversity=d)
        assert len(recs) == 10 and len(set(recs)) == 10


def test_personalised_beats_popularity(setup):
    _, _, test, rec = setup
    pop = evaluate(rec, test, method="popularity")
    hyb = evaluate(rec, test, method="hybrid")
    assert hyb["precision@10"] > pop["precision@10"]
    assert hyb["coverage"] > pop["coverage"]


def test_diversity_knob_increases_variety(setup):
    _, _, test, rec = setup
    low = evaluate(rec, test, diversity=0.0)
    high = evaluate(rec, test, diversity=0.8)
    assert high["list_diversity"] > low["list_diversity"]


def test_mmr_zero_diversity_matches_ranking():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    sim = np.zeros((4, 4))
    assert mmr(scores, sim, 3, 0.0) == [1, 3, 2]


def test_unknown_method_raises(setup):
    rec = setup[3]
    with pytest.raises(ValueError):
        rec.scores(next(iter(rec.users)), method="magic")
