"""Run every model, write outputs/results.md and outputs/tradeoff.png.

    python run.py                 # MovieLens if data/ has it, else synthetic
    python run.py --source synthetic
    python run.py --user 42       # print one viewer's history and picks
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from data import load  # noqa: E402
from evaluate import evaluate, split_last_n  # noqa: E402
from recommenders import Recommender  # noqa: E402

OUT = Path(__file__).parent / "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="auto", choices=["auto", "movielens", "synthetic"])
    ap.add_argument("--user", type=int)
    args = ap.parse_args()

    (items, events), name = load(args.source)
    train, test = split_last_n(events)
    rec = Recommender(items, train)
    print(f"Data: {name} | {events['user_id'].nunique():,} viewers | "
          f"{len(items):,} titles | {len(events):,} watches")

    if args.user is not None:
        show_user(rec, items, train, args.user)
        return

    rows = [evaluate(rec, test, method=m) for m in ["popularity", "cf", "content", "hybrid"]]
    sweep = [evaluate(rec, test, method="hybrid", diversity=d) for d in [0, 0.2, 0.4, 0.6, 0.8]]
    table = pd.DataFrame(rows).round(3)
    sweep_df = pd.DataFrame(sweep).round(3)
    print(table.to_string(index=False))
    print(sweep_df.to_string(index=False))

    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sweep_df["list_diversity"], sweep_df["precision@10"], marker="o", color="#1f5fa8")
    for _, r in sweep_df.iterrows():
        ax.annotate(f"λ={r['diversity']}", (r["list_diversity"], r["precision@10"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Diversity of each list (1 − mean genre similarity)")
    ax.set_ylabel("Precision@10")
    ax.set_title("What variety costs in accuracy (hybrid + MMR)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "tradeoff.png", dpi=150)

    (OUT / "results.md").write_text(
        f"# Results ({name} data)\n\n## Model comparison\n\n{table.to_markdown(index=False)}\n\n"
        f"## Diversity sweep (hybrid)\n\n{sweep_df.to_markdown(index=False)}\n\n"
        "![trade-off](tradeoff.png)\n")
    print(f"Wrote {OUT/'results.md'} and {OUT/'tradeoff.png'}")


def show_user(rec, items, train, user):
    title = items.set_index("item_id")["title"]
    hist = train[train["user_id"] == user].tail(8)["item_id"]
    print("Recently watched:", *[f"  - {title[i]}" for i in hist], sep="\n")
    for d in (0.0, 0.6):
        picks = rec.recommend(user, k=10, diversity=d)
        print(f"\nRecommended (diversity={d}):", *[f"  - {title[i]}" for i in picks], sep="\n")


if __name__ == "__main__":
    main()
