# What to Watch Next: a recommender that measures variety as well as accuracy

Streaming services like BBC iPlayer recommend programmes. A recommender tuned
only for clicks tends to keep showing the same popular hits. This project
builds four recommenders, then measures accuracy **and** three things a public
service broadcaster also cares about:

- **Coverage:** how much of the catalogue ever gets recommended.
- **Diversity:** how different the programmes on one list are.
- **Novelty:** how far the list goes beyond the obvious hits.

It then adds a diversity dial (MMR re-ranking) and shows how much accuracy
each step of extra variety costs.

## Run it

```bash
pip install pandas numpy scipy scikit-learn matplotlib tabulate pytest
python run.py --source synthetic   # works straight away on simulated data
pytest -q                          # 7 tests
```

For real data, download **ml-latest-small** from
https://grouplens.org/datasets/movielens/ and copy `movies.csv` and
`ratings.csv` into `data/`. Then run `python run.py`. It picks up the real
data automatically and writes `outputs/results.md` and `outputs/tradeoff.png`.

`python run.py --user 42` prints one viewer's history next to their picks,
with diversity at 0 and at 0.6. Use it as a sanity check, and for a
screenshot in your README.

## How it works

| File | What it does |
| --- | --- |
| `src/data.py` | Loads MovieLens and drops titles with fewer than 10 ratings, or generates long-tail synthetic data |
| `src/recommenders.py` | Popularity baseline, item-item collaborative filtering (cosine), content-based (TF-IDF on genres), a hybrid blend, and MMR re-ranking |
| `src/evaluate.py` | Holds out each viewer's 5 latest watches, then scores precision@10, recall@10, hit rate, coverage, list diversity and novelty |
| `run.py` | Runs everything and writes the results table and the trade-off chart |

Results on the synthetic data (yours on MovieLens will differ):

| Model | Precision@10 | Coverage |
| --- | --- | --- |
| Popularity | 0.074 | 7% |
| Collaborative filtering | 0.078 | 10% |
| Content-based | 0.030 | 91% |
| Hybrid | 0.091 | 26% |

The hybrid model is 23% more accurate than popularity and recommends 3.5
times as much of the catalogue. At diversity 0.6, list diversity rises from
0.74 to 0.86 for about 13% less precision.

## Make it yours (do at least two before you list it)

1. **Run it on MovieLens** and put your own numbers on your CV.
2. **Cold start:** a brand-new viewer has no history. Add a fallback that asks
   for 3 favourite genres. How does precision change for users with under 5
   watches?
3. **Add tags.** MovieLens ships `tags.csv`. Join the tags into the text that
   TF-IDF reads. Does content-based improve?
4. **A tiny dashboard.** `pip install streamlit`, then write a 30-line app with
   a user picker and a diversity slider that shows the list changing.
5. **Write a 300-word "responsible recommender" note.** Which diversity setting
   would you ship, and why? This is a strong thing to talk about in a BBC
   interview.

## Things to be ready to explain

- Why the latest 5 watches are held out, rather than a random 5.
- Why popularity is a hard baseline to beat on precision.
- What cosine similarity is measuring in the item-item matrix.
- Why accuracy alone is the wrong target for a public service broadcaster.
