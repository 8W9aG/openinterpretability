"""Feasibility study for a concept map of a model's neurons (issue #3290).

Each neuron is placed by the meaning of the text it fires on. A pole's text is its newest
record's top examples, cut to ±20 tokens around the peak with the peak marked. A neuron's
vector is the mean embedding of its positive pole's texts minus its negative pole's.
A missing pole is zero, and the vector is L2-normalised. The study measures whether that
places neurons usefully, against the registry's own explanations, and what a full map
would cost.

The heavy libraries stay out of the project. Run it in a throwaway environment::

    uv run --with sentence-transformers --with umap-learn --with leidenalg \\
        python scripts/concept_map_study.py

It writes docs/analyses/concept-map-feasibility.jsonl, one row per model.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path("docs/analyses/concept-map-feasibility.jsonl")
MODELS = ("openai-community/gpt2", "Qwen/Qwen2.5-0.5B")  # the two with the most records
EMBEDDERS = ("sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2")
WINDOW = 20  # tokens each side of the peak
FULL_NEURONS = 116_736  # Qwen2.5-0.5B: 24 layers x 4,864
TEXTS_PER_POLE = 10


def window(example) -> str:
    """The example's text within WINDOW tokens of its peak, with the peak marked."""
    tokens, peak = example.tokens, example.max_token_index
    left, right = max(0, peak - WINDOW), min(len(tokens), peak + WINDOW + 1)
    parts = [f"«{t}»" if i == peak else t for i, t in enumerate(tokens[left:right], left)]
    return "".join(parts).strip()


def poles(model: str) -> dict:
    """Each pole's texts, explanation and status, from its newest record, by neuron."""
    from openinterpretability.builder.wiki import load_archive

    newest = {}
    for record in load_archive(Path("wiki")):
        ident = record.submission.identifier
        if ident.model_name != model or ident.kind != "mlp_neuron_pole":
            continue
        old = newest.get(ident.slug)
        if old is None or record.verdict.timestamp > old.verdict.timestamp:
            newest[ident.slug] = record
    out: dict = {}
    for record in newest.values():
        ident, sub = record.submission.identifier, record.submission
        texts = [window(e) for e in sub.top_examples if e.tokens][:TEXTS_PER_POLE]
        if texts:
            out.setdefault((ident.layer, ident.neuron_idx), {})[ident.pole] = (
                texts,
                sub.natural_language_explanation,
                record.verdict.status,
            )
    return out


def vectors(neurons: dict, encode) -> dict:
    """Context and explanation vectors, norms, pole means and the corpus-mean embedding."""
    keys = sorted(neurons)
    dim = len(encode(["x"])[0])
    ctx, exp, norms, one_pole, means = [], [], [], [], {}
    for key in keys:
        c, e = np.zeros(dim), np.zeros(dim)
        for pole, (texts, explanation, _) in neurons[key].items():
            sign = 1.0 if pole == "positive" else -1.0
            means[(key, pole)] = encode(texts).mean(0)
            c += sign * means[(key, pole)]
            e += sign * encode([explanation])[0]
        norms.append(float(np.linalg.norm(c)))
        ctx.append(c / (np.linalg.norm(c) or 1.0))
        exp.append(e / (np.linalg.norm(e) or 1.0))
        one_pole.append(len(neurons[key]) == 1)
    corpus = np.mean(list(means.values()), axis=0)
    return {
        "keys": keys,
        "ctx": np.array(ctx),
        "exp": np.array(exp),
        "norms": np.array(norms),
        "one_pole": np.array(one_pole),
        "means": means,
        "corpus": corpus,
    }


def retrieval(ctx, exp) -> dict:
    """How often a neuron's context vector finds its own explanation, against chance."""
    sims = ctx @ exp.T
    ranks = (sims > sims.diagonal()[:, None]).sum(1)  # explanations scoring above its own
    n = len(ctx)
    return {
        "neurons": n,
        "top1": float((ranks == 0).mean()),
        "top10": float((ranks < 10).mean()),
        "chance_top1": 1 / n,
        "chance_top10": min(1.0, 10 / n),
        "median_rank": float(np.median(ranks)) + 1,
    }


def pair_correlation(ctx, exp, pairs=200_000, seed=0) -> float:
    """Spearman correlation of context cosine and explanation cosine over random pairs."""
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, len(ctx), pairs), rng.integers(0, len(ctx), pairs)
    keep = i != j
    a, b = (ctx[i] * ctx[j]).sum(1)[keep], (exp[i] * exp[j]).sum(1)[keep]
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


def clumping(v: dict) -> dict:
    """Mean cosine in one-pole and two-pole neurons, before and after removing the corpus mean."""
    out = {}
    for name, ctx in (("raw", v["ctx"]), ("centred", _centred(v))):
        one, two = ctx[v["one_pole"]], ctx[~v["one_pole"]]
        out[name] = {
            "one_pole": _mean_cos(one),
            "two_pole": _mean_cos(two),
            "across": float((one @ two.T).mean()) if len(one) and len(two) else None,
        }
    return out


def _centred(v: dict):
    """Context vectors from pole means with the corpus-mean embedding taken out of each."""
    rows = []
    for k in v["keys"]:
        c = sum(
            (1.0 if p == "positive" else -1.0) * (v["means"][(k, p)] - v["corpus"])
            for p in ("positive", "negative")
            if (k, p) in v["means"]
        )
        rows.append(c / (np.linalg.norm(c) or 1.0))
    return np.array(rows)


def _mean_cos(x) -> float | None:
    if len(x) < 2:
        return None
    s = x @ x.T
    return float((s.sum() - np.trace(s)) / (len(x) * (len(x) - 1)))


def two_poles(v: dict) -> dict:
    """The cosine between a two-pole neuron's positive and negative pole means."""
    cos = []
    for k in v["keys"]:
        if (k, "positive") in v["means"] and (k, "negative") in v["means"]:
            a, b = v["means"][(k, "positive")], v["means"][(k, "negative")]
            cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
    return {
        "neurons": len(cos),
        "median_cos": float(np.median(cos)) if cos else None,
        "share_below_0.3": float(np.mean(np.array(cos) < 0.3)) if cos else None,
    }


def clusters(ctx, exp, keys, k: int) -> dict:
    """Leiden clusters on the k-NN graph, and whether explanation-similar pairs share one."""
    import igraph as ig
    import leidenalg

    sims = ctx @ ctx.T
    np.fill_diagonal(sims, -np.inf)
    nn = np.argsort(-sims, axis=1)[:, :k]
    edges = {tuple(sorted((i, int(j)))) for i in range(len(ctx)) for j in nn[i]}
    graph = ig.Graph(n=len(ctx), edges=list(edges))
    part = leidenalg.find_partition(graph, leidenalg.ModularityVertexPartition, seed=0)
    member = np.array(part.membership)
    esims = exp @ exp.T
    layers = np.array([key[0] for key in keys])
    cross = layers[:, None] != layers[None, :]
    similar = (esims > 0.6) & cross
    rng = np.random.default_rng(0)
    rand_i, rand_j = rng.integers(0, len(ctx), 50_000), rng.integers(0, len(ctx), 50_000)
    si, sj = np.nonzero(np.triu(similar, 1))
    return {
        "k": k,
        "clusters": int(member.max() + 1),
        "modularity": float(part.modularity),
        "similar_cross_layer_pairs": int(len(si)),
        "same_cluster_if_similar": float((member[si] == member[sj]).mean()) if len(si) else None,
        "same_cluster_random": float((member[rand_i] == member[rand_j]).mean()),
    }


def layout_stability(ctx, n_neighbors=(10, 15), seeds=(0, 1)) -> dict:
    """UMAP layouts across seeds and k: the overlap of each point's 10 nearest 2-D neighbours."""
    import umap

    layouts, seconds = {}, {}
    for k in n_neighbors:
        for seed in seeds:
            start = time.perf_counter()
            layouts[(k, seed)] = umap.UMAP(
                n_neighbors=k, metric="cosine", random_state=seed
            ).fit_transform(ctx)
            seconds[(k, seed)] = time.perf_counter() - start
    base = layouts[(n_neighbors[0], seeds[0])]
    return {
        "umap_seconds": round(float(np.mean(list(seconds.values()))), 2),
        "overlap_other_seed": _overlap(base, layouts[(n_neighbors[0], seeds[1])]),
        "overlap_other_k": _overlap(base, layouts[(n_neighbors[1], seeds[0])]),
    }


def _overlap(a, b, k=10) -> float:
    """The mean share of each point's k nearest neighbours in ``a`` that are also so in ``b``."""

    def nn(x):
        d = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d, np.inf)
        return np.argsort(d, axis=1)[:, :k]

    na, nb = nn(a), nn(b)
    return float(np.mean([len(set(na[i]) & set(nb[i])) / k for i in range(len(a))]))


def neighbour_overlap(x, y, k=10) -> float:
    """The overlap of 10 nearest neighbours (cosine) between two embeddings of the same neurons."""

    def nn(v):
        s = v @ v.T
        np.fill_diagonal(s, -np.inf)
        return np.argsort(-s, axis=1)[:, :k]

    a, b = nn(x), nn(y)
    return float(np.mean([len(set(a[i]) & set(b[i])) / k for i in range(len(x))]))


def study(model: str) -> dict:
    """Every measure for one model (module docstring)."""
    from sentence_transformers import SentenceTransformer

    neurons = poles(model)
    texts = [t for pole in neurons.values() for texts, _, _ in pole.values() for t in texts]
    embedders = {name: SentenceTransformer(name, device="cpu") for name in EMBEDDERS}
    start = time.perf_counter()
    embedders[EMBEDDERS[0]].encode(texts[:500], batch_size=64)
    per_text = (time.perf_counter() - start) / min(500, len(texts))
    vecs = {
        name: vectors(neurons, lambda t, m=m: m.encode(t, batch_size=64, normalize_embeddings=True))
        for name, m in embedders.items()
    }
    v = vecs[EMBEDDERS[0]]
    full_texts = FULL_NEURONS * 2 * TEXTS_PER_POLE
    return {
        "model": model,
        "neurons": len(v["keys"]),
        "one_pole": int(v["one_pole"].sum()),
        "texts": len(texts),
        "embedder": EMBEDDERS[0],
        "retrieval": retrieval(v["ctx"], v["exp"]),
        "retrieval_mpnet": retrieval(vecs[EMBEDDERS[1]]["ctx"], vecs[EMBEDDERS[1]]["exp"]),
        "pair_spearman": pair_correlation(v["ctx"], v["exp"]),
        "embedder_neighbour_overlap": neighbour_overlap(v["ctx"], vecs[EMBEDDERS[1]]["ctx"]),
        "norm_quartiles": [float(q) for q in np.quantile(v["norms"], [0.25, 0.5, 0.75])],
        "clumping": clumping(v),
        "two_poles": two_poles(v),
        "clusters": [clusters(v["ctx"], v["exp"], v["keys"], k) for k in (10, 15)],
        "layout": layout_stability(v["ctx"]),
        "cost": {
            "seconds_per_text_cpu": round(per_text, 5),
            "full_texts": full_texts,
            "full_embed_hours_cpu": round(full_texts * per_text / 3600, 1),
        },
    }


def main() -> None:
    """Study each model in MODELS and write one row each."""
    random.seed(0)
    rows = []
    for model in MODELS:
        row = study(model)
        print(json.dumps(row)[:400], flush=True)
        rows.append(row)
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
