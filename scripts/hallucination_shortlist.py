"""Find a prompt that pythia-70m answers clearly wrong, and the units behind it (issue #4781).

Each candidate has one right single-token answer. A candidate is kept when all of these hold:

- the model's top token is wrong;
- the top token is a content word, not a function word, a newline or punctuation;
- the model is confident: the top token leads the next one by at least :data:`MIN_LEAD`;
- the right answer is one token, so it can be explained too.

For the pick, it also writes the **contrast**: each unit's first-order attribution to
logit(wrong) − logit(right), by :func:`openinterpretability.explain.attribute`. Those are the
units that favour the wrong token over the right one.

Run: ``uv run python scripts/hallucination_shortlist.py`` (float32, CPU, about a minute).
It writes ``docs/analyses/hallucination-shortlist.jsonl``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

MODEL = "EleutherAI/pythia-70m"
MIN_LEAD = 0.2  # the wrong token's lead in probability over the next token
OUT = Path("docs/analyses/hallucination-shortlist.jsonl")
PICK = ("A week has seven days and a year has twelve", " months")
FUNCTION_WORDS = {"the", "a", "an", "of", "and", "to", "in", "is", "it", "that", "on",
                  "for", "as", "with", "at", "by", "from", "this", "be", "or", "are",
                  "was", "about", "one", "not", "all", "his", "her", "their"}  # fmt: skip

# (prompt, the right answer). Families: units copied from a first clause, capitals,
# opposites, counting, days and months, and well-known pairs.
CANDIDATES = [
    ("A minute has sixty seconds and an hour has sixty", " minutes"),
    ("A week has seven days and a year has twelve", " months"),
    ("A hand has five fingers and a foot has five", " toes"),
    ("A dozen has twelve eggs and a year has twelve", " months"),
    ("An hour has sixty minutes and a minute has sixty", " seconds"),
    ("A spider has eight legs and a dog has four", " legs"),
    ("A car has four wheels and a bike has two", " wheels"),
    ("A triangle has three sides and a square has", " four"),
    ("A year has twelve months and a week has seven", " days"),
    ("A day has twenty-four hours and an hour has sixty", " minutes"),
    ("A cat has four legs and a bird has two", " legs"),
    ("The capital of France is the city of", " Paris"),
    ("The capital of Spain is the city of", " Madrid"),
    ("The capital of Italy is the city of", " Rome"),
    ("The capital of Germany is the city of", " Berlin"),
    ("The capital of Japan is the city of", " Tokyo"),
    ("The Eiffel Tower is in the city of", " Paris"),
    ("The Colosseum is in the city of", " Rome"),
    ("The opposite of hot is", " cold"),
    ("The opposite of up is", " down"),
    ("The opposite of black is", " white"),
    ("The opposite of day is", " night"),
    ("One, two, three, four,", " five"),
    ("Monday, Tuesday, Wednesday,", " Thursday"),
    ("January, February, March,", " April"),
    ("The day after Monday is", " Tuesday"),
    ("The month after January is", " February"),
    ("Two plus two equals", " four"),
    ("Salt and", " pepper"),
    ("Bread and", " butter"),
    ("The sky is blue and the grass is", " green"),
    ("Snow is white and coal is", " black"),
    ("Fish swim and birds", " fly"),
    ("A cow says moo and a dog says", " woof"),
    ("Ice is cold and fire is", " hot"),
]


def is_content(token: str) -> bool:
    """A word of letters, at least 3 long, and not a function word."""
    word = token.strip()
    return word.isalpha() and len(word) >= 3 and word.lower() not in FUNCTION_WORDS


@torch.no_grad()
def score(runner, prompt: str, right: str) -> dict:
    """The prompt's top tokens, the right answer's rank, and whether it is kept."""
    ids = runner.tokenizer.encode(right, add_special_tokens=False)
    probs = torch.softmax(runner.forward(runner.encode(prompt))[0, -1].float(), -1)
    top = probs.topk(5)
    tops = [
        (runner.tokenizer.decode([int(i)]), round(float(p), 4))
        for p, i in zip(top.values, top.indices, strict=True)
    ]
    rank = int((probs > probs[ids[0]]).sum()) + 1 if len(ids) == 1 else None
    wrong = tops[0][0] != right
    kept = (wrong and is_content(tops[0][0]) and tops[0][1] - tops[1][1] >= MIN_LEAD
            and len(ids) == 1)  # fmt: skip
    return {"prompt": prompt, "right": right, "top": tops, "right_rank": rank,
            "right_p": round(float(probs[ids[0]]), 4) if len(ids) == 1 else None,
            "wrong": wrong, "kept": kept}  # fmt: skip


def contrast(runner, means, prompt: str, wrong: str, right: str, top: int = 20) -> list[dict]:
    """The units that favour ``wrong`` over ``right`` most: attribution to their logit gap."""
    from openinterpretability.explain import PoleAttribution, attribute

    tok = [runner.tokenizer.encode(t, add_special_tokens=False)[0] for t in (wrong, right)]
    a, _ = attribute(runner, prompt, means, tok[0], -1)
    b, _ = attribute(runner, prompt, means, tok[1], -1)
    gap = {u: a[u][0] - b.get(u, (0.0, 0.0))[0] for u in a}
    ranked = sorted(gap.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top]
    return [{"unit": PoleAttribution(*u, 0.0, 0.0).label, "favours_wrong_by": round(v, 4)}
            for u, v in ranked]  # fmt: skip


def main() -> None:
    """Score every candidate, write the shortlist and the pick's contrast, and print them."""
    from openinterpretability.cli import _load_runner
    from openinterpretability.evidence import read_corpus
    from openinterpretability.explain import unit_means

    runner = _load_runner(MODEL, "cpu", "float32")  # bfloat16 rounds pythia's logits
    rows = [score(runner, p, r) for p, r in CANDIDATES]
    means = unit_means(runner, MODEL, read_corpus(Path("corpus.jsonl"))[0], 20_000)
    pick = next(r for r in rows if r["prompt"] == PICK[0])
    pick["contrast"] = contrast(runner, means, PICK[0], pick["top"][0][0], PICK[1])
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    for r in rows:
        mark = "kept" if r["kept"] else ("wrong" if r["wrong"] else "right")
        right = f"right {r['right']!r} rank {r['right_rank']}"
        print(f"{mark:5}  {r['prompt']!r} → {r['top'][0]} ({right})")
    print(f"\n{sum(r['kept'] for r in rows)} kept of {len(rows)} → {OUT}")


if __name__ == "__main__":
    main()
