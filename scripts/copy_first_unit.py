"""Why pythia-70m copies the first clause's unit: absent or outvoted (issue #5015).

#4781 verified a route by which pythia-70m answers " days" to "A week has seven days and a
year has twelve". Copy heads carry the first clause's unit to the answer. This script asks
whether the right unit (" months") is absent from the model's processing of the prompt, or
present and outvoted. It reads the shortlist's six wrong prompts and three prompts that
copying answers right (:data:`PROMPTS`), in float32 on the CPU. For each it measures:

1. ``single``: the second clause alone, zero-shot and after :data:`SHOTS`. A model that
   answers it right knows the fact without a clause to copy.
2. ``enrichment``: the right and copied tokens' ranks under the logit lens at the second
   clause's subject (" year"). The lens reads each layer's residual stream and MLP output.
   It reads the answer position's residual stream too.
3. ``attention``: what share of the answer position's attention :data:`HEADS` give to the
   first clause's unit and to the second clause's subject.
4. ``scaling``: the answer when the copy heads' outputs are scaled by each of
   :data:`ALPHAS`. It also switches off :data:`DRAWS` random sets of three heads of layers
   3 and 4. A head is switched off by setting its output to zero. ``four_off`` switches off
   the copy heads and L4/H1 together.
5. ``writers``: each layer 4 and 5 neuron pole's direct write to logit(right) −
   logit(copied) at the answer position. It reads the prompt and its single clause.

The rows go to ``--out``::

    uv run python scripts/copy_first_unit.py --out papers/copy-first-unit/results.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import click
import torch

MODEL = "EleutherAI/pythia-70m"
# (prompt, right answer, the first clause's unit, the second clause's subject, its clause)
PROMPTS = (
    ("A minute has sixty seconds and an hour has sixty", " minutes", " seconds", " hour",
     "An hour has sixty"),
    ("A week has seven days and a year has twelve", " months", " days", " year",
     "A year has twelve"),
    ("A hand has five fingers and a foot has five", " toes", " fingers", " foot",
     "A foot has five"),
    ("A dozen has twelve eggs and a year has twelve", " months", " eggs", " year",
     "A year has twelve"),
    ("An hour has sixty minutes and a minute has sixty", " seconds", " minutes", " minute",
     "A minute has sixty"),
    ("A year has twelve months and a week has seven", " days", " months", " week",
     "A week has seven"),
    ("A spider has eight legs and a dog has four", " legs", " legs", " dog", "A dog has four"),
    ("A car has four wheels and a bike has two", " wheels", " wheels", " bike", "A bike has two"),
    ("A cat has four legs and a bird has two", " legs", " legs", " bird", "A bird has two"),
)  # fmt: skip
# Four facts with no unit of the prompts, for the few-shot form of the single clause.
SHOTS = ("A triangle has three sides. A square has four corners. "
         "An octopus has eight arms. A pentagon has five sides. ")  # fmt: skip
COPY_HEADS = ((3, 3), (4, 0), (4, 7))  # the verified route's copy heads (#4781)
HEADS = (*COPY_HEADS, (4, 1))  # and L4/H1, the candidate copy-suppression head (#5063)
ALPHAS = (1.0, 0.5, 0.0, -0.5)  # scales of the copy heads' outputs
DRAWS = 20  # random sets of three heads switched off, the control
SEED = 0
TOP_WRITERS = 5  # the neuron poles listed on each side
FOCUS = (5, 2014)  # L5/N2014+, the prompt's largest single attribution (#5015, question 2)


def token(runner, text: str) -> int:
    """The id of ``text``, which must be one token."""
    ids = runner.tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{text!r} is {len(ids)} tokens")
    return ids[0]


def position(runner, prompt: str, text: str) -> int:
    """The position of the first token of ``prompt`` that is ``text``."""
    ids = runner.tokenizer.encode(prompt)
    return ids.index(token(runner, text))


def rank(logits: torch.Tensor, tok: int) -> int:
    """``tok``'s rank in ``logits``, from 1."""
    return int((logits > logits[tok]).sum()) + 1


def lens(runner, vector: torch.Tensor) -> torch.Tensor:
    """``vector`` read through the final norm and the unembedding."""
    norm, unembed = runner.final_norm(), runner.model.get_output_embeddings()
    return unembed(norm(vector)).float()


@torch.no_grad()
def residuals(runner, text: str) -> torch.Tensor:
    """The residual stream after each layer, before the final norm: ``[layers, tokens, d]``.

    Hugging Face returns the last layer's state after the final norm, so the input of the
    final norm takes its place here.
    """
    seen = []
    hook = runner.final_norm().register_forward_pre_hook(lambda _m, a: seen.append(a[0]))
    try:
        out = runner.model(**runner.encode(text), output_hidden_states=True)
    finally:
        hook.remove()
    return torch.stack([*out.hidden_states[1:-1], seen[0]])[:, 0].float()


@torch.no_grad()
def last_logits(runner, prompt: str, heads=None) -> torch.Tensor:
    """The answer position's logits, with ``heads`` (head interventions) in place."""
    with runner.hooks(None, None, heads or {}):
        return runner.forward(runner.encode(prompt))[0, -1].float()


def answer(runner, logits: torch.Tensor, right: int, copied: int) -> dict:
    """The top token, and the right and copied tokens' ranks and probabilities."""
    probs = torch.softmax(logits, -1)
    return {"top": runner.tokenizer.decode([int(logits.argmax())]),
            "right_rank": rank(logits, right), "right_p": round(float(probs[right]), 5),
            "copied_rank": rank(logits, copied), "copied_p": round(float(probs[copied]), 5),
            "gap": round(float(logits[right] - logits[copied]), 4)}  # fmt: skip


def single(runner, case) -> list[dict]:
    """Measure 1: the second clause alone, zero-shot and few-shot."""
    _, right, copied, _, clause = case
    r, c = token(runner, right), token(runner, copied)
    return [{"form": form, "text": text, **answer(runner, last_logits(runner, text), r, c)}
            for form, text in (("zero-shot", clause), ("few-shot", SHOTS + clause))]  # fmt: skip


@torch.no_grad()
def enrichment(runner, case) -> dict:
    """Measure 2: the logit lens at the subject, by layer, for the residual and each MLP."""
    prompt, right, copied, subject, _ = case
    r, c = token(runner, right), token(runner, copied)
    at = position(runner, prompt, subject)
    mlp: dict[int, torch.Tensor] = {}

    def keep(layer):
        def hook(out):
            mlp[layer] = out[0].detach()

        return hook

    with runner.hooks(None, {la: keep(la) for la in range(runner.n_layers)}):
        states = residuals(runner, prompt)
    rows = []
    for layer in range(runner.n_layers):
        resid = lens(runner, states[layer, at])
        write = lens(runner, mlp[layer][at])
        last = lens(runner, states[layer, -1])
        rows.append({"layer": layer, "resid_right": rank(resid, r), "resid_copied": rank(resid, c),
                     "mlp_right": rank(write, r), "mlp_copied": rank(write, c),
                     "last_right": rank(last, r), "last_copied": rank(last, c)})  # fmt: skip
    return {"subject_position": at, "layers": rows,
            "best_mlp_right": min(x["mlp_right"] for x in rows),
            "best_resid_right": min(x["resid_right"] for x in rows)}  # fmt: skip


@torch.no_grad()
def attention(eager, case) -> list[dict]:
    """Measure 3: the answer position's attention to the first unit and to the subject."""
    prompt, _, copied, subject, _ = case
    unit, at = position(eager, prompt, copied), position(eager, prompt, subject)
    out = eager.model(**eager.encode(prompt), output_attentions=True)
    rows = []
    for layer, head in HEADS:
        pattern = out.attentions[layer][0, head, -1].float()
        rows.append({"head": f"L{layer}/H{head}", "to_unit": round(float(pattern[unit]), 4),
                     "to_subject": round(float(pattern[at]), 4)})  # fmt: skip
    return rows


def scaled(runner, heads, alpha: float) -> dict:
    """Head interventions that scale ``heads``' outputs by ``alpha``, past the first token."""
    by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        by_layer.setdefault(layer, []).append(head)

    def at(layer):
        def intervene(z):
            z, d = z.clone(), runner.head_dim(layer)
            for head in by_layer[layer]:
                z[..., 1:, head * d : (head + 1) * d] *= alpha
            return z

        return intervene

    return {layer: at(layer) for layer in by_layer}


def scaling(runner, case, draws: list) -> dict:
    """Measure 4: the copy heads scaled by each alpha, and random heads switched off."""
    prompt, right, copied, _, _ = case
    r, c = token(runner, right), token(runner, copied)
    copy = []
    for a in ALPHAS:
        logits = last_logits(runner, prompt, scaled(runner, COPY_HEADS, a))
        copy.append({"alpha": a, **answer(runner, logits, r, c)})
    rand = [answer(runner, last_logits(runner, prompt, scaled(runner, d, 0.0)), r, c)
            for d in draws]  # fmt: skip
    four = answer(runner, last_logits(runner, prompt, scaled(runner, HEADS, 0.0)), r, c)
    return {"copy_heads": copy, "random_off": rand, "four_off": four}


@torch.no_grad()
def writers(runner, case) -> dict:
    """Measure 5: each layer 4 and 5 pole's direct write to logit(right) − logit(copied)."""
    prompt, right, copied, _, clause = case
    r, c = token(runner, right), token(runner, copied)
    unembed = runner.model.get_output_embeddings().weight.float()
    norm = runner.final_norm()
    direction = (unembed[r] - unembed[c]) * norm.weight.float()  # LayerNorm's gain, folded in
    out = {}
    for name, text in (("prompt", prompt), ("clause", clause)):
        hidden = residuals(runner, text)[-1, -1]
        scale = hidden.var(unbiased=False).add(norm.eps).sqrt()  # the norm's divisor here
        acts = runner.run_with_cache(text, [4, 5])[1]
        found = []
        for layer in (4, 5):
            a = acts[layer][0, -1].float()
            w = runner.down_proj(layer).weight.float()  # [d_model, d_mlp]
            w = w - w.mean(0, keepdim=True)  # LayerNorm subtracts the mean
            per = a * (direction @ w) / scale
            found += [(float(v), layer, n, float(a[n])) for n, v in enumerate(per)]
        found.sort()
        pole = lambda v, la, n, a: {"unit": f"L{la}/N{n}{'+' if a > 0 else '−'}",  # noqa: E731
                                    "write": round(v, 4), "activation": round(a, 4)}  # fmt: skip
        out[name] = {"to_right": [pole(*f) for f in reversed(found[-TOP_WRITERS:])],
                     "to_copied": [pole(*f) for f in found[:TOP_WRITERS]],
                     "sum_to_right": round(sum(v for v, *_ in found if v > 0), 4),
                     "sum_to_copied": round(sum(v for v, *_ in found if v < 0), 4),
                     "focus": next(pole(*f) for f in found if f[1:3] == FOCUS)}  # fmt: skip
    return out


@click.command()
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(out: Path) -> None:
    """Measure the six wrong prompts and three controls, and write one row per prompt."""
    from openinterpretability.cli import _load_runner

    runner = _load_runner(MODEL, "cpu", "float32")
    eager = _load_runner(MODEL, "cpu", "float32", None, "eager")
    rng = random.Random(SEED)
    pool = [(la, h) for la in (3, 4) for h in range(runner.n_heads) if (la, h) not in HEADS]
    draws = [rng.sample(pool, len(COPY_HEADS)) for _ in range(DRAWS)]
    rows = []
    for case in PROMPTS:
        prompt, right, copied, subject, clause = case
        r, c = token(runner, right), token(runner, copied)
        row = {"prompt": prompt, "right": right, "copied": copied, "subject": subject,
               "clause": clause, "wrong": right != copied,
               "answer": answer(runner, last_logits(runner, prompt), r, c),
               "single": single(runner, case), "enrichment": enrichment(runner, case),
               "attention": attention(eager, case), "scaling": scaling(runner, case, draws),
               "writers": writers(runner, case)}  # fmt: skip
        rows.append(row)
        click.echo(
            f"{prompt!r}: {row['answer']['top']!r}, right rank {row['answer']['right_rank']}"
        )
    rows.append({"table": "draws", "heads": [[f"L{la}/H{h}" for la, h in d] for d in draws]})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    click.echo(f"wrote {len(rows)} rows → {out}")


if __name__ == "__main__":
    main()
