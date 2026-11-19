"""Is pythia-70m's L4/H1 a copy-suppression head? (issue #5063).

McDougall et al. (2023) describe copy suppression: a head attends to a token that earlier
layers predict, and lowers that token's logit. This script tests pythia-70m's L4/H1 for it
in three ways, in float32 on the CPU:

1. ``ov``: the OV test of their Section 3.1, for every head. Each of :data:`OV_TOKENS`
   seeded tokens goes from its effective embedding through the head's OV circuit to the
   unembedding. The score is the share of tokens whose own logit is among the
   :data:`OV_RANK` most negative. The effective embedding is the token embedding plus
   the layer-0 MLP's write.
2. ``cspa``: a copy-suppression-preserving ablation (their Section 4) on :data:`WINDOWS`
   corpus windows. Mean ablation (MA) sets the head's output to its mean. CSPA does the
   same, then adds back each source token's write along that token's unembedding, when
   it lowers that token. Effect explained = 1 − KL(CSPA) / KL(MA), from the clean output.
   The control adds back the writes that raise the token instead. Only their OV part is
   kept. Their QK part is not.
3. ``prompts``: the shortlist's six wrong prompts (#5015). It records the head's direct
   write at the answer position to the copied unit and to the right unit. The final norm's scale
   is frozen at the real residual stream's. It also gives the answer with the head's
   output scaled by each of :data:`SCALES`.

The rows go to ``--out``::

    uv run python scripts/copy_suppression.py --out docs/analyses/copy-suppression-l4h1.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import click
import torch

MODEL = "EleutherAI/pythia-70m"
HEAD = (4, 1)
OV_TOKENS = 4000
OV_RANK = 10  # the paper's "among the 10 most negative"
WINDOWS = 300
WINDOW_TOKENS = 64
SCALES = (0.0, 1.0, 2.0, 4.0)
SEED = 0
# The shortlist's six wrong prompts: (prompt, right answer, copied unit), as in #5015.
PROMPTS = (
    ("A minute has sixty seconds and an hour has sixty", " minutes", " seconds"),
    ("A week has seven days and a year has twelve", " months", " days"),
    ("A hand has five fingers and a foot has five", " toes", " fingers"),
    ("A dozen has twelve eggs and a year has twelve", " months", " eggs"),
    ("An hour has sixty minutes and a minute has sixty", " seconds", " minutes"),
    ("A year has twelve months and a week has seven", " days", " months"),
)


def layer(runner, index: int):
    """The GPT-NeoX layer ``index``."""
    return runner.model.gpt_neox.layers[index]


def unembedding(runner) -> torch.Tensor:
    """``[vocab, d]``: the unembedding with the final norm's gain folded in."""
    w = runner.model.get_output_embeddings().weight.float()
    return w * runner.final_norm().weight.float()


@torch.no_grad()
def ov_scores(runner, tokens: list[int]) -> list[dict]:
    """Measure 1: each head's share of tokens whose own logit is among the most negative."""
    emb = runner.model.get_input_embeddings().weight.float()[tokens]
    first = layer(runner, 0)
    effective = emb + first.mlp(first.post_attention_layernorm(emb[:, None]))[:, 0].float()
    w_u = unembedding(runner)
    w_u = w_u - w_u.mean(0, keepdim=True)
    ids = torch.tensor(tokens)
    rows = []
    d = runner.head_dim(0)
    for la in range(runner.n_layers):
        block = layer(runner, la)
        x = block.input_layernorm(effective[:, None])[:, 0].float()
        qkv = block.attention.query_key_value(x).float().view(len(tokens), runner.n_heads, 3 * d)
        for h in range(runner.n_heads):
            v = qkv[:, h, 2 * d :]
            out = v @ runner.head_output_weights(la, h).float().T  # [tokens, d_model]
            hits = 0
            for start in range(0, len(tokens), 500):
                logits = out[start : start + 500] @ w_u.T
                own = logits.gather(1, ids[start : start + 500, None])
                hits += int(((logits < own).sum(1) < OV_RANK).sum())
            rows.append({"head": f"L{la}/H{h}", "own_in_bottom": round(hits / len(tokens), 4)})
    return rows


@torch.no_grad()
def cspa(runner, eager, texts: list[str], means) -> dict:
    """Measure 2: KL from the clean output under MA, CSPA and its control, and the share."""
    la, h = HEAD
    d = runner.head_dim(la)
    w_o = runner.head_output_weights(la, h).float()
    w_u = unembedding(runner)
    unit = w_u / w_u.norm(dim=-1, keepdim=True)
    mean_z = means.heads[la][h * d : (h + 1) * d].float()
    totals = {"ma": 0.0, "cspa": 0.0, "control": 0.0}
    count = 0
    for text in texts:
        enc = eager.encode(text)
        ids = enc["input_ids"][0]
        seen = {}
        hook = layer(eager, la).attention.query_key_value.register_forward_hook(
            lambda _m, _a, out, seen=seen: seen.__setitem__("qkv", out.detach())
        )
        try:
            out = eager.model(**enc, output_attentions=True)
        finally:
            hook.remove()
        clean = torch.log_softmax(out.logits[0].float(), -1)
        pattern = out.attentions[la][0, h].float()  # [dest, source]
        v = seen["qkv"][0].float().view(len(ids), runner.n_heads, 3 * d)[:, h, 2 * d :]
        write = v @ w_o.T  # each source's write, before its attention weight
        along = (write * unit[ids]).sum(-1)  # its component along its own token's unembedding
        kept = {"cspa": along.clamp(max=0.0), "control": along.clamp(min=0.0)}
        for name in totals:
            add = None if name == "ma" else pattern @ (kept[name][:, None] * unit[ids])
            logp = _ablated(eager, enc, mean_z, add)
            kl = (clean.exp() * (clean - logp)).sum(-1)[1:]
            totals[name] += float(kl.sum())
        count += len(ids) - 1
    mean = {k: v / count for k, v in totals.items()}
    return {"positions": count, "kl": {k: round(v, 6) for k, v in mean.items()},
            "explained": round(1 - mean["cspa"] / mean["ma"], 4),
            "control_explained": round(1 - mean["control"] / mean["ma"], 4)}  # fmt: skip


def _ablated(runner, enc, mean_z, add) -> torch.Tensor:
    """Log-probabilities with the head mean-ablated, plus ``add`` on its output (or none)."""
    la, h = HEAD
    d = runner.head_dim(la)

    def heads(z):
        z = z.clone()
        z[..., 1:, h * d : (h + 1) * d] = mean_z.to(z)
        return z

    def plus(_module, _args, output):
        output = output.clone()
        output[0, 1:] += add[1:].to(output)
        return output

    handle = runner.attn_out(la).register_forward_hook(plus) if add is not None else None
    try:
        with runner.hooks(None, None, {la: heads}):
            logits = runner.forward(enc)[0].float()
    finally:
        if handle is not None:
            handle.remove()
    return torch.log_softmax(logits, -1)


@torch.no_grad()
def prompts(runner) -> list[dict]:
    """Measure 3: the head's direct write to the copied and right units, and its scaling."""
    la, h = HEAD
    d = runner.head_dim(la)
    w_o = runner.head_output_weights(la, h).float()
    w_u = unembedding(runner)
    rows = []
    for prompt, right, copied in PROMPTS:
        r, c = (runner.tokenizer.encode(t, add_special_tokens=False)[0] for t in (right, copied))
        enc = runner.encode(prompt)
        seen = {}

        def keep(z, seen=seen):
            seen["z"] = z[0, -1, h * d : (h + 1) * d].detach().float()

        final = []
        hook = runner.final_norm().register_forward_pre_hook(
            lambda _m, a, final=final: final.append(a[0])
        )
        try:
            with runner.hooks(None, None, {la: keep}):
                runner.forward(enc)
        finally:
            hook.remove()
        resid = final[0][0, -1].float()
        scale = resid.var(unbiased=False).add(runner.final_norm().eps).sqrt()
        out = w_o @ seen["z"]
        logits = (w_u @ (out - out.mean())) / scale
        logits = logits - logits.mean()
        row = {"prompt": prompt, "right": right, "copied": copied,
               "direct_copied": round(float(logits[c]), 4),
               "direct_right": round(float(logits[r]), 4), "scaled": []}  # fmt: skip
        for alpha in SCALES:
            row["scaled"].append({"alpha": alpha, **_answer(runner, enc, alpha, r, c)})
        rows.append(row)
    return rows


def _answer(runner, enc, alpha: float, right: int, copied: int) -> dict:
    """The answer with the head's output scaled by ``alpha``, past the first token."""
    la, h = HEAD
    d = runner.head_dim(la)

    def scale(z):
        z = z.clone()
        z[..., 1:, h * d : (h + 1) * d] *= alpha
        return z

    with runner.hooks(None, None, {la: scale}):
        logits = runner.forward(enc)[0, -1].float()
    probs = torch.softmax(logits, -1)
    top = runner.tokenizer.decode([int(logits.argmax())])
    return {"top": top, "p_copied": round(float(probs[copied]), 4),
            "p_right": round(float(probs[right]), 4)}  # fmt: skip


@click.command()
@click.option("--corpus", type=click.Path(dir_okay=False, path_type=Path), default="corpus.jsonl")
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(corpus: Path, out: Path) -> None:
    """Run the three measures and write their rows."""
    from openinterpretability.cli import _load_runner
    from openinterpretability.evidence import read_corpus
    from openinterpretability.explain import unit_means
    from openinterpretability.harness.explanation_scores import cut_windows

    runner = _load_runner(MODEL, "cpu", "float32")
    eager = _load_runner(MODEL, "cpu", "float32", None, "eager")
    texts = read_corpus(corpus)[0]
    means = unit_means(runner, MODEL, texts, 20_000)
    rng = random.Random(SEED)
    tokens = rng.sample(range(runner.model.get_input_embeddings().weight.shape[0]), OV_TOKENS)
    ov = ov_scores(runner, tokens)
    click.echo(f"OV: L4/H1 {next(r for r in ov if r['head'] == 'L4/H1')['own_in_bottom']}")
    windows = cut_windows(runner, texts[-200:], WINDOW_TOKENS)[:WINDOWS]
    ablation = cspa(runner, eager, windows, means)
    click.echo(
        f"CSPA explains {ablation['explained']:.1%}, control {ablation['control_explained']:.1%}"
    )
    rows = [{"table": "ov", "rows": ov}, {"table": "cspa", **ablation},
            *({"table": "prompt", **r} for r in prompts(runner))]  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    click.echo(f"wrote {len(rows)} rows → {out}")


if __name__ == "__main__":
    main()
