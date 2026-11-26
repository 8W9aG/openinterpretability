"""ROME edits of unrelated facts at one layer of pythia-70m, and the poles that keep (#5167).

ROME (Meng et al., 2022) inserts a fact with a rank-one change to one MLP's output matrix:
Ŵ = W + Λuᵀ, with u = C⁻¹k* and Λ = (v* − Wk* − b) / (uᵀk*). Here C is the second moment of
the layer's hidden activations, k* the hidden activation at the subject's last token, and v*
the output that makes the model predict the new object. The edit changes each neuron's
write and no neuron's input weights. So in pythia-70m, which has a parallel residual, every
pole at or below the edit layer fires as before. Poles above it change through the
residual stream.

For each fact the script edits the base model, tests every VERIFIED pole with the batched
keep test of ``scripts/random_delta_keep.py``, and restores the weights. It then does the
same with a random rank-one change of the same norm. Each pole row above the edit layer
gives the alignment of Λ with the pole's input weights. Each edit's row gives how u loads
on the VERIFIED neurons of the edit layer.

    uv run python scripts/rome_keep.py --device cuda --out papers/rome-keep/results.jsonl
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from random_delta_keep import (  # noqa: E402
    base_columns,
    base_rows,
    cached_readings,
    changed,
    first_position_activations,
    probe_activations,
    run_condition,
)

from openinterpretability.interference.keep import (  # noqa: E402
    SAMPLE_TOKENS,
    sample_windows,
    verified_records,
)

MODEL = "EleutherAI/pythia-70m"
LAYER = 2
# Unrelated counterfactual facts, each with a one-token new object: (subject, relation, object).
FACTS = (
    ("The Eiffel Tower", " is located in the city of", " Rome"),
    ("Barack Obama", " was born in the country of", " France"),
    ("The Amazon River", " flows through the continent of", " Europe"),
    ("Mount Everest", " is part of the mountain range called the", " Alps"),
    ("Albert Einstein", " is best known for his work in", " music"),
    ("The Great Wall", " was built in the country of", " Brazil"),
    ("William Shakespeare", " wrote his plays in the language of", " German"),
    ("The Pacific Ocean", " is the largest", " desert"),
    ("Toyota", " is a company based in the country of", " Canada"),
    ("The Moon", " orbits the planet", " Mars"),
)  # fmt: skip
PREFIXES = 10  # corpus prefixes over which k* and v* are fitted (the paper uses 20 random ones)
COV_TOKENS = 100_000  # tokens for the second moment C
RIDGE = 1e-4  # times the mean diagonal of C, for a stable solve
V_STEPS, V_LR, KL_WEIGHT, CLAMP = 40, 0.5, 0.0625, 4.0  # our settings for the fit of v*


def down(model, layer: int):
    """The MLP output projection of ``layer`` in a GPT-NeoX model.

    Args:
        model: The model.
        layer: The layer.
    """
    return model.gpt_neox.layers[layer].mlp.dense_4h_to_h


def param_name(layer: int) -> str:
    """The parameter name of ``layer``'s output projection weight.

    Args:
        layer: The layer.
    """
    return f"gpt_neox.layers.{layer}.mlp.dense_4h_to_h.weight"


def rome_update(weight, bias, cov, key, value) -> tuple[torch.Tensor, torch.Tensor]:
    """ROME's rank-one update: ``(Λ, u)`` such that ``(W + Λuᵀ)k* + b = v*``.

    Args:
        weight: The output matrix W, ``[d_model, hidden]``.
        bias: Its bias b, ``[d_model]``.
        cov: The second moment C of the hidden activations, ``[hidden, hidden]``.
        key: k*, ``[hidden]``.
        value: v*, ``[d_model]``.

    Returns:
        Λ, ``[d_model]``, and u = C⁻¹k*, ``[hidden]``.
    """
    u = torch.linalg.solve(cov, key)
    lam = (value - weight @ key - bias) / (u @ key)
    return lam, u


def random_rank_one(lam, u, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Gaussian Λ′ and u′ with the norms of ``lam`` and ``u``, so ``‖Λ′u′ᵀ‖ = ‖Λuᵀ‖``.

    Args:
        lam: Λ of the edit.
        u: u of the edit.
        seed: The seed of the draw.
    """
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(lam.shape, generator=generator, dtype=torch.float64)
    b = torch.randn(u.shape, generator=generator, dtype=torch.float64)
    lam_norm, u_norm = float(lam.norm()), float(u.norm())
    return ((a / a.norm() * lam_norm).to(lam.dtype).to(lam.device),
            (b / b.norm() * u_norm).to(u.dtype).to(u.device))  # fmt: skip


def alignment(w_in, gain, lam) -> float:
    """|cos| of a neuron's input weights with Λ as its layer norm scales it.

    Args:
        w_in: The neuron's input weights, ``[d_model]``.
        gain: The gain of the layer norm that feeds the MLP, ``[d_model]``.
        lam: Λ, the direction that the edit adds to the residual stream.
    """
    scaled = gain * lam
    return float((w_in @ scaled).abs() / (w_in.norm() * scaled.norm()).clamp_min(1e-30))


def hidden_at(model, ids: torch.Tensor, layer: int) -> torch.Tensor:
    """The hidden activations that ``layer``'s output projection reads, ``[batch, seq, hidden]``.

    Args:
        model: The model.
        ids: Token ids, ``[batch, seq]``.
        layer: The layer.
    """
    got = {}

    def keep(_m, args, _out) -> None:
        """Keep the projection's input. Return None, so that its output stays.

        Args:
            _m: The module.
            args: Its positional inputs, the hidden activations first.
            _out: Its output.
        """
        got["h"] = args[0].detach()

    hook = down(model, layer).register_forward_hook(keep)
    try:
        with torch.no_grad():
            model(ids.to(next(model.parameters()).device))
    finally:
        hook.remove()
    return got["h"]


def second_moment(model, windows: list[torch.Tensor], layer: int) -> torch.Tensor:
    """C: the uncentred second moment of the hidden activations, with a small ridge.

    Args:
        model: The model.
        windows: Batches of token ids.
        layer: The layer.
    """
    total, count = None, 0
    for batch in windows:
        h = hidden_at(model, batch, layer).reshape(-1, down(model, layer).in_features).double()
        total = h.T @ h if total is None else total + h.T @ h
        count += len(h)
    cov = total / count
    ridge = RIDGE * cov.diagonal().mean()
    return (cov + ridge * torch.eye(len(cov), dtype=cov.dtype, device=cov.device)).float()


def contexts(tokenizer, windows: list[torch.Tensor], subject: str, n: int = PREFIXES) -> list:
    """``n`` prompts that end with the subject, each after a short corpus prefix.

    Args:
        tokenizer: The model's tokenizer.
        windows: Batches of corpus token ids, which give the prefixes.
        subject: The subject.
        n: How many.

    Returns:
        ``[(prefix ids, subject ids)]``, so that the subject's last token is known.
    """
    rows = [row for b in windows for row in b]
    subject_ids = tokenizer(" " + subject)["input_ids"]
    # Prefixes of 5 or 10 tokens, as in the paper, taken from the start of fixed windows.
    return [(rows[i][: 5 if i % 2 else 10].tolist(), subject_ids) for i in range(n)]


def fit_value(model, tokenizer, ctxs, fact, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    """k* and v* of one fact: the mean hidden activation and the fitted output at the subject.

    v* is the MLP's mean output at the subject's last token plus a δ that Adam fits. The
    loss is the new object's negative log-probability after each context and the relation,
    plus a KL term on the prediction after "{subject} is a". The term keeps the subject's
    other knowledge, as ROME does.

    Args:
        model: The model.
        tokenizer: Its tokenizer.
        ctxs: The contexts, from :func:`contexts`.
        fact: ``(subject, relation, object)``.
        layer: The layer.

    Returns:
        k* and v*.
    """
    subject, relation, obj = fact
    device = next(model.parameters()).device
    rel_ids, target = tokenizer(relation)["input_ids"], tokenizer(obj)["input_ids"][0]
    prompts = [(p + s + rel_ids, len(p) + len(s) - 1) for p, s in ctxs]
    essence = tokenizer(" " + subject)["input_ids"]
    essence_ids, essence_pos = essence + tokenizer(" is a")["input_ids"], len(essence) - 1
    keys, outs = [], []
    for ids, pos in prompts:
        t = torch.tensor([ids], device=device)
        keys.append(hidden_at(model, t, layer)[0, pos])
        outs.append(down(model, layer)(keys[-1]).detach())
    key, base_out = torch.stack(keys).mean(0), torch.stack(outs).mean(0)
    delta = torch.zeros_like(base_out, requires_grad=True)
    with torch.no_grad():
        base_logp = torch.log_softmax(
            model(torch.tensor([essence_ids], device=device)).logits[0, -1], -1
        )
    optimizer = torch.optim.Adam([delta], lr=V_LR)
    for _ in range(V_STEPS):
        loss = KL_WEIGHT * _essence_kl(model, essence_ids, essence_pos, delta, base_logp, layer)
        for ids, pos in prompts:
            logits = _with_delta(model, ids, pos, delta, layer)
            loss = loss - torch.log_softmax(logits[0, -1], -1)[target] / len(prompts)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():  # δ stays within a multiple of the output's norm
            limit = CLAMP * base_out.norm()
            if delta.norm() > limit:
                delta.mul_(limit / delta.norm())
    return key, (base_out + delta).detach()


def _with_delta(model, ids: list[int], pos: int, delta, layer: int):
    """The logits with ``delta`` added to ``layer``'s MLP output at position ``pos``.

    Args:
        model: The model.
        ids: The prompt's token ids.
        pos: The position of the subject's last token.
        delta: The vector to add.
        layer: The layer.
    """

    def add(_m, _args, out):
        """The projection's output with ``delta`` added at ``pos``.

        Args:
            _m: The module.
            _args: Its positional inputs.
            out: Its output.
        """
        out = out.clone()
        out[0, pos] = out[0, pos] + delta
        return out

    hook = down(model, layer).register_forward_hook(add)
    try:
        return model(torch.tensor([ids], device=delta.device)).logits
    finally:
        hook.remove()


def _essence_kl(model, ids: list[int], pos: int, delta, base_logp, layer: int):
    """KL from the unedited prediction after "{subject} is a" to the one with ``delta``.

    Args:
        model: The model.
        ids: The essence prompt's token ids.
        pos: The position of the subject's last token in it.
        delta: The vector to add.
        base_logp: The unedited log-probabilities at the prompt's last token.
        layer: The layer.
    """
    logp = torch.log_softmax(_with_delta(model, ids, pos, delta, layer)[0, -1], -1)
    return (base_logp.exp() * (base_logp - logp)).sum()


def object_prob(model, tokenizer, fact) -> float:
    """P(the new object) after "{subject}{relation}", with no prefix.

    Args:
        model: The model, as it is now.
        tokenizer: Its tokenizer.
        fact: ``(subject, relation, object)``.
    """
    subject, relation, obj = fact
    device = next(model.parameters()).device
    ids = torch.tensor([tokenizer(subject + relation)["input_ids"]], device=device)
    with torch.no_grad():
        probs = torch.softmax(model(ids).logits[0, -1].float(), -1)
    return float(probs[tokenizer(obj)["input_ids"][0]])


def u_loading(u: torch.Tensor, neurons: list[int]) -> dict:
    """How u loads on ``neurons``: the percentile of each one's |u_j| among all the layer's.

    Args:
        u: u of the edit, ``[hidden]``.
        neurons: The VERIFIED neurons of the edit layer.

    Returns:
        The median percentile, 0.5 for neurons that load as a random set would, and the
        percentiles themselves.
    """
    ranks = u.abs().argsort().argsort().float() / (len(u) - 1)
    got = [round(float(ranks[n]), 4) for n in neurons]
    return {"median_percentile": round(statistics.median(got), 4) if got else None,
            "percentiles": got}  # fmt: skip


def with_alignment(rows: list[dict], model, records, lam) -> list[dict]:
    """``rows`` with each pole above the edit layer given its alignment with ``lam``.

    Args:
        rows: The pole rows of one condition, from ``random_delta_keep.run_condition``.
        model: The model, whose input weights the edit leaves as they are.
        records: The VERIFIED records, in the rows' order.
        lam: Λ of the condition.
    """
    for row, record in zip(rows, records, strict=True):
        ident = record.submission.identifier
        if ident.layer > LAYER:
            layer = model.gpt_neox.layers[ident.layer]
            w_in = layer.mlp.dense_h_to_4h.weight[ident.neuron_idx].detach()
            gain = layer.post_attention_layernorm.weight.detach()
            row["alignment"] = round(alignment(w_in, gain, lam.to(w_in.device)), 6)
    return rows


@click.command()
@click.option("--wiki-dir", default="wiki", type=click.Path(file_okay=False, path_type=Path))
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--device", default="cpu", show_default=True)
def main(wiki_dir, out, device) -> None:
    """Edit each fact in turn with ROME and with a random rank-one change, and test each pole.

    Args:
        wiki_dir: The registry that holds the VERIFIED records.
        out: The JSONL file to write. It is written again after each condition.
        device: The device of the model.
    """
    from openinterpretability.cli import _load_runner

    records = verified_records(wiki_dir, MODEL)
    runner = _load_runner(MODEL, device, "float32", None)
    model, tokenizer = runner.model, runner.tokenizer
    for p in model.parameters():
        p.requires_grad_(False)
    batches = sample_windows(tokenizer)
    base = base_columns(runner, batches, records)
    cache, firsts = probe_activations(runner, records), first_position_activations(runner, records)
    on_base = {r.submission.identifier.slug: cached_readings(runner, r, cache, firsts)
               for r in records}  # fmt: skip
    lines = base_rows(records, on_base, MODEL)
    cov = second_moment(model, sample_windows(tokenizer, COV_TOKENS), LAYER)
    ctx_windows = sample_windows(tokenizer, COV_TOKENS)[-3:]  # prefixes from the end of C's sample
    verified = sorted({r.submission.identifier.neuron_idx for r in records
                       if r.submission.identifier.layer == LAYER})  # fmt: skip
    layer = down(model, LAYER)
    for i, fact in enumerate(FACTS):
        key, value = fit_value(model, tokenizer, contexts(tokenizer, ctx_windows, fact[0]), fact,
                               LAYER)  # fmt: skip
        lam, u = rome_update(layer.weight.detach(), layer.bias.detach(), cov, key, value)
        for kind, (a, b) in (("rome", (lam, u)), ("random", random_rank_one(lam, u, i))):
            change = {param_name(LAYER): torch.outer(a, b).cpu()}
            fields = {"model": MODEL, "condition": kind, "fact": i, "edit_layer": LAYER,
                      "sample_tokens": SAMPLE_TOKENS}  # fmt: skip
            before = object_prob(model, tokenizer, fact)
            rows = run_condition(runner, batches, records, base, on_base, change, fields)
            with changed(model, change):
                after = object_prob(model, tokenizer, fact)
            rows[0] |= {"object_prob_before": round(before, 6),
                        "object_prob_after": round(after, 6),
                        "u_loading": u_loading(b.cpu(), verified)}  # fmt: skip
            lines += [rows[0], *with_alignment(rows[1:], model, records, a)]
            click.echo(json.dumps({k: v for k, v in rows[0].items()
                                   if k not in ("flipped", "u_loading")}), err=True)  # fmt: skip
            out.write_text("".join(json.dumps(r) + "\n" for r in lines))


if __name__ == "__main__":
    main()
