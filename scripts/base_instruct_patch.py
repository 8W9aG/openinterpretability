"""How much of the base-to-instruct difference each layer's VERIFIED poles carry (#5140).

The test of Minder et al. (2025), §3.2, on native MLP neuron poles, in two designs. Into
the instruct model: the instruct model runs with one layer at its base activation, except
the chosen poles. Into the base model: the base model runs, and the chosen poles take the
instruct activation (``hooks.replace_pole``). The KL from the instruct model's next-token
distribution, over the response tokens, says how far the patch moves the output. Each
condition patches one layer:

* ``verified``: the layer's VERIFIED poles.
* ``random``: as many random poles of the same layer, not VERIFIED, in :data:`DRAWS` draws.
* ``other``: the VERIFIED poles with the instruct activations of another conversation,
  aligned by position. It tells a general shift of the tuned activations apart from
  information about the prompt (the review of Prakash et al. 2024).
* ``all``: every neuron of the layer.

The conversations are first user turns from UltraChat 200k with the instruct model's own
greedy responses. Nothing is written in ``wiki/``::

    uv run python scripts/base_instruct_patch.py --out papers/instruct-difference/results.jsonl
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE, INSTRUCT, SLUG = "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen--Qwen2.5-0.5B"
DATASET, SPLIT_FILE = (
    "HuggingFaceH4/ultrachat_200k",
    "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet",
)
CONVERSATIONS, RESPONSE_TOKENS, DRAWS, SEED = 128, 64, 5, 0
EARLY = (1, 10)  # response tokens 2 to 10, 0-based: the paper's "first 9 tokens" after one


def kl_per_position(target: torch.Tensor, patched: torch.Tensor) -> torch.Tensor:
    """KL(target ‖ patched) of the next-token distributions at each position.

    Args:
        target: The instruct model's logits, ``[seq, vocab]``.
        patched: The patched base model's logits, ``[seq, vocab]``.

    Returns:
        The KL at each position, ``[seq]``.

    Examples:
        >>> same = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
        >>> kl_per_position(same, same).tolist()
        [0.0, 0.0]
    """
    log_p = torch.log_softmax(target.float(), -1)  # float32: MPS has no float64
    log_q = torch.log_softmax(patched.float(), -1)
    return (log_p.exp() * (log_p - log_q)).sum(-1)


def response_means(kl: torch.Tensor, start: int, end: int) -> tuple[float, float]:
    """The mean KL over the predictions of response tokens ``start`` to ``end``, all and early.

    The logits at position ``i`` predict token ``i + 1``, so the response's predictions are
    at ``start - 1`` to ``end - 2``.

    Args:
        kl: The KL at each position.
        start: The first response token's position.
        end: One past the last response token.

    Returns:
        The mean over every response token, and over response tokens 2 to 10.

    Examples:
        >>> response_means(torch.arange(10.0), 4, 8)
        (4.5, 5.0)
    """
    over = kl[start - 1 : end - 1]
    early = over[EARLY[0] : EARLY[1]]
    return float(over.mean()), float(early.mean()) if early.numel() else float("nan")


def random_poles(width: int, exclude: set[tuple[int, str]], count: int, seed: int) -> list:
    """``count`` poles of a layer drawn with ``seed``, none of them in ``exclude``.

    Args:
        width: The layer's neurons.
        exclude: The ``(neuron, pole)`` pairs not to draw.
        count: How many to draw.
        seed: The draw's seed.

    Returns:
        The ``(neuron, pole)`` pairs.

    Examples:
        >>> random_poles(3, {(0, "positive")}, 5, 0) == random_poles(3, {(0, "positive")}, 5, 0)
        True
        >>> (0, "positive") in random_poles(3, {(0, "positive")}, 5, 0)
        False
    """
    pool = [(n, p) for n in range(width) for p in ("positive", "negative") if (n, p) not in exclude]
    return random.Random(seed).sample(pool, min(count, len(pool)))


def patch_poles(poles: list, source: torch.Tensor):
    """An intervention that gives each pole of ``poles`` its activation in ``source``.

    A positive pole takes the source's positive part and keeps its own negative part, and a
    negative pole the reverse (``hooks.replace_pole``). Positions past the end of
    ``source`` keep the base activation.

    Args:
        poles: The ``(neuron, pole)`` pairs.
        source: The instruct activation of the layer, ``[seq, width]``.

    Returns:
        The intervention, for ``hooks``'s ``interventions``.

    Examples:
        >>> base = torch.tensor([[[1.0, -1.0], [2.0, -2.0]]])
        >>> src = torch.tensor([[-3.0, 3.0], [4.0, -4.0]])
        >>> patch_poles([(0, "positive"), (1, "negative")], src)(base).tolist()
        [[[0.0, 0.0], [4.0, -4.0]]]
    """
    by_pole = {p: torch.tensor(sorted({n for n, q in poles if q == p}), dtype=torch.long)
               for p in ("positive", "negative")}  # fmt: skip

    def intervene(act: torch.Tensor) -> torch.Tensor:
        """The base activation with the poles' parts swapped.

        Args:
            act: The layer's activation, ``[1, seq, width]``.

        Returns:
            The patched activation.
        """
        out = act.clone()
        n = min(act.shape[1], source.shape[0])
        for pole, index in by_pole.items():
            if not index.numel():
                continue
            here, src = act[0, :n, index], source[:n, index].to(act.device, act.dtype)
            if pole == "positive":
                out[0, :n, index] = here.clamp(max=0) + src.clamp(min=0)
            else:
                out[0, :n, index] = here.clamp(min=0) + src.clamp(max=0)
        return out

    return intervene


def conditions(poles: list, width: int, layer: int) -> dict[str, list]:
    """Each condition's poles at one layer: verified, the random draws and every neuron.

    Args:
        poles: The layer's VERIFIED ``(neuron, pole)`` pairs.
        width: The layer's neurons.
        layer: The layer, which seeds the random draws.

    Returns:
        The poles of each condition, by name.
    """
    out = {
        "verified": poles,
        "all": [(n, p) for n in range(width) for p in ("positive", "negative")],
    }
    for draw in range(DRAWS):
        out[f"random{draw}"] = random_poles(
            width, set(poles), len(poles), SEED + 1000 * layer + draw
        )
    return out


def load_prompts(n: int = CONVERSATIONS, seed: int = SEED) -> list[str]:
    """``n`` first user turns of UltraChat 200k's ``test_sft`` split, drawn with ``seed``.

    Args:
        n: How many.
        seed: The draw's seed.

    Returns:
        The user messages.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, SPLIT_FILE, repo_type="dataset")
    prompts = pq.read_table(path, columns=["prompt"]).column("prompt").to_pylist()
    return random.Random(seed).sample([p for p in prompts if p.strip()], n)


def conversation(instruct, user: str) -> tuple[torch.Tensor, int]:
    """The chat template with ``user``'s turn and the instruct model's greedy response.

    Args:
        instruct: The instruct runner.
        user: The user's message.

    Returns:
        The token ids ``[1, seq]``, and the position of the response's first token.
    """
    tok = instruct.tokenizer
    text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True)  # fmt: skip
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(instruct.device)
    with torch.no_grad():
        out = instruct.model.generate(ids, max_new_tokens=RESPONSE_TOKENS, do_sample=False,
                                      pad_token_id=tok.eos_token_id)  # fmt: skip
    return out.cpu(), ids.shape[1]


def instruct_pass(instruct, ids: torch.Tensor, layers: list[int]) -> tuple:
    """The instruct model's logits ``[seq, vocab]`` and each layer's MLP activation.

    Args:
        instruct: The instruct runner.
        ids: The conversation's token ids.
        layers: The layers to read.

    Returns:
        The logits, and each layer's activation ``[seq, width]``, on the CPU.
    """
    logits, cache = instruct.run_with_cache(ids, layers)
    return logits[0].float().cpu(), {k: v[0].float().cpu() for k, v in cache.items()}


def batched_kl(runner, conv: dict, patches: list) -> list[tuple[float, float]]:
    """Each patch's KL from the instruct model, from one batched pass with a row per patch.

    Args:
        runner: The model that runs downstream.
        conv: The conversation: ``ids``, ``start`` and the instruct ``target`` logits.
        patches: For each row, a function from its activation ``[1, seq, width]`` to the
            patched one, or None to leave it.

    Returns:
        Each row's mean KL over every response token, and over response tokens 2 to 10.
    """
    ids, start, end = conv["ids"], conv["start"], conv["ids"].shape[1]

    def intervene(act: torch.Tensor) -> torch.Tensor:
        """Each row of the batch with its own patch.

        Args:
            act: The layer's activation, ``[rows, seq, width]``.

        Returns:
            The patched activation.
        """
        rows = [act[b : b + 1] if f is None else f(act[b : b + 1]) for b, f in enumerate(patches)]
        return torch.cat(rows)

    batch = ids.repeat(len(patches), 1)
    logits = runner.run_with_interventions(batch, {conv["layer"]: intervene})
    # Only the response's predictions, and on the device: the full vocabulary is large.
    target = conv["target"][start - 1 : end - 1].to(logits.device)
    n = end - start + 1
    return [response_means(kl_per_position(target, logits[b, start - 1 : end - 1]), 1, n)
            for b in range(len(patches))]  # fmt: skip


def into_base(base, conv: dict, poles: dict) -> list:
    """The issue's design: the chosen poles take the instruct activation in the base model.

    Args:
        base: The base runner.
        conv: The conversation: ``ids``, ``start``, the instruct ``target`` logits, its
            instruct activations ``inst`` and another conversation's, ``other``.
        poles: The VERIFIED ``(neuron, pole)`` pairs of each layer.

    Returns:
        Rows of ``layer``, ``condition`` and ``kl``: the mean over every response token, and
        over response tokens 2 to 10. ``none`` is the base model alone.
    """
    rows = []
    for layer, verified in sorted(poles.items()):
        inst = conv["inst"][layer]
        named = conditions(verified, inst.shape[1], layer)
        patches = {k: patch_poles(v, inst) for k, v in named.items()}
        patches |= {"other": patch_poles(verified, conv["other"][layer])}
        patches |= {"none": None} if not rows else {}  # the base model alone, once
        found = batched_kl(base, conv | {"layer": layer}, list(patches.values()))
        for name, kl in zip(patches, found, strict=True):
            at = None if name == "none" else layer
            rows.append({"design": "into_base", "layer": at, "condition": name, "kl": kl})
    return rows


def into_instruct(instruct, conv: dict, poles: dict) -> list:
    """The paper's design: the instruct model, with one layer at its base activation.

    The chosen poles keep their instruct activation.

    Args:
        instruct: The instruct runner.
        conv: The conversation, as for :func:`into_base`, with ``base_acts``: the base
            model's activations by layer.
        poles: The VERIFIED ``(neuron, pole)`` pairs of each layer.

    Returns:
        Rows as for :func:`into_base`. ``base`` is the layer wholly at its base activation,
        the reference. The whole instruct layer would give 0.
    """
    rows = []
    for layer, verified in sorted(poles.items()):
        inst, at_base = conv["inst"][layer], conv["base_acts"][layer]
        named = {k: v for k, v in conditions(verified, inst.shape[1], layer).items() if k != "all"}
        sources = {k: inst for k in named} | {"base": inst, "other": conv["other"][layer]}
        chosen = named | {"base": [], "other": verified}

        def patch(name: str, chosen=chosen, sources=sources, at_base=at_base):
            fn = patch_poles(chosen[name], sources[name])
            return lambda act: fn(at_base[None].to(act.device, act.dtype))

        found = batched_kl(instruct, conv | {"layer": layer}, [patch(k) for k in chosen])
        rows += [{"design": "into_instruct", "layer": layer, "condition": k, "kl": v}
                 for k, v in zip(chosen, found, strict=True)]  # fmt: skip
    return rows


def partner(lengths: list[int], i: int) -> int:
    """Another conversation at least as long as conversation ``i``, or the longest other one.

    Args:
        lengths: Each conversation's length.
        i: The conversation.

    Returns:
        The next such conversation after ``i``, in order.

    Examples:
        >>> partner([5, 3, 9, 4], 0), partner([5, 3, 9, 4], 2), partner([5, 3, 9, 4], 3)
        (2, 0, 0)
    """
    n = len(lengths)
    later = [(i + k) % n for k in range(1, n)]
    return next(
        (j for j in later if lengths[j] >= lengths[i]), max(later, key=lambda j: lengths[j])
    )


def verified_by_layer(wiki_dir: Path) -> dict[int, list]:
    """The base model's VERIFIED ``(neuron, pole)`` pairs, by layer.

    Args:
        wiki_dir: The wiki folder.

    Returns:
        The pairs of each layer.
    """
    from conformal_line import newest_verified

    out: dict[int, list] = {}
    for record in newest_verified(wiki_dir).get(SLUG, []):
        ident = record["submission"]["identifier"]
        out.setdefault(ident["layer"], []).append((ident["neuron_idx"], ident["pole"]))
    return {layer: sorted(set(pairs)) for layer, pairs in out.items()}


@click.command()
@click.option("--wiki-dir", type=click.Path(file_okay=False, path_type=Path), default=Path("wiki"))
@click.option("--device", default="auto", show_default=True)
@click.option("--conversations", default=CONVERSATIONS, show_default=True)
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(wiki_dir, device, conversations, out):  # pragma: no cover - loads two models
    """Patch the instruct model's poles into the base model, and read the KL that is left.

    Args:
        wiki_dir: The wiki folder.
        device: Where both models run.
        conversations: How many conversations.
        out: The JSONL file for the rows.
    """
    from openinterpretability.cli import _load_runner

    poles = verified_by_layer(wiki_dir)
    base = _load_runner(BASE, device, "float32", None)
    instruct = _load_runner(INSTRUCT, device, "float32", None)
    convs = [conversation(instruct, user) for user in load_prompts(conversations)]
    layers = sorted(poles)
    reads = [instruct_pass(instruct, ids, layers) for ids, _ in convs]
    lengths = [ids.shape[1] for ids, _ in convs]
    rows = [{"table": "setup", "base": BASE, "instruct": INSTRUCT, "conversations": len(convs),
             "poles": {str(k): len(v) for k, v in poles.items()}, "draws": DRAWS}]  # fmt: skip
    for i, (ids, start) in enumerate(convs):
        other = reads[partner(lengths, i)][1]
        conv = {"ids": ids, "start": start, "target": reads[i][0], "inst": reads[i][1],
                "other": {k: v[: ids.shape[1]] for k, v in other.items()},
                "base_acts": instruct_pass(base, ids, layers)[1]}  # fmt: skip
        found = into_base(base, conv, poles) + into_instruct(instruct, conv, poles)
        rows += [r | {"table": "kl", "conversation": i} for r in found]
        click.echo(f"conversation {i + 1} of {len(convs)}", err=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    click.echo(f"wrote {len(rows)} rows → {out}")


if __name__ == "__main__":
    main()
