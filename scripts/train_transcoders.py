"""Train skip transcoders for one model's MLPs: the learned-basis baseline where no public
dictionary exists (Qwen2.5-0.5B, the registry's default model).

Each transcoder reads its layer's MLP input (``mlp_in``) and reconstructs the MLP's output,
with a linear skip term (Paulo, Shabalin & Belrose, 2025, arXiv:2501.18823). It trains
with a top-k activation and an AuxK loss against dead latents (Gao et al., 2024,
arXiv:2406.04093), then becomes a JumpReLU with one threshold, chosen on held-out text so
that its L0 matches k, as Llama Scope converted theirs. A JumpReLU latent's activation
depends only on its own pre-activation, which the harness needs (``Dictionary.feature``).

The data is ``NeelNanda/pile-10k`` without the scan corpus's documents, so no scanned
text was trained on. One forward pass feeds every layer's transcoder. Each layer's
weights go to ``<out>/layer_<layer>.safetensors`` in the layout
``openinterp.harness.dictionaries`` reads, with training statistics in
``<out>/layer_<layer>.json``.

    uv run python scripts/train_transcoders.py --layers 2,12,21 --tokens 3000000 \\
        --out exchange/qwen-transcoders-pilot
"""

from __future__ import annotations

import json
import random
import time
from contextlib import suppress
from pathlib import Path

import click
import torch
from torch import nn

PILE = ("NeelNanda/pile-10k", "127bfedcd5047750df5ccf3a12979a47bfa0bafa")
PILE_FILE = "data/train-00000-of-00001-4746b8785c874cc7.parquet"
PREFIX_CHARS = 200  # a scan document is matched to its pile-10k row by its first characters
VALIDATION_DOCS = 100


def training_texts(scan_corpus: Path) -> tuple[list[str], list[str], int]:
    """pile-10k's documents outside the scan corpus, split into training and validation
    documents, and how many scan documents were left out."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(PILE[0], PILE_FILE, repo_type="dataset", revision=PILE[1])
    texts = pq.read_table(path).column("text").to_pylist()
    lines = scan_corpus.read_text(encoding="utf-8").splitlines()
    scanned = {json.loads(line)["text"][:PREFIX_CHARS] for line in lines if line.strip()}
    kept = [t for t in texts if t[:PREFIX_CHARS] not in scanned]
    return kept[:-VALIDATION_DOCS], kept[-VALIDATION_DOCS:], len(texts) - len(kept)


def sequences(tokenizer, texts: list[str], seq_len: int, seed: int) -> list[list[int]]:
    """Every document cut into pieces of at most ``seq_len`` tokens, in a seeded order."""
    pieces = []
    for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]:
        pieces += [ids[i : i + seq_len] for i in range(0, len(ids), seq_len)]
    pieces = [p for p in pieces if len(p) >= 32]
    random.Random(seed).shuffle(pieces)
    return pieces


class _Stop(Exception):
    pass


@torch.no_grad()
def mlp_io(runner, rows: list[list[int]], layers: list[int]) -> dict[int, tuple]:
    """Each layer's MLP input and output on every token but each row's first, ``[T, d]``."""
    width = max(map(len, rows))
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    mask = torch.zeros(len(rows), width, dtype=torch.bool)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.tensor(r)
        mask[i, : len(r)] = True
    keep = mask.clone()
    keep[:, 0] = False  # the attention sink; the harness's fit leaves it out too
    keep = keep.to(runner.device)
    seen: dict[int, list] = {layer: [] for layer in layers}
    last = max(layers)

    def read(layer):
        return lambda _m, args: seen[layer].append(args[0][keep].float())

    def write(layer):
        def hook(_m, _a, out):
            seen[layer].append(out[keep].float())
            if layer == last:
                raise _Stop

        return hook

    handles = []
    for layer in layers:
        handles.append(runner.mlp(layer).register_forward_pre_hook(read(layer)))
        handles.append(runner.mlp_write(layer).register_forward_hook(write(layer)))
    try:
        with suppress(_Stop):
            runner.model(input_ids=ids.to(runner.device), attention_mask=mask.to(runner.device))
    finally:
        for h in handles:
            h.remove()
    return {layer: (seen[layer][0], seen[layer][1]) for layer in layers}


class TopKTranscoder(nn.Module):
    """Trains on inputs and outputs scaled to an average norm of sqrt(d)."""

    def __init__(self, d_in: int, d_out: int, n: int, k: int, k_aux: int, generator):
        super().__init__()
        w_dec = torch.randn(n, d_out, generator=generator)
        w_dec /= w_dec.norm(dim=-1, keepdim=True)
        self.w_dec = nn.Parameter(w_dec)
        self.w_enc = nn.Parameter(w_dec.T.clone() if d_in == d_out else torch.randn(d_in, n))
        self.b_enc = nn.Parameter(torch.zeros(n))
        self.b_dec = nn.Parameter(torch.zeros(d_out))
        self.w_skip = nn.Parameter(torch.zeros(d_out, d_in))
        self.k, self.k_aux = k, k_aux
        self.register_buffer("idle", torch.zeros(n, dtype=torch.long))  # tokens since it fired
        self.register_buffer("scale_in", torch.tensor(0.0))
        self.register_buffer("scale_out", torch.tensor(0.0))

    def pre(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w_enc + self.b_enc

    def loss(self, x: torch.Tensor, y: torch.Tensor, dead_after: int, aux_coef: float) -> dict:
        pre = self.pre(x)
        top = pre.topk(self.k, dim=-1)
        acts = torch.zeros_like(pre).scatter(-1, top.indices, top.values.relu())
        linear = self.b_dec + x @ self.w_skip.T
        recon = acts @ self.w_dec + linear
        error = y - recon
        mse = error.pow(2).sum(-1).mean()
        fired = (acts > 0).any(0)
        self.idle += x.shape[0]
        self.idle[fired] = 0
        dead = self.idle > dead_after
        aux = torch.zeros((), device=x.device)
        if dead.any():  # dead latents learn to explain what the live ones miss
            k_aux = min(self.k_aux, int(dead.sum()))
            masked = pre.masked_fill(~dead, -torch.inf).topk(k_aux, dim=-1)
            # No ReLU: a dead latent's pre-activation is usually negative, and a ReLU
            # would zero the very gradient that is meant to revive it.
            aux_acts = torch.zeros_like(pre).scatter(-1, masked.indices, masked.values)
            target = error.detach()
            aux_err = (aux_acts @ self.w_dec - target).pow(2).sum(-1).mean()
            aux = aux_err / target.pow(2).sum(-1).mean().clamp(min=1e-8)
        variance = (y - y.mean(0)).pow(2).sum(-1).mean()
        return {
            "loss": mse / variance + aux_coef * aux,
            "fvu": float((mse / variance).detach()),
            "dead": float(dead.float().mean()),
        }

    @torch.no_grad()
    def renorm(self) -> None:
        self.w_dec /= self.w_dec.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def threshold_for(self, x: torch.Tensor) -> float:
        """The single JumpReLU threshold that keeps the k·T largest pre-activations of these
        T tokens, so the converted L0 is k on average."""
        pre = self.pre(x).relu().flatten().cpu()  # hundreds of millions of values
        keep = self.k * x.shape[0]
        return float(pre.kthvalue(pre.numel() - keep + 1).values)

    @torch.no_grad()
    def evaluate(self, x: torch.Tensor, y: torch.Tensor, threshold: float | None) -> dict:
        pre = self.pre(x)
        if threshold is None:
            top = pre.topk(self.k, dim=-1)
            acts = torch.zeros_like(pre).scatter(-1, top.indices, top.values.relu())
        else:
            acts = pre * (pre > threshold)
        recon = acts @ self.w_dec + self.b_dec + x @ self.w_skip.T
        variance = (y - y.mean(0)).pow(2).sum(-1).sum()
        latents = (acts > 0).float()
        return {
            "explained_variance": round(1 - float((y - recon).pow(2).sum() / variance), 4),
            "mean_l0": round(float(latents.sum(-1).mean()), 2),
            "fired_share": round(float(latents.amax(0).mean()), 4),
        }

    @torch.no_grad()
    def export(self, threshold: float) -> dict[str, torch.Tensor]:
        """Weights on the model's own scale: x is the raw MLP input, the output the raw MLP
        output (W_skip ``[d_out, d_in]``, applied as ``x @ W_skip.T``)."""
        s_in, s_out = float(self.scale_in), float(self.scale_out)
        n = self.b_enc.shape[0]
        return {
            "W_enc": (self.w_enc * s_in).contiguous(),
            "b_enc": self.b_enc.clone(),
            "threshold": torch.full((n,), threshold),
            "W_dec": (self.w_dec / s_out).contiguous(),
            "b_dec": self.b_dec / s_out,
            "W_skip": (self.w_skip * (s_in / s_out)).contiguous(),
        }


@click.command()
@click.option("--model", "model_name", default="Qwen/Qwen2.5-0.5B", show_default=True)
@click.option("--revision", default=None, help="Model commit (default: the latest).")
@click.option("--layers", required=True, help="Comma-separated layers, or 'all'.")
@click.option("--tokens", type=int, default=None, help="Training tokens (default: all).")
@click.option("--width", type=int, default=16384, show_default=True)
@click.option("--k", type=int, default=32, show_default=True)
@click.option("--k-aux", type=int, default=256, show_default=True)
@click.option("--aux-coef", type=float, default=1 / 32, show_default=True)
@click.option("--dead-after", type=int, default=2_000_000, show_default=True)
@click.option("--seq-len", type=int, default=1024, show_default=True)
@click.option("--forward-rows", type=int, default=16, show_default=True)
@click.option("--batch", type=int, default=4096, show_default=True)
@click.option("--lr", type=float, default=2e-4, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--scan-corpus", type=click.Path(exists=True, path_type=Path), default="corpus.jsonl")
@click.option("--out", type=click.Path(path_type=Path), required=True)
def main(
    model_name, revision, layers, tokens, width, k, k_aux, aux_coef, dead_after, seq_len,
    forward_rows, batch, lr, seed, device, scan_corpus, out,
):  # fmt: skip
    from openinterp.cli import _load_runner

    torch.manual_seed(seed)
    out.mkdir(parents=True, exist_ok=True)
    runner = _load_runner(model_name, device, "float32", revision)
    runner.model.requires_grad_(False)
    layers = range(runner.n_layers) if layers == "all" else map(int, layers.split(","))
    layers = list(layers)
    train_texts, valid_texts, excluded = training_texts(scan_corpus)
    click.echo(f"{len(train_texts)} training documents; {excluded} scan documents left out")
    rows = sequences(runner.tokenizer, train_texts, seq_len, seed)
    valid_rows = sequences(runner.tokenizer, valid_texts, seq_len, seed)
    total = sum(len(r) - 1 for r in rows)
    budget = min(tokens or total, total)
    click.echo(f"{total:,} training tokens available; training on {budget:,}")

    d = runner.model.config.hidden_size
    gen = torch.Generator().manual_seed(seed)
    tcs = {layer: TopKTranscoder(d, d, width, k, k_aux, gen).to(runner.device) for layer in layers}
    opts = {layer: torch.optim.Adam(tc.parameters(), lr=lr) for layer, tc in tcs.items()}
    steps = budget // batch
    warmup, decay_from = min(500, steps // 10), int(steps * 0.8)

    def lr_at(step: int) -> float:
        if step < warmup:
            return lr * (step + 1) / warmup
        if step >= decay_from:
            return lr * max(0.0, (steps - step) / max(1, steps - decay_from))
        return lr

    log = (out / "train.jsonl").open("w")
    step, seen, start = 0, 0, time.time()
    for i in range(0, len(rows), forward_rows):
        if seen >= budget:
            break
        io = mlp_io(runner, rows[i : i + forward_rows], layers)
        n_tokens = next(iter(io.values()))[0].shape[0]
        order = torch.randperm(n_tokens, device=runner.device)
        for layer in layers:
            x, y = io[layer]
            tc = tcs[layer]
            if float(tc.scale_in) == 0:  # average norm sqrt(d), from the first batch
                tc.scale_in.fill_(d**0.5 / float(x.norm(dim=-1).mean()))
                tc.scale_out.fill_(d**0.5 / float(y.norm(dim=-1).mean()))
            x, y = x[order] * tc.scale_in, y[order] * tc.scale_out
            for b, s in enumerate(range(0, n_tokens - batch + 1, batch)):
                for group in opts[layer].param_groups:
                    group["lr"] = lr_at(step + b)
                result = tc.loss(x[s : s + batch], y[s : s + batch], dead_after, aux_coef)
                opts[layer].zero_grad(set_to_none=True)
                result["loss"].backward()
                opts[layer].step()
                tc.renorm()
            log.write(json.dumps({"step": step, "layer": layer, "tokens": seen,
                                  "fvu": round(result["fvu"], 4),
                                  "dead": round(result["dead"], 4)}) + "\n")  # fmt: skip
        step += n_tokens // batch
        seen += n_tokens
        log.flush()
        if (i // forward_rows) % 20 == 0:
            rate = seen / (time.time() - start)
            eta = (budget - seen) / max(rate, 1) / 3600
            click.echo(f"{seen:,}/{budget:,} tokens, {rate:,.0f} tok/s, ~{eta:.1f} h left")

    # Held-out text: choose each threshold on half of it, evaluate on the other half.
    io = mlp_io(runner, valid_rows[: 2 * forward_rows], layers)
    for layer in layers:
        tc = tcs[layer]
        x, y = io[layer]
        x, y = x * tc.scale_in, y * tc.scale_out
        half = x.shape[0] // 2
        threshold = tc.threshold_for(x[:half])
        stats = {
            "model": model_name,
            "revision": revision,
            "layer": layer,
            "input_site": "mlp_in",
            "data": {
                "dataset": PILE[0],
                "revision": PILE[1],
                "excluded_scan_docs": excluded,
                "validation_docs": VALIDATION_DOCS,
            },  # fmt: skip
            "tokens": seen,
            "width": width,
            "k": k,
            "k_aux": k_aux,
            "aux_coef": aux_coef,
            "dead_after": dead_after,
            "lr": lr,
            "batch": batch,
            "seq_len": seq_len,
            "seed": seed,
            "threshold": threshold,
            "held_out_tokens": x.shape[0] - half,
            "topk": tc.evaluate(x[half:], y[half:], None),
            "jumprelu": tc.evaluate(x[half:], y[half:], threshold),
        }
        from safetensors.torch import save_file

        save_file({k_: v.cpu() for k_, v in tc.export(threshold).items()},
                  out / f"layer_{layer}.safetensors")  # fmt: skip
        (out / f"layer_{layer}.json").write_text(json.dumps(stats, indent=2) + "\n")
        click.echo(f"L{layer}: top-k {stats['topk']}  JumpReLU {stats['jumprelu']}")
    click.echo(f"Wrote {out}")


if __name__ == "__main__":
    main()
