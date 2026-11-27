"""A fitting harness for the phase 4a arms A–F of the copy-family fix (#7006, #5220 §3).

Each arm changes the base model in its own way, so that it gives the right unit on the
copy family's train prompts (``docs/analyses/copy-family.jsonl``, #5232):

- **A** trains a low-rank change of the copy heads' OV, a deliberate negative control.
- **B** trains a low-rank change of the MLP outputs of the fact layers.
- **C** trains the same change on every MLP.
- **D** trains it on random heads, with as many parameters as B.
- **E** puts a few-shot prefix before each prompt, and changes no weight.
- **F** solves a closed-form edit at B's middle layer (MEMIT's least squares), and its
  AlphaEdit variant, which leaves the retain prompts' keys unchanged.

Each fit of A–D and F is written as a draft patch (``docs/patches.md``, #5229), checked at
zero strength, and given the interference report of #5230. Each fit also writes one row
to ``fits.jsonl``, with its learning measures and the report's summary. Sub-issue #7007
runs the arms on the real prompts. No fit writes to the registry.

    uv run python scripts/fix_arms.py --device cuda --out runs/fix-arms
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fix_closed_form import fit_closed_form  # noqa: E402
from fix_family import (  # noqa: E402
    FAMILY,
    MODEL,
    SEED,
    SHOTS,
    Family,
    Fit,
    Item,
    Layout,
    checked,
    draft,
    evaluate,
    few_shot,
    last_logits,
    read_family,
    token,
)  # fmt: skip

LAMBDAS = (0.0, 0.1, 1.0, 10.0)  # the weights of the retain KL term (#5220 §3)
RANKS = (1, 4)  # the ranks of each trained change
BETA = 1.0  # the weight of the unlikelihood of the copied unit
STEPS, BATCH, LR = 200, 16, 1e-2  # optimiser steps, prompts per step, learning rate
INIT = 0.01  # the scale of the random start of each A factor. Each B factor starts at zero.


@dataclass(frozen=True)
class Site:
    """One weight that a trained arm changes, and the head it is limited to, if any."""

    target: str
    head: int | None = None


def shape(graph, site: Site) -> tuple[int, int]:
    """The ``[out, in]`` shape of the weight at ``site``.

    Args:
        graph: The model's canonical graph.
        site: The weight.
    """
    out, inp = graph.weight(site.target).shape
    return int(out), int(inp)


def site_params(graph, site: Site, rank: int) -> int:
    """The number of trained parameters of a rank-``rank`` change at ``site``.

    A change limited to one head trains only the head's columns of its A factor.

    Args:
        graph: The model's canonical graph.
        site: The weight.
        rank: The rank of the change.
    """
    out, inp = shape(graph, site)
    width = inp // graph.heads if site.head is not None else inp
    return rank * (out + width)


def sites(arm: str, graph, layout: Layout, rank: int = 1, seed: int = SEED) -> list[Site]:
    """The weights that arm ``arm`` (A to D) changes.

    D draws heads at random, outside A's heads, until it has about as many trained
    parameters as B at the same rank.

    Args:
        arm: ``A``, ``B``, ``C`` or ``D``.
        graph: The model's canonical graph.
        layout: Where the arms act.
        rank: The rank of each change, which sets D's parameter count.
        seed: The seed of D's draw.

    Raises:
        ValueError: ``arm`` is not a trained arm.
    """
    if arm == "A":
        return [Site(f"blocks.{layer}.attn.W_O", head) for layer, head in layout.copy_heads]
    if arm == "B":
        return [Site(f"blocks.{layer}.mlp.W_out") for layer in layout.fact_layers]
    if arm == "C":
        return [Site(f"blocks.{layer}.mlp.W_out") for layer in range(len(graph.layers))]
    if arm == "D":
        budget = sum(site_params(graph, s, rank) for s in sites("B", graph, layout))
        pool = [Site(f"blocks.{layer}.attn.W_O", head)
                for layer in range(len(graph.layers)) for head in range(graph.heads)
                if (layer, head) not in layout.copy_heads]  # fmt: skip
        random.Random(seed).shuffle(pool)
        per = site_params(graph, pool[0], rank)
        return pool[: max(1, min(len(pool), round(budget / per)))]
    raise ValueError(f"arm {arm!r} is not a trained arm")


def head_mask(graph, site: Site) -> torch.Tensor:
    """A ``[1, in]`` mask of the columns that a change at ``site`` may use.

    Args:
        graph: The model's canonical graph.
        site: The weight. With a head, only that head's slice of ``hook_z`` is open.
    """
    _, inp = shape(graph, site)
    mask = torch.ones(1, inp)
    if site.head is not None:
        width = inp // graph.heads
        mask.zero_()
        mask[:, site.head * width : (site.head + 1) * width] = 1.0
    return mask


class Adapters:
    """Trainable low-rank changes at a list of sites, added through forward hooks.

    Each change is ``B @ (A * mask)``. B starts at zero, so the model is unchanged at
    the start. The model's own weights are never written.

    Args:
        graph: The model's canonical graph.
        chosen: The sites to change.
        rank: The rank of each change.
        seed: The seed of the A factors' random start.
    """

    def __init__(self, graph, chosen: list[Site], rank: int, seed: int = SEED):
        generator = torch.Generator().manual_seed(seed)
        self.graph, self.sites, self.handles = graph, chosen, []
        self.factors: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for site in chosen:
            out, inp = shape(graph, site)
            a = (torch.randn(rank, inp, generator=generator) * INIT).requires_grad_()
            b = torch.zeros(out, rank, requires_grad=True)
            self.factors.append((a, b, head_mask(graph, site)))

    def parameters(self) -> list[torch.Tensor]:
        """The trained tensors."""
        return [t for a, b, _ in self.factors for t in (a, b)]

    def delta(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The A and B factors of change ``i``, with A masked.

        Args:
            i: The change's index in :attr:`sites`.
        """
        a, b, mask = self.factors[i]
        return (a * mask.to(a.device)).detach(), b.detach()

    def __enter__(self):
        """Hook each change into the model, and return the adapters."""
        from openinterpretability.patches.graph import parameter_hook

        for site, (a, b, mask) in zip(self.sites, self.factors, strict=True):

            def add(x, a=a, b=b, mask=mask):
                """The change's term ``x @ ΔWᵀ``, with ΔW = B @ (A * mask).

                Args:
                    x: The input of the changed layer.
                    a: The change's A factor.
                    b: The change's B factor.
                    mask: The columns that A may use.
                """
                return x @ (b @ (a * mask.to(a.device))).to(x.device).t()

            self.handles += parameter_hook(self.graph, site.target, add)
        return self

    def __exit__(self, *_):
        """Remove every hook, so that the model is the base model again.

        Args:
            *_: The exception, if any, which is not handled here.
        """
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def to(self, device) -> Adapters:
        """Move every factor to ``device``, as a leaf that still trains.

        Args:
            device: The model's device.
        """
        self.factors = [(a.detach().to(device).requires_grad_(),
                         b.detach().to(device).requires_grad_(),
                         mask.to(device)) for a, b, mask in self.factors]  # fmt: skip
        return self


def fit_loss(model, tokenizer, items: list[Item], retain: tuple | None, lam: float,
             beta: float = BETA) -> torch.Tensor:  # fmt: skip
    """CE(right unit) + β·unlikelihood(copied unit) on ``items``, + λ·KL(base ‖ fixed) on retain.

    Args:
        model: The model, with the trained changes hooked in.
        tokenizer: Its tokenizer.
        items: The train prompts of this step.
        retain: The retain prompts and the base model's log-probabilities at their last
            positions, or None when λ is 0.
        lam: The weight of the KL term.
        beta: The weight of the unlikelihood term.
    """
    logp = torch.log_softmax(last_logits(model, tokenizer, [i.prompt for i in items]), -1)
    rows = torch.arange(len(items), device=logp.device)
    right = torch.tensor([token(tokenizer, i.right) for i in items], device=logp.device)
    copied = torch.tensor([token(tokenizer, i.copied) for i in items], device=logp.device)
    p_copied = logp[rows, copied].exp().clamp(max=1 - 1e-6)
    loss = -logp[rows, right].mean() - beta * torch.log1p(-p_copied).mean()
    if retain is not None and lam:
        prompts, base = retain
        fixed = torch.log_softmax(last_logits(model, tokenizer, prompts), -1)
        loss = loss + lam * torch.sum(base.exp() * (base - fixed), -1).mean()
    return loss


def train(runner, adapters: Adapters, family: Family, lam: float, steps: int = STEPS,
          seed: int = SEED) -> list[float]:  # fmt: skip
    """Fit ``adapters`` with Adam, and return the loss at each step.

    Args:
        runner: The base model's runner.
        adapters: The changes to fit.
        family: The family. Its train prompts are fitted, and its retain prompts are kept.
        lam: The weight of the retain KL term.
        steps: The optimiser steps.
        seed: The seed of the batches.
    """
    model, tokenizer = runner.model, runner.tokenizer
    retain = None
    if lam and family.retain:
        prompts = [i.prompt for i in family.retain]
        with torch.no_grad():
            retain = (prompts, torch.log_softmax(last_logits(model, tokenizer, prompts), -1))
    opt = torch.optim.Adam(adapters.parameters(), lr=LR)
    rng, losses = random.Random(seed), []
    with adapters:
        for _ in range(steps):
            batch = rng.sample(family.train, min(BATCH, len(family.train)))
            loss = fit_loss(model, tokenizer, batch, retain, lam)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
    return losses


def fit_trained(runner, arm: str, family: Family, layout: Layout, lam: float, rank: int,
                steps: int = STEPS) -> Fit:  # fmt: skip
    """Fit arm ``arm`` (A to D) at one λ and rank.

    Args:
        runner: The base model's runner.
        arm: ``A``, ``B``, ``C`` or ``D``.
        family: The family.
        layout: Where the arms act.
        lam: The weight of the retain KL term.
        rank: The rank of each change.
        steps: The optimiser steps.
    """
    from openinterpretability.patches.graph import Graph

    graph = Graph(runner.model)
    device = next(runner.model.parameters()).device
    adapters = Adapters(graph, sites(arm, graph, layout, rank), rank).to(device)
    losses = train(runner, adapters, family, lam, steps)
    ops = [(s.target, *adapters.delta(i)) for i, s in enumerate(adapters.sites)]
    return Fit(arm, {"lambda": lam, "rank": rank}, ops, losses)


def report(model_name: str, path: Path, out: Path, corpus: Path, tokens: int, wiki_dir: Path,
           device: str) -> dict:  # fmt: skip
    """Run ``open-interpretability interference --patch`` on the draft, and return its summary.

    Args:
        model_name: The base model.
        path: The draft file.
        out: Where the report's rows go.
        corpus: The corpus of the report's sample.
        tokens: Corpus tokens in the sample.
        wiki_dir: The registry that the report reads.
        device: The model's device.
    """
    from openinterpretability.cli import cli

    evidence = out.with_suffix(".summary.json")
    cli.main(["interference", "--model", model_name, "--patch", str(path), "--out", str(out),
              "--evidence-out", str(evidence), "--corpus", str(corpus), "--tokens", str(tokens),
              "--wiki-dir", str(wiki_dir), "--device", device],
             standalone_mode=False)  # fmt: skip
    return json.loads(evidence.read_text(encoding="utf-8"))


@dataclass
class Run:
    """What every fit of one run shares."""

    runner: object
    model_name: str
    family: Family
    out: Path
    corpus: Path
    tokens: int
    wiki_dir: Path
    device: str
    with_report: bool = True


def record(run: Run, fit: Fit) -> dict:
    """Write ``fit``'s draft, check it, measure it and report its interference.

    Args:
        run: The run.
        fit: The fit.

    Returns:
        The fit's row of ``fits.jsonl``.
    """
    from openinterpretability.patches.apply import applied

    model = run.runner.model
    revision = getattr(model.config, "_commit_hash", None) or "main"
    path = run.out / "drafts" / f"{fit.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = draft(fit, f"{run.model_name}@{revision}", run.family)
    path.write_text(json.dumps(body, indent=1) + "\n",
                    encoding="utf-8")  # fmt: skip
    _, ops = checked(path, model)
    with applied(model, ops):
        found = evaluate(model, run.runner.tokenizer, run.family)
    row = {"table": "fit", "arm": fit.arm, **fit.settings, "draft": str(path),
           "params": sum(a.numel() + b.numel() for _, a, b in fit.ops),
           "targets": [t for t, _, _ in fit.ops], **found,
           "loss": [round(fit.losses[0], 4), round(fit.losses[-1], 4)] if fit.losses
           else None}  # fmt: skip
    if run.with_report:
        reports = run.out / "reports" / f"{fit.name}.jsonl"
        reports.parent.mkdir(parents=True, exist_ok=True)
        row["interference"] = report(run.model_name, path, reports, run.corpus, run.tokens,
                                     run.wiki_dir, run.device)  # fmt: skip
    return row


def prompt_row(run: Run, shots: int = SHOTS) -> dict:
    """Arm E's row: the learning measures with a few-shot prefix. It has no patch.

    Args:
        run: The run.
        shots: The examples in the prefix.
    """
    prefix = few_shot(run.family, shots)
    found = evaluate(run.runner.model, run.runner.tokenizer, run.family, prefix)
    return {"table": "fit", "arm": "E", "shots": shots, "prefix": prefix, "draft": None,
            "params": 0, **found}  # fmt: skip


def fits(run: Run, arms: str, layout: Layout, batches: list[torch.Tensor],
         lambdas=LAMBDAS, ranks=RANKS, steps: int = STEPS):  # fmt: skip
    """Each fit of ``arms``, as a row, in order: A–D over λ and rank, E, then F.

    F's ranks are ``ranks`` and the edit as solved, which is not cut to a low rank.

    Args:
        run: The run.
        arms: The arms to fit, such as ``ABCDEF``.
        layout: Where the arms act.
        batches: Corpus windows, for arm F's second moment.
        lambdas: The λ sweep.
        ranks: The rank sweep.
        steps: The optimiser steps of each trained fit.
    """
    for arm in (a for a in arms if a in "ABCD"):
        for lam in lambdas:
            for rank in ranks:
                fit = fit_trained(run.runner, arm, run.family, layout, lam, rank, steps)
                yield record(run, fit)
    if "E" in arms:
        yield prompt_row(run)
    if "F" in arms:
        for variant in ("memit", "alphaedit"):
            for rank in (*ranks, None):
                fit = fit_closed_form(run.runner, run.family, layout, batches, rank, variant)
                yield record(run, fit)


@click.command()
@click.option("--model", "model_name", default=MODEL, show_default=True)
@click.option("--family", "family_file", default=FAMILY, show_default=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path))  # fmt: skip
@click.option("--out", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--arms", default="ABCDEF", show_default=True)
@click.option("--corpus", default="corpus.jsonl", type=click.Path(path_type=Path))
@click.option("--tokens", default=20_000, show_default=True)
@click.option("--wiki-dir", default="wiki", type=click.Path(path_type=Path))
@click.option("--no-report", is_flag=True, help="Skip the interference report of each fit.")
@click.option("--device", default="cpu", show_default=True)
def main(model_name, family_file, out, arms, corpus, tokens, wiki_dir, no_report,
         device):  # pragma: no cover  # fmt: skip
    """Fit the arms on the family's train prompts, and write ``fits.jsonl`` under OUT.

    Args:
        model_name: The base model.
        family_file: The family's JSONL file.
        out: The run's folder.
        arms: The arms to fit.
        corpus: The corpus of the reports and of arm F's second moment.
        tokens: Corpus tokens in each sample.
        wiki_dir: The registry that the reports read.
        no_report: Skip the reports.
        device: The model's device.
    """
    from openinterpretability.cli import _load_runner
    from openinterpretability.interference.keep import sample_windows

    torch.manual_seed(SEED)
    runner = _load_runner(model_name, device, "float32")
    for p in runner.model.parameters():
        p.requires_grad_(False)
    run = Run(runner, model_name, read_family(family_file), out, corpus, tokens, wiki_dir,
              device, not no_report)  # fmt: skip
    batches = sample_windows(runner.tokenizer, tokens, corpus)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "fits.jsonl", "w", encoding="utf-8") as f:
        for row in fits(run, arms, Layout(), batches):
            f.write(json.dumps(row) + "\n")
            f.flush()
            click.echo(json.dumps({k: row[k] for k in ("arm", "heldout", "retain") if k in row}))


if __name__ == "__main__":
    main()
