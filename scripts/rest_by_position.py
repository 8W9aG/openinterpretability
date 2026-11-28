"""Read where the MLP neurons rest at each position of long windows (#7202).

A layer's rest is the median over its neurons of each neuron's median pre-activation h. The
paper papers/why-neurons-rest/README.md uses the same measure. This script gives the rest for
each bucket of positions in held-out windows of 2,048 tokens. The windows join the windows
of the corpus sample in order. An end-of-text token follows each document, as in the
training text. No small model trains on the corpus sample, so all of it is held out.

    uv run python scripts/rest_by_position.py small --condition pythia_size_batch_32x_window \\
        --seed 0 --save-dir ckpt_window --out rows.jsonl
    uv run python scripts/rest_by_position.py pythia --step 2000 --out rows.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).parent))

from default_on_poles import load_split, write_row
from pretrained_push import PYTHIA, pick_steps
from resting_conditions import CONDITIONS, PYTHIA_WINDOW
from train_resting import TOKENIZER, build_model, checkpoint_path

from openinterpretability.cli import resolve_device
from openinterpretability.corpus import DEFAULT_SAMPLE_SEED, DEFAULT_SAMPLE_TOKENS

# The first position of each bucket. A bucket ends where the next starts, and the last one
# ends at the window's end.
BUCKETS = (0, 1, 16, 128, 512)
BATCH = 4  # long windows in each forward pass


def long_windows(ids: torch.Tensor, docs: torch.Tensor, eos: int, length: int) -> torch.Tensor:
    """Join the windows of each document, with ``eos`` after each document, and cut them again.

    Args:
        ids: The short windows, ``[windows, tokens]``, in the order of their documents.
        docs: The document of each window.
        eos: The end-of-text token.
        length: The tokens in each long window.

    Returns:
        The long windows, ``[windows, length]``. The tokens after the last whole window drop.

    Examples:
        >>> ids = torch.tensor([[1, 2], [3, 4], [5, 6]])
        >>> long_windows(ids, torch.tensor([0, 0, 1]), 9, 4).tolist()
        [[1, 2, 3, 4], [9, 5, 6, 9]]
    """
    stream: list[int] = []
    for i, window in enumerate(ids.tolist()):
        stream.extend(window)
        if i + 1 == len(ids) or docs[i + 1] != docs[i]:
            stream.append(eos)
    whole = len(stream) // length * length
    return torch.tensor(stream[:whole]).view(-1, length)


def bucket_names(length: int) -> list[str]:
    """The name of each bucket of positions in a window of ``length`` tokens.

    Args:
        length: The tokens in a window.

    Returns:
        One name a bucket, as "first-last", for the buckets that start inside the window.

    Examples:
        >>> bucket_names(128)
        ['0-0', '1-15', '16-127']
    """
    starts = [s for s in BUCKETS if s < length]
    ends = [*starts[1:], length]
    return [f"{s}-{e - 1}" for s, e in zip(starts, ends, strict=True)]


def buckets(length: int) -> list[tuple[str, int, int]]:
    """Each bucket of a window of ``length`` tokens: its name, first position and end.

    Args:
        length: The tokens in a window.

    Returns:
        ``(name, start, end)`` for each bucket, with ``end`` past its last position.

    Examples:
        >>> buckets(20)
        [('0-0', 0, 1), ('1-15', 1, 16), ('16-19', 16, 20)]
    """
    return [(n, int(n.split("-")[0]), int(n.split("-")[1]) + 1) for n in bucket_names(length)]


def bucket_stats(x: torch.Tensor, h: torch.Tensor) -> dict[str, dict[str, float]]:
    """One layer's rest, spread and mean input share in each bucket of positions.

    The rest is the median over neurons of each neuron's median h. The spread is the median
    over neurons of each neuron's SD of h. The mean input share is the norm of the inputs'
    mean over the norm of an input, on average, as in train_resting's ``_layer_stats``. A
    median of an even count is the lower middle value, as torch.median gives it.

    Args:
        x: The MLP's inputs, ``[windows, positions, width]``.
        h: Its pre-activations, ``[windows, positions, neurons]``.

    Returns:
        ``{"rest": …, "h_sd": …, "mean_input_share": …}``, each by bucket name.

    Examples:
        >>> h = torch.arange(8.0).view(1, 4, 2)  # positions 0-3, two neurons
        >>> x = torch.tensor([[[1.0, 0.0], [1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]])
        >>> out = bucket_stats(x, h)
        >>> out["rest"], out["mean_input_share"]["0-0"]
        ({'0-0': 0.0, '1-3': 4.0}, 1.0)
    """
    out: dict[str, dict[str, float]] = {"rest": {}, "h_sd": {}, "mean_input_share": {}}
    for name, start, end in buckets(h.shape[1]):
        part = h[:, start:end].reshape(-1, h.shape[-1])
        inputs = x[:, start:end].reshape(-1, x.shape[-1])
        out["rest"][name] = round(part.median(0).values.median().item(), 4)
        sd = part.std(0).median().item() if len(part) > 1 else 0.0
        out["h_sd"][name] = round(sd, 4)
        share = inputs.mean(0).norm() / inputs.norm(dim=1).mean()
        out["mean_input_share"][name] = round(share.item(), 4)
    return out


def energy_shares(x: torch.Tensor) -> dict[str, float]:
    """The share of the inputs' energy in their mean, and in each position's own mean.

    The energy is the mean squared norm of an input. The global share is the squared norm of
    the mean of every input. The position share is the mean squared norm of each position's
    mean over the windows. So it holds what the inputs at one position have in common, such
    as a position's vector. It includes the global share. Both shares divide by the energy.

    Args:
        x: One MLP's inputs, ``[windows, positions, width]``.

    Returns:
        ``{"global_share": …, "position_share": …}``.

    Examples:
        >>> x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
        >>> energy_shares(x)  # each position the same in both windows
        {'global_share': 0.5, 'position_share': 1.0}
    """
    energy = (x**2).sum(-1).mean()
    mean = x.reshape(-1, x.shape[-1]).mean(0)
    by_position = (x.mean(0) ** 2).sum(-1).mean()
    return {"global_share": round((mean @ mean / energy).item(), 4),
            "position_share": round((by_position / energy).item(), 4)}  # fmt: skip


@torch.no_grad()
def read_rests(model, modules: list, windows: torch.Tensor) -> list[dict]:
    """Run ``windows`` through ``model`` and give each layer's ``bucket_stats``.

    Args:
        model: A causal language model.
        modules: The layer whose output is h, one a block, such as GPT-2's ``c_fc``.
        windows: The windows, ``[windows, tokens]``.

    Returns:
        For each layer, the statistics of each bucket, and the ``energy_shares`` of its
        inputs after the first position.
    """
    seen: list[list[torch.Tensor]] = [[] for _ in modules]
    taken: list[list[torch.Tensor]] = [[] for _ in modules]

    def keep(i):
        """A hook that keeps one layer's inputs and pre-activations.

        Args:
            i: The layer's index in ``modules``.

        Returns:
            The hook.
        """

        def hook(_module, args, out):
            """Keep the input and the output of the layer.

            Args:
                _module: The layer.
                args: Its inputs. The first is the MLP's input.
                out: Its output, the pre-activations.
            """
            taken[i].append(args[0].cpu())
            seen[i].append(out.cpu())

        return hook

    handles = [m.register_forward_hook(keep(i)) for i, m in enumerate(modules)]
    device = next(model.parameters()).device
    try:
        for batch in windows.split(BATCH):
            model(batch.to(device))
    finally:
        for handle in handles:
            handle.remove()
    out = []
    for x, h in zip(taken, seen, strict=True):
        inputs = torch.cat(x).float()
        out.append({**bucket_stats(inputs, torch.cat(h).float()),
                    **energy_shares(inputs[:, 1:])})  # fmt: skip
    return out


def held_out(tokenizer, corpus: Path, length: int) -> torch.Tensor:
    """The corpus sample's windows, cut to ``length`` tokens with ``long_windows``.

    Args:
        tokenizer: The model's tokenizer.
        corpus: The corpus file.
        length: The tokens in each window.

    Returns:
        The windows.
    """
    split = load_split(tokenizer, corpus, DEFAULT_SAMPLE_TOKENS, DEFAULT_SAMPLE_SEED)
    order = torch.argsort(split.docs, stable=True)  # each document's windows together
    return long_windows(split.ids[order], split.docs[order], tokenizer.eos_token_id, length)


@click.group()
def cli():
    """Read the rest by position in small models' checkpoints and in Pythia-70m."""


@cli.command()
@click.option("--condition", "name", type=click.Choice(sorted(CONDITIONS)), required=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--save-dir", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.option("--step", "only", multiple=True, type=int, help="Default: every saved step.")
@click.option("--length", type=int, default=PYTHIA_WINDOW, show_default=True)
@click.option("--corpus", type=click.Path(path_type=Path), default=Path("corpus.jsonl"))
@click.option("--device", default="auto", show_default=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
def small(name, seed, save_dir, only, length, corpus, device, out):  # pragma: no cover - GPU
    """Append a row for each saved checkpoint of one small run to ``--out``.

    Args:
        name: The run's condition.
        seed: The run's seed.
        save_dir: The folder of its checkpoints.
        only: The steps to read.
        length: The tokens in each window. It can't pass the model's positions.
        corpus: The corpus file.
        device: The device.
        out: The JSONL file.
    """
    from transformers import AutoTokenizer

    condition = CONDITIONS[name]
    if length > condition.window:
        raise click.UsageError(f"{name} has {condition.window} positions, not {length}")
    windows = held_out(AutoTokenizer.from_pretrained(TOKENIZER), corpus, length)
    files = save_dir.glob(checkpoint_path(Path(), name, seed, "*").name)
    device = resolve_device(device)
    for step in pick_steps(sorted(int(f.stem.rsplit("step", 1)[1]) for f in files), only):
        state = torch.load(checkpoint_path(save_dir, name, seed, step), map_location="cpu")
        model = build_model(condition, state["transformer.wte.weight"].shape[0], seed)
        model.load_state_dict(state)
        model = model.to(device).eval()
        layers = read_rests(model, [b.mlp.c_fc for b in model.transformer.h], windows)
        write_row(out, {"part": "rest_by_position", "condition": name, "seed": seed,
                        "step": step, "length": length, "windows": len(windows),
                        "layers": layers})  # fmt: skip
        click.echo(f"{name} seed {seed} step {step}: L1 {layers[1]['rest']}", err=True)


@cli.command()
@click.option("--step", "steps", multiple=True, type=int, required=True)
@click.option("--length", type=int, default=PYTHIA_WINDOW, show_default=True)
@click.option("--corpus", type=click.Path(path_type=Path), default=Path("corpus.jsonl"))
@click.option("--device", default="auto", show_default=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
def pythia(steps, length, corpus, device, out):  # pragma: no cover - downloads the model
    """Append a row for each of Pythia-70m's checkpoints at ``--step`` to ``--out``.

    Args:
        steps: The checkpoints.
        length: The tokens in each window.
        corpus: The corpus file.
        device: The device.
        out: The JSONL file.
    """
    from transformers import AutoTokenizer, GPTNeoXForCausalLM

    windows = held_out(AutoTokenizer.from_pretrained(PYTHIA), corpus, length)
    device = resolve_device(device)
    for step in steps:
        model = GPTNeoXForCausalLM.from_pretrained(
            PYTHIA, revision=f"step{step}", dtype=torch.float32
        ).to(device).eval()  # fmt: skip
        modules = [layer.mlp.dense_h_to_4h for layer in model.gpt_neox.layers]
        layers = read_rests(model, modules, windows)
        write_row(out, {"part": "rest_by_position", "model": PYTHIA, "step": step,
                        "length": length, "windows": len(windows), "layers": layers})  # fmt: skip
        click.echo(f"pythia step {step}: L1 {layers[1]['rest']}", err=True)


if __name__ == "__main__":
    cli()
