"""How the last token's state changes through the layers, for each prompt (issue #4917).

The maintainer asked on 2026-11-18 to see the embedding change over the layers in 3D, as a
frequency waterfall does. For each prompt and each layer, at the last position:

- **The logit lens:** the residual stream through the final norm and the unembedding, as
  probabilities over one shared set of candidate tokens. The set is the final top
  :data:`TOP` of every prompt, plus the claim's answer, so the prompts can be compared. The
  last layer is the model's real output.
- **The spectrum:** the residual stream at every token, scaled to unit RMS, as the model's
  norms see it. It is projected onto :data:`COMPONENTS` principal directions, fit on every
  prompt's tokens and layers together, so one direction means the same thing everywhere.
  The directions run from the most variance to the least, as frequencies run in a
  spectrum. The page reads it token by token, as the bracket paper's figure does.

The page's 3D explorer draws both as waterfalls, layer after layer. They are reported, not
gated.
"""

from __future__ import annotations

import torch

TOP = 1  # each prompt's final top tokens that join the candidate set: more crowds the axis
COMPONENTS = 16  # principal directions in the spectrum: a record holds one per token and layer


@torch.no_grad()
def states(runner, prompt: str, hooks=None) -> torch.Tensor:
    """The residual stream at every position after the embedding and each layer.

    The result is ``[layers + 1, tokens, d_model]``.

    ``hooks`` is a pair from :func:`openinterpretability.explain.mean_ablation`, to switch
    units off.
    """
    neurons, heads = hooks or ({}, {})
    with runner.hooks(neurons, None, heads):
        out = runner.model(**runner.encode(prompt), output_hidden_states=True)
    return torch.stack([h[0].float().cpu() for h in out.hidden_states])


@torch.no_grad()
def lens(runner, hidden: torch.Tensor) -> torch.Tensor:
    """Each layer's probabilities over the vocabulary, through the final norm.

    HF applies the final norm to the last hidden state already, so it is not applied again.
    """
    hidden = hidden[:, -1]  # the lens reads the last position, where the answer is made
    norm, unembed = runner.final_norm(), runner.model.get_output_embeddings()
    weight = next(unembed.parameters())
    rows = []
    for i, h in enumerate(hidden):
        x = h.to(weight.device, weight.dtype)  # the model's own dtype, for its norm
        if norm is not None and i < len(hidden) - 1:
            x = norm(x)
        rows.append(torch.softmax(unembed(x).float(), -1).cpu())
    return torch.stack(rows)


def candidates(probs: dict[str, torch.Tensor], answer: int) -> list[int]:
    """The answer and every prompt's final top :data:`TOP` tokens, in first-seen order."""
    out = [answer]
    for p in probs.values():
        out += [int(t) for t in p[-1].topk(TOP).indices if int(t) not in out]
    return out


def basis(hidden: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """The mean and the principal directions of every prompt's scaled layers."""
    rows = torch.cat([_scaled(h).reshape(-1, h.shape[-1]) for h in hidden.values()])
    mean = rows.mean(0)
    _, _, v = torch.linalg.svd(rows - mean, full_matrices=False)
    return mean, v[: min(COMPONENTS, v.shape[0])]


def _scaled(h: torch.Tensor) -> torch.Tensor:
    return h / h.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-9)


def measures(runner, means, prompts: list[str], answer: int, route: list,
             labels: list[str]) -> dict[str, dict]:  # fmt: skip
    """Each prompt's layers on the shared axes, clean and with units switched off.

    Every prompt gets ``clean`` and ``route_off``: the whole route mean-ablated. The first
    prompt, the original, also gets ``unit_off``: each route unit mean-ablated on its own,
    keyed by its label. Each is a ``LayerView`` dict: a lens row and a spectrum row per
    layer. The axes come from the clean runs only, so a switched-off run reads on them.
    """
    from openinterpretability.explain import mean_ablation

    hidden = {p: states(runner, p) for p in prompts}
    probs = {p: lens(runner, h) for p, h in hidden.items()}
    tokens = candidates(probs, answer)
    names = [runner.tokenizer.decode([t]) for t in tokens]
    mean, directions = basis(hidden)

    def view(h: torch.Tensor, p: torch.Tensor | None = None) -> dict:
        p = lens(runner, h) if p is None else p
        table = [{n: round(float(row[t]), 5) for n, t in zip(names, tokens, strict=True)}
                 for row in p]  # fmt: skip
        spectrum = ((_scaled(h) - mean) @ directions.T).tolist()  # [layer][token][direction]
        return {
            "lens": table,
            "spectrum": [[[round(x, 3) for x in tok] for tok in layer] for layer in spectrum],
        }

    off = mean_ablation(route, means, runner)
    out = {p: {"clean": view(h, probs[p]), "route_off": view(states(runner, p, off))}
           for p, h in hidden.items()}  # fmt: skip
    first = prompts[0]
    out[first]["unit_off"] = {
        label: view(states(runner, first, mean_ablation([unit], means, runner)))
        for unit, label in zip(route, labels, strict=True)
    }
    return out
