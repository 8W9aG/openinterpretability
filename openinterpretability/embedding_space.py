"""A model's embedding space in 3D, continuous through its layers (issue #4929).

The answer pages place each prompt's tokens in this space, layer by layer, in the style of
Goodfire's 3D embedding clouds. :func:`fit` builds the space once per model, from a sample
of the scan corpus:

1. It takes :data:`POINTS` token positions from windows of :data:`WINDOW` tokens. Each
   window's first token is left out: it is an attention sink.
2. At the embedding and after each layer, each state is scaled to unit RMS, as the model's
   norms see it. The cloud is centred and reduced to its top 3 principal directions.
3. Each layer's 3D cloud is rotated onto the layer before it (orthogonal Procrustes) and
   scaled to a spread of 1. The same tokens then move a little from one layer to the next.
   So the space shifts continuously with depth, and doesn't flip or jump.

With a ``basis`` (:func:`route_directions`, issue #4934), the 3 directions are chosen within
the subspace that an answer's route writes to. So the axes show the space where the causal
heads and neurons act.

:func:`project` puts any state into the space with the same mean, directions and rotation.
The file holds each layer's mean and 3×d projection, and the projected cloud.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch

POINTS = 1500  # corpus token positions in the cloud
WINDOW = 64  # tokens per corpus window
SPACE_DIR = Path("data/embedding-space")  # one file per model, <model slug>.json


def path_for(model: str, root: Path = SPACE_DIR) -> Path:
    """``data/embedding-space/<model slug>.json``."""
    return root / f"{model.replace('/', '--')}.json"


def _scaled(h: torch.Tensor) -> torch.Tensor:
    """Each state at unit RMS, as the model's norms see it."""
    return h / h.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-9)


@torch.no_grad()
def corpus_states(runner, texts, points: int = POINTS, seed: int = 0) -> torch.Tensor:
    """``[layers + 1, points, d_model]``: the states of ``points`` corpus positions."""
    ids = [i for t in texts for i in runner.tokenizer.encode(t)]
    windows = [ids[i : i + WINDOW] for i in range(0, len(ids) - WINDOW + 1, WINDOW)]
    per = max(1, -(-points // max(1, len(windows))))  # positions per window, rounded up
    rng, out = random.Random(seed), []
    for window in windows:
        if sum(o.shape[1] for o in out) >= points:
            break
        hidden = runner.model(**runner.encode(torch.tensor([window])), output_hidden_states=True)
        stack = torch.stack([h[0].float().cpu() for h in hidden.hidden_states])
        keep = sorted(rng.sample(range(1, len(window)), min(per, len(window) - 1)))
        out.append(stack[:, keep])
    return torch.cat(out, dim=1)[:, :points]


def route_directions(runner, means, route, prompt: str) -> torch.Tensor:
    """``[units, d_model]``: the unit direction each route unit writes to the residual stream.

    A neuron pole writes along its down-projection column, negated for a negative pole. A
    head writes ``W_O (z − z̄)``, here averaged over the prompt's tokens past the first.
    """
    from openinterpretability.explain import _captured

    with torch.no_grad():
        _, _, heads = _captured(runner, runner.encode(prompt), bool(means.heads))
    out = []
    for layer, index, pole in route:
        if pole == "head":
            d = runner.head_dim(layer)
            cols = slice(index * d, (index + 1) * d)
            z = heads[layer][0, 1:, cols].float().cpu() - means.heads[layer][cols]
            v = runner.head_output_weights(layer, index).float().cpu() @ z.mean(0)
        else:
            v = runner.neuron_output_direction(layer, index).float().cpu()
            v = -v if pole == "negative" else v
        out.append(v / v.norm().clamp(min=1e-9))
    return torch.stack(out)


def fit(states: torch.Tensor, basis: torch.Tensor | None = None) -> dict:
    """The space of ``states`` (``[layers, points, d]``): means, projections and the cloud.

    ``basis`` (``[k, d]``), when given, limits the 3 directions to the span of its rows.
    """
    q = torch.linalg.qr(basis.T.double())[0].T.float() if basis is not None else None
    means, projections, cloud = [], [], []
    previous = None
    for layer in states:
        x = _scaled(layer)
        mean = x.mean(0)
        if q is None:
            _, _, v = torch.linalg.svd(x - mean, full_matrices=False)
            w = v[:3]  # [3, d]
        else:  # the top 3 directions within the basis's span
            _, _, v = torch.linalg.svd((x - mean) @ q.T, full_matrices=False)
            w = v[:3] @ q
            if w.shape[0] < 3:  # a route of 1 or 2 units: the corpus fills the rest
                rest = (x - mean) - ((x - mean) @ q.T) @ q
                w = torch.cat([w, torch.linalg.svd(rest, full_matrices=False)[2][: 3 - w.shape[0]]])
        coords = (x - mean) @ w.T
        if previous is not None:  # rotate onto the layer before: the cloud moves, not flips
            u, _, vt = torch.linalg.svd(coords.T @ previous)
            rotation = u @ vt
            coords, w = coords @ rotation, rotation.T @ w
        spread = coords.pow(2).sum(-1).mean().sqrt().clamp(min=1e-9)
        coords, w = coords / spread, w / spread
        means.append(mean)
        projections.append(w)
        cloud.append(coords)
        previous = coords
    return {"mean": torch.stack(means), "w": torch.stack(projections), "cloud": torch.stack(cloud)}


def project(space: dict, states: torch.Tensor) -> torch.Tensor:
    """``states`` (``[layers, tokens, d]``) in the space: ``[layers, tokens, 3]``."""
    mean, w = space["mean"], space["w"]
    return torch.einsum("ltd,lkd->ltk", _scaled(states) - mean[:, None], w)


def save(space: dict, path: Path, model: str, meta: dict) -> None:
    """Write the space as JSON: the cloud to 3 decimals, the rest to 6 significant figures."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def sig(t: torch.Tensor):
        return json.loads(
            json.dumps(t.tolist(), default=float), parse_float=lambda s: float(f"{float(s):.6g}")
        )

    data = {"model": model, **meta, "layers": int(space["cloud"].shape[0]),
            "mean": sig(space["mean"]), "w": sig(space["w"]),
            "cloud": [[[round(x, 3) for x in p] for p in layer]
                      for layer in space["cloud"].tolist()]}  # fmt: skip
    path.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")


def load(path: Path) -> dict | None:
    """The space at ``path`` as tensors, or None when it isn't there."""
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"mean": torch.tensor(data["mean"]), "w": torch.tensor(data["w"])}
