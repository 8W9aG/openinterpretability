"""Behaviour: what a change does to the model's predictions (#5230).

Damage in the parts the registry doesn't explain shows here. Two measures, each on both
sides of a comparison:

- :func:`token_kl`: the mean, over every token of a corpus sample, of KL(first ‖ second)
  between the two sides' next-token distributions.
- :func:`answer_probabilities`: the probability of each explained prompt's answer, from the
  explanation records in ``wiki/explanations/``.
"""

from __future__ import annotations

from pathlib import Path

import torch


@torch.no_grad()
def _log_probs(side, batch: torch.Tensor) -> torch.Tensor:
    with side.entered() as runner:
        logits = runner.model(input_ids=batch.to(runner.device)).logits
    return torch.log_softmax(logits.float(), dim=-1).cpu()


def token_kl(first, second, batches: list[torch.Tensor]) -> float:
    """The mean token-level KL(first ‖ second), in nats, first tokens left out."""
    total, count = 0.0, 0
    for batch in batches:
        p, q = _log_probs(first, batch)[:, 1:], _log_probs(second, batch)[:, 1:]
        total += float((p.exp() * (p - q)).sum(-1).sum())
        count += p.shape[0] * p.shape[1]
    return total / max(count, 1)


def explained_prompts(wiki_dir: Path, model: str) -> list[dict]:
    """Each explained prompt of ``model``: its prompt, position and answer token id."""
    from openinterpretability.core.explanation_records import load_all

    out, seen = [], set()
    for records in load_all(Path(wiki_dir)).values():
        for _, record in records:
            key = (record.prompt, record.position)
            if record.model_name != model or key in seen:
                continue
            seen.add(key)
            token = record.inputs.token_id if record.inputs else None
            out.append({"prompt": record.prompt, "position": record.position,
                        "answer": record.answer, "token_id": token})  # fmt: skip
    return out


@torch.no_grad()
def answer_probabilities(side, prompts: list[dict]) -> list[float | None]:
    """The probability of each prompt's answer token on ``side``; None when it has no id."""
    out: list[float | None] = []
    with side.entered() as runner:
        for p in prompts:
            if p["token_id"] is None:
                out.append(None)
                continue
            ids = torch.tensor([runner.tokenizer.encode(p["prompt"])], device=runner.device)
            logits = runner.model(input_ids=ids).logits[0, p["position"]].float()
            out.append(round(float(torch.softmax(logits, -1)[p["token_id"]]), 6))
    return out
