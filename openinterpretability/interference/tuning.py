"""A sequence of narrow fine-tunes, and what each stage does to a model (#5125).

Each stage trains the whole model on one narrow task, as Luo et al. (2023) do. The tasks
come from Hub datasets, each pinned to one commit and read from its parquet files. Every
task uses one prompt template, :data:`TEMPLATE`, so that two models trained on one sequence
differ only in where they start. After each stage, these measures show what changed:

- :func:`heldout_loss`: the mean loss on the response tokens of a task's held-out pairs.
- :func:`stage_kl`: KL(before ‖ after) on the response tokens of the stage's own held-out
  pairs, the quantity of RL's Razor (Shenfeld et al., 2025).
- :func:`arc_accuracy`: zero-shot ARC-Easy accuracy, the general benchmark.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass

import torch

TEMPLATE = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
MAX_TOKENS = 512
DOC_CHARS = 1_500  # an XSum article is cut to this many characters, to fit MAX_TOKENS


@dataclass(frozen=True)
class Task:
    """One narrow task: a Hub dataset at one commit, and how a row becomes a pair."""

    name: str
    repo: str
    revision: str
    train: str
    heldout: str
    columns: tuple[str, ...]
    instruction: str


TASKS = (
    Task("sst2", "stanfordnlp/sst2", "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb",
         "data/train-00000-of-00001.parquet", "data/validation-00000-of-00001.parquet",
         ("sentence", "label"),
         "Classify the sentiment of the sentence as positive or negative."),
    Task("xsum", "EdinburghNLP/xsum", "7d4d486c2f8ef850b1a11aead99b894ff3dd7da9",
         "data/train-00000-of-00001.parquet", "data/validation-00000-of-00001.parquet",
         ("document", "summary"), "Summarise the article in one sentence."),
    Task("gsm8k", "openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866",
         "main/train-00000-of-00001.parquet", "main/test-00000-of-00001.parquet",
         ("question", "answer"), "Solve the maths problem. End with the answer after ####."),
)  # fmt: skip
ARC = Task("arc_easy", "allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453", "",
           "ARC-Easy/test-00000-of-00001.parquet", ("question", "choices", "answerKey"),
           "")  # fmt: skip


def pair(task: Task, row: dict) -> tuple[str, str]:
    """``(prompt, response)`` for one row of ``task``.

    >>> pair(TASKS[0], {"sentence": "a fine film", "label": 1})[1]
    'positive'
    """
    text, target = (row[c] for c in task.columns[:2])
    if task.name == "sst2":
        target = "positive" if target == 1 else "negative"
    if task.name == "xsum":
        text = text[:DOC_CHARS]
    return TEMPLATE.format(instruction=task.instruction, input=text.strip()), target.strip()


def load_rows(task: Task, split: str) -> list[dict]:
    """Every row of ``task``'s ``split`` file (``train`` or ``heldout``), from the Hub."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(task.repo, getattr(task, split), repo_type="dataset",
                           revision=task.revision)  # fmt: skip
    return pq.read_table(path, columns=list(task.columns)).to_pylist()


def sample(rows: Sequence, count: int, seed: int) -> list:
    """``count`` rows drawn without replacement with ``seed``, or all rows when fewer.

    >>> sample(list(range(10)), 3, 0) == sample(list(range(10)), 3, 0)
    True
    """
    if count >= len(rows):
        return list(rows)
    return [rows[i] for i in sorted(random.Random(seed).sample(range(len(rows)), count))]


def encode(tokenizer, prompt: str, response: str) -> tuple[list[int], int]:
    """The ids of ``prompt + response`` and the end of sequence, and where the response starts.

    The ids are cut to :data:`MAX_TOKENS`. The prompt is cut first from its start, so the
    response and the instruction's end stay.
    """
    head = tokenizer.encode(prompt, add_special_tokens=False)
    tail = tokenizer.encode(response + tokenizer.eos_token, add_special_tokens=False)
    tail = tail[: MAX_TOKENS - 1]
    head = head[max(0, len(head) + len(tail) - MAX_TOKENS) :]
    return head + tail, len(head)


def collate(rows: list[tuple[list[int], int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(ids, attention, mask)``, right-padded. ``mask`` marks each response token."""
    width = max(len(ids) for ids, _ in rows)
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    attention = torch.zeros(len(rows), width, dtype=torch.long)
    mask = torch.zeros(len(rows), width, dtype=torch.bool)
    for i, (row, start) in enumerate(rows):
        ids[i, : len(row)] = torch.tensor(row)
        attention[i, : len(row)] = 1
        mask[i, start : len(row)] = True
    return ids, attention, mask


def _token_log_probs(model, ids, attention, device) -> torch.Tensor:
    """Log-probabilities of the next token at each position but the last, as float32."""
    autocast = torch.autocast("cuda", torch.bfloat16) if device == "cuda" else nullcontext()
    with autocast:
        logits = model(input_ids=ids.to(device), attention_mask=attention.to(device)).logits
    return torch.log_softmax(logits[:, :-1].float(), dim=-1)


def response_loss(model, ids, attention, mask, device) -> tuple[torch.Tensor, int]:
    """The summed loss on the response tokens, and how many there are."""
    logp = _token_log_probs(model, ids, attention, device)
    target = ids[:, 1:].to(device)
    picked = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    keep = mask[:, 1:].to(device)
    return -(picked * keep).sum(), int(keep.sum())


def train_stage(model, tokenizer, pairs: list[tuple[str, str]], *, device: str, epochs: int,
                lr: float, batch: int, accumulate: int, seed: int,
                progress: Callable[[str], None] = lambda _: None) -> list[float]:  # fmt: skip
    """Train every weight of ``model`` on ``pairs``, with AdamW at a constant ``lr``.

    Each optimiser step takes ``batch * accumulate`` pairs. The loss is the mean over the
    response tokens of a step. The model is left in eval mode with no gradients.

    Returns:
        The loss of each optimiser step.
    """
    encoded = [encode(tokenizer, p, r) for p, r in pairs]
    model.train()
    model.requires_grad_(True)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    rng, losses = random.Random(seed), []
    per_step = batch * accumulate
    for epoch in range(epochs):
        order = list(range(len(encoded)))
        rng.shuffle(order)
        for start in range(0, len(order) - per_step + 1, per_step):
            step = [encoded[i] for i in order[start : start + per_step]]
            parts = [collate(step[i : i + batch]) for i in range(0, per_step, batch)]
            tokens = sum(int(m[:, 1:].sum()) for _, _, m in parts)
            total = 0.0
            for ids, attention, mask in parts:
                loss, _ = response_loss(model, ids, attention, mask, device)
                (loss / max(tokens, 1)).backward()
                total += float(loss)
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            losses.append(total / max(tokens, 1))
            if len(losses) % 50 == 0:
                progress(f"epoch {epoch + 1}, step {len(losses)}: loss {losses[-1]:.3f}")
    del optimiser
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.eval()
    model.requires_grad_(False)
    return losses


@torch.no_grad()
def heldout_loss(model, tokenizer, pairs: list[tuple[str, str]], device: str,
                 batch: int = 4) -> float:  # fmt: skip
    """The mean loss, in nats, over every response token of ``pairs``."""
    encoded = [encode(tokenizer, p, r) for p, r in pairs]
    total, count = 0.0, 0
    for i in range(0, len(encoded), batch):
        loss, n = response_loss(model, *collate(encoded[i : i + batch]), device)
        total, count = total + float(loss), count + n
    return total / max(count, 1)


@torch.no_grad()
def stage_kl(before, after, tokenizer, pairs: list[tuple[str, str]], device: str,
             batch: int = 2) -> float:  # fmt: skip
    """The mean KL(before ‖ after), in nats, over every response token of ``pairs``.

    The response tokens are the dataset's references, not samples from either model.
    """
    encoded = [encode(tokenizer, p, r) for p, r in pairs]
    total, count = 0.0, 0
    for i in range(0, len(encoded), batch):
        ids, attention, mask = collate(encoded[i : i + batch])
        p = _token_log_probs(before, ids, attention, device)
        q = _token_log_probs(after, ids, attention, device)
        kl = (p.exp() * (p - q)).sum(-1)
        keep = mask[:, 1:].to(device)
        total, count = total + float((kl * keep).sum()), count + int(keep.sum())
    return total / max(count, 1)


def arc_prompt(question: str) -> str:
    """The zero-shot prompt of one ARC question.

    >>> arc_prompt("Why?")
    'Question: Why?\\nAnswer:'
    """
    return f"Question: {question.strip()}\nAnswer:"


@torch.no_grad()
def arc_accuracy(model, tokenizer, rows: list[dict], device: str) -> tuple[int, int]:
    """``(correct, items)``: how many ARC rows the model answers right, zero-shot.

    The model's answer is the choice whose tokens have the highest mean log-probability
    after :func:`arc_prompt`. The mean over tokens corrects for the length of a choice.
    """
    correct = 0
    for row in rows:
        head = tokenizer.encode(arc_prompt(row["question"]), add_special_tokens=False)
        options = [(head + tokenizer.encode(" " + text.strip(), add_special_tokens=False),
                    len(head)) for text in row["choices"]["text"]]  # fmt: skip
        ids, attention, mask = collate(options)
        logp = _token_log_probs(model, ids, attention, device)
        picked = logp.gather(-1, ids[:, 1:].to(device).unsqueeze(-1)).squeeze(-1)
        keep = mask[:, 1:].to(device)
        scores = (picked * keep).sum(-1) / keep.sum(-1).clamp(min=1)
        best = row["choices"]["label"][int(scores.argmax())]
        correct += best == row["answerKey"]
    return correct, len(rows)
