"""How much LLMLingua-2 would cut from the prose the agents read, and what it would lose.

The numbers behind docs/analyses/prompt-compression.md (TODO "Performance"). Only prose
is scored: each agent's system prompt, its Claude Code definition and the methodology
skills that definition tells it to read. The evidence (token windows, activations,
lenses) is never compressed: it is the data the proposer explains and the reviewer tests.

For each paragraph of each text: LLMLingua-2's mean keep probability per word, which
says which parts carry information. For each text and rate: the compressed size, and how
many of its literals survive (inline code, numbers, paths), since a skill whose
thresholds or command names are dropped no longer says the same thing.

Writes one JSON line per paragraph (``"kind": "paragraph"``) and per text and rate
(``"kind": "compression"``), and the compressed texts to ``--compressed-dir`` for
reading. LLMLingua isn't a project dependency; run it with ``uv run --with``:

    uv run --with llmlingua python scripts/prompt_compression.py \\
        --out docs/analyses/prompt-compression.jsonl --compressed-dir /tmp/compressed
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import click
import torch

from openinterp.agents import ProposerAgent, ReviewerAgent
from openinterp.agents.judge import ExplanationJudge

MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
RATES = (0.7, 0.5, 0.33)
AGENTS = {
    "proposer": (ProposerAgent, "neuron-proposer"),
    "reviewer": (ReviewerAgent, "neuron-reviewer"),
    "judge": (ExplanationJudge, "explanation-judge"),
}
SKILL_PATH = re.compile(r"\.claude/skills/[\w-]+/SKILL\.md")
# Literals a reader acts on: `code`, numbers (0.25, 10–20, §5), and file paths.
LITERAL = re.compile(r"`[^`\n]+`|§?\d+(?:\.\d+)?%?|[\w.-]+/[\w./<>-]+")
CHUNK_TOKENS = 512


def prose_texts(root: Path) -> dict[str, str]:
    """Every prose text an agent reads, by name: system prompts, definitions, skills."""
    texts: dict[str, str] = {}
    for role, (agent, definition) in AGENTS.items():
        texts[f"{role}/system"] = agent.system_prompt
        body = (root / ".claude/agents" / f"{definition}.md").read_text()
        texts[f"{role}/definition"] = body.split("---", 2)[2].strip()
        for path in dict.fromkeys(SKILL_PATH.findall(body)):
            texts[f"skill/{Path(path).parent.name}"] = (root / path).read_text()
    return texts


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


@torch.no_grad()
def word_keep_probs(compressor, text: str) -> list[tuple[str, float]]:
    """Each word of ``text`` with its keep probability (the mean over its subwords)."""
    words = text.split()
    if not words:
        return []
    probs: list[float] = []
    for start in range(0, len(words), CHUNK_TOKENS // 3):
        chunk = words[start : start + CHUNK_TOKENS // 3]
        enc = compressor.tokenizer(
            chunk, is_split_into_words=True, return_tensors="pt", truncation=True
        )
        keep = compressor.model(**enc).logits.softmax(-1)[0, :, 1]
        sums, counts = [0.0] * len(chunk), [0] * len(chunk)
        for i, word in enumerate(enc.word_ids()):
            if word is not None:
                sums[word] += keep[i].item()
                counts[word] += 1
        probs += [s / max(c, 1) for s, c in zip(sums, counts, strict=True)]
    return list(zip(words, probs, strict=True))


def literal_survival(original: str, compressed: str) -> tuple[int, int]:
    """How many of ``original``'s literals appear verbatim in ``compressed``, of how many."""
    found = LITERAL.findall(original)
    return sum(lit in compressed for lit in found), len(found)


def paragraph_rows(compressor, name: str, text: str) -> list[dict]:
    rows = []
    for index, para in enumerate(paragraphs(text)):
        scored = word_keep_probs(compressor, para)
        mean = sum(p for _, p in scored) / len(scored)
        low = [w for w, p in sorted(scored, key=lambda wp: wp[1])[:8]]
        row = {"kind": "paragraph", "text": name, "index": index, "chars": len(para)}
        rows.append(row | {"mean_keep": round(mean, 3), "lowest": low, "head": para[:80]})
    return rows


def compression_rows(compressor, name: str, text: str, out_dir: Path | None) -> list[dict]:
    rows = []
    for rate in RATES:
        compressed = compressor.compress_prompt(text, rate=rate, force_tokens=["\n"])
        short = compressed["compressed_prompt"]
        kept, total = literal_survival(text, short)
        if out_dir is not None:
            path = out_dir / f"{name.replace('/', '--')}.rate{rate}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(short)
        row = {"kind": "compression", "text": name, "rate": rate, "chars": len(text)}
        row |= {"compressed_chars": len(short), "literals": total, "literals_kept": kept}
        rows.append(row)
    return rows


@click.command()
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--compressed-dir", type=click.Path(path_type=Path), default=None)
@click.option("--root", type=click.Path(path_type=Path), default=Path("."))
def main(out: Path, compressed_dir: Path | None, root: Path) -> None:
    from llmlingua import PromptCompressor

    compressor = PromptCompressor(model_name=MODEL, use_llmlingua2=True, device_map="cpu")
    rows = []
    for name, text in prose_texts(root).items():
        rows += paragraph_rows(compressor, name, text)
        rows += compression_rows(compressor, name, text, compressed_dir)
        click.echo(f"{name}: {len(text)} chars", err=True)
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


if __name__ == "__main__":
    main()
