"""``open-interpretability answers``: claims about how a model produces one answer (#4778).

The steps of one attempt, each run by a Claude Code agent or by hand:

1. ``answers propose-prompt`` writes the proposer's prompt from the newest explanation of
   the prompt and the earlier claims on it. The ``answer-proposer`` agent writes the claim.
2. ``answers check`` checks the claim's token counts and unit labels.
3. ``answers review-prompt`` writes the reviewer's prompt. It shows only the claim. The
   ``answer-reviewer`` agent writes the probes.
4. ``answers review`` measures the claim, decides it and writes the record to
   ``wiki/answers/``.
"""

from __future__ import annotations

from pathlib import Path

import click

from openinterpretability.cli import _load_runner, cli, wiki_dir_option, with_model_options


@cli.group(name="answers")
def answers_group() -> None:
    """Answer experiments: how a model produces one answer (docs/PROTOCOL.md §4)."""


def _read_claim(path: Path):
    from openinterpretability.core.answer_records import AnswerClaim

    return AnswerClaim.model_validate_json(path.read_text(encoding="utf-8"))


@answers_group.command(name="propose-prompt")
@click.argument("prompt")
@click.option("--model", "model_name", required=True, help="Hugging Face id.")
@click.option("--position", default=-1, show_default=True)
@click.option("--answer", default=None, help="The answer token. Default: the explanation's.")
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), default=None,
              help="Default: answer-<prompt id> in the exchange folder.")  # fmt: skip
@wiki_dir_option()
def propose_prompt_command(prompt, model_name, position, answer, out_dir, wiki_dir) -> None:
    """Write the proposer's prompt for the next answer claim on PROMPT."""
    from openinterpretability.answer_prompts import proposer_prompt
    from openinterpretability.core.answer_records import (
        MAX_ATTEMPTS,
        load,
        next_attempt,
        working_dir,
    )
    from openinterpretability.core.explanation_records import prompt_id, prompt_records

    found = prompt_records(Path(wiki_dir), model_name, prompt, position)
    newest = max(found, key=lambda r: r.written, default=None)
    answer = answer or (newest.answer if newest is not None else None)
    if answer is None:
        raise click.UsageError("no explanation of this prompt is recorded: pass --answer")
    pid = prompt_id(model_name, position, prompt)
    earlier = load(Path(wiki_dir), model_name, pid)
    attempt = next_attempt(earlier)
    if attempt is None:
        done = "a claim is VERIFIED" if len(earlier) < MAX_ATTEMPTS else f"{MAX_ATTEMPTS} claims"
        raise click.UsageError(f"no more attempts on this prompt: {done}")
    out = out_dir or working_dir(pid)
    out.mkdir(parents=True, exist_ok=True)
    claim = out / f"claim-{attempt}.json"
    text = proposer_prompt(str(claim), model_name, prompt, position, answer, newest, earlier,
                           attempt)  # fmt: skip
    path = out / f"proposer-{attempt}.prompt.md"
    path.write_text(text + "\n", encoding="utf-8")
    click.echo(f"attempt {attempt}: {path} (the claim goes to {claim})")


@answers_group.command(name="check")
@click.option("--claim", "claim_file", required=True, type=click.Path(exists=True, path_type=Path))
def check_command(claim_file: Path) -> None:
    """Check a claim's token counts and unit labels, with the model's tokenizer only."""
    from transformers import AutoTokenizer

    from openinterpretability.answer_review import check_claim

    claim = _read_claim(claim_file)
    tokenizer = AutoTokenizer.from_pretrained(claim.model_name, revision=claim.model_revision)
    problems = check_claim(tokenizer, claim)
    for problem in problems:
        click.echo(f"problem: {problem}")
    if problems:
        raise SystemExit(1)
    click.echo("ok")


@answers_group.command(name="review-prompt")
@click.option("--claim", "claim_file", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Default: reviewer-<attempt>.prompt.md beside the claim.")  # fmt: skip
def review_prompt_command(claim_file: Path, out: Path | None) -> None:
    """Write the reviewer's prompt for a claim. It shows the claim, and no measure."""
    from openinterpretability.answer_prompts import reviewer_prompt

    claim = _read_claim(claim_file)
    probes = claim_file.parent / f"answer-{claim.attempt}.probes.json"
    out = out or claim_file.parent / f"reviewer-{claim.attempt}.prompt.md"
    out.write_text(reviewer_prompt(str(probes), claim) + "\n", encoding="utf-8")
    click.echo(f"{out} (the probes go to {probes})")


@answers_group.command(name="review")
@click.option("--claim", "claim_file", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--probes", "probes_file", required=True,
              type=click.Path(exists=True, path_type=Path))  # fmt: skip
@click.option("--corpus", default="corpus.jsonl", show_default=True,
              type=click.Path(path_type=Path))  # fmt: skip
@click.option("--mean-tokens", default=20_000, show_default=True,
              help="Corpus tokens for each unit's mean, as explain uses (cached).")  # fmt: skip
@click.option("--seed", type=int, default=None,
              help="Seed of the matched random sets. By default the review draws one and "
              "records it, so the proposer can't tune against them (#4892).")  # fmt: skip
@wiki_dir_option()
@with_model_options
def review_command(claim_file, probes_file, corpus, mean_tokens, seed, wiki_dir, device, dtype):
    """Measure a claim on the reviewer's probes, decide it, and write its record."""
    from openinterpretability.answer_review import review
    from openinterpretability.core.answer_records import (
        AnswerProbes,
        load,
        next_attempt,
        record_path,
    )
    from openinterpretability.evidence import read_corpus
    from openinterpretability.explain import unit_means

    claim = _read_claim(claim_file)
    probes = AnswerProbes.model_validate_json(probes_file.read_text(encoding="utf-8"))
    expected = next_attempt(load(Path(wiki_dir), claim.model_name, claim.prompt_id))
    if expected != claim.attempt:
        raise click.UsageError(f"the claim is attempt {claim.attempt}; the next is {expected}")
    runner = _load_runner(claim.model_name, device, dtype, claim.model_revision)
    texts = read_corpus(corpus)[0]
    means = unit_means(runner, claim.model_name, texts, mean_tokens)
    try:
        record = review(runner, means, claim, probes, seed=seed, texts=texts)
    except ValueError as error:
        raise click.UsageError(str(error)) from error
    path = record_path(Path(wiki_dir), record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json() + "\n", encoding="utf-8")  # compact: it is large
    v = record.verdict
    failed = ", ".join(v.failed) or "none"
    click.echo(f"{v.status} (failed gates: {failed}) → {path}")


@answers_group.command(name="remeasure")
@click.option("--record", "record_file", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path))  # fmt: skip
@click.option("--corpus", default="corpus.jsonl", show_default=True,
              type=click.Path(path_type=Path))  # fmt: skip
@click.option("--mean-tokens", default=20_000, show_default=True)
@wiki_dir_option()
@with_model_options
def remeasure_command(record_file, corpus, mean_tokens, wiki_dir, device, dtype):
    """Measure an answer record's claim and probes again, as a new record (#4917).

    Records are write-once, so the new one names the old one in ``remeasures``. It holds
    the route's reading on every prompt, which records before #4917 lack. It is not an
    attempt: the claim and the probes are the same.
    """
    from openinterpretability.answer_review import review
    from openinterpretability.core.answer_records import AnswerRecord, record_path
    from openinterpretability.evidence import read_corpus
    from openinterpretability.explain import unit_means

    old = AnswerRecord.model_validate_json(record_file.read_text(encoding="utf-8"))
    c = old.claim
    runner = _load_runner(c.model_name, device, dtype, c.model_revision)
    texts = read_corpus(corpus)[0]
    means = unit_means(runner, c.model_name, texts, mean_tokens)
    seed = old.verdict.measures.null_seed  # the same random sets; records before 1.1 used 0
    record = review(runner, means, c, old.probes, old.verdict.policy, remeasures=old.id,
                    texts=texts, seed=0 if seed is None else seed)  # fmt: skip
    path = record_path(Path(wiki_dir), record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json() + "\n", encoding="utf-8")  # compact: it is large
    v = record.verdict
    click.echo(f"{v.status} (was {old.verdict.status}; failed gates: "
               f"{', '.join(v.failed) or 'none'}) → {path}")  # fmt: skip


@answers_group.command(name="space")
@click.option("--model", "model_name", required=True, help="Hugging Face id.")
@click.option("--corpus", default="corpus.jsonl", show_default=True,
              type=click.Path(path_type=Path))  # fmt: skip
@click.option("--points", default=1500, show_default=True, help="Corpus token positions.")
@click.option("--seed", default=0, show_default=True)
@with_model_options
def space_command(model_name, corpus, points, seed, device, dtype):
    """Fit the model's 3D embedding space on the corpus, for the answer pages (#4929).

    It writes data/embedding-space/<model slug>.json. Re-measure an answer record
    (`answers remeasure`) to place its prompts in the space.
    """
    import hashlib

    from openinterpretability import embedding_space as es
    from openinterpretability.evidence import read_corpus

    texts = read_corpus(corpus)[0]
    runner = _load_runner(model_name, device, dtype)
    space = es.fit(es.corpus_states(runner, texts, points, seed))
    digest = (
        hashlib.sha256(Path(corpus).read_bytes()).hexdigest() if Path(corpus).exists() else None
    )
    path = es.path_for(model_name)
    es.save(space, path, model_name, {"points": points, "seed": seed, "window": es.WINDOW,
                                      "corpus": str(corpus), "corpus_sha256": digest})  # fmt: skip
    click.echo(f"{space['cloud'].shape[1]} points × {space['cloud'].shape[0]} layers → {path}")
