"""The sister search over many model pairs, with each model read once per stage.

:func:`~openinterp.harness.sisters.find_sisters` gives the method. A search per pair reads
both models over the whole corpus twice, so its cost grows with the number of pairs. This
module splits the same computation into four stages:

1. **Source series.** Each source model runs to its deepest new pole. Each pole's value
   at every token goes to the CPU.
2. **Forward.** Each other model runs over every layer. Each layer is correlated with
   every incoming series, at the boundaries that the two tokenizations share.
3. **Match series and rotations.** Each other model runs to the deepest layer needed. It
   keeps each match's series, and correlates each source pole with a rotated basis of its
   top match's layer. Rotations go a few layers at a time, under a memory budget.
4. **Backward.** Each source model runs over every layer, correlated with the incoming
   match series.

Each stage loads one model at a time and frees it before the next. Hooks keep only the
rows that a stage reads, not the whole ``[batch, seq, width]`` activation.
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass, field

import torch

from openinterp.harness.hooks import NeuronHookRunner, _StopForward
from openinterp.harness.scan import BATCH_TOKENS, plan_batches
from openinterp.harness.sisters import (
    Correlations,
    Sisters,
    Tokens,
    Unit,
    _top,
    pole_columns,
    rotation,
    shared_boundaries,
    tokenize,
)

# A model as a stage loads it: (name, revision).
ModelKey = tuple[str, str | None]
# The largest total size of the rotations that one pass holds on the device.
ROTATION_BYTES = 2 * 1024**3
# The largest total size of the correlation sums that one pass keeps, on the CPU.
CORRELATION_BYTES = 3 * 1024**3


@dataclass
class Job:
    """One ordered pair: the ``units`` of ``source`` to match in ``other``.

    The search fills in the fields after ``other_name``. ``other_name`` seeds the
    rotations, as ``find_sisters`` does.
    """

    source: ModelKey
    other: ModelKey
    units: list[Unit]
    other_name: str
    found: dict[Unit, Sisters] = field(default_factory=dict)
    tokens: int = 0  # positions paired
    readable: int = 0  # the source's tokens, less each text's first
    source_revision: str | None = None
    other_revision: str | None = None
    source_layers: int = 0
    other_layers: int = 0
    # Working state between stages.
    rows: list[tuple[list[int], list[int]]] = field(default_factory=list)  # per document
    matches: list[list] = field(default_factory=list)
    match_units: list[Unit] = field(default_factory=list)
    match_series: list[torch.Tensor | None] = field(default_factory=list)  # per document
    back: list = field(default_factory=list)  # per match unit: its best source partner
    by_layer: dict[int, list[int]] = field(default_factory=dict)
    rotated: dict[int, Correlations] = field(default_factory=dict)


@dataclass
class _Model:
    """What the search keeps of one model between stages."""

    tokens: list[Tokens] = field(default_factory=list)
    units: list[Unit] = field(default_factory=list)  # its new poles, as a source
    series: list[torch.Tensor | None] = field(default_factory=list)  # per document


def search(
    jobs: Sequence[Job],
    load: Callable[[str, str | None], NeuronHookRunner],
    texts: Sequence[str],
    *,
    k: int = 5,
    batch_tokens: int = BATCH_TOKENS,
    rotation_bytes: int = ROTATION_BYTES,
    correlation_bytes: int = CORRELATION_BYTES,
    progress: Callable[[str, int], None] | None = None,
) -> None:
    """Fill in each job's sisters, in four stages over the corpus (module docstring)."""
    run = _Stages(
        list(jobs), load, texts, k, batch_tokens, rotation_bytes, correlation_bytes, progress
    )
    for key in _keys(j.source for j in jobs):
        run.source_series(key)
    for key in _keys(j.other for j in jobs):
        run.forward(key)
    for key in _keys(j.other for j in jobs):
        run.match_series(key)
    for key in _keys(j.source for j in jobs):
        run.backward(key)
    for job in jobs:
        job.found = run.result(job)
        job.rows, job.match_series, job.rotated, job.back = [], [], {}, []


def _keys(keys) -> list[ModelKey]:
    return list(dict.fromkeys(keys))


@dataclass
class _Stages:
    jobs: list[Job]
    load: Callable[[str, str | None], NeuronHookRunner]
    texts: Sequence[str]
    k: int
    batch_tokens: int
    rotation_bytes: int
    correlation_bytes: int
    progress: Callable[[str, int], None] | None
    models: dict[ModelKey, _Model] = field(default_factory=dict)

    def tick(self, stage: str, n: int) -> None:
        if self.progress:
            self.progress(stage, n)

    def source_series(self, key: ModelKey) -> None:
        """Stage 1: each new pole of ``key`` at every token of every text."""
        jobs = [j for j in self.jobs if j.source == key]
        runner = self.load(*key)
        model = self.models[key] = _Model([tokenize(runner, t) for t in self.texts])
        model.units = list(dict.fromkeys(u for j in jobs for u in j.units))
        everything = [list(range(len(t.ids))) for t in model.tokens]
        model.series = [None] * len(self.texts)
        columns = {}
        for layer, idx, _ in model.units:
            columns.setdefault(layer, []).append(idx)
        for docs, rows in _passes(runner, model.tokens, everything, columns, self.batch_tokens):
            series = _strengths(rows, columns, model.units)
            for d, part in _split(series, docs, [len(everything[d]) for d in docs]):
                model.series[d] = part
            self.tick("source", sum(len(everything[d]) for d in docs))
        readable = sum(max(len(t.ids) - 1, 0) for t in model.tokens)
        for job in jobs:
            job.readable, job.source_layers = readable, runner.n_layers
            job.source_revision = _revision(runner)
        del runner
        _free()

    def forward(self, key: ModelKey) -> None:
        """Stage 2: each job's ``k`` best poles over every layer of ``key``."""
        jobs = [j for j in self.jobs if j.other == key]
        runner = self.load(*key)
        tokens = self.models.setdefault(key, _Model()).tokens or [
            tokenize(runner, t) for t in self.texts
        ]
        self.models[key].tokens = tokens
        for job in jobs:
            src = self.models[job.source].tokens
            job.rows = [_boundaries(a, b) for a, b in zip(src, tokens, strict=True)]
            job.other_layers, job.other_revision = runner.n_layers, _revision(runner)
        every = list(range(runner.n_layers))
        corr = self._correlate(runner, tokens, jobs, "forward")
        widths = {layer: runner.mlp_width(layer) for layer in every}
        for job in jobs:
            job.matches = _top(corr[id(job)], every, widths, self.k)
            job.tokens = corr[id(job)].n
        del runner
        _free()

    def match_series(self, key: ModelKey) -> None:
        """Stage 3: each match's series, and each pole's rotated reference."""
        jobs = [j for j in self.jobs if j.other == key]
        for job in jobs:
            job.match_units = list(
                dict.fromkeys((m.layer, m.neuron_idx, m.pole) for ms in job.matches for m in ms)
            )
            job.match_series = [None] * len(self.texts)
            job.by_layer = {}
            for i, ms in enumerate(job.matches):
                if ms:
                    job.by_layer.setdefault(ms[0].layer, []).append(i)
            job.rotated = {layer: Correlations(len(i)) for layer, i in job.by_layer.items()}
        runner = self.load(*key)
        tokens = self.models[key].tokens
        turned = sorted({layer for j in jobs for layer in j.by_layer})
        for n, group in enumerate(_groups(turned, runner, self.rotation_bytes) or [[]]):
            self._match_pass(runner, jobs, tokens, group, first=n == 0)
        del runner
        _free()

    def _match_pass(self, runner, jobs, tokens, group, first: bool) -> None:
        """One pass of stage 3: the match series (on the first pass) and ``group``'s turns."""
        name = jobs[0].other_name
        turns = {
            layer: rotation(name, layer, runner.mlp_width(layer)).to(runner.device)
            for layer in group
        }
        layers = set(group) | ({u[0] for j in jobs for u in j.match_units} if first else set())
        if not layers:
            return
        wanted = _union(tokens, [[r[1] for r in j.rows] for j in jobs])
        columns = dict.fromkeys(layers)
        for docs, rows in _passes(runner, tokens, wanted, columns, self.batch_tokens):
            for job in jobs:
                sel = _select(docs, wanted, [job.rows[d][1] for d in docs])
                reads = {layer for layer in group if layer in job.by_layer}
                reads |= {u[0] for u in job.match_units} if first else set()
                picked = {layer: _take(rows[layer], sel) for layer in reads}
                if first:
                    series = _strengths(picked, None, job.match_units)
                    counts = [len(job.rows[d][1]) for d in docs]
                    for d, part in _split(series, docs, counts):
                        job.match_series[d] = part
                own = self._sources(job, docs, job.units).to(runner.device)
                for layer in group:
                    if layer in job.by_layer:
                        xs = own[:, job.by_layer[layer]]
                        job.rotated[layer].add_sources(xs)
                        job.rotated[layer].add_targets(
                            layer, xs, pole_columns(picked[layer] @ turns[layer])
                        )
            self.tick("matches", sum(len(wanted[d]) for d in docs))

    def backward(self, key: ModelKey) -> None:
        """Stage 4: each match's best partner over every layer of the source ``key``."""
        jobs = [j for j in self.jobs if j.source == key]
        runner = self.load(*key)
        tokens = self.models[key].tokens
        every = list(range(runner.n_layers))
        back = self._correlate(runner, tokens, jobs, "backward")
        widths = {layer: runner.mlp_width(layer) for layer in every}
        for job in jobs:
            best = _top(back[id(job)], every, widths, 1)
            job.back = [m[0] for m in best]
        del runner
        _free()

    def _correlate(self, runner, tokens, jobs: list[Job], stage: str) -> dict[int, Correlations]:
        """Correlate every layer of ``runner`` with each job's incoming series.

        In the forward stage the series are the source poles, at the other model's side of
        each boundary. In the backward stage they are the matches, at the source's side.
        Jobs go in groups whose running totals fit :data:`CORRELATION_BYTES`, one pass each.
        """
        every = list(range(runner.n_layers))
        side = 1 if stage == "forward" else 0
        columns = 2 * sum(runner.mlp_width(layer) for layer in every)
        corr: dict[int, Correlations] = {}
        for group in _job_groups(jobs, columns, self.correlation_bytes, stage):
            group_corr = {id(j): Correlations(_width(j, stage)) for j in group}
            wanted = _union(tokens, [[r[side] for r in j.rows] for j in group])
            columns_ = dict.fromkeys(every)
            for docs, rows in _passes(runner, tokens, wanted, columns_, self.batch_tokens):
                for job in group:
                    x = self._incoming(job, docs, stage).to(runner.device)
                    sel = _select(docs, wanted, [job.rows[d][side] for d in docs])
                    group_corr[id(job)].add_sources(x)
                    for layer in every:
                        group_corr[id(job)].add_targets(
                            layer, x, pole_columns(_take(rows[layer], sel))
                        )
                self.tick(stage, group_corr[id(group[0])].n)
            corr.update(group_corr)
        return corr

    def _incoming(self, job: Job, docs: list[int], stage: str) -> torch.Tensor:
        if stage == "forward":
            return self._sources(job, docs, job.units)
        return _cat([job.match_series[d] for d in docs], len(job.match_units))

    def _sources(self, job: Job, docs: list[int], units: list[Unit]) -> torch.Tensor:
        """``[n, len(units)]``: the source's series at ``job``'s boundaries in ``docs``."""
        model = self.models[job.source]
        cols = [model.units.index(u) for u in units]
        parts = [
            model.series[d][job.rows[d][0]][:, cols]
            for d in docs
            if job.rows[d][0] and model.series[d] is not None
        ]
        return torch.cat(parts) if parts else torch.zeros(0, len(cols))

    def result(self, job: Job) -> dict[Unit, Sisters]:
        """Each of ``job.units`` with its matches, back partners and rotated reference."""
        back_of = dict(zip(job.match_units, job.back, strict=True))
        reference = {}
        for layer, idx in job.by_layer.items():
            r = job.rotated[layer].pearson(layer).amax(dim=1)
            reference.update({i: float(v) for i, v in zip(idx, r.tolist(), strict=True)})
        return {
            unit: Sisters(
                job.matches[i],
                [back_of[(m.layer, m.neuron_idx, m.pole)] for m in job.matches[i]],
                reference.get(i, 0.0),
            )
            for i, unit in enumerate(job.units)
        }


def _boundaries(a: Tokens, b: Tokens) -> tuple[list[int], list[int]]:
    """The shared boundaries of one text, or none if either model reads no token of it."""
    return shared_boundaries(a.ends, b.ends) if a.ids and b.ids else ([], [])


def _union(tokens: list[Tokens], per_job: list[list[list[int]]]) -> list[list[int]]:
    """Per document: the sorted positions that any job reads."""
    return [sorted({p for job in per_job for p in job[d]}) for d in range(len(tokens))]


def _select(docs: list[int], wanted: list[list[int]], chosen: list[list[int]]) -> torch.Tensor:
    """Indices into a pass's rows (``wanted`` of ``docs``, in order) of the ``chosen`` ones."""
    out, offset = [], 0
    for d, picks in zip(docs, chosen, strict=True):
        where = {p: i for i, p in enumerate(wanted[d])}
        out += [offset + where[p] for p in picks]
        offset += len(wanted[d])
    return torch.tensor(out, dtype=torch.long)


def _take(rows: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
    """``rows[sel]``, without a copy when ``sel`` is every row in order."""
    if sel.shape[0] == rows.shape[0] and bool((sel == torch.arange(sel.shape[0])).all()):
        return rows
    return rows[sel.to(rows.device)]


def _split(series: torch.Tensor, docs: list[int], counts: list[int]):
    """Cut a pass's ``[n, P]`` rows into one CPU block per document, in ``docs`` order."""
    offset = 0
    for d, n in zip(docs, counts, strict=True):
        yield d, series[offset : offset + n].cpu()
        offset += n


def _cat(parts: list[torch.Tensor | None], width: int) -> torch.Tensor:
    parts = [p for p in parts if p is not None and p.shape[0]]
    return torch.cat(parts) if parts else torch.zeros(0, width)


def _strengths(rows: dict, columns: dict | None, units: list[Unit]) -> torch.Tensor:
    """``[n, len(units)]`` pole strengths from a pass's rows.

    With ``columns``, each layer's rows hold only the listed neurons, in that order.
    """
    cols = []
    for layer, idx, pole in units:
        at = columns[layer].index(idx) if columns else idx
        act = rows[layer][:, at]
        cols.append(act.clamp(min=0) if pole == "positive" else (-act).clamp(min=0))
    return torch.stack(cols, dim=1)


def _width(job: Job, stage: str) -> int:
    return len(job.units) if stage == "forward" else len(job.match_units)


def _job_groups(jobs: list[Job], columns: int, budget: int, stage: str) -> list[list[Job]]:
    """``jobs`` in order, cut into groups whose running totals fit in ``budget`` bytes.

    Each source series costs ``columns`` float64 sums (:class:`Correlations`). A job over
    the budget on its own gets a group of its own.
    """
    groups, size = [], budget + 1
    for job in jobs:
        need = _width(job, stage) * columns * 8
        if size + need > budget:
            groups.append([])
            size = 0
        groups[-1].append(job)
        size += need
    return groups


def _groups(layers: list[int], runner, budget: int) -> list[list[int]]:
    """``layers`` in order, cut into groups whose rotations fit in ``budget`` bytes."""
    groups, size = [], budget + 1
    for layer in layers:
        need = runner.mlp_width(layer) ** 2 * 4
        if size + need > budget:
            groups.append([])
            size = 0
        groups[-1].append(layer)
        size += need
    return groups


def _passes(
    runner: NeuronHookRunner,
    tokens: list[Tokens],
    wanted: list[list[int]],
    columns: dict[int, list[int] | None],
    batch_tokens: int,
) -> Iterator[tuple[list[int], dict[int, torch.Tensor]]]:
    """Run every text with a wanted position, and yield (documents, ``{layer: rows}``).

    The rows are each document's ``wanted`` positions, in order. Where ``columns`` lists
    neurons for a layer, only those are kept. The pass stops after the deepest layer.
    """
    docs = [d for d, t in enumerate(tokens) if t.ids and wanted[d]]
    lengths = [len(tokens[d].ids) for d in docs]
    start = 0
    for size in plan_batches(lengths, batch_tokens):
        group, start = docs[start : start + size], start + size
        yield (
            group,
            _gather(runner, [tokens[d].ids for d in group], [wanted[d] for d in group], columns),
        )


def _gather(runner, seqs, positions, columns) -> dict[int, torch.Tensor]:
    """Run ``seqs`` (right-padded), keeping only ``positions`` inside each layer's hook."""
    width = max(map(len, seqs))
    pad = getattr(runner.tokenizer, "pad_token_id", None) or 0
    ids = torch.tensor([s + [pad] * (width - len(s)) for s in seqs])
    mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs])
    rows = [(row, p) for row, pos in enumerate(positions) for p in pos]
    index = torch.tensor(rows, dtype=torch.long).reshape(-1, 2).to(runner.device)
    keep = {
        layer: None if cols is None else torch.tensor(cols, device=runner.device)
        for layer, cols in columns.items()
    }
    out: dict[int, torch.Tensor] = {}
    last = max(columns)

    def recorder(layer: int):
        def record(act: torch.Tensor) -> None:
            picked = act[index[:, 0], index[:, 1]]
            out[layer] = (picked if keep[layer] is None else picked[:, keep[layer]]).float()
            if layer == last:
                raise _StopForward

        return record

    hooks = runner.hooks({layer: recorder(layer) for layer in columns})
    with torch.no_grad(), suppress(_StopForward), hooks:
        runner.forward(runner.encode({"input_ids": ids, "attention_mask": mask}))
    return out


def _revision(runner) -> str | None:
    from openinterp.harness.environment import run_environment

    return run_environment(runner).model_revision


def _free() -> None:
    """Collect the model that a stage dropped, and empty the device's cache."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
