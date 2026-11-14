"""The lenses and attribution that follow a scan (:func:`openinterp.evidence.scan_evidence`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from openinterp.evidence import ScanRequest, _Texts


def read_lenses(runner, request: ScanRequest, ident, spec, corpus: _Texts, scan):
    """Return the vocabulary lens and, for a neuron, the top examples' input attribution.

    A head has no lens, because it writes a subspace, not one direction that a lens can
    read.
    """
    from openinterp.core import costs
    from openinterp.harness.dictionaries import load_spec
    from openinterp.harness.input_attribution import input_attribution
    from openinterp.harness.jlens import direction_lens, vocabulary_lens
    from openinterp.harness.scan import cut_pieces
    from openinterp.harness.superpositions import superposition_write

    lens = attribution = None
    kind = request.unit
    if kind != "head":
        click.echo("Reading the output direction through the logit and Jacobian lenses…", err=True)
        # The first 100 documents, each cut to its first window when scanning windows.
        texts, window = corpus.texts[:100], request.window_tokens
        pieces = cut_pieces(runner.tokenizer, texts, window, corpus.stride)
        lens_texts = [p.text for p in pieces if p.start == 0]
        with costs.stage("lenses"):
            if kind == "neuron":
                lens = vocabulary_lens(
                    runner, lens_texts, ident.layer, ident.neuron_idx, ident.pole
                )
            else:  # a feature or a superposition: the direction that it writes
                direction = (
                    superposition_write(runner, ident, request.superposition)
                    if kind == "superposition"
                    else load_spec(spec, ident.layer, runner.device).w_dec[ident.feature_idx]
                )
                lens = direction_lens(runner, lens_texts, ident.layer, direction)
    if kind == "neuron":
        click.echo("Attributing the top examples' peaks to their context tokens…", err=True)
        with costs.stage("input_attribution"):
            attribution = input_attribution(
                runner,
                scan.examples,
                ident.layer,
                ident.neuron_idx,
                ident.pole,
                skip_first=scan.first_token_floor,
            )
    return lens, attribution
