"""Dictionary cross-reference: which latents read, normalised writes, and wide layers."""

from types import SimpleNamespace

import pytest
import torch

from openinterpretability.core.exceptions import ArchitectureNotSupportedError
from openinterpretability.harness import crossref
from openinterpretability.harness.crossref import _align, align_layer, latent_reads, latent_writes
from tests.test_crossref import N_FEATURES, _spec, gpt2_runner  # noqa: F401  (fixture)


def test_a_normalised_mlp_output_has_no_comparable_latent_write():
    runner = SimpleNamespace(mlp_write=lambda layer: "norm", down_proj=lambda layer: "proj")
    with pytest.raises(ArchitectureNotSupportedError, match="normalised before the residual"):
        latent_writes(runner, 0, SimpleNamespace(w_dec=torch.zeros(2, 2)))


def test_a_residual_mid_dictionary_reads_with_its_encoder_as_it_is():
    w_enc = torch.randn(4, 3)
    spec = SimpleNamespace(input_site="resid_mid")
    read = latent_reads(None, spec, 0, SimpleNamespace(w_enc=w_enc))
    torch.testing.assert_close(read, w_enc.T)


def test_a_dictionary_on_the_mlp_output_has_only_a_write_side(
    tmp_path,
    monkeypatch,
    gpt2_runner,  # noqa: F811
):
    torch.manual_seed(5)
    w_enc, w_dec = torch.randn(32, N_FEATURES), torch.randn(N_FEATURES, 32)
    spec = _spec(tmp_path, monkeypatch, "out-sae", "tiny-gpt2", "mlp_out", w_enc, w_dec)
    assert latent_reads(gpt2_runner, spec, 1, SimpleNamespace(w_enc=w_enc)) is None
    write, read = align_layer(gpt2_runner, spec, 1, [(3, "positive")], k=2)[(3, "positive")]
    assert read is None and len(write.matches) == 2


def test_a_pole_past_the_first_chunk_is_matched_in_its_own_chunk():
    g = torch.Generator().manual_seed(0)
    width = crossref.CHUNK + 20
    rows = torch.randn(width, 8, generator=g)
    latents = crossref._unit(torch.randn(10, 8, generator=g))
    found, _ = _align({"out": rows}, latents, [(width - 1, "positive")], k=3)
    cos = crossref._unit(rows[width - 1 : width]) @ latents.T
    expected = cos[0].topk(3).indices.tolist()
    assert [m.feature_idx for m in found[(width - 1, "positive")]] == expected
