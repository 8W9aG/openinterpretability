"""The one-pass token sample of papers/why-neurons-rest (scripts/pythia_tokens.py, #6935)."""

import io
import sys
from pathlib import Path

import numpy as np
import torch
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import pythia_tokens  # noqa: E402
from pythia_tokens import chunks, main, read_ids, without  # noqa: E402


def test_chunks_cover_the_ids_in_order():
    """The chunks join back to the ids, and only the last one is short."""
    ids = np.arange(7)
    parts = list(chunks(ids, 3))
    assert [len(p) for p in parts] == [3, 3, 1]
    assert np.concatenate(parts).tolist() == ids.tolist()


def test_without_cuts_every_copy_of_a_held_out_document():
    """Each copy of a document is cut, and the count says how many."""
    text, cuts = without("aDOCbDOCc", ["DOC"])
    assert (text, cuts) == ("abc", 2)


def test_without_cuts_to_the_end_when_the_document_runs_past_it(monkeypatch):
    """A document whose start is at the end of the text is cut to the end."""
    monkeypatch.setattr(pythia_tokens, "PREFIX", 5)
    assert without("keep DOC", ["DOCUMENT-START-AND-MORE"]) == ("keep DOC", 0)
    assert without("keep DOCUMENT-ST", ["DOCUMENT-START-AND-MORE"]) == ("keep ", 1)


def test_without_matches_on_the_prefix_only(monkeypatch):
    """Only the first PREFIX characters must match, then the document's length is cut."""
    monkeypatch.setattr(pythia_tokens, "PREFIX", 3)
    assert without("xxABCzzzyy", ["ABCdef"]) == ("xxyy", 1)


def test_without_skips_an_empty_document():
    """An empty held-out document matches nothing, so it never cuts the whole text."""
    assert without("some text", [""]) == ("some text", 0)


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_read_ids_reads_by_range_in_chunks(monkeypatch):
    """The ranges cover the first count ids, in requests of CHUNK_BYTES."""
    data = np.arange(10, dtype=np.uint16).tobytes()
    ranges = []

    def urlopen(request, timeout):
        start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        ranges.append((start, end))
        return _Response(data[start : end + 1])

    monkeypatch.setattr(pythia_tokens, "CHUNK_BYTES", 4)
    monkeypatch.setattr(pythia_tokens.urllib.request, "urlopen", urlopen)
    ids = read_ids(5, "https://example.invalid/shard.bin")
    assert ids.tolist() == [0, 1, 2, 3, 4]
    assert ranges == [(0, 3), (4, 7), (8, 9)]


class _Tokenizer:
    """Decodes ids as letters and encodes letters as ids, one per character."""

    eos_token_id = 99

    def decode(self, ids):
        return "".join(chr(ord("a") + i) for i in ids)

    def __call__(self, text, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [ord(c) - ord("a") for c in text]}


def test_main_cuts_held_out_text_and_ends_with_eos(monkeypatch, tmp_path):
    """The saved vector is the GPT-2 ids of the text without held-out documents, then EOS."""
    import huggingface_hub
    import train_resting  # noqa: F401  first, as in main: it loads the CLI in its order
    import transformers

    import openinterpretability.cli_evidence as cli_evidence

    monkeypatch.setattr(pythia_tokens, "CHUNK_IDS", 4)
    monkeypatch.setattr(pythia_tokens, "read_ids", lambda count, url: np.arange(count) % 6)
    monkeypatch.setattr(huggingface_hub, "hf_hub_url", lambda *a, **k: "https://example.invalid")
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda name: _Tokenizer())
    )
    monkeypatch.setattr(cli_evidence, "read_corpus", lambda path: (["cd"],))
    out = tmp_path / "tokens.pt"
    result = CliRunner().invoke(main, ["--tokens", "8", "--out", str(out)])
    assert result.exit_code == 0, result.output
    # ids 0..5, 0, 1 decode in chunks of 4 as "abcd" and "efab"; "cd" is cut from the first.
    assert torch.load(out).tolist() == [0, 1, 4, 5, 0, 1, 99]
    assert "1 held-out cuts" in result.output
