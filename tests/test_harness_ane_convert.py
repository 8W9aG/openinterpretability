"""The coremltools probe when the package is not installed."""

import sys

from openinterpretability.harness.ane import convert


def test_without_coremltools_nothing_is_available_and_no_version(monkeypatch):
    # A None entry in sys.modules makes the import raise ImportError.
    monkeypatch.setitem(sys.modules, "coremltools", None)
    assert convert.available() is False
    assert convert.version() is None
