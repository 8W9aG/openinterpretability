from openinterpretability.core import pause


def test_pause_file_defaults_to_the_cache_folder(monkeypatch, tmp_path):
    monkeypatch.delenv(pause.ENV, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert pause.pause_file() == tmp_path / "openinterpretability" / "pause"
