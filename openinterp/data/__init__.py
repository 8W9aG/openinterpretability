from importlib.resources import files


def sample_corpus() -> list[str]:
    """A small, diverse built-in corpus for quick scans. Use --corpus for real audits."""
    text = files(__name__).joinpath("sample_corpus.txt").read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]
