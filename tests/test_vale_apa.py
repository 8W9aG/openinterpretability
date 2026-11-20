"""The APA rules of Vale catch each fault and pass the good form (.vale/styles/APA, #3931)."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "vale"
RULES = {"APA.EtAl", "APA.CitationComma", "APA.Ampersand", "APA.Latin", "APA.Percent"}

pytestmark = pytest.mark.skipif(shutil.which("vale") is None, reason="needs vale (vale.sh)")


def _alerts(name: str) -> list[dict]:
    out = subprocess.run(["vale", "--output=JSON", str(FIXTURES / name)],
                         capture_output=True, text=True, check=False).stdout  # fmt: skip
    return [a for alerts in json.loads(out).values() for a in alerts if a["Check"] in RULES]


def test_each_rule_catches_its_fault():
    found = _alerts("apa_bad.py")
    assert {a["Check"] for a in found} == RULES
    assert len(found) == 8  # one per fault line of the fixture
    assert all(a["Severity"] == "error" for a in found)


def test_the_good_forms_pass():
    assert _alerts("apa_good.py") == []


def test_every_rule_is_tested():
    styles = Path(__file__).parents[1] / ".vale" / "styles" / "APA"
    assert {f"APA.{p.stem}" for p in styles.glob("*.yml")} == RULES
