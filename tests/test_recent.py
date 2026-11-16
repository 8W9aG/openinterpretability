"""A model's page lists its most recent experiments (openinterp/builder/recent.py, #3276)."""

import re

from openinterp.builder.compiler import compile_site
from openinterp.builder.recent import RECENT_ON_MODEL
from tests.test_compiler import MODEL, experiment


def test_a_model_page_lists_its_newest_experiments_first(tmp_path, record):
    records = [
        experiment(record, idx=i, status="VERIFIED" if i % 2 else "REJECTED", counts=(10, 0, 5, 0),
                   hours=i, text=f"Explanation {i}.")
        for i in range(RECENT_ON_MODEL + 3)
    ]  # fmt: skip
    compile_site(records, tmp_path)
    page = (tmp_path / "models" / MODEL / "index.html").read_text(encoding="utf-8")
    section = page.split("Most recent experiments", 1)[1].split("</section>", 1)[0]
    shown = [int(n) for n in re.findall(r"Explanation (\d+)\.", section)]
    newest = list(range(RECENT_ON_MODEL + 2, 2, -1))
    assert shown == newest
    assert "REJECTED" in section and "VERIFIED" in section
