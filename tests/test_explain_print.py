"""What `explain` prints for the units of an answer (explain_print.py)."""

from openinterpretability.explain import Explanation, PoleAttribution
from openinterpretability.explain_groups import GroupAttribution
from openinterpretability.explain_print import print_result


def test_a_hub_and_the_groups_are_printed(capsys):
    hub = {"hub": "L1-N2", "href": "models/m/hubs/L1-N2.html", "active_share": 0.6,
           "default_on": True}  # fmt: skip
    pole = PoleAttribution(1, 2, "positive", 0.5, 1.0, hub=hub)
    groups = [
        GroupAttribution("superposition", "L1/Sx", 3, 0.4, "VERIFIED", "reads b", -0.2),
        GroupAttribution("circuit", "C4", 2, -0.1),
    ]
    result = Explanation("a b", -1, " c", 2.0, [pole], 1.0, 0.0, groups=groups)
    print_result(result)
    out = capsys.readouterr().out
    assert "hub L1-N2 (" in out and "default-on, active on 60% of its scan" in out
    assert "Superpositions and circuits (they overlap with the units above):" in out
    assert "L1/Sx (superposition, 3 members)" in out and "-0.20  VERIFIED" in out
    assert "reads b" in out and "C4 (circuit, 2 members)" in out
