"""The literature map's callout cards and citations (builder/literature_map_view.py, #5798)."""

from openinterpretability.builder.literature_map_view import (
    CARDS_PER_SIDE,
    EDGE,
    HEIGHT,
    WIDTH,
    Callout,
    Point,
    callouts,
    citations,
    crosses,
)

TITLE = "A long title for one paper that wraps onto more than one line"


def _used(name, x, y, cites=1, section="Probing"):
    return Point(name, TITLE, "used", x, y, credit="A, 2026", cites=cites, section=section)


def _crossings(cards):
    same = [
        (a, b) for i, a in enumerate(cards) for b in cards[i + 1 :] if (a.x < 500) == (b.x < 500)
    ]
    return [pair for pair in same if crosses(*pair)]


def test_cards_stay_inside_the_map_and_their_lines_never_cross():
    # 40 used papers crowd the middle, as the repo's do: far more than fit on a side.
    points = [_used(f"p{i:02d}", 300 + (i * 37) % 400, 60 + (i * 53) % 600, cites=i % 7)
              for i in range(40)]  # fmt: skip
    cards = callouts(points)
    assert 0 < len(cards) <= 2 * CARDS_PER_SIDE
    assert all(c.box[1] >= EDGE and c.box[3] <= HEIGHT - EDGE for c in cards)
    assert _crossings(cards) == []


def test_the_most_cited_papers_take_the_cards_and_background_is_left_out():
    points = [_used("low", 300, 100, cites=0), _used("high", 310, 300, cites=9),
              _used("gelu", 320, 500, cites=20, section="Model architecture")]  # fmt: skip
    assert [c.point.name for c in callouts(points)] == ["low", "high"]
    many = [_used(f"p{i}", 300, 40 + 60 * i, cites=i) for i in range(CARDS_PER_SIDE + 2)]
    names = {c.point.name for c in callouts(many)}
    assert len(names) <= CARDS_PER_SIDE and "p0" not in names and "p1" not in names


def test_a_card_takes_the_side_nearer_its_dot():
    cards = callouts([_used("west", WIDTH / 2 - 1, 300), _used("east", WIDTH / 2 + 1, 300)])
    assert {c.point.name: c.x < WIDTH / 2 for c in cards} == {"west": True, "east": False}


def test_crossing_lines_swap_their_cards_first():
    # Three dots on one row. Stacked in dot order, the middle card's line crosses the third's,
    # so those two cards swap, and then no line crosses.
    points = [_used(f"p{i}", x, 40, cites=3 - i) for i, x in enumerate((220, 300, 220))]
    cards = callouts(points)
    assert [c.point.name for c in sorted(cards, key=lambda c: c.y)] == ["p0", "p2", "p1"]
    assert _crossings(cards) == []


def test_lines_that_still_cross_after_a_swap_drop_the_less_cited_card():
    points = [_used(f"p{i}", x, 40, cites=3 - i) for i, x in enumerate((480, 260, 220))]
    cards = callouts(points)
    assert sorted(c.point.name for c in cards) == ["p0", "p2"]  # p1 is the less cited of its pair
    assert _crossings(cards) == []


def test_two_lines_that_meet_only_at_an_end_do_not_cross():
    a = Callout(_used("a", 300, 100), 14, 80, ("t",))
    b = Callout(_used("b", 300, 400), 14, 380, ("t",))
    assert not crosses(a, b)


def test_citations_count_each_page_once_and_skip_the_index_and_the_page_itself(tmp_path):
    lit = tmp_path / "docs" / "literature"
    lit.mkdir(parents=True)
    (lit / "README.md").write_text("[a](a.md) [b](b.md)")
    (lit / "a.md").write_text("[a](a.md) [b](b.md#method) [b again](./b.md)")
    (lit / "b.md").write_text("[dictionaries](../dictionaries.md)")
    (tmp_path / "papers" / "p").mkdir(parents=True)
    (tmp_path / "papers" / "p" / "README.md").write_text("[a](../../docs/literature/a.md)")
    (tmp_path / "docs" / "x.md").write_text("[b](literature/b.md) [d](dictionaries.md)")
    assert citations(lit, {"a", "b"}) == {"a": 1, "b": 2}


def test_a_card_for_a_dot_at_the_bottom_moves_up_into_the_map():
    (card,) = callouts([_used("low", 300, HEIGHT - 10)])
    assert card.box[3] == HEIGHT - EDGE


def test_a_side_with_no_candidate_has_no_card():
    from openinterpretability.builder.literature_map_view import _side

    assert _side([], 14) == []


def test_a_side_drops_its_least_cited_cards_until_the_rest_fit():
    from openinterpretability.builder.literature_map_view import _side

    chosen = [_used(f"p{i}", 300, 360, cites=20 - i) for i in range(11)]  # most-cited first
    cards = _side(chosen, 14)
    assert 0 < len(cards) < 11 and cards[0].y >= EDGE and cards[-1].box[3] <= HEIGHT - EDGE
    assert {c.point.name for c in cards} == {f"p{i}" for i in range(len(cards))}
