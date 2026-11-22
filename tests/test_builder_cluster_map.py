"""Cluster map badges and region outlines in the edge cases."""

from openinterpretability.builder.cluster_map import Badge, hull, place_badges, region_outline


def test_a_badge_with_no_free_spot_stays_at_its_centre():
    centre = Badge(number=1, name="a", slot=1, size=3, x=100.0, y=100.0)
    everything = [(0.0, 0.0, 1000.0, 1000.0)]  # one box that blocks the whole map
    (badge,) = place_badges([centre], everything, 1000.0, 1000.0)
    assert (badge.x, badge.y) == (100.0, 100.0)


def test_the_hull_of_two_points_is_the_points_in_order():
    assert hull([(3.0, 1.0), (1.0, 2.0), (3.0, 1.0)]) == [(1.0, 2.0), (3.0, 1.0)]


def test_points_on_one_line_get_no_outline():
    line = [(float(i), 2.0 * i) for i in range(6)]
    assert region_outline(line) == ""


def test_touching_clusters_never_share_a_family_and_clash_on_the_lightest_borders():
    from openinterpretability.builder.cluster_map import CLASH, FAMILY, SLOTS, touching_slots

    numbers = list(range(1, SLOTS + 1))
    # A ring: each cluster touches the next. The border from 1 to 2 is the heaviest.
    touches = {n: {} for n in numbers}
    for a, b in zip(numbers, numbers[1:] + numbers[:1], strict=True):
        w = 9 if (a, b) == (1, 2) else 1
        touches[a][b] = touches[b][a] = w
    found = touching_slots(numbers, touches)
    assert sorted(found.values()) == numbers  # each cluster its own colour
    borders = [(a, b) for a in numbers for b in touches[a]]
    assert all(FAMILY[found[a]] != FAMILY[found[b]] for a, b in borders)
    assert frozenset((found[1], found[2])) not in CLASH


def test_too_many_clusters_or_no_fit_falls_back_to_the_order():
    from openinterpretability.builder.cluster_map import SLOTS, colour_slots, touching_slots

    many = list(range(1, SLOTS + 2))
    assert touching_slots(many, {}) == colour_slots(many)
    # Seven clusters that all touch need seven hue families, and the palette has six.
    seven = list(range(1, 8))
    everyone = {n: {m: 1 for m in seven if m != n} for n in seven}
    assert touching_slots(seven, everyone) == colour_slots(seven)
