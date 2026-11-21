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
