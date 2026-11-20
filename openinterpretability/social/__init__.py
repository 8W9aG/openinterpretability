"""Post new results to social networks (issue #4463).

Each new VERIFIED experiment, and each paper when it is published, gets one post that links
to its page on the site. The posts come from the site's Atom feeds: ``feed.xml`` (VERIFIED
experiments) and ``research/published.xml`` (published papers only). So a preprint or a
literature review is never posted.

- :mod:`.feed`: read an Atom feed.
- :mod:`.message`: the text of a post, cut to fit and screened as the site is.
- :mod:`.poster`: the :class:`~.poster.Poster` interface, and the X client.
- :mod:`.run`: which entries are new, the per-run and daily limits, and the state file.
"""
