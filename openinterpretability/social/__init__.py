"""Post new results to social networks (issue #4463).

Each paper when it is published, and each site post, gets one post that links to its page on
the site (#5533). The posts come from the site's Atom feeds: ``research/published.xml``
(published papers only) and ``posts/posts.xml`` (site posts, never drafts). So a preprint, a
draft, a literature review or a VERIFIED unit is never posted.

- :mod:`.feed`: read an Atom feed.
- :mod:`.message`: the text of a post, cut to fit and screened as the site is.
- :mod:`.poster`: the :class:`~.poster.Poster` interface, and the X client.
- :mod:`.run`: which entries are new, and the state file.
"""
