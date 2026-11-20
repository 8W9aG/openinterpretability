# We are on X

The project now has an account on X, at [x.com/openinterpret](https://x.com/openinterpret). It posts new papers and new posts from the site, so you can follow the work without watching the site.

## What it posts

- **Published papers.** Each paper posts once, on the day it is published: its title, the first sentence of its summary and its link. A preprint doesn't post.
- **Posts like this one.** Each new post on the site posts once: its title, its first paragraph and its link. A draft doesn't post.

At most 15 posts go out in a day. The site's own feeds carry the same entries, if you prefer a feed reader: `research/published.xml` for papers and `posts/posts.xml` for posts. New VERIFIED units have their own feed, `feed.xml`, and don't post to X.

## How it works

After each deploy, `open-interpretability social post` reads the site's feeds and posts each entry it hasn't posted before. It keeps the ids it has posted, so an entry posts once. [`docs/development.md`](../../docs/development.md) describes it under *Posts to X*.
