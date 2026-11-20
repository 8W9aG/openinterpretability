# We are on X

The project now has an account on X, at [x.com/openinterpret](https://x.com/openinterpret). It posts new results from the registry and the research section, so you can follow the work without watching the site.

## What it posts

- **Published papers.** Each paper posts once, on the day it is published: its title, the first sentence of its summary and its link. A preprint doesn't post.
- **VERIFIED units.** Each newly VERIFIED experiment posts its unit, its verdict and a link to its page. The post holds no text from the record, because any contributor can write a record. The explanation is on the unit's page.

At most 15 posts go out in a day. The site's own feeds carry the same entries, if you prefer a feed reader: `feed.xml` for VERIFIED units and `research/papers.xml` for papers.

## How it works

After each deploy, `open-interpretability social post` reads the site's feeds and posts each entry it hasn't posted before. It keeps the ids it has posted, so an entry posts once. [`docs/development.md`](../../docs/development.md) describes it under *Posts to X*.
