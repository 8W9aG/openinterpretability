# Writing a post

The site's Posts section ([open-interpretability.com/posts](https://open-interpretability.com/posts/index.html)) holds short pieces. A post shows how to use a tool, or says what the project is doing. A full study with a method and results is a paper (`papers/`, the `write-paper` skill), not a post.

## Layout

```text
posts/
├── README.md        the list of posts, with their dates
└── <slug>/
    ├── README.md    the post
    └── media/       its figures, if it has any
```

A post has the same form as a paper, and the same code builds it (`openinterpretability/builder/research.py`). So the same rules apply:

- **The title.** `README.md` starts with a level 1 heading.
- **The summary.** The first paragraph after the title is the summary. The index and the feed show it, so it must make sense alone.
- **The list.** Add a line to `posts/README.md` in the form `- YYYY-MM-DD: [Title](slug/README.md)`. The date is the day the post goes out. The index and the feed put the newest post first, and the feed takes its dates from this list.
- **Links.** A link to another post's `README.md` goes to that post's page. A link to any other file in the repository goes to the file on GitHub.
- **Media.** The site copies `media/`. Link a figure as `media/<name>`. The first image in the post is its picture on the index.
- **Maths.** `$…$` and `$$…$$` render as they do in a paper.
- **Personal data.** The build removes personal data and taken-down text from the post and its text files, as it does on every page.

A post has no fixed sections, no `results.jsonl` and no PDF.

## Drafts

To merge a post before it goes out, mark it as a draft in `posts/README.md`. Write `Draft` in place of the date: `- Draft: [Title](slug/README.md)`.

- **Where it builds.** A draft builds at `posts/drafts/<slug>/`, with its media and a "Draft" badge. Local builds and deploys do the same, so you can read it on the real site.
- **Unlisted.** No index, feed or sitemap lists a draft, search doesn't find it, and no page links to it. Its page tells search engines not to index it.
- **Public by its URL.** Anyone with the link can read a draft. So a draft must not hold anything secret.
- **Publish it.** Change `Draft` to the day it goes out. The post then builds at `posts/<slug>/`, and the draft's URL stops working.

## Output that a post shows

When a post shows the output of a command, run the command and paste its real output. Don't type output by hand, and don't change it. Say which model and which version ran it.

When a post names a command's options, a test checks them against the command's `--help`. `tests/test_posts.py` does this for the post on `explain`. Add the same check for a post on another command, so that the post fails when an option changes.

## Check it

```bash
make build                        # writes dist/posts/
uv run pytest tests/test_posts.py
make lint vale
```

Open `dist/posts/index.html` and the post's page, in light and dark. The feed is `dist/posts/posts.xml`.
