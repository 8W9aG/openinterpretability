# Security policy

The registry is trusted only if each VERIFIED result comes from the protocol, measured on the real model. So a security problem here is any way to break that trust, as well as the usual software problems.

## How to report

Report privately with GitHub's private vulnerability reporting: open the repository's **Security** tab and click **Report a vulnerability**. Don't open a public issue or PR, and don't push a record that uses the problem.

Include:

- what the problem is, and which files, commands or workflows it uses
- the steps to reproduce it, or a record that shows it
- what you think the effect is, for example a false VERIFIED or a leaked key.

The maintainer replies in the private advisory. A fix that changes the decision rule is a protocol change ([docs/PROTOCOL.md](docs/PROTOCOL.md) §6) and is listed in [CHANGELOG.md](CHANGELOG.md). When the fix is public, the advisory is published with credit to you, unless you ask for no credit.

Only `main` and the latest release get fixes.

## What to report

### A leaked or lost signing key

A record signed by a key in `.github/allowed_signers` skips CI reproduction ([docs/development.md](docs/development.md#pull-requests), [docs/PROTOCOL.md](docs/PROTOCOL.md) §5). So a person who has a listed private key can merge records that were never measured. Report it at once if:

- you find a listed private key in public, for example in a commit, a log or a paste
- you lose a key that is listed with your name, or you think someone else has it.

If the key is yours, also stop using it: set `OPENINTERP_AUTO_SIGN=0`, or move `~/.ssh/openinterp_signing` away.

What the maintainer does:

1. Find the start of the exposure: the earliest time that someone else could have had the key. If you don't know it, use the key's `valid-after`. Then no signature by the key is trusted.
2. In a commit to `main`, add `valid-before="YYYYMMDDHHMMSSZ"` (UTC) to the key's line in `.github/allowed_signers`, at the start of the exposure. Don't remove the line. CI checks each signature at the time it landed on `main` ([docs/PROTOCOL.md](docs/PROTOCOL.md) §5). So signatures that landed before the exposure stay trusted, and the other signatures by the key vouch for nothing. CI reads the file from the base branch, so the change applies to every PR after the commit.
3. List the records that the key signed in its exposure: `openinterp signatures`. Reproduce them: `openinterp reproduce --include-signed $(openinterp signatures --paths-only)`. Until then, the nightly run reproduces each one when its rotation gets to it, and does not skip it.
4. Deal with each record that doesn't reproduce in a PR that a maintainer reviews, and list them in the advisory.
   If the key signed `wiki/errata.jsonl` in its exposure, `openinterp check` now fails on `main`. Read every erratum in the file again, then sign the file with a trusted key ([docs/PROTOCOL.md](docs/PROTOCOL.md) §5).
5. If the owner still contributes, they make a new key and add it to `.github/allowed_signers` in its own commit to `main` (see [Rotating a signing key](#rotating-a-signing-key)).

If you remove a key's line, every signature by the key fails, including the signatures from before the exposure.

### Rotating a signing key

Each line in `.github/allowed_signers` has `valid-after`, the UTC day that the key was added. A key has no expiry until it gets `valid-before`. Rotate a key when you stop using its machine, or when you think the key is old:

1. Make a new key (see [docs/development.md](docs/development.md#pull-requests)).
2. In one commit to `main`, add the new key's line with `valid-after` set to today, and add `valid-before` set to the current time to the old key's line.

The old key's signatures that landed before its `valid-before` stay trusted, so no record is reproduced. After `valid-before`, the old key signs nothing: `openinterp sign` refuses, and automatic signing stops with a warning. You can also give a new key a `valid-before` in the future, so that it expires. Rotate it before that date, or CI reproduces your records.

### A way to get a false VERIFIED past CI

Report any way to merge a record whose verdict doesn't follow from its probes and the model. Examples:

- a record that passes `openinterp check` and `openinterp reproduce`, but its numbers don't come from its model commit
- a gate that a record can pass without the evidence the gate is for, for example by a change to its stored policy
- a way past a CI rule, for example write-once records or pages that change only in Discussion
- a way to make CI trust a key that is not in the base branch's `.github/allowed_signers`, or skip a reproduction that it must run
- a PR that changes files outside `wiki/` but merges automatically (`.github/workflows/auto-merge.yml`).

### Other software problems

Also report these:

- code that runs from a record, a corpus, a model or a dictionary file
- secrets or deploy credentials that a workflow can expose
- script injection on the site from the text of a record or a page.

## A request to remove text

Records quote the corpora that their scans read: top examples, scored windows and the other pole's examples. If text that you wrote, or that names you, appears in the registry and you want it removed, report it privately in the same way. Don't quote it in a public issue. Send the passage as it appears, and a link to a page that shows it.

Corpora are never screened (`docs/PROTOCOL.md` §4), so this is the only way that a passage leaves what the project publishes. Records stay signed and write-once, because CI reproduces them from their text. So a takedown hides the passage. It doesn't rewrite the records.

What the maintainer does:

1. Save the passage to a file outside the repository, and run `uv run python scripts/takedown.py --passage <file> --id <yyyy-mm-dd-x> --reason "<a few words, no personal data>"`. It prints how many records and pages hold the passage. It appends a takedown to `wiki/takedowns.jsonl`, and removes the passage from the experiment pages above their Discussion sections.
2. Sign the file: `openinterp sign wiki/takedowns.jsonl`. Open a pull request with the file, its signature and the pages. Don't name the requester or quote the passage in the pull request.
3. After it merges, build and deploy the site. Reply in the advisory.

The takedown file holds no text. It names a passage by a hash of each run of 8 words, lowercased (`openinterp/core/takedowns.py`). So it finds the passage in every record and field, even where punctuation or case differs. A passage needs at least 8 words.

What changes:

- **The site and `registry.parquet`.** Each build replaces the passage with `[removed]` in text, and with `█` in each token that it covers, as it does for personal data (`openinterp/builder/pii.py`). The same applies to the Neuronpedia export.
- **The experiment pages in `wiki/`.** `openinterp check` lets a page change outside its Discussion section only by this removal, and only when an allowed key signed the takedown file. A branch can't add a takedown of its own.

What doesn't change:

- **The record files and their signatures.** They still hold the passage, so `openinterp reproduce` still works.
- **Git history.** Earlier commits still hold the passage.
- **What others already have.** Clones, forks, and the Zenodo archives of earlier releases.

### When the text must leave the repository

Do this only if the law requires it, for example under a court order or a valid data-protection request that the takedown doesn't satisfy. It can't be undone, and the maintainer decides.

1. Do steps 1–3 above first, so that the site stops showing the passage at once.
2. In a pull request that a maintainer merges by hand, remove the records and pages that hold the passage, and their signatures. `openinterp check` reports the removed records as write-once breaks, so it can't merge by itself. Take the removed records out of every errata rule and list that covers them.
3. Rewrite history with `git filter-repo`, to remove the files from every commit. Then force-push every branch and tag. Every clone and open pull request must start again from the new history, and commit links from before the rewrite stop working.
4. Ask GitHub Support to remove cached views and pull-request references that still hold the text.
5. Ask Zenodo to restrict the release versions that hold the passage. They are archives, so Zenodo decides.
6. Say in the advisory what can't be recalled: forks and clones that other people have.

## What not to report here

A hypothesis you think is wrong, or probes you think are too easy, is a scientific disagreement, not a security problem. Do it in public, so that it is part of the record: run a re-review or a revision (`openinterp next`), or open an issue.
