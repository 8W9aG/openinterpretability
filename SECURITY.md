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
2. In a commit to `main`, add `valid-before="YYYYMMDDHHMMSSZ"` (UTC) to the key's line in `.github/allowed_signers`, at the start of the exposure. Don't delete the line. CI checks each signature at the time it landed on `main` ([docs/PROTOCOL.md](docs/PROTOCOL.md) §5). So signatures that landed before the exposure stay trusted, and the other signatures by the key vouch for nothing. CI reads the file from the base branch, so the change applies to every PR after the commit.
3. List the records that the key signed in its exposure: `openinterp signatures`. Reproduce them: `openinterp reproduce --include-signed $(openinterp signatures --paths-only)`. Until then, the nightly run reproduces each one when its rotation gets to it, and does not skip it.
4. Deal with each record that doesn't reproduce in a PR that a maintainer reviews, and list them in the advisory.
   If the key signed `wiki/errata.jsonl` in its exposure, `openinterp check` now fails on `main`. Read every erratum in the file again, then sign the file with a trusted key ([docs/PROTOCOL.md](docs/PROTOCOL.md) §5).
5. If the owner still contributes, they make a new key and add it to `.github/allowed_signers` in its own commit to `main` (see [Rotating a signing key](#rotating-a-signing-key)).

If you delete a key's line, every signature by the key fails, including the signatures from before the exposure.

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

## What not to report here

A hypothesis you think is wrong, or probes you think are too easy, is a scientific disagreement, not a security problem. Do it in public, so that it is part of the record: run a re-review or a revision (`openinterp next`), or open an issue.
