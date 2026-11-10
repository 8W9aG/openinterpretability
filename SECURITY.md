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

1. Remove the key's line from `.github/allowed_signers` in a commit to `main`. CI reads that file from the base branch, so the key stops counting for every PR after the commit.
2. Find the records that the key signed. Each `<id>.json.sig` under `wiki/` that verifies with an allowed-signers file that holds only that key is one (`ssh-keygen -Y find-principals`).
3. Reproduce those records: `openinterp reproduce --include-signed <paths>`. After step 1 their signatures don't verify, so the nightly run fails on them until this step is done.
4. Deal with each record that doesn't reproduce in a PR that a maintainer reviews, and list them in the advisory.
5. If the owner still contributes, they make a new key and add it to `.github/allowed_signers` in its own commit to `main`.

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
