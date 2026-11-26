---
name: single-source-of-truth
description: Keep one definition for each constant, threshold, list of names or rule, and import or read it everywhere else. Use before adding a literal (a number, a status, a model name, a path, a class name) to code, templates, docs or workflows. Also use when changing a value that may be copied elsewhere, and when reviewing a diff.
---

# Single source of truth

Every concept in this repo has one definition. Everything else imports it, reads it, or is pinned to it by a test. A value typed twice drifts: someone changes one copy, CI stays green, and the site, the docs or a rule quietly disagree.

**Read [`project.md`](project.md) too.** It lists where this project's definitions live, its examples, and its own copies that must exist. This file holds the rule and no value of a project (#6631).

## Before you add a literal

1. **Name the concept**, not the value. "The request timeout" is the concept. `30` is the value.
2. **Search for the value and the concept** across everything, not only Python:

   ```bash
   git grep -n -e '\b30\b' -e 'request_timeout' -e 'request timeout'
   ```

   Search the spellings a copy could use: the number (`0.25`, `.25`, `25%`), the identifier, and the words in prose and comments. For a list of names (statuses, steps, models), search for two of its members together.
3. **If a definition exists, use it.** If only copies exist, make one the source (see below) and point the others at it in the same change.
4. **If nothing exists**, define it once in the place that fits (`project.md` lists them) and import it.

## Where definitions live

`project.md` gives the project's table of concepts and their sources. A value used by one module only stays a module-level constant there, with a comment saying what it is. Move it to a shared place when a second module needs it, not before.

A stored record can keep the default it was made with. Read such a default from the record when one is at hand, and from the named constant otherwise. Never retype it.

## Templates

Jinja templates can't import Python, so pass values in. Never retype a Python list or map in a template.

- Site-wide values go in the environment's `globals`. Whole modules can be globals, so a template reads a module's constant with no per-constant wiring. Functions work too.
- Page-specific values go in the view the page renders from.
- Macros imported with `{% from "_macros.html" import … %}` see globals, so a macro can read a global directly.
- The stylesheet is built from the rendered pages' classes, so class names may live in Python.
- SQL in a Python string is an f-string that interpolates the constant.

## When a copy has to exist

Some copies can't import their source:

- **Workflows.** A workflow can copy a value on purpose, for example so that a fork's PR can't widen what CI downloads.
- **Docs.** Docs and skills quote numbers so a reader doesn't need the code.
- **Rendered output and fixtures** that state an expected value.

For each one:

1. Add a test that the copy agrees with the source. `project.md` names the test module, unless an existing test module owns the concept. Format the value from the source in the test, and search for that text in the copy, so the test fails when either side moves.
2. Name the source at each copy: a comment, or in the prose itself when that reads naturally. Skip it when the text already points there (an env var or config key next to the number).
3. Check the test catches drift: change the copy, see it fail, change it back.

Numbers that can't drift need no test: maths ("0.5 is chance", the median percentile), and the version a feature arrived in.

Prose that explains a value in passing in a docstring is fine when the docstring also names the source. Don't add a test for every passing mention. Add one where a reader would act on the number.

## Changing a value

1. Change the source.
2. Search for the old value and the concept's names (as above) to find copies the tests don't cover, and update them.
3. Follow `project.md` for a value that changes a decision rule.

## Reviewing a diff

Flag:

- a new literal that equals an existing constant, or a new list whose members match an existing list.
- a template `{% set … = { … } %}` map keyed by values that exist in Python.
- a number in docs or a workflow with no test pinning it and no comment naming its source.
- a source changed without its copies.

Suggest the import or global to use, not only "this is duplicated".
