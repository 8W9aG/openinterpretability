"""Subject–verb agreement prompts: the templates, nouns and pairs (issues #3225 and #3686).

``scripts/sva_circuit.py`` traces a circuit on these prompts. The ``grammatical-number``
finder in :mod:`openinterp.superposition_finders` probes them. Each prompt ends where
the verb comes next.
"""

from __future__ import annotations

import random

# Nouns whose singular and plural are one token each in Qwen2.5's tokenizer.
NOUNS = [
    *("author", "pilot", "surgeon", "farmer", "manager", "customer", "officer", "teacher"),
    *("senator", "guard", "doctor", "student", "dancer", "lawyer", "writer", "painter"),
    *("driver", "player", "worker", "singer", "judge", "nurse", "minister", "boy", "girl"),
    *("king", "friend", "parent", "dog", "cat"),
]
IRREGULAR = {"child": "children", "woman": "women", "man": "men"}
PREPOSITIONS = ("near", "behind", "beside", "with")
VERBS = ("liked", "saw", "knew", "met")
TEMPLATES = ("simple", "across_pp", "across_rc", "within_rc")


def plural(noun: str) -> str:
    """The plural of ``noun``."""
    return IRREGULAR.get(noun, noun + "s")


def prompt(template: str, a: str, b: str, prep: str, verb: str) -> str:
    """The prompt. ``a`` is the head noun and ``b`` the second noun, both inflected."""
    return {
        "simple": f"The {a}",
        "across_pp": f"The {a} {prep} the {b}",
        "across_rc": f"The {a} that the {b} {verb}",
        "within_rc": f"The {a} that the {b}",
    }[template]


def pairs(template: str, n: int, rng: random.Random) -> list[dict]:
    """``n`` (clean, corrupted, sign) pairs. Half of them have a singular clean noun.

    The verb agrees with the head noun, except within a relative clause, where it agrees
    with the clause's subject. Only that noun's number differs between the two prompts.
    The other noun's number is drawn at random, and is the same in both.
    """
    nouns = [*NOUNS, *IRREGULAR]
    out = []
    for i in range(n):
        a, b = rng.sample(nouns, 2)
        prep, verb = rng.choice(PREPOSITIONS), rng.choice(VERBS)
        singular = i % 2 == 0
        agree, other = (b, a) if template == "within_rc" else (a, b)
        other = other if rng.random() < 0.5 else plural(other)
        clean, corrupted = (agree, plural(agree)) if singular else (plural(agree), agree)

        def text(form, other=other, prep=prep, verb=verb):
            head, second = (other, form) if template == "within_rc" else (form, other)
            return prompt(template, head, second, prep, verb)

        out.append({"clean": text(clean), "corrupted": text(corrupted),
                    "sign": 1.0 if singular else -1.0})  # fmt: skip
    return out
