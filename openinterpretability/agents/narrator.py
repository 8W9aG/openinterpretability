"""Narrator agent: writes why a model gives one answer, from the evidence alone (issue #3743).

``open-interpretability explain --level long|paragraph`` gives the narrator the level-1
evidence (:func:`openinterpretability.explain_narrate.evidence`). The narrator writes a long
form that follows the computation from the prompt to the answer, and a one-paragraph
summary of it. Each claim cites the unit that supports it. Only VERIFIED explanations are
facts, and the text must say how much of the answer is unexplained.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from openinterpretability.agents.base import LLMAgent, recorded_model
from openinterpretability.core.agent_names import EXTERNAL_PREFIX, canonical_agent
from openinterpretability.core.fence import FENCE_RULE


class Section(BaseModel):  # noqa: D101 (a docstring would join the JSON schema that the agent reads)
    title: str = Field(description="A short heading for this step of the computation.")
    text: str = Field(description="One or more paragraphs. Cite a unit for every claim.")


class Narrative(BaseModel):  # noqa: D101 (a docstring would join the JSON schema that the agent reads)
    sections: list[Section] = Field(
        min_length=1,
        description="The long form: a few sections in the order of the computation, from "
        "the prompt to the answer. The last section says what is not known yet.",
    )
    paragraph: str = Field(
        description="One paragraph that shortens the long form. It keeps the citations and "
        "the explained and unexplained shares."
    )


SYSTEM_PROMPT = f"""You explain why a language model gives one answer to one prompt, for a \
public registry of audited neurons, attention heads, superpositions and circuits.

You get the evidence as JSON, and nothing else. Follow these rules:

1. Use only the evidence. Do not add what you know about language models, transformers or \
the task. If the evidence does not say it, do not write it.
2. Cite a unit for every claim, in its own notation: `L3/N7+` or `L3/N7-` for a neuron \
pole, `L2/H10` for an attention head, `L20/S0123456789ab` for a superposition and `C4` for \
a circuit. Cite only units that are in the evidence.
3. A unit's explanation is a fact only when its status is VERIFIED. Every other status \
(unstudied, NEEDS_REVISION, REJECTED, stopped) means its role is not known. You may say \
what its attribution and ablation show, but call it unverified.
4. Say how much of the answer is explained and how much is not: the explained share and \
the prompt coverage, against random units matched by kind and layer. When little is \
explained, the honest text is mostly "this is not known yet". Say so plainly. When \
coverage_caution is true, never call the answer explained or understood: say that keeping \
the units brings the answer back, which shows they are sufficient on this prompt, not how \
they produce it. Use heads_only, poles_only and the minimal set to say which units matter.
5. answer_experiment is the newest claim about how the model produces this answer: a route \
from the sources through units to the answer, tested by ablation, patching and a reviewer's \
prompts. Call the answer understood only when its status is VERIFIED. Then describe the \
route in order, cite its units and give its necessity, sufficiency and patching. A VERIFIED \
claim is about this prompt only. Any other status means how the answer is produced is not \
known yet, whatever the coverage.
6. End the long form with what is not known yet, and cite the next experiments from the \
evidence that would explain it.
7. Shares are of first-order attribution on this prompt. A VERIFIED explanation describes \
its unit on a scan corpus, so it may not cover this prompt. Say so if you rely on one.

{FENCE_RULE}"""


class ReportedNarrative(Narrative):  # noqa: D101 (a docstring would join the JSON schema)
    model: str = Field(
        description="Your exact model ID, for example claude-sonnet-5-5. The record names "
        "this model as the narrator."
    )


class Narrator(LLMAgent):
    """Writes the long form and the paragraph from the evidence (module docstring)."""

    system_prompt = SYSTEM_PROMPT
    role = "narrator"

    def narrate(self, evidence: dict) -> Narrative:
        """The long form and the paragraph for ``evidence``.

        An ``external/`` narrator also reports its model ID. ``self.model`` then becomes
        ``external/<that ID>``, so the record names the model that wrote the text.
        """
        task = (
            "Explain why the model gives this answer, from this evidence only.\n\n"
            f"```json\n{json.dumps(evidence, indent=1)}\n```"
        )
        if not self.model.startswith(EXTERNAL_PREFIX):
            return self.structured(Narrative, task)
        # An external answerer reports the model that wrote the text, and the record names
        # that model, not the placeholder such as external/claude-code (#5966).
        told = self.structured(ReportedNarrative, task)
        self.model = recorded_model(EXTERNAL_PREFIX + canonical_agent(told.model)) or self.model
        return Narrative.model_validate(told.model_dump(exclude={"model"}))
