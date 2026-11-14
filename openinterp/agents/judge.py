"""Judge agent: scores an explanation on held-out windows from the explanation alone.

Detection asks which windows activate the pole. Fuzzing asks whether the marked tokens
in each window are the ones that activate it (Paulo et al., 2025). The judge never sees
activations. A :class:`ScoringTask` holds only the windows' text and markings. The
labels are measured again when the answers are scored.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from openinterp.agents.base import LLMAgent
from openinterp.core.costs import ExperimentCosts
from openinterp.core.fence import FENCE_RULE, fence
from openinterp.core.schemas import ComponentIdentifier, DictionaryProfile, HeadProfile
from openinterp.score_controls import ControlTask

MARK_OPEN, MARK_CLOSE = "<<", ">>"


class TaskWindow(BaseModel):
    """One held-out window as the judge sees it: text, tokens and the marked tokens."""

    text: str
    tokens: list[str]
    highlighted: list[int] = Field(default_factory=list)


class ScoringTask(BaseModel):
    """What ``open-interpretability score --emit-task`` writes: everything but the labels."""

    identifier: ComponentIdentifier
    head_profile: HeadProfile | None = None  # a head's strength is defined by its mean output
    dictionary_profile: DictionaryProfile | None = None  # and a feature's by its dictionary
    # v0.8: the unit isn't measured at a window's first token (NeuronAuditSubmission)
    first_token_floor: bool | None = None
    explanation: str
    threshold: float
    window_tokens: int
    quantiles: int
    quantile_edges: list[float]
    seed: int
    corpus: str
    corpus_sha256: str | None = None
    candidate_windows: int
    active_windows: int
    detection: list[TaskWindow]
    fuzzing: list[TaskWindow]
    controls: list[ControlTask] = Field(default_factory=list)  # judged on the same windows
    fuzzing_decoys: str | None = None  # "matched-shape" since issue #3505 (ExplanationScores)
    costs: ExperimentCosts | None = None  # of drawing the windows


class DetectionAnswer(BaseModel):  # noqa: D101 (a docstring would join the JSON schema that the agent reads)
    active: list[int] = Field(
        description="Numbers of the texts in which, going by the explanation, the neuron "
        "fires on at least one token. Leave out every other text."
    )


class FuzzingAnswer(BaseModel):  # noqa: D101 (a docstring would join the JSON schema that the agent reads)
    correct: list[int] = Field(
        description="Numbers of the texts whose marked tokens are exactly where, going by the "
        "explanation, the neuron fires. Leave out texts marked wrongly."
    )


def _show(tokens: Sequence[str], marked: Sequence[int] = ()) -> str:
    hot = set(marked)
    text = "".join(f"{MARK_OPEN}{t}{MARK_CLOSE}" if i in hot else t for i, t in enumerate(tokens))
    return text.replace("\n", "⏎")


class ExplanationJudge(LLMAgent):
    """The judge: applies an explanation to windows, never seeing activations."""

    role = "judge"
    system_prompt = (
        "You test how well a short explanation of a neuron (or attention head, or dictionary "
        "latent) in a language "
        "model predicts where it fires. You see only the explanation and some short texts, "
        "never its activations. Apply the explanation literally, as written: don't guess "
        "what the neuron really does, and don't reward texts for being on a related topic "
        "if the explanation's condition isn't met. Texts are excerpts and may start or end "
        "mid-word; line breaks are shown as ⏎. " + FENCE_RULE
    )

    @staticmethod
    def detection_prompt(
        explanation: str, windows: Sequence[TaskWindow], noun: str = "neuron"
    ) -> str:
        """The detection prompt: which of ``windows`` make the unit fire."""
        texts = "\n".join(f"{n}. {_show(w.tokens)}" for n, w in enumerate(windows, 1))
        return (
            f"Explanation of the {noun}:\n\n{fence(explanation)}\n\n"
            f"Any number of the texts below, possibly none or all, make the {noun} fire on at "
            "least one token. List the numbers of the texts in which the explanation says it "
            f"fires.\n\n{fence(texts)}"
        )

    @staticmethod
    def fuzzing_prompt(
        explanation: str, windows: Sequence[TaskWindow], noun: str = "neuron"
    ) -> str:
        """The fuzzing prompt: in which ``windows`` the marked tokens are where the unit fires."""
        texts = "\n".join(
            f"{n}. {_show(w.tokens, w.highlighted)}" for n, w in enumerate(windows, 1)
        )
        return (
            f"Explanation of the {noun}:\n\n{fence(explanation)}\n\n"
            f"In each text below, some tokens are marked {MARK_OPEN}like this{MARK_CLOSE}. In "
            f"some texts the marks are exactly the tokens where the {noun} fires; in the rest "
            f"they are on other tokens, or the {noun} doesn't fire in that text at all. List "
            f"the numbers of the texts whose marks are where the explanation says the {noun} "
            f"fires.\n\n{fence(texts)}"
        )

    def detect(
        self, explanation: str, windows: Sequence[TaskWindow], noun: str = "neuron"
    ) -> DetectionAnswer:
        """Ask the judge the detection question."""
        prompt = self.detection_prompt(explanation, windows, noun)
        return self.structured(DetectionAnswer, prompt)

    def fuzz(
        self, explanation: str, windows: Sequence[TaskWindow], noun: str = "neuron"
    ) -> FuzzingAnswer:
        """Ask the judge the fuzzing question."""
        return self.structured(FuzzingAnswer, self.fuzzing_prompt(explanation, windows, noun))


def answered(numbers: Sequence[int], count: int, what: str) -> list[bool]:
    """Per-window yes/no from a judge's list of 1-based text numbers."""
    bad = sorted({n for n in numbers if not 1 <= n <= count})
    if bad:
        raise ValueError(f"{what} answer names texts {bad}, but there are only {count}")
    chosen = set(numbers)
    return [n in chosen for n in range(1, count + 1)]
