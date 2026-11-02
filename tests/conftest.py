from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from openinterp.core.schemas import (
    ActivatingExample,
    AdversarialTestResult,
    AuditRecord,
    CausalMetrics,
    CounterfactualPair,
    Histogram,
    LookaheadProfile,
    NeuronAuditSubmission,
    NeuronIdentifier,
    VerificationVerdict,
)

# Tiny models gain nothing from threads, and competing for CPU with other jobs has
# hung the suite. Covers bare `pytest` and CI too, not just `make test`.
torch.set_num_threads(1)

VOCAB, D_MODEL, D_MLP, N_LAYERS = 16, 8, 12, 3


class ToyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up_proj = nn.Linear(D_MODEL, D_MLP)
        self.act = nn.SiLU()  # signed, so both poles occur
        self.down_proj = nn.Linear(D_MLP, D_MODEL)

    def forward(self, x):
        return self.down_proj(self.act(self.up_proj(x)))


class ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = ToyMLP()

    def forward(self, x):
        return x + self.mlp(x)


class ToyModel(nn.Module):
    """Llama-shaped residual MLP stack: ``model.layers[i].mlp.down_proj``."""

    def __init__(self, n_layers: int = N_LAYERS) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(ToyBlock() for _ in range(n_layers))
        self.lm_head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        for block in self.model.layers:
            x = block(x)
        return SimpleNamespace(logits=self.lm_head(x))


@pytest.fixture
def toy_model() -> ToyModel:
    torch.manual_seed(0)
    return ToyModel().eval()


@pytest.fixture
def tiny_gpt2():
    """A randomly initialised GPT-2 built from config: real HF architecture, no download."""
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    config = GPT2Config(n_layer=2, n_embd=32, n_head=2, n_positions=64, vocab_size=VOCAB)
    return GPT2LMHeadModel(config).eval()


@pytest.fixture
def identifier() -> NeuronIdentifier:
    return NeuronIdentifier(
        model_name="Qwen/Qwen2.5-0.5B", layer=3, neuron_idx=117, pole="positive"
    )


@pytest.fixture
def submission(identifier) -> NeuronAuditSubmission:
    return NeuronAuditSubmission(
        identifier=identifier,
        natural_language_explanation="Fires on the closing quote of reported speech.",
        activating_tokens=['"', "'"],
        counterfactuals=[
            CounterfactualPair(
                base_prompt='"Hello," she said.',
                counterfactual_prompt="Hello, she said.",
                expected_delta=0.8,
                observed_delta=0.71,
            )
        ],
        lookahead=LookaheadProfile(
            is_lookahead=False, target_upcoming_tokens=[], kl_divergence=0.002
        ),
        submitting_agent_model="openai/gpt-test",
        max_activation=2.5,
        top_examples=[
            ActivatingExample(
                text='"Hi," he said.',
                tokens=['"', "Hi", ",", '"', " he", " said", "."],
                activations=[0.1, 0.0, 0.2, 2.5, 0.3, 0.0, 0.0],
                max_activation=2.5,
                max_token_index=3,
            )
        ],
        activation_histogram=Histogram(bin_edges=[0.0, 1.0, 2.0, 3.0], counts=[90, 8, 2]),
        model_num_layers=24,
    )


@pytest.fixture
def verdict() -> VerificationVerdict:
    return VerificationVerdict(
        status="VERIFIED",
        confidence_score=0.86,
        adversarial_test_results=[
            AdversarialTestResult(
                kind="hard_negative",
                prompt="The quotation marks were missing.",
                passed=True,
                expected_active=False,
                activation=0.05,
            )
        ],
        reviewer_agent_model="anthropic/claude-test",
        timestamp=datetime(2026, 10, 1, 12, 0, tzinfo=UTC),
        causal_metrics=CausalMetrics(
            sufficiency_score=0.6, necessity_score=0.4, hydra_mitigated=True
        ),
    )


@pytest.fixture
def record(submission, verdict) -> AuditRecord:
    return AuditRecord(submission=submission, verdict=verdict)


class FakeTokenizer:
    """Character-level tokenizer over the toy vocabulary, with right padding."""

    pad_token = eos_token = "\0"

    def _id(self, ch: str) -> int:
        return 1 + ord(ch) % (VOCAB - 1)

    def __call__(self, texts, return_tensors="pt", padding=True, return_offsets_mapping=False):
        if isinstance(texts, str):  # one text, as lists, with each token's character span
            ids = self.encode(texts)
            spans = [(i, i + 1) for i in range(len(ids))]
            return {"input_ids": ids, "attention_mask": [1] * len(ids), "offset_mapping": spans}
        ids = [[self._id(c) for c in t] or [0] for t in texts]
        width = max(map(len, ids))
        return {
            "input_ids": torch.tensor([row + [0] * (width - len(row)) for row in ids]),
            "attention_mask": torch.tensor(
                [[1] * len(row) + [0] * (width - len(row)) for row in ids]
            ),
        }

    def encode(self, text, add_special_tokens=True):
        return [self._id(c) for c in text]

    def decode(self, ids):
        return "".join(chr(96 + i) for i in ids)


@pytest.fixture
def toy_runner(toy_model):
    from openinterp.harness.hooks import NeuronHookRunner

    return NeuronHookRunner(toy_model, FakeTokenizer())


@pytest.fixture(autouse=True)
def _no_auto_sign(monkeypatch):
    """Never sign test records with the developer's real key (tests opt back in)."""
    monkeypatch.setenv("OPENINTERP_AUTO_SIGN", "0")


@pytest.fixture(autouse=True)
def _no_costs_ledger(monkeypatch):
    """Keep test commands out of a loop's real costs ledger (tests opt back in)."""
    from openinterp.core.costs import LEDGER_ENV

    monkeypatch.delenv(LEDGER_ENV, raising=False)


@pytest.fixture(autouse=True)
def _scan_cache(tmp_path_factory, monkeypatch):
    """Cache layer scans in a temporary directory, never the developer's real cache."""
    from openinterp.harness.layer_cache import ENV

    monkeypatch.setenv(ENV, str(tmp_path_factory.mktemp("scan-cache")))
