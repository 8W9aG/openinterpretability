import pytest
import torch

from openinterp.core.schemas import HeadIdentifier, NeuronIdentifier
from openinterp.harness.hooks import NeuronHookRunner
from openinterp.harness.patchscope import patchscope, write_direction
from openinterp.harness.units import HeadUnit, NeuronUnit
from tests.conftest import N_LAYERS, FakeTokenizer

PROMPTS = {"short": "ab{x}c", "long": "hello {x} means"}


@pytest.fixture
def runner(tiny_gpt2):
    return NeuronHookRunner(tiny_gpt2, FakeTokenizer())


def greedy(runner, ids, steps):
    seq = torch.tensor([ids])
    for _ in range(steps):
        seq = torch.cat([seq, runner.forward({"input_ids": seq})[:, -1:].argmax(-1)], dim=1)
    return runner.tokenizer.decode(seq[0, len(ids) :].tolist())


def test_readings_cover_prompts_layers_and_scales(toy_runner):
    direction = toy_runner.neuron_output_direction(N_LAYERS - 1, 3)
    readout = patchscope(toy_runner, N_LAYERS - 1, direction, prompts=PROMPTS, early_layer=1)
    keys = [(r.prompt, r.layer, r.scale) for r in readout.readings]
    assert keys == [(p, layer, s) for p in PROMPTS for layer in (1, 2) for s in (1.0, 2.0, 4.0)]
    assert all(r.text and r.control_text for r in readout.readings)
    assert readout.prompts == PROMPTS and readout.placeholder == "?"
    again = patchscope(toy_runner, N_LAYERS - 1, direction, prompts=PROMPTS, early_layer=1)
    assert again == readout  # greedy decoding and a seeded control
    early = patchscope(toy_runner, 0, direction, prompts=PROMPTS)
    assert {r.layer for r in early.readings} == {0}  # no earlier layer than the unit's own


def test_injecting_the_residual_it_replaces_changes_nothing(runner):
    """Replacing the placeholder's residual with itself, at its own norm, must leave the
    continuation exactly as an unpatched model writes it."""
    ids = [runner.tokenizer._id(c) for c in "ab?c"]
    captured = []
    handle = runner.layers[0].register_forward_hook(
        lambda _m, _a, out: captured.append((out[0] if isinstance(out, tuple) else out)[0, 2])
    )
    with torch.no_grad():
        runner.forward({"input_ids": torch.tensor([ids])})
    handle.remove()
    readout = patchscope(
        runner, 0, captured[0], prompts={"p": "ab{x}c"}, scales=(1.0,), max_new_tokens=6
    )
    assert readout.readings[0].text == greedy(runner, ids, 6)


def test_the_next_block_reads_the_scaled_direction_at_the_placeholder(toy_runner):
    ids = [toy_runner.tokenizer._id(c) for c in "ab?c"]
    reads = []
    handle = toy_runner.layers[0].register_forward_hook(
        lambda _m, _a, out: reads.append(out[0, 2].clone())
    )
    with torch.no_grad():
        toy_runner.forward({"input_ids": torch.tensor([ids])})
    handle.remove()
    norm = float(reads[0].norm())
    handle = toy_runner.layers[1].register_forward_pre_hook(
        lambda _m, args: reads.append(args[0][:, 2].clone())
    )
    direction = toy_runner.neuron_output_direction(0, 5)
    try:
        patchscope(toy_runner, 0, direction, prompts={"p": "ab{x}c"}, scales=(3.0,),
                   max_new_tokens=1)  # fmt: skip
    finally:
        handle.remove()
    pole, control = reads[-1]
    torch.testing.assert_close(pole, direction / direction.norm() * norm * 3.0)
    torch.testing.assert_close(control.norm(), torch.tensor(norm * 3.0))


def test_write_direction_is_the_signed_w_out_and_heads_have_none(toy_runner):
    def ident(pole):
        return NeuronIdentifier(model_name="toy", layer=1, neuron_idx=4, pole=pole)

    w = toy_runner.neuron_output_direction(1, 4)
    assert torch.equal(write_direction(toy_runner, NeuronUnit(ident("positive"))), w)
    assert torch.equal(write_direction(toy_runner, NeuronUnit(ident("negative"))), -w)
    head = HeadUnit(HeadIdentifier(model_name="toy", layer=1, head_idx=0), (0.0,))
    assert write_direction(toy_runner, head) is None


def test_the_placeholder_must_be_one_token(toy_runner):
    direction = toy_runner.neuron_output_direction(0, 0)
    with pytest.raises(ValueError, match="one token"):
        patchscope(toy_runner, 0, direction, prompts=PROMPTS, placeholder="??")


def test_readout_rendered_on_both_pages(tmp_path, record):
    from openinterp.builder.compiler import compile_site
    from openinterp.builder.wiki import write_experiment_page
    from openinterp.core.patchscope import PatchscopeReading, PatchscopeReadout

    readout = PatchscopeReadout(
        prompts={"category": "banana: fruit\n{x}:"},
        placeholder="?",
        max_new_tokens=12,
        control_seed=0,
        readings=[
            PatchscopeReading(prompt="category", layer=2, scale=s, text=t, control_text=c)
            for s, t, c in [(2.0, "crop", "emotion"), (1.0, "a|b", "period")]
        ],
    )
    sub = record.submission.model_copy(update={"patchscope": readout})
    rec = record.model_copy(update={"submission": sub})
    md = write_experiment_page(rec, tmp_path / "wiki").read_text()
    assert "### Patchscope" in md and "| Prompt | Layer | | ×1 | ×2 |" in md
    assert "| category | 2 | unit | a\\|b | crop |" in md  # by scale, cells escaped
    assert "| | | control | period | emotion |" in md
    assert "category `banana: fruit ⏎ ?:`" in md
    assert "### Patchscope" not in write_experiment_page(record, tmp_path / "old").read_text()

    compile_site([rec], tmp_path / "site")
    (page,) = (tmp_path / "site" / "models").rglob("N117.html")
    html = page.read_text()
    assert "Patchscope" in html and ">crop</td>" in html and ">emotion</td>" in html
