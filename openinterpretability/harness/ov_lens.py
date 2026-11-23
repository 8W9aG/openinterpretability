"""A head's OV circuit read through the vocabulary, from its weights alone (#5062).

The circuit ``W_U · W_OV · W_Eᵀ`` maps each token's embedding to the logits that the
head's write of it gives. ``W_OV = W_O · W_V``, with the norm before attention folded into
``W_V`` (:func:`.wiring.attention_reads`). ``W_U`` carries the final norm's gain, and its
centring for a LayerNorm, as the neuron lens does (:mod:`.jlens`). The full circuit is
vocabulary by vocabulary, so each measure here reads it through ``d_model``-sized pieces.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from openinterpretability.core.profiles import OVLens, OVMapping
from openinterpretability.harness.hooks import NeuronHookRunner
from openinterpretability.harness.wiring import attention_reads

CHUNK = 256  # tokens whose logits are held at once


def ov_circuit(runner: NeuronHookRunner, layer: int, head: int) -> torch.Tensor:
    """``W_OV = W_O · W_V`` for one head, ``[d_model, d_model]``: its write of an input.

    Args:
        runner: The model.
        layer: The head's layer.
        head: The head's index.

    Returns:
        ``W_OV``, on the CPU, in float32.

    Raises:
        ValueError: The architecture's attention weights aren't supported.
    """
    reads = attention_reads(runner, layer)
    if reads is None:
        raise ValueError("this model's attention weights aren't supported")
    w_v = reads["v"][head].float()  # [head_dim, d_model]
    return runner.head_output_weights(layer, head).float().cpu() @ w_v.cpu()


def unembedding(runner: NeuronHookRunner) -> torch.Tensor:
    """``W_U`` with the final norm's gain folded in, and centred for a LayerNorm.

    Args:
        runner: The model.

    Returns:
        ``[vocab, d_model]``, on the CPU, in float32.
    """
    unembed = runner.model.get_output_embeddings().weight.detach().float().cpu()
    norm = runner.final_norm()
    weight = getattr(norm, "weight", None)
    if weight is not None:
        unembed = unembed * weight.detach().float().cpu()
    if isinstance(norm, nn.LayerNorm):
        unembed = unembed - unembed.mean(-1, keepdim=True)
    return unembed


def copying_score(embed: torch.Tensor, ov: torch.Tensor, unembed: torch.Tensor) -> float:
    """``Σλ / Σ|λ|`` over the eigenvalues of ``W_U · W_OV · W_Eᵀ`` (Olsson et al., 2022).

    Its nonzero eigenvalues are those of ``W_Eᵀ · W_U · W_OV``, which is ``d × d``.

    Args:
        embed: ``W_E``, ``[vocab, d]``.
        ov: ``W_OV``, ``[d, d]``.
        unembed: ``W_U``, ``[vocab, d]``.

    Returns:
        A value from −1 (suppresses) to +1 (copies).

    Examples:
        >>> e = torch.eye(3)
        >>> copying_score(e, torch.eye(3), e), copying_score(e, -torch.eye(3), e)
        (1.0, -1.0)
    """
    eig = torch.linalg.eigvals(embed.T.double() @ unembed.double() @ ov.double())
    total = eig.abs().sum()
    return 0.0 if total == 0 else round(float(eig.real.sum() / total), 6)


def own_ranks(
    embed: torch.Tensor, ov: torch.Tensor, unembed: torch.Tensor, tokens: torch.Tensor
) -> torch.Tensor:
    """Each token's rank among the logits of its own OV write, 1 for the highest.

    Args:
        embed: ``W_E``, ``[vocab, d]``.
        ov: ``W_OV``, ``[d, d]``.
        unembed: ``W_U``, ``[vocab, d]``.
        tokens: The token ids to rank.

    Returns:
        One rank per token.

    Examples:
        >>> e = torch.eye(3)
        >>> own_ranks(e, torch.eye(3), e, torch.arange(3)).tolist()
        [1, 1, 1]
        >>> own_ranks(e, -torch.eye(3), e, torch.arange(3)).tolist()
        [3, 3, 3]
    """
    ranks = []
    for start in range(0, len(tokens), CHUNK):
        ids = tokens[start : start + CHUNK]
        logits = (embed[ids] @ ov.T) @ unembed.T  # [chunk, vocab]
        own = logits.gather(1, ids[:, None])
        ranks.append((logits > own).sum(1) + 1)
    return torch.cat(ranks)


def mappings(
    runner: NeuronHookRunner,
    embed: torch.Tensor,
    ov: torch.Tensor,
    unembed: torch.Tensor,
    token_ids: Sequence[int],
    k: int = 5,
) -> list[OVMapping]:
    """Each token, with the ``k`` tokens that its OV write raises most.

    Args:
        runner: The model, for its tokenizer.
        embed: ``W_E``, ``[vocab, d]``.
        ov: ``W_OV``, ``[d, d]``.
        unembed: ``W_U``, ``[vocab, d]``.
        token_ids: The tokens to map, in order. A repeated token is mapped once.
        k: How many tokens to list for each.

    Returns:
        One mapping per distinct token.
    """
    decode = runner.tokenizer.decode
    out = []
    for token in dict.fromkeys(token_ids):  # in order, once each
        logits = unembed @ (ov @ embed[token])
        top = logits.topk(k).indices.tolist()
        out.append(OVMapping(token=decode([token]), top=[decode([t]) for t in top]))
    return out


@torch.no_grad()
def ov_lens(
    runner: NeuronHookRunner,
    layer: int,
    head: int,
    attended: Sequence[int] = (),
    *,
    tokens: int = 2000,
    seed: int = 0,
    k: int = 10,
) -> OVLens:
    """Read one head's OV circuit through the vocabulary.

    Args:
        runner: The model.
        layer: The head's layer.
        head: The head's index.
        attended: Token ids to map, such as those the top examples' peaks attend to.
        tokens: How many vocabulary tokens to rank, drawn with ``seed``.
        seed: The draw's seed.
        k: The top and bottom places that count as copying and suppression.

    Returns:
        The lens.
    """
    embed = runner.model.get_input_embeddings().weight.detach().float().cpu()
    ov, unembed = ov_circuit(runner, layer, head), unembedding(runner)
    vocab = min(embed.shape[0], unembed.shape[0])
    generator = torch.Generator().manual_seed(seed)
    sample = torch.randperm(vocab, generator=generator)[: min(tokens, vocab)]
    ranks = own_ranks(embed, ov, unembed, sample)
    return OVLens(
        copying_score=copying_score(embed[:vocab], ov, unembed[:vocab]),
        copy_share=round(float((ranks <= k).float().mean()), 6),
        suppression_share=round(float((ranks > unembed.shape[0] - k).float().mean()), 6),
        median_rank=int(ranks.median()),
        tokens=len(sample),
        seed=seed,
        k=k,
        mappings=mappings(runner, embed, ov, unembed, [t for t in attended if t < vocab]),
    )
