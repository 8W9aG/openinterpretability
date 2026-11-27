"""The training conditions of scripts/train_resting.py, and the sizes they share.

Each condition changes one thing about how a small GPT-2 trains. The Pythia conditions
can change several. The paper is papers/why-neurons-rest/README.md.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from figure_svg import LEAKY_SLOPE

N_LAYER = 4
N_EMBD = 256
N_HEAD = 8
BATCH = 32
LEARNING_RATE = 3e-4
SGD_MOMENTUM = 0.9
ADAM_BETAS = (0.9, 0.999)  # torch's default
# Pythia's training setup (Biderman et al., 2023) uses AdamW at 1e-3, betas (0.9, 0.95)
# and weight decay 0.01. It has no dropout and has parallel attention. It warms up over 1%
# of 143,000 steps.
# Its warm-up is 1,430 of the 4,000 steps that the checkpoint test reads. 715 keeps that
# share in a 2,000-step run.
PYTHIA_LEARNING_RATE = 1e-3
PYTHIA_BETAS = (0.9, 0.95)
PYTHIA_WEIGHT_DECAY = 0.01
PYTHIA_WARMUP = 715
PYTHIA_BATCH_FACTOR = 4  # Pythia's batch is 500 times ours. 4 times fits the time we have.
# Gradient accumulation reaches 32 times in about an hour a run on one GPU (#6499).
PYTHIA_FULL_BATCH_FACTOR = 32
TRAIN_WINDOW = 128  # tokens in each training window. A test checks it is default_on_poles'.
PYTHIA_WINDOW = 2048  # Pythia's training window
# Pythia-70m is 512 wide, with 6 layers and 8 heads (Biderman et al., 2023).
PYTHIA_70M_LAYERS = 6
PYTHIA_70M_WIDTH = 512
SGD_LEARNING_RATE = 0.1  # the best of 0.01, 0.03, 0.1 and 0.3 over 500 steps that trained stably


@dataclass(frozen=True)
class Condition:
    """One way to train: the baseline, with the fields that the condition changes."""

    activation: str = "gelu"
    dropout: float = 0.1
    weight_decay: float = 0.0
    mlp_ratio: int = 4
    fixed_rest: float | None = None  # h is centred per neuron, then moved to this value
    learned_rest: bool = False  # h is centred per neuron, then moved to a learned value
    rest_lr_factor: float = 1.0  # multiplies the learning rate of the learned value
    optimizer: str = "adamw"  # "adamw", or "sgd" with momentum SGD_MOMENTUM
    learning_rate: float = LEARNING_RATE
    leaky_slope: float = LEAKY_SLOPE  # the slope of leaky ReLU below 0
    output_shift: float = 0.0  # a constant added to the activation's output
    centre_output: bool = False  # the activation's output is centred per neuron over each batch
    # The LayerNorms whose input is centred over the batch's tokens. It is "blocks", "final",
    # "all" (no norm sees a constant offset), or a list such as "final,L2.ln_1" (#6465).
    # See resting_modules.centred_norm_names.
    centre_norms: str = ""
    batch: int = BATCH  # windows in each step
    warmup_steps: int = 0  # the learning rate grows linearly over these steps, then holds
    adam_betas: tuple[float, float] = ADAM_BETAS
    parallel: bool = False  # attention and the MLP read the same input, as in GPT-NeoX
    n_layer: int = N_LAYER
    n_embd: int = N_EMBD  # the MLP is mlp_ratio times as wide
    window: int = TRAIN_WINDOW  # tokens in each training window, and the model's positions


# The mean output of ReLU and of GELU at init, where h has an SD of about 0.31. For ReLU it
# is 0.31/√(2π), and for GELU about 0.31² × φ(0). The shifts swap the two means. They test
# whether the mean output sets the sign of the output's push.
INIT_MEAN_RELU = 0.124
INIT_MEAN_GELU = 0.038

BASE = Condition()
CONDITIONS = {
    "gelu": BASE,
    "relu": replace(BASE, activation="relu"),
    "tanh": replace(BASE, activation="tanh"),
    "mirrored_gelu": replace(BASE, activation="mirrored_gelu"),
    "dropout_0": replace(BASE, dropout=0.0),
    "dropout_0.3": replace(BASE, dropout=0.3),
    # With no dropout, a constant offset in the MLP's output stays constant on each token.
    "relu_dropout_0": replace(BASE, activation="relu", dropout=0.0),
    "weight_decay_0.1": replace(BASE, weight_decay=0.1),
    "mlp_1x": replace(BASE, mlp_ratio=1),
    "mlp_16x": replace(BASE, mlp_ratio=16),
    "rest_0": replace(BASE, fixed_rest=0.0),
    "rest_-1": replace(BASE, fixed_rest=-1.0),
    "rest_learned": replace(BASE, learned_rest=True),
    "rest_learned_fast": replace(BASE, learned_rest=True, rest_lr_factor=30.0),
    "softplus": replace(BASE, activation="softplus"),
    "leaky_relu": replace(BASE, activation="leaky_relu"),
    "leaky_relu_0.01": replace(BASE, activation="leaky_relu", leaky_slope=0.01),
    "silu": replace(BASE, activation="silu"),
    "lr_1e-4": replace(BASE, learning_rate=1e-4),
    "lr_1e-3": replace(BASE, learning_rate=1e-3),
    "relu_lr_1e-4": replace(BASE, activation="relu", learning_rate=1e-4),
    "relu_lr_1e-3": replace(BASE, activation="relu", learning_rate=1e-3),
    # rest_0 with another activation: the push on the rest when every activation rests at 0.
    "relu_rest_0": replace(BASE, activation="relu", fixed_rest=0.0),
    "softplus_rest_0": replace(BASE, activation="softplus", fixed_rest=0.0),
    "leaky_relu_rest_0": replace(BASE, activation="leaky_relu", fixed_rest=0.0),
    "silu_rest_0": replace(BASE, activation="silu", fixed_rest=0.0),
    "tanh_rest_0": replace(BASE, activation="tanh", fixed_rest=0.0),
    # Fixed rests below 0: the push on the rest at each of several depths.
    "relu_rest_-0.5": replace(BASE, activation="relu", fixed_rest=-0.5),
    "relu_rest_-1": replace(BASE, activation="relu", fixed_rest=-1.0),
    "relu_rest_-2": replace(BASE, activation="relu", fixed_rest=-2.0),
    "relu_rest_-4": replace(BASE, activation="relu", fixed_rest=-4.0),
    "softplus_rest_-1": replace(BASE, activation="softplus", fixed_rest=-1.0),
    "softplus_rest_-2": replace(BASE, activation="softplus", fixed_rest=-2.0),
    "softplus_rest_-4": replace(BASE, activation="softplus", fixed_rest=-4.0),
    "rest_-0.5": replace(BASE, fixed_rest=-0.5),
    "leaky_relu_rest_-1": replace(BASE, activation="leaky_relu", fixed_rest=-1.0),
    "silu_rest_-1": replace(BASE, activation="silu", fixed_rest=-1.0),
    # Centring the output removes the output's push, so only the spread term pushes h.
    "gelu_output_centred": replace(BASE, centre_output=True),
    "silu_output_centred": replace(BASE, activation="silu", centre_output=True),
    "relu_output_shift": replace(
        BASE, activation="relu", output_shift=INIT_MEAN_GELU - INIT_MEAN_RELU
    ),
    "gelu_output_shift": replace(BASE, output_shift=INIT_MEAN_RELU - INIT_MEAN_GELU),
    "relu_norms_centred": replace(BASE, activation="relu", centre_norms="blocks"),
    "relu_all_norms_centred": replace(BASE, activation="relu", centre_norms="all"),
    # Every norm centred and no dropout: a constant offset in the residual stream then has no
    # route to the loss.
    "relu_norms_centred_dropout_0": replace(
        BASE, activation="relu", centre_norms="blocks", dropout=0.0
    ),
    "relu_all_norms_centred_dropout_0": replace(
        BASE, activation="relu", centre_norms="all", dropout=0.0
    ),
    # Only the block LayerNorms can see a constant offset.
    "relu_final_norm_centred_dropout_0": replace(
        BASE, activation="relu", centre_norms="final", dropout=0.0
    ),
    # Pythia's setup, one part at a time, on the no-dropout baseline, then all parts.
    "pythia_lr": replace(BASE, dropout=0.0, learning_rate=PYTHIA_LEARNING_RATE),
    "pythia_warmup": replace(
        BASE, dropout=0.0, learning_rate=PYTHIA_LEARNING_RATE, warmup_steps=PYTHIA_WARMUP
    ),
    "pythia_adam": replace(
        BASE, dropout=0.0, adam_betas=PYTHIA_BETAS, weight_decay=PYTHIA_WEIGHT_DECAY
    ),
    "pythia_parallel": replace(BASE, dropout=0.0, parallel=True),
    "pythia_batch": replace(BASE, dropout=0.0, batch=BATCH * PYTHIA_BATCH_FACTOR),
    "pythia_all": replace(
        BASE,
        dropout=0.0,
        learning_rate=PYTHIA_LEARNING_RATE,
        warmup_steps=PYTHIA_WARMUP,
        adam_betas=PYTHIA_BETAS,
        weight_decay=PYTHIA_WEIGHT_DECAY,
        parallel=True,
        batch=BATCH * PYTHIA_BATCH_FACTOR,
    ),
    # Pythia's batch and Pythia-70m's size, each on its own (#6499).
    "pythia_batch_32x": replace(BASE, dropout=0.0, batch=BATCH * PYTHIA_FULL_BATCH_FACTOR),
    "sgd": replace(BASE, optimizer="sgd", learning_rate=SGD_LEARNING_RATE),
    "relu_sgd": replace(BASE, activation="relu", optimizer="sgd", learning_rate=SGD_LEARNING_RATE),
}
CONDITIONS["pythia_size"] = replace(
    CONDITIONS["pythia_all"], n_layer=PYTHIA_70M_LAYERS, n_embd=PYTHIA_70M_WIDTH
)
# Pythia-70m's size and 32 times our batch together (#6806).
CONDITIONS["pythia_size_batch_32x"] = replace(
    CONDITIONS["pythia_size"], batch=BATCH * PYTHIA_FULL_BATCH_FACTOR
)
# Pythia-70m's size with Pythia's window, at the same tokens per step as the 32x batch (#7140).
CONDITIONS["pythia_size_batch_32x_window"] = replace(
    CONDITIONS["pythia_size_batch_32x"],
    window=PYTHIA_WINDOW,
    batch=BATCH * PYTHIA_FULL_BATCH_FACTOR * TRAIN_WINDOW // PYTHIA_WINDOW,
)
CONDITIONS |= {f"gelu_{n[5:]}": replace(c, activation="gelu")  # each ReLU norm condition (#3193)
               for n, c in CONDITIONS.items() if n[:5] == "relu_" and c.centre_norms}  # fmt: skip
CONDITIONS |= {f"gelu_final_L{i}_ln{j}_dropout_0":  # the final norm and one block norm (#6465)
               replace(BASE, centre_norms=f"final,L{i}.ln_{j}", dropout=0.0)
               for i in range(N_LAYER) for j in (1, 2)}  # fmt: skip
CONDITIONS |= {f"gelu_final_all_ln{j}_dropout_0":  # the final norm and every ln_j (#6785)
               replace(BASE, dropout=0.0,
                       centre_norms=",".join(["final", *(f"L{i}.ln_{j}" for i in range(N_LAYER))]))
               for j in (1, 2)}  # fmt: skip
