"""Library of common neuron functions ready to drop into a graph."""

from __future__ import annotations

import math

from .neuron import NeuronDef, module_neuron, neuron
from .port import Port
from .torch_backend import (
    default_attention_config,
    default_linear_config,
    default_lm_head_config,
    default_logit_softcap_config,
    default_merge_heads_config,
    default_mlp_config,
    default_qk_gain_config,
    default_repeat_kv_config,
    default_residual_add_config,
    default_residual_mix_config,
    default_reshape_heads_config,
    default_rotary_embedding_config,
    default_rms_norm_config,
    default_scaled_dot_product_attention_config,
    default_token_embedding_config,
)


@neuron(
    inputs=[Port("x", range=(-10, 10), precision=0.001)],
    outputs=[Port("y", range=(0, 1), precision=0.001)],
)
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


@neuron(
    inputs=[Port("x", range=(-10, 10), precision=0.001)],
    outputs=[Port("y", range=(0, 10), precision=0.001)],
)
def relu(x):
    return max(0.0, x)


@neuron(
    inputs=[Port("x", range=(-10, 10), precision=0.001)],
    outputs=[Port("y", range=(-1, 1), precision=0.001)],
)
def tanh_neuron(x):
    return math.tanh(x)


@neuron(
    inputs=[Port("x", range=(-10, 10), precision=0.001)],
    outputs=[Port("y", range=(0, 1), precision=1.0)],
    name="threshold",
)
def threshold(x):
    return 1.0 if x >= 0.0 else 0.0


@neuron(
    inputs=[Port("x", range=(-10, 10), precision=0.001)],
    outputs=[Port("y", range=(-10, 10), precision=0.001)],
    name="identity",
)
def identity(x):
    return x


@neuron(
    inputs=[Port("x", range=(-10, 10), precision=0.001)],
    outputs=[Port("y", range=(-10, 10), precision=0.001)],
    name="negate",
)
def negate(x):
    return -x


@neuron(
    inputs=[
        Port("a", range=(-10, 10), precision=0.001),
        Port("b", range=(-10, 10), precision=0.001),
    ],
    outputs=[Port("sum", range=(-20, 20), precision=0.001)],
    name="add",
)
def add(a, b):
    return a + b


@neuron(
    inputs=[
        Port("a", range=(-10, 10), precision=0.001),
        Port("b", range=(-10, 10), precision=0.001),
    ],
    outputs=[Port("product", range=(-100, 100), precision=0.001)],
    name="multiply",
)
def multiply(a, b):
    return a * b


@neuron(
    inputs=[Port("x", range=(-10, 10), precision=0.001)],
    outputs=[Port("y", range=(0, 10), precision=0.001)],
    name="gaussian",
)
def gaussian(x):
    return math.exp(-x * x)


@neuron(
    inputs=[Port("x", range=(0.001, 10), precision=0.001)],
    outputs=[Port("y", range=(-5, 5), precision=0.001)],
    name="log",
)
def log_neuron(x):
    return math.log(max(x, 1e-7))


# ── ReLU variants ─────────────────────────────────────────────────────

@neuron(
    inputs=[Port("x", range=(-10, 10), precision=0.001)],
    outputs=[Port("y", range=(-0.1, 10), precision=0.001)],
    name="leaky_relu",
)
def leaky_relu(x):
    return x if x >= 0.0 else 0.01 * x


@neuron(
    inputs=[Port("x", range=(-10, 10), precision=0.001)],
    outputs=[Port("y", range=(-2.5, 10), precision=0.001)],
    name="prelu",
)
def prelu(x):
    return x if x >= 0.0 else 0.25 * x


@neuron(
    inputs=[Port("x", range=(-10, 10), precision=0.001)],
    outputs=[Port("y", range=(0, 6), precision=0.001)],
    name="relu6",
)
def relu6(x):
    return min(max(0.0, x), 6.0)


# ── Exponential linear units ─────────────────────────────────────────

@neuron(
    inputs=[Port("x", range=(-10, 10), precision=0.001)],
    outputs=[Port("y", range=(-1, 10), precision=0.001)],
    name="elu",
)
def elu(x):
    return x if x >= 0.0 else math.exp(x) - 1.0


_SELU_ALPHA = 1.6732632423543772
_SELU_LAMBDA = 1.0507009873554805


@neuron(
    inputs=[Port("x", range=(-10, 10), precision=0.001)],
    outputs=[Port("y", range=(-2, 20), precision=0.001)],
    name="selu",
)
def selu(x):
    return _SELU_LAMBDA * (x if x >= 0.0 else _SELU_ALPHA * (math.exp(x) - 1.0))


# ── Smooth alternatives ──────────────────────────────────────────────

@neuron(
    inputs=[Port("x", range=(-10, 10), precision=0.001)],
    outputs=[Port("y", range=(-1, 10), precision=0.001)],
    name="gelu",
)
def gelu(x):
    return x * 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@neuron(
    inputs=[Port("x", range=(-10, 10), precision=0.001)],
    outputs=[Port("y", range=(-1, 10), precision=0.001)],
    name="silu",
)
def silu(x):
    return x / (1.0 + math.exp(-x))


@neuron(
    inputs=[Port("x", range=(-10, 10), precision=0.001)],
    outputs=[Port("y", range=(-1, 10), precision=0.001)],
    name="mish",
)
def mish(x):
    sp = math.log(1.0 + math.exp(x))
    return x * math.tanh(sp)


# ── Classic smooth ────────────────────────────────────────────────────

@neuron(
    inputs=[Port("x", range=(-10, 10), precision=0.001)],
    outputs=[Port("y", range=(0, 10), precision=0.001)],
    name="softplus",
)
def softplus(x):
    return math.log(1.0 + math.exp(x))


@neuron(
    inputs=[Port("x", range=(-10, 10), precision=0.001)],
    outputs=[Port("y", range=(-1, 1), precision=0.001)],
    name="softsign",
)
def softsign(x):
    return x / (1.0 + abs(x))


# ── Hard approximations ──────────────────────────────────────────────

@neuron(
    inputs=[Port("x", range=(-10, 10), precision=0.001)],
    outputs=[Port("y", range=(0, 1), precision=0.001)],
    name="hard_sigmoid",
)
def hard_sigmoid(x):
    return max(0.0, min(1.0, x / 6.0 + 0.5))


@neuron(
    inputs=[Port("x", range=(-10, 10), precision=0.001)],
    outputs=[Port("y", range=(-1, 1), precision=0.001)],
    name="hard_tanh",
)
def hard_tanh(x):
    return max(-1.0, min(1.0, x))


@neuron(
    inputs=[Port("x", range=(-10, 10), precision=0.001)],
    outputs=[Port("y", range=(-1, 10), precision=0.001)],
    name="hard_swish",
)
def hard_swish(x):
    return x * max(0.0, min(1.0, x / 6.0 + 0.5))


# ── Output-layer activations ─────────────────────────────────────────

@neuron(
    inputs=[
        Port("a", range=(-10, 10), precision=0.001),
        Port("b", range=(-10, 10), precision=0.001),
    ],
    outputs=[
        Port("p_a", range=(0, 1), precision=0.001),
        Port("p_b", range=(0, 1), precision=0.001),
    ],
    name="softmax_2",
)
def softmax_2(a, b):
    m = max(a, b)
    ea = math.exp(a - m)
    eb = math.exp(b - m)
    s = ea + eb
    return ea / s, eb / s


@neuron(
    inputs=[
        Port("a", range=(-10, 10), precision=0.001),
        Port("b", range=(-10, 10), precision=0.001),
    ],
    outputs=[
        Port("lp_a", range=(-20, 0), precision=0.001),
        Port("lp_b", range=(-20, 0), precision=0.001),
    ],
    name="logsoftmax_2",
)
def logsoftmax_2(a, b):
    m = max(a, b)
    lse = m + math.log(math.exp(a - m) + math.exp(b - m))
    return a - lse, b - lse


# ── I/O terminals ─────────────────────────────────────────────────────

# passthrough nodes used as graph I/O terminals
@neuron(
    inputs=[Port("in", range=(-100, 100), precision=0.001)],
    outputs=[Port("out", range=(-100, 100), precision=0.001)],
    name="input",
)
def input_node(x):
    return x


@neuron(
    inputs=[Port("in", range=(-100, 100), precision=0.001)],
    outputs=[Port("out", range=(-100, 100), precision=0.001)],
    name="output",
)
def output_node(x):
    return x


token_embedding_module = module_neuron(
    name="token_embedding",
    module_type="token_embedding",
    input_ports=[Port("token_ids", range=(0, 65535), precision=1.0, dtype="tokens")],
    output_ports=[
        Port("hidden", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("weight", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    module_config=default_token_embedding_config(),
)

linear_module = module_neuron(
    name="linear",
    module_type="linear",
    input_ports=[Port("x", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[Port("y", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config=default_linear_config(),
)

rms_norm_module = module_neuron(
    name="rms_norm",
    module_type="rms_norm",
    input_ports=[Port("x", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[Port("y", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config=default_rms_norm_config(),
)

reshape_heads_module = module_neuron(
    name="reshape_heads",
    module_type="reshape_heads",
    input_ports=[Port("x", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[Port("heads", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config=default_reshape_heads_config(),
)

merge_heads_module = module_neuron(
    name="merge_heads",
    module_type="merge_heads",
    input_ports=[Port("heads", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[Port("x", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config=default_merge_heads_config(),
)

repeat_kv_module = module_neuron(
    name="repeat_kv",
    module_type="repeat_kv",
    input_ports=[Port("x", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[Port("y", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config=default_repeat_kv_config(),
)

rotary_embedding_module = module_neuron(
    name="rotary_embedding",
    module_type="rotary_embedding",
    input_ports=[
        Port("q", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("k", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    output_ports=[
        Port("q_rot", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("k_rot", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    module_config=default_rotary_embedding_config(),
)

qk_gain_module = module_neuron(
    name="qk_gain",
    module_type="qk_gain",
    input_ports=[Port("q", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[Port("q_scaled", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config=default_qk_gain_config(),
)

scaled_dot_product_attention_module = module_neuron(
    name="scaled_dot_product_attention",
    module_type="scaled_dot_product_attention",
    input_ports=[
        Port("q", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("k", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("v", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    output_ports=[Port("y", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config=default_scaled_dot_product_attention_config(),
)

residual_mix_module = module_neuron(
    name="residual_mix",
    module_type="residual_mix",
    input_ports=[
        Port("x", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("x0", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    output_ports=[Port("mixed", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config=default_residual_mix_config(),
)

causal_self_attention_module = module_neuron(
    name="causal_self_attention",
    module_type="causal_self_attention",
    input_ports=[Port("x", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[Port("attn_out", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config=default_attention_config(),
)

residual_add_module = module_neuron(
    name="residual_add",
    module_type="residual_add",
    input_ports=[
        Port("residual", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("delta", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    output_ports=[Port("y", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config=default_residual_add_config(),
)

mlp_relu2_module = module_neuron(
    name="mlp_relu2",
    module_type="mlp_relu2",
    input_ports=[Port("x", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[Port("y", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config=default_mlp_config(),
)

tied_lm_head_module = module_neuron(
    name="tied_lm_head",
    module_type="tied_lm_head",
    input_ports=[
        Port("hidden", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("weight", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    output_ports=[Port("logits", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config={},
)

lm_head_module = module_neuron(
    name="lm_head",
    module_type="lm_head",
    input_ports=[Port("hidden", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[Port("logits", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config=default_lm_head_config(),
)

logit_softcap_module = module_neuron(
    name="logit_softcap",
    module_type="logit_softcap",
    input_ports=[Port("logits", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[Port("softcapped", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config=default_logit_softcap_config(),
)

token_cross_entropy_module = module_neuron(
    name="token_cross_entropy",
    module_type="token_cross_entropy",
    input_ports=[
        Port("logits", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("targets", range=(0, 65535), precision=1.0, dtype="tokens"),
    ],
    output_ports=[Port("loss", range=(0, 100), precision=0.0001, dtype="loss")],
    module_config={},
)



layer_norm_module = module_neuron(
    name="layer_norm",
    module_type="layer_norm",
    input_ports=[Port("x", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[Port("y", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config={"model_dim": 128, "eps": 1e-5},
)

dropout_module = module_neuron(
    name="dropout",
    module_type="dropout",
    input_ports=[Port("x", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[Port("y", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config={"p": 0.1},
)

gelu_module = module_neuron(
    name="gelu",
    module_type="gelu",
    input_ports=[Port("x", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[Port("y", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config={},
)

swiglu_module = module_neuron(
    name="swiglu",
    module_type="swiglu",
    input_ports=[Port("x", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[Port("y", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config={"model_dim": 128, "mlp_mult": 4, "multiple_of": 256},
)

absolute_position_embedding_module = module_neuron(
    name="absolute_position_embedding",
    module_type="absolute_position_embedding",
    input_ports=[Port("x", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[Port("y", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config={"max_seq_len": 1024, "model_dim": 128},
)

kv_cache_read_module = module_neuron(
    name="kv_cache_read",
    module_type="kv_cache_read",
    input_ports=[
        Port("k", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("v", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("cache_k", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("cache_v", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    output_ports=[
        Port("k_out", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("v_out", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    module_config={},
)

kv_cache_write_module = module_neuron(
    name="kv_cache_write",
    module_type="kv_cache_write",
    input_ports=[
        Port("k", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("v", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    output_ports=[
        Port("k_out", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("v_out", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    module_config={},
)

router_logits_module = module_neuron(
    name="router_logits",
    module_type="router_logits",
    input_ports=[Port("x", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[Port("logits", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config={"model_dim": 128, "experts": 8},
)

topk_route_module = module_neuron(
    name="topk_route",
    module_type="topk_route",
    input_ports=[Port("logits", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[
        Port("weights", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("indices", range=(0, 100), precision=1.0, dtype="tensor"),
    ],
    module_config={"top_k": 2},
)

expert_dispatch_module = module_neuron(
    name="expert_dispatch",
    module_type="expert_dispatch",
    input_ports=[
        Port("x", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("weights", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("indices", range=(0, 100), precision=1.0, dtype="tensor"),
    ],
    output_ports=[Port("y", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config={"model_dim": 128, "experts": 8, "mlp_mult": 4},
)

expert_combine_module = module_neuron(
    name="expert_combine",
    module_type="expert_combine",
    input_ports=[Port("x", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[Port("y", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config={},
)

load_balance_loss_module = module_neuron(
    name="load_balance_loss",
    module_type="load_balance_loss",
    input_ports=[
        Port("router_logits", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("weights", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("indices", range=(0, 100), precision=1.0, dtype="tensor"),
    ],
    output_ports=[
        Port("aux_loss", range=(0, 1_000_000), precision=0.001, dtype="tensor"),
        Port("router_logits_out", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    module_config={"experts": 8},
)

aux_loss_add_module = module_neuron(
    name="aux_loss_add",
    module_type="aux_loss_add",
    input_ports=[
        Port("main_loss", range=(0, 100), precision=0.0001, dtype="loss"),
        Port("aux_loss", range=(0, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    output_ports=[Port("loss", range=(0, 100), precision=0.0001, dtype="loss")],
    module_config={"coef": 0.01},
)

dataset_source_module = module_neuron(
    name="dataset_source",
    module_type="dataset_source",
    input_ports=[],
    output_ports=[
        Port("tokens", range=(0, 65535), precision=1.0, dtype="tokens"),
        Port("targets", range=(0, 65535), precision=1.0, dtype="tokens"),
    ],
    module_config={"dataset_names": [], "seq_len": 64},
)

_BUILTIN_ATTR_MAP: dict[str, NeuronDef] = {
    "sigmoid": sigmoid,
    "relu": relu,
    "tanh_neuron": tanh_neuron,
    "threshold": threshold,
    "identity": identity,
    "negate": negate,
    "add": add,
    "multiply": multiply,
    "gaussian": gaussian,
    "log_neuron": log_neuron,
    "leaky_relu": leaky_relu,
    "prelu": prelu,
    "relu6": relu6,
    "elu": elu,
    "selu": selu,
    "gelu": gelu,
    "silu": silu,
    "mish": mish,
    "softplus": softplus,
    "softsign": softsign,
    "hard_sigmoid": hard_sigmoid,
    "hard_tanh": hard_tanh,
    "hard_swish": hard_swish,
    "softmax_2": softmax_2,
    "logsoftmax_2": logsoftmax_2,
    "input_node": input_node,
    "output_node": output_node,
    "token_embedding_module": token_embedding_module,
    "linear_module": linear_module,
    "rms_norm_module": rms_norm_module,
    "reshape_heads_module": reshape_heads_module,
    "merge_heads_module": merge_heads_module,
    "repeat_kv_module": repeat_kv_module,
    "rotary_embedding_module": rotary_embedding_module,
    "qk_gain_module": qk_gain_module,
    "scaled_dot_product_attention_module": scaled_dot_product_attention_module,
    "residual_mix_module": residual_mix_module,
    "causal_self_attention_module": causal_self_attention_module,
    "residual_add_module": residual_add_module,
    "mlp_relu2_module": mlp_relu2_module,
    "tied_lm_head_module": tied_lm_head_module,
    "lm_head_module": lm_head_module,
    "logit_softcap_module": logit_softcap_module,
    "layer_norm_module": layer_norm_module,
    "dropout_module": dropout_module,
    "gelu_module": gelu_module,
    "swiglu_module": swiglu_module,
    "absolute_position_embedding_module": absolute_position_embedding_module,
    "kv_cache_read_module": kv_cache_read_module,
    "kv_cache_write_module": kv_cache_write_module,
    "router_logits_module": router_logits_module,
    "topk_route_module": topk_route_module,
    "expert_dispatch_module": expert_dispatch_module,
    "expert_combine_module": expert_combine_module,
    "load_balance_loss_module": load_balance_loss_module,
    "aux_loss_add_module": aux_loss_add_module,
    "token_cross_entropy_module": token_cross_entropy_module,
    "dataset_source_module": dataset_source_module,
}


class BuiltinNeurons:
    """Public helper exposing the built-in neuron catalog."""

    sigmoid = sigmoid
    relu = relu
    tanh_neuron = tanh_neuron
    threshold = threshold
    identity = identity
    negate = negate
    add = add
    multiply = multiply
    gaussian = gaussian
    log_neuron = log_neuron
    leaky_relu = leaky_relu
    prelu = prelu
    relu6 = relu6
    elu = elu
    selu = selu
    gelu = gelu
    silu = silu
    mish = mish
    softplus = softplus
    softsign = softsign
    hard_sigmoid = hard_sigmoid
    hard_tanh = hard_tanh
    hard_swish = hard_swish
    softmax_2 = softmax_2
    logsoftmax_2 = logsoftmax_2
    input_node = input_node
    output_node = output_node
    token_embedding_module = token_embedding_module
    linear_module = linear_module
    rms_norm_module = rms_norm_module
    reshape_heads_module = reshape_heads_module
    merge_heads_module = merge_heads_module
    repeat_kv_module = repeat_kv_module
    rotary_embedding_module = rotary_embedding_module
    qk_gain_module = qk_gain_module
    scaled_dot_product_attention_module = scaled_dot_product_attention_module
    residual_mix_module = residual_mix_module
    causal_self_attention_module = causal_self_attention_module
    residual_add_module = residual_add_module
    mlp_relu2_module = mlp_relu2_module
    tied_lm_head_module = tied_lm_head_module
    lm_head_module = lm_head_module
    logit_softcap_module = logit_softcap_module
    layer_norm_module = layer_norm_module
    dropout_module = dropout_module
    gelu_module = gelu_module
    swiglu_module = swiglu_module
    absolute_position_embedding_module = absolute_position_embedding_module
    kv_cache_read_module = kv_cache_read_module
    kv_cache_write_module = kv_cache_write_module
    router_logits_module = router_logits_module
    topk_route_module = topk_route_module
    expert_dispatch_module = expert_dispatch_module
    expert_combine_module = expert_combine_module
    load_balance_loss_module = load_balance_loss_module
    aux_loss_add_module = aux_loss_add_module
    token_cross_entropy_module = token_cross_entropy_module
    dataset_source_module = dataset_source_module

    @classmethod
    def all(cls) -> list[NeuronDef]:
        """Return the full built-in neuron catalog."""
        return list(_BUILTIN_ATTR_MAP.values())

    @classmethod
    def get(cls, name: str) -> NeuronDef:
        """Lookup a built-in by attribute name or serialised display name."""
        neuron_def = _BUILTIN_ATTR_MAP.get(name)
        if neuron_def is not None:
            return neuron_def

        for candidate in _BUILTIN_ATTR_MAP.values():
            if candidate.name == name:
                return candidate

        raise KeyError(f"Unknown built-in neuron: {name}")


BUILTIN_NEURONS: list[NeuronDef] = BuiltinNeurons.all()

BUILTIN_ATTR_MAP: dict[str, NeuronDef] = dict(_BUILTIN_ATTR_MAP)

BUILTIN_MAP: dict[str, NeuronDef] = {n.name: n for n in BUILTIN_NEURONS}

__all__ = [
    "BuiltinNeurons",
    "BUILTIN_NEURONS",
    "BUILTIN_MAP",
    "BUILTIN_ATTR_MAP",
    "sigmoid",
    "relu",
    "tanh_neuron",
    "threshold",
    "identity",
    "negate",
    "add",
    "multiply",
    "gaussian",
    "log_neuron",
    "leaky_relu",
    "prelu",
    "relu6",
    "elu",
    "selu",
    "gelu",
    "silu",
    "mish",
    "softplus",
    "softsign",
    "hard_sigmoid",
    "hard_tanh",
    "hard_swish",
    "softmax_2",
    "logsoftmax_2",
    "input_node",
    "output_node",
    "token_embedding_module",
    "linear_module",
    "rms_norm_module",
    "reshape_heads_module",
    "merge_heads_module",
    "repeat_kv_module",
    "rotary_embedding_module",
    "qk_gain_module",
    "scaled_dot_product_attention_module",
    "residual_mix_module",
    "causal_self_attention_module",
    "residual_add_module",
    "mlp_relu2_module",
    "tied_lm_head_module",
    "lm_head_module",
    "logit_softcap_module",
    "layer_norm_module",
    "dropout_module",
    "gelu_module",
    "swiglu_module",
    "absolute_position_embedding_module",
    "kv_cache_read_module",
    "kv_cache_write_module",
    "router_logits_module",
    "topk_route_module",
    "expert_dispatch_module",
    "expert_combine_module",
    "load_balance_loss_module",
    "aux_loss_add_module",
    "token_cross_entropy_module",
    "dataset_source_module",
]
