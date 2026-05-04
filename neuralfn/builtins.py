"""Library of common neuron functions ready to drop into a graph."""

from __future__ import annotations

import math

from .neuron import NeuronDef, module_neuron, neuron
from .port import Port
from .semantic import DEFAULT_SEMANTIC_VOCAB_REF, NUM_SEMANTIC_DIMS, NUM_VOCAB_DIMS
from .torch_backend import (
    default_attention_config,
    default_fused_attention_config,
    default_gpt_config,
    default_kv_pca_config,
    default_kv_quant_unpack_config,
    default_linear_config,
    default_loss_scale_config,
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


def _normalized_builtin_port(port: Port) -> Port:
    keep_constraints = port.dtype in {"tokens", "bool"}
    return Port(
        name=port.name,
        range=port.range if keep_constraints else None,
        precision=port.precision if keep_constraints else None,
        dtype=port.dtype,
    )


def _normalize_builtin_ports(*neurons: NeuronDef) -> None:
    for neuron_def in neurons:
        neuron_def.input_ports = [_normalized_builtin_port(port) for port in neuron_def.input_ports]
        neuron_def.output_ports = [_normalized_builtin_port(port) for port in neuron_def.output_ports]


@neuron(
    inputs=[Port("x")],
    outputs=[Port("y")],
)
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


@neuron(
    inputs=[Port("x")],
    outputs=[Port("y")],
)
def relu(x):
    return max(0.0, x)


@neuron(
    inputs=[Port("x")],
    outputs=[Port("y")],
)
def tanh_neuron(x):
    return math.tanh(x)


@neuron(
    inputs=[Port("x")],
    outputs=[Port("y", range=(0, 1), precision=1.0, dtype="bool")],
    name="threshold",
)
def threshold(x):
    return 1.0 if x >= 0.0 else 0.0


@neuron(
    inputs=[Port("x")],
    outputs=[Port("y")],
    name="identity",
)
def identity(x):
    return x


@neuron(
    inputs=[Port("x")],
    outputs=[Port("y")],
    name="negate",
)
def negate(x):
    return -x


@neuron(
    inputs=[Port("a"), Port("b")],
    outputs=[Port("sum")],
    name="add",
)
def add(a, b):
    return a + b


@neuron(
    inputs=[Port("a"), Port("b")],
    outputs=[Port("product")],
    name="multiply",
)
def multiply(a, b):
    return a * b


@neuron(
    inputs=[Port("x")],
    outputs=[Port("y")],
    name="gaussian",
)
def gaussian(x):
    return math.exp(-x * x)


@neuron(
    inputs=[Port("x")],
    outputs=[Port("y")],
    name="log",
)
def log_neuron(x):
    return math.log(max(x, 1e-7))


# ── ReLU variants ─────────────────────────────────────────────────────

@neuron(
    inputs=[Port("x")],
    outputs=[Port("y")],
    name="leaky_relu",
)
def leaky_relu(x):
    return x if x >= 0.0 else 0.01 * x


@neuron(
    inputs=[Port("x")],
    outputs=[Port("y")],
    name="prelu",
)
def prelu(x):
    return x if x >= 0.0 else 0.25 * x


@neuron(
    inputs=[Port("x")],
    outputs=[Port("y")],
    name="relu6",
)
def relu6(x):
    return min(max(0.0, x), 6.0)


# ── Exponential linear units ─────────────────────────────────────────

@neuron(
    inputs=[Port("x")],
    outputs=[Port("y")],
    name="elu",
)
def elu(x):
    return x if x >= 0.0 else math.exp(x) - 1.0


_SELU_ALPHA = 1.6732632423543772
_SELU_LAMBDA = 1.0507009873554805


@neuron(
    inputs=[Port("x")],
    outputs=[Port("y")],
    name="selu",
)
def selu(x):
    return _SELU_LAMBDA * (x if x >= 0.0 else _SELU_ALPHA * (math.exp(x) - 1.0))


# ── Smooth alternatives ──────────────────────────────────────────────

@neuron(
    inputs=[Port("x")],
    outputs=[Port("y")],
    name="gelu",
)
def gelu(x):
    return x * 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@neuron(
    inputs=[Port("x")],
    outputs=[Port("y")],
    name="silu",
)
def silu(x):
    return x / (1.0 + math.exp(-x))


@neuron(
    inputs=[Port("x")],
    outputs=[Port("y")],
    name="mish",
)
def mish(x):
    sp = math.log(1.0 + math.exp(x))
    return x * math.tanh(sp)


# ── Classic smooth ────────────────────────────────────────────────────

@neuron(
    inputs=[Port("x")],
    outputs=[Port("y")],
    name="softplus",
)
def softplus(x):
    return math.log(1.0 + math.exp(x))


@neuron(
    inputs=[Port("x")],
    outputs=[Port("y")],
    name="softsign",
)
def softsign(x):
    return x / (1.0 + abs(x))


# ── Hard approximations ──────────────────────────────────────────────

@neuron(
    inputs=[Port("x")],
    outputs=[Port("y")],
    name="hard_sigmoid",
)
def hard_sigmoid(x):
    return max(0.0, min(1.0, x / 6.0 + 0.5))


@neuron(
    inputs=[Port("x")],
    outputs=[Port("y")],
    name="hard_tanh",
)
def hard_tanh(x):
    return max(-1.0, min(1.0, x))


@neuron(
    inputs=[Port("x")],
    outputs=[Port("y")],
    name="hard_swish",
)
def hard_swish(x):
    return x * max(0.0, min(1.0, x / 6.0 + 0.5))


# ── Output-layer activations ─────────────────────────────────────────

@neuron(
    inputs=[Port("a"), Port("b")],
    outputs=[Port("p_a"), Port("p_b")],
    name="softmax_2",
)
def softmax_2(a, b):
    m = max(a, b)
    ea = math.exp(a - m)
    eb = math.exp(b - m)
    s = ea + eb
    return ea / s, eb / s


@neuron(
    inputs=[Port("a"), Port("b")],
    outputs=[Port("lp_a"), Port("lp_b")],
    name="logsoftmax_2",
)
def logsoftmax_2(a, b):
    m = max(a, b)
    lse = m + math.log(math.exp(a - m) + math.exp(b - m))
    return a - lse, b - lse


# ── I/O terminals ─────────────────────────────────────────────────────

# passthrough nodes used as graph I/O terminals
@neuron(
    inputs=[Port("in")],
    outputs=[Port("out")],
    name="input",
)
def input_node(x):
    return x


@neuron(
    inputs=[Port("in")],
    outputs=[Port("out")],
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

fused_causal_attention_module = module_neuron(
    name="fused_causal_attention",
    module_type="fused_causal_attention",
    input_ports=[Port("x", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[Port("attn_out", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config=default_fused_attention_config(),
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

kv_pca_encode_module = module_neuron(
    name="kv_pca_encode",
    module_type="kv_pca_encode",
    input_ports=[
        Port("k", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("v", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    output_ports=[
        Port("k_comp", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("v_comp", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    module_config=default_kv_pca_config(),
)

kv_pca_decode_module = module_neuron(
    name="kv_pca_decode",
    module_type="kv_pca_decode",
    input_ports=[
        Port("k_comp", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("v_comp", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    output_ports=[
        Port("k", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("v", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    module_config=default_kv_pca_config(),
)

kv_quant_pack_module = module_neuron(
    name="kv_quant_pack",
    module_type="kv_quant_pack",
    input_ports=[
        Port("k", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("v", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    output_ports=[Port("packed", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config={},
)

kv_quant_unpack_module = module_neuron(
    name="kv_quant_unpack",
    module_type="kv_quant_unpack",
    input_ports=[Port("packed", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[
        Port("k", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("v", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    module_config=default_kv_quant_unpack_config(),
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
    module_config={"top_k": 2, "experts": 8},
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

loss_scale_module = module_neuron(
    name="loss_scale",
    module_type="loss_scale",
    input_ports=[Port("loss", range=(0, 100), precision=0.0001, dtype="loss")],
    output_ports=[Port("scaled_loss", range=(0, 100), precision=0.0001, dtype="loss")],
    module_config=default_loss_scale_config(),
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

bitlinear_ternary_module = module_neuron(
    name="bitlinear_ternary",
    module_type="bitlinear_ternary",
    input_ports=[Port("x", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[Port("y", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config={"input_dim": 128, "output_dim": 128},
)

randmap_adapter_module = module_neuron(
    name="randmap_adapter",
    module_type="randmap_adapter",
    input_ports=[Port("x", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[Port("y", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config={"model_dim": 128, "adapter_dim": 32},
)

mamba_module = module_neuron(
    name="mamba",
    module_type="mamba",
    input_ports=[Port("x", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[Port("y", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config={"model_dim": 128, "d_state": 16, "d_conv": 4, "expand": 2},
)

denoise_head_module = module_neuron(
    name="denoise_head",
    module_type="denoise_head",
    input_ports=[Port("x", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[Port("logits", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config={"model_dim": 128, "vocab_size": 1024},
)

mask_scheduler_module = module_neuron(
    name="mask_scheduler",
    module_type="mask_scheduler",
    input_ports=[
        Port("tokens", range=(0, 65535), precision=1.0, dtype="tokens"),
        Port("timesteps", range=(0, 1), precision=0.001, dtype="tensor"),
    ],
    output_ports=[Port("noisy_tokens", range=(0, 65535), precision=1.0, dtype="tokens")],
    module_config={"vocab_size": 1024, "mask_token_id": 0},
)

random_timesteps_module = module_neuron(
    name="random_timesteps",
    module_type="random_timesteps",
    input_ports=[Port("tokens", range=(0, 65535), precision=1.0, dtype="tokens")],
    output_ports=[Port("timesteps", range=(0, 1), precision=0.001, dtype="tensor")],
    module_config={},
)

jepa_mask_module = module_neuron(
    name="jepa_mask",
    module_type="jepa_mask",
    input_ports=[Port("tokens", range=(0, 65535), precision=1.0, dtype="tokens")],
    output_ports=[
        Port("masked_tokens", range=(0, 65535), precision=1.0, dtype="tokens"),
        Port("mask", range=(0, 1), precision=0.001, dtype="tensor"),
    ],
    module_config={"mask_ratio": 0.5, "mask_token_id": 0, "mask_strategy": "random", "num_blocks": 4, "min_block_ratio": 0.1, "max_block_ratio": 0.25},
)

latent_pool_module = module_neuron(
    name="latent_pool",
    module_type="latent_pool",
    input_ports=[
        Port("x", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("mask", range=(0, 1), precision=0.001, dtype="tensor"),
    ],
    output_ports=[Port("pooled", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config={},
)

jepa_projector_module = module_neuron(
    name="jepa_projector",
    module_type="jepa_projector",
    input_ports=[Port("x", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[Port("y", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config={"input_dim": 128, "latent_dim": 128},
)

jepa_predictor_module = module_neuron(
    name="jepa_predictor",
    module_type="jepa_predictor",
    input_ports=[Port("x", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[Port("y", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config={"latent_dim": 128},
)

latent_mse_loss_module = module_neuron(
    name="latent_mse_loss",
    module_type="latent_mse_loss",
    input_ports=[
        Port("pred", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("target", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    output_ports=[Port("loss", range=(0, 100), precision=0.0001, dtype="loss")],
    module_config={},
)

byte_patch_embed_module = module_neuron(
    name="byte_patch_embed",
    module_type="byte_patch_embed",
    input_ports=[Port("tokens", range=(0, 255), precision=1.0, dtype="tokens")],
    output_ports=[Port("x", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config={"model_dim": 128, "patch_size": 4, "stride": 4, "vocab_size": 256},
)

byte_patch_merge_module = module_neuron(
    name="byte_patch_merge",
    module_type="byte_patch_merge",
    input_ports=[
        Port("x", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("target_tokens", range=(0, 255), precision=1.0, dtype="tokens"),
    ],
    output_ports=[Port("y", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config={},
)

act_halt_gate_module = module_neuron(
    name="act_halt_gate",
    module_type="act_halt_gate",
    input_ports=[Port("x", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[Port("halt", range=(0, 1), precision=0.001, dtype="tensor")],
    module_config={"model_dim": 128},
)

act_weighted_sum_module = module_neuron(
    name="act_weighted_sum",
    module_type="act_weighted_sum",
    input_ports=[
        Port("states", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("weights", range=(0, 1), precision=0.001, dtype="tensor"),
    ],
    output_ports=[Port("y", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config={},
)

universal_transformer_module = module_neuron(
    name="universal_transformer",
    module_type="universal_transformer",
    input_ports=[Port("x", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[
        Port("y", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("halt_weights", range=(0, 1), precision=0.001, dtype="tensor"),
    ],
    module_config={"model_dim": 128, "num_heads": 4, "mlp_mult": 4.0, "max_steps": 4, "halt_epsilon": 0.01},
)

ttt_linear_module = module_neuron(
    name="ttt_linear",
    module_type="ttt_linear",
    input_ports=[Port("x", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[Port("y", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config={"input_dim": 128, "output_dim": 128, "hidden_dim": 16},
)

semantic_data_source_module = module_neuron(
    name="semantic_data_source",
    module_type="semantic_data_source",
    input_ports=[],
    output_ports=[
        Port("sem_targets", range=(-100, 65535), precision=1.0, dtype="tokens"),
    ],
    module_config={
        "seq_len": NUM_SEMANTIC_DIMS,
        "semantic_vocab_ref": DEFAULT_SEMANTIC_VOCAB_REF,
        "emit_router_vecs": False,
        "router_vec_dim": NUM_VOCAB_DIMS,
    },
)

semantic_projector_module = module_neuron(
    name="semantic_projector",
    module_type="semantic_projector",
    input_ports=[Port("hidden", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[
        Port("semantic_vec", range=(-1, 1), precision=0.001, dtype="tensor"),
        Port("residual", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("topic_logits", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    module_config={
        "input_dim": 128,
        "semantic_dim": NUM_SEMANTIC_DIMS,
        "residual_dim": 64,
        "n_sig_buckets": 4096,
        "semantic_vocab_ref": DEFAULT_SEMANTIC_VOCAB_REF,
    },
)

semantic_alignment_loss_module = module_neuron(
    name="semantic_alignment_loss",
    module_type="semantic_alignment_loss",
    input_ports=[
        Port("pred_logits", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("target", range=(-100, 65535), precision=1.0, dtype="tokens"),
    ],
    output_ports=[Port("loss", range=(0, 100), precision=0.0001, dtype="loss")],
    module_config={"ignore_index": -100, "semantic_vocab_ref": DEFAULT_SEMANTIC_VOCAB_REF},
)

semantic_hasher_module = module_neuron(
    name="semantic_hasher",
    module_type="semantic_hasher",
    input_ports=[Port("semantic_vec", range=(-1, 1), precision=0.001, dtype="tensor")],
    output_ports=[Port("bucket_indices", range=(0, 1_000_000), precision=1.0, dtype="tensor")],
    module_config={"dim": NUM_SEMANTIC_DIMS, "tables": 8, "planes": 12, "seed": 42},
)

semantic_moe_router_module = module_neuron(
    name="semantic_moe_router",
    module_type="semantic_moe_router",
    input_ports=[Port("semantic_vec", range=(-1, 1), precision=0.001, dtype="tensor")],
    output_ports=[
        Port("expert_weights", range=(0, 1), precision=0.001, dtype="tensor"),
        Port("expert_indices", range=(0, 100), precision=1.0, dtype="tensor"),
    ],
    module_config={"n_experts": 32, "semantic_dim": NUM_SEMANTIC_DIMS, "top_k": 2},
)

semantic_hash_router_module = module_neuron(
    name="semantic_hash_router",
    module_type="semantic_hash_router",
    input_ports=[
        Port("semantic_vec", range=(-1, 1), precision=0.001, dtype="tensor"),
        Port("bucket_indices", range=(0, 1_000_000), precision=1.0, dtype="tensor"),
        Port("topic_logits", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("sem_targets", range=(-100, 65535), precision=1.0, dtype="tokens"),
    ],
    output_ports=[
        Port("expert_weights", range=(0, 1), precision=0.001, dtype="tensor"),
        Port("expert_indices", range=(0, 100), precision=1.0, dtype="tensor"),
    ],
    module_config={
        "n_experts": NUM_VOCAB_DIMS,
        "semantic_dim": NUM_SEMANTIC_DIMS,
        "top_k": 2,
        "tables": 8,
        "n_buckets": 4096,
        "ignore_index": -100,
        "semantic_vocab_ref": DEFAULT_SEMANTIC_VOCAB_REF,
        "routing_source": "topic_logits",
    },
)

causal_chunk_state_module = module_neuron(
    name="causal_chunk_state",
    module_type="causal_chunk_state",
    input_ports=[Port("hidden", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[Port("chunk_state", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config={"chunk_size": 32, "mode": "prefix"},
)

semantic_chunk_projector_module = module_neuron(
    name="semantic_chunk_projector",
    module_type="semantic_chunk_projector",
    input_ports=[Port("chunk_state", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[
        Port("semantic_vec", range=(-1, 1), precision=0.001, dtype="tensor"),
        Port("residual", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("topic_logits", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    module_config={
        "input_dim": 128,
        "semantic_dim": NUM_SEMANTIC_DIMS,
        "residual_dim": 64,
        "n_sig_buckets": 4096,
        "semantic_vocab_ref": DEFAULT_SEMANTIC_VOCAB_REF,
    },
)

semantic_chunk_hasher_module = module_neuron(
    name="semantic_chunk_hasher",
    module_type="semantic_chunk_hasher",
    input_ports=[Port("semantic_vec", range=(-1, 1), precision=0.001, dtype="tensor")],
    output_ports=[Port("bucket_indices", range=(0, 1_000_000), precision=1.0, dtype="tensor")],
    module_config={"dim": NUM_SEMANTIC_DIMS, "tables": 8, "planes": 12, "seed": 42},
)

semantic_moe_jepa_evo_router_module = module_neuron(
    name="semantic_moe_jepa_evo_router",
    module_type="semantic_moe_jepa_evo_router",
    input_ports=[
        Port("semantic_vec", range=(-1, 1), precision=0.001, dtype="tensor"),
        Port("bucket_indices", range=(0, 1_000_000), precision=1.0, dtype="tensor"),
        Port("topic_logits", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("sem_targets", range=(-100, 65535), precision=1.0, dtype="tokens"),
    ],
    output_ports=[
        Port("expert_weights", range=(0, 1), precision=0.001, dtype="tensor"),
        Port("expert_indices", range=(0, 1000), precision=1.0, dtype="tensor"),
        Port("route_logits", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    module_config={
        "semantic_dim": NUM_SEMANTIC_DIMS,
        "top_k": 2,
        "shared_experts": 2,
        "free_experts": 8,
        "tables": 8,
        "n_buckets": 4096,
        "ignore_index": -100,
        "semantic_vocab_ref": DEFAULT_SEMANTIC_VOCAB_REF,
    },
)

broadcast_chunk_routes_module = module_neuron(
    name="broadcast_chunk_routes",
    module_type="broadcast_chunk_routes",
    input_ports=[
        Port("hidden", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("expert_weights", range=(0, 1), precision=0.001, dtype="tensor"),
        Port("expert_indices", range=(0, 1000), precision=1.0, dtype="tensor"),
    ],
    output_ports=[
        Port("routing_weights", range=(0, 1), precision=0.001, dtype="tensor"),
        Port("routing_indices", range=(0, 1000), precision=1.0, dtype="tensor"),
    ],
    module_config={"chunk_size": 32},
)

route_balance_loss_module = module_neuron(
    name="route_balance_loss",
    module_type="route_balance_loss",
    input_ports=[Port("route_logits", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[Port("loss", range=(0, 100), precision=0.0001, dtype="loss")],
    module_config={},
)

route_selection_loss_module = module_neuron(
    name="route_selection_loss",
    module_type="route_selection_loss",
    input_ports=[
        Port("route_logits", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("sem_targets", range=(-100, 65535), precision=1.0, dtype="tokens"),
    ],
    output_ports=[Port("loss", range=(0, 100), precision=0.0001, dtype="loss")],
    module_config={"semantic_vocab_ref": DEFAULT_SEMANTIC_VOCAB_REF, "shared_experts": 2, "free_experts": 8, "ignore_index": -100},
)

route_distillation_loss_module = module_neuron(
    name="route_distillation_loss",
    module_type="route_distillation_loss",
    input_ports=[
        Port("student_route_logits", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("target_topic_logits", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    output_ports=[Port("loss", range=(0, 100), precision=0.0001, dtype="loss")],
    module_config={"semantic_vocab_ref": DEFAULT_SEMANTIC_VOCAB_REF, "shared_experts": 2, "free_experts": 8},
)

broadcast_expert_routes_module = module_neuron(
    name="broadcast_expert_routes",
    module_type="broadcast_expert_routes",
    input_ports=[
        Port("hidden", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("expert_weights", range=(0, 1), precision=0.001, dtype="tensor"),
        Port("expert_indices", range=(0, 100), precision=1.0, dtype="tensor"),
    ],
    output_ports=[
        Port("routing_weights", range=(0, 1), precision=0.001, dtype="tensor"),
        Port("routing_indices", range=(0, 100), precision=1.0, dtype="tensor"),
    ],
    module_config={},
)

routed_attention_experts_module = module_neuron(
    name="routed_attention_experts",
    module_type="routed_attention_experts",
    input_ports=[
        Port("hidden", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("expert_weights", range=(0, 1), precision=0.001, dtype="tensor"),
        Port("expert_indices", range=(0, 100), precision=1.0, dtype="tensor"),
    ],
    output_ports=[Port("hidden_out", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config={
        "model_dim": 128,
        "num_heads": 4,
        "num_kv_heads": 4,
        "rope_base": 10000.0,
        "qk_gain_init": 1.0,
        "experts": NUM_VOCAB_DIMS,
        "top_k": 2,
        "is_causal": True,
    },
)

attentionless_decoder_module = module_neuron(
    name="attentionless_decoder",
    module_type="attentionless_decoder",
    input_ports=[
        Port("bucket_indices", range=(0, 1_000_000), precision=1.0, dtype="tensor"),
        Port("expert_output", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    output_ports=[Port("logits", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config={"semantic_dim": NUM_SEMANTIC_DIMS, "residual_dim": 64, "vocab_size": 256, "n_buckets": 256},
)

softmax_distillation_loss_module = module_neuron(
    name="softmax_distillation_loss",
    module_type="softmax_distillation_loss",
    input_ports=[
        Port("teacher_logits", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("student_logits", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    output_ports=[Port("loss", range=(0, 100), precision=0.0001, dtype="loss")],
    module_config={},
)

# ── Fine-tuning operators ─────────────────────────────────────────────
# Parameter-efficient fine-tuning (LoRA, qLoRA), SFT, DPO, and RLHF/PPO.
# These auto-populate the editor node library and are wired into graphs
# by ``neuralfn/torch_templates.py`` via the ``adapter_type`` branch on
# ``BlockSpec`` and the objective-aware root-graph builders.

lora_linear_module = module_neuron(
    name="lora_linear",
    module_type="lora_linear",
    input_ports=[Port("x", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[Port("y", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config={
        "input_dim": 128,
        "output_dim": 128,
        "rank": 8,
        "alpha": 16.0,
        "dropout": 0.0,
        "bias": False,
        "merge_on_eval": False,
    },
)

nf4_linear_module = module_neuron(
    name="nf4_linear",
    module_type="nf4_linear",
    input_ports=[Port("x", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[Port("y", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config={
        "input_dim": 128,
        "output_dim": 128,
        "rank": 8,
        "alpha": 16.0,
        "dropout": 0.0,
        "bias": False,
        "group_size": 64,
        "compute_dtype": "bf16",
    },
)

masked_token_cross_entropy_module = module_neuron(
    name="masked_token_cross_entropy",
    module_type="masked_token_cross_entropy",
    input_ports=[
        Port("logits", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("targets", range=(0, 65535), precision=1.0, dtype="tokens"),
        Port("loss_mask", range=(0, 1), precision=0.001, dtype="tensor"),
    ],
    output_ports=[Port("loss", range=(0, 100), precision=0.0001, dtype="loss")],
    module_config={"ignore_index": -100},
)

reference_forward_module = module_neuron(
    name="reference_forward",
    module_type="reference_forward",
    input_ports=[Port("tokens", range=(0, 65535), precision=1.0, dtype="tokens")],
    output_ports=[Port("ref_logits", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config={
        "ref_graph_path": "",
        "ref_weights_path": "",
        "vocab_size": 256,
        "model_dim": 128,
        "num_layers": 4,
        "num_heads": 4,
    },
)

sft_dataset_source_module = module_neuron(
    name="sft_dataset_source",
    module_type="sft_dataset_source",
    input_ports=[],
    output_ports=[
        Port("tokens", range=(0, 65535), precision=1.0, dtype="tokens"),
        Port("targets", range=(0, 65535), precision=1.0, dtype="tokens"),
        Port("loss_mask", range=(0, 1), precision=0.001, dtype="tensor"),
    ],
    module_config={
        "dataset_names": [],
        "seq_len": 64,
        "prompt_field": "prompt",
        "response_field": "response",
        "format": "chat",
        "mask_prompt": True,
    },
)

# DPO / reward-model operators --------------------------------------------------

sequence_logp_module = module_neuron(
    name="sequence_logp",
    module_type="sequence_logp",
    input_ports=[
        Port("logits", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("targets", range=(0, 65535), precision=1.0, dtype="tokens"),
        Port("loss_mask", range=(0, 1), precision=0.001, dtype="tensor"),
    ],
    output_ports=[Port("logp", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config={"ignore_index": -100},
)

dpo_pairwise_loss_module = module_neuron(
    name="dpo_pairwise_loss",
    module_type="dpo_pairwise_loss",
    input_ports=[
        Port("policy_logp_chosen", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("policy_logp_rejected", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("ref_logp_chosen", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("ref_logp_rejected", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    output_ports=[
        Port("loss", range=(0, 100), precision=0.0001, dtype="loss"),
        Port("chosen_reward", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("rejected_reward", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    module_config={"beta": 0.1, "label_smoothing": 0.0, "loss_type": "sigmoid"},
)

dpo_dataset_source_module = module_neuron(
    name="dpo_dataset_source",
    module_type="dpo_dataset_source",
    input_ports=[],
    output_ports=[
        Port("tokens_chosen", range=(0, 65535), precision=1.0, dtype="tokens"),
        Port("tokens_rejected", range=(0, 65535), precision=1.0, dtype="tokens"),
        Port("targets_chosen", range=(0, 65535), precision=1.0, dtype="tokens"),
        Port("targets_rejected", range=(0, 65535), precision=1.0, dtype="tokens"),
        Port("mask_chosen", range=(0, 1), precision=0.001, dtype="tensor"),
        Port("mask_rejected", range=(0, 1), precision=0.001, dtype="tensor"),
    ],
    module_config={
        "dataset_names": [],
        "seq_len": 64,
        "prompt_field": "prompt",
        "chosen_field": "chosen",
        "rejected_field": "rejected",
    },
)

reward_head_module = module_neuron(
    name="reward_head",
    module_type="reward_head",
    input_ports=[Port("hidden", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[Port("scalar", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config={"model_dim": 128, "pool": "last"},
)

preference_bce_loss_module = module_neuron(
    name="preference_bce_loss",
    module_type="preference_bce_loss",
    input_ports=[
        Port("reward_chosen", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("reward_rejected", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    output_ports=[Port("loss", range=(0, 100), precision=0.0001, dtype="loss")],
    module_config={},
)

# PPO operators ----------------------------------------------------------------

value_head_module = module_neuron(
    name="value_head",
    module_type="value_head",
    input_ports=[Port("hidden", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    output_ports=[Port("value", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config={"model_dim": 128},
)

ppo_clipped_loss_module = module_neuron(
    name="ppo_clipped_loss",
    module_type="ppo_clipped_loss",
    input_ports=[
        Port("logp_new", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("logp_old", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("advantages", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("value_new", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("value_old", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("returns", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    output_ports=[
        Port("policy_loss", range=(0, 100), precision=0.0001, dtype="tensor"),
        Port("value_loss", range=(0, 100), precision=0.0001, dtype="tensor"),
        Port("loss", range=(0, 100), precision=0.0001, dtype="loss"),
    ],
    module_config={"clip_range": 0.2, "vf_coef": 0.5, "ent_coef": 0.0},
)

kl_penalty_module = module_neuron(
    name="kl_penalty",
    module_type="kl_penalty",
    input_ports=[
        Port("logp_policy", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("logp_ref", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("rewards", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    output_ports=[Port("shaped_rewards", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config={"kl_coef": 0.1},
)

reward_forward_module = module_neuron(
    name="reward_forward",
    module_type="reward_forward",
    input_ports=[Port("tokens", range=(0, 65535), precision=1.0, dtype="tokens")],
    output_ports=[Port("reward", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor")],
    module_config={
        "reward_graph_path": "",
        "reward_weights_path": "",
        "model_dim": 128,
    },
)

ppo_rollout_source_module = module_neuron(
    name="ppo_rollout_source",
    module_type="ppo_rollout_source",
    input_ports=[],
    output_ports=[
        Port("tokens", range=(0, 65535), precision=1.0, dtype="tokens"),
        Port("targets", range=(0, 65535), precision=1.0, dtype="tokens"),
        Port("loss_mask", range=(0, 1), precision=0.001, dtype="tensor"),
        Port("logp_old", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("value_old", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("advantages", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("returns", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    module_config={"seq_len": 64, "rollout_length": 64},
)

gae_compute_module = module_neuron(
    name="gae_compute",
    module_type="gae_compute",
    input_ports=[
        Port("rewards", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("values", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    output_ports=[
        Port("advantages", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
        Port("returns", range=(-1_000_000, 1_000_000), precision=0.001, dtype="tensor"),
    ],
    module_config={"gamma": 1.0, "lambda_": 0.95},
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
    "fused_causal_attention_module": fused_causal_attention_module,
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
    "kv_pca_encode_module": kv_pca_encode_module,
    "kv_pca_decode_module": kv_pca_decode_module,
    "kv_quant_pack_module": kv_quant_pack_module,
    "kv_quant_unpack_module": kv_quant_unpack_module,
    "router_logits_module": router_logits_module,
    "topk_route_module": topk_route_module,
    "expert_dispatch_module": expert_dispatch_module,
    "expert_combine_module": expert_combine_module,
    "load_balance_loss_module": load_balance_loss_module,
    "aux_loss_add_module": aux_loss_add_module,
    "loss_scale_module": loss_scale_module,
    "token_cross_entropy_module": token_cross_entropy_module,
    "dataset_source_module": dataset_source_module,
    "bitlinear_ternary_module": bitlinear_ternary_module,
    "randmap_adapter_module": randmap_adapter_module,
    "mamba_module": mamba_module,
    "denoise_head_module": denoise_head_module,
    "mask_scheduler_module": mask_scheduler_module,
    "random_timesteps_module": random_timesteps_module,
    "jepa_mask_module": jepa_mask_module,
    "latent_pool_module": latent_pool_module,
    "jepa_projector_module": jepa_projector_module,
    "jepa_predictor_module": jepa_predictor_module,
    "latent_mse_loss_module": latent_mse_loss_module,
    "byte_patch_embed_module": byte_patch_embed_module,
    "byte_patch_merge_module": byte_patch_merge_module,
    "act_halt_gate_module": act_halt_gate_module,
    "act_weighted_sum_module": act_weighted_sum_module,
    "universal_transformer_module": universal_transformer_module,
    "ttt_linear_module": ttt_linear_module,
    "semantic_data_source_module": semantic_data_source_module,
    "semantic_projector_module": semantic_projector_module,
    "semantic_alignment_loss_module": semantic_alignment_loss_module,
    "semantic_hasher_module": semantic_hasher_module,
    "semantic_moe_router_module": semantic_moe_router_module,
    "semantic_hash_router_module": semantic_hash_router_module,
    "causal_chunk_state_module": causal_chunk_state_module,
    "semantic_chunk_projector_module": semantic_chunk_projector_module,
    "semantic_chunk_hasher_module": semantic_chunk_hasher_module,
    "semantic_moe_jepa_evo_router_module": semantic_moe_jepa_evo_router_module,
    "broadcast_chunk_routes_module": broadcast_chunk_routes_module,
    "route_balance_loss_module": route_balance_loss_module,
    "route_selection_loss_module": route_selection_loss_module,
    "route_distillation_loss_module": route_distillation_loss_module,
    "broadcast_expert_routes_module": broadcast_expert_routes_module,
    "routed_attention_experts_module": routed_attention_experts_module,
    "attentionless_decoder_module": attentionless_decoder_module,
    "softmax_distillation_loss_module": softmax_distillation_loss_module,
    "lora_linear_module": lora_linear_module,
    "nf4_linear_module": nf4_linear_module,
    "masked_token_cross_entropy_module": masked_token_cross_entropy_module,
    "reference_forward_module": reference_forward_module,
    "sft_dataset_source_module": sft_dataset_source_module,
    "sequence_logp_module": sequence_logp_module,
    "dpo_pairwise_loss_module": dpo_pairwise_loss_module,
    "dpo_dataset_source_module": dpo_dataset_source_module,
    "reward_head_module": reward_head_module,
    "preference_bce_loss_module": preference_bce_loss_module,
    "value_head_module": value_head_module,
    "ppo_clipped_loss_module": ppo_clipped_loss_module,
    "kl_penalty_module": kl_penalty_module,
    "reward_forward_module": reward_forward_module,
    "ppo_rollout_source_module": ppo_rollout_source_module,
    "gae_compute_module": gae_compute_module,
}

_normalize_builtin_ports(*_BUILTIN_ATTR_MAP.values())


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
    fused_causal_attention_module = fused_causal_attention_module
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
    kv_pca_encode_module = kv_pca_encode_module
    kv_pca_decode_module = kv_pca_decode_module
    kv_quant_pack_module = kv_quant_pack_module
    kv_quant_unpack_module = kv_quant_unpack_module
    router_logits_module = router_logits_module
    topk_route_module = topk_route_module
    expert_dispatch_module = expert_dispatch_module
    expert_combine_module = expert_combine_module
    load_balance_loss_module = load_balance_loss_module
    aux_loss_add_module = aux_loss_add_module
    loss_scale_module = loss_scale_module
    token_cross_entropy_module = token_cross_entropy_module
    dataset_source_module = dataset_source_module
    bitlinear_ternary_module = bitlinear_ternary_module
    randmap_adapter_module = randmap_adapter_module
    mamba_module = mamba_module
    denoise_head_module = denoise_head_module
    mask_scheduler_module = mask_scheduler_module
    random_timesteps_module = random_timesteps_module
    jepa_mask_module = jepa_mask_module
    latent_pool_module = latent_pool_module
    jepa_projector_module = jepa_projector_module
    jepa_predictor_module = jepa_predictor_module
    latent_mse_loss_module = latent_mse_loss_module
    byte_patch_embed_module = byte_patch_embed_module
    byte_patch_merge_module = byte_patch_merge_module
    act_halt_gate_module = act_halt_gate_module
    act_weighted_sum_module = act_weighted_sum_module
    universal_transformer_module = universal_transformer_module
    ttt_linear_module = ttt_linear_module
    semantic_data_source_module = semantic_data_source_module
    semantic_projector_module = semantic_projector_module
    semantic_alignment_loss_module = semantic_alignment_loss_module
    semantic_hasher_module = semantic_hasher_module
    semantic_moe_router_module = semantic_moe_router_module
    semantic_hash_router_module = semantic_hash_router_module
    causal_chunk_state_module = causal_chunk_state_module
    semantic_chunk_projector_module = semantic_chunk_projector_module
    semantic_chunk_hasher_module = semantic_chunk_hasher_module
    semantic_moe_jepa_evo_router_module = semantic_moe_jepa_evo_router_module
    broadcast_chunk_routes_module = broadcast_chunk_routes_module
    route_balance_loss_module = route_balance_loss_module
    route_selection_loss_module = route_selection_loss_module
    route_distillation_loss_module = route_distillation_loss_module
    broadcast_expert_routes_module = broadcast_expert_routes_module
    routed_attention_experts_module = routed_attention_experts_module
    attentionless_decoder_module = attentionless_decoder_module
    softmax_distillation_loss_module = softmax_distillation_loss_module
    lora_linear_module = lora_linear_module
    nf4_linear_module = nf4_linear_module
    masked_token_cross_entropy_module = masked_token_cross_entropy_module
    reference_forward_module = reference_forward_module
    sft_dataset_source_module = sft_dataset_source_module
    sequence_logp_module = sequence_logp_module
    dpo_pairwise_loss_module = dpo_pairwise_loss_module
    dpo_dataset_source_module = dpo_dataset_source_module
    reward_head_module = reward_head_module
    preference_bce_loss_module = preference_bce_loss_module
    value_head_module = value_head_module
    ppo_clipped_loss_module = ppo_clipped_loss_module
    kl_penalty_module = kl_penalty_module
    reward_forward_module = reward_forward_module
    ppo_rollout_source_module = ppo_rollout_source_module
    gae_compute_module = gae_compute_module

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
    "fused_causal_attention_module",
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
    "kv_pca_encode_module",
    "kv_pca_decode_module",
    "kv_quant_pack_module",
    "kv_quant_unpack_module",
    "router_logits_module",
    "topk_route_module",
    "expert_dispatch_module",
    "expert_combine_module",
    "load_balance_loss_module",
    "aux_loss_add_module",
    "loss_scale_module",
    "token_cross_entropy_module",
    "dataset_source_module",
    "bitlinear_ternary_module",
    "randmap_adapter_module",
    "mamba_module",
    "denoise_head_module",
    "mask_scheduler_module",
    "random_timesteps_module",
    "jepa_mask_module",
    "latent_pool_module",
    "jepa_projector_module",
    "jepa_predictor_module",
    "latent_mse_loss_module",
    "byte_patch_embed_module",
    "byte_patch_merge_module",
    "act_halt_gate_module",
    "act_weighted_sum_module",
    "universal_transformer_module",
    "ttt_linear_module",
    "semantic_data_source_module",
    "semantic_projector_module",
    "semantic_alignment_loss_module",
    "semantic_hasher_module",
    "semantic_moe_router_module",
    "semantic_hash_router_module",
    "causal_chunk_state_module",
    "semantic_chunk_projector_module",
    "semantic_chunk_hasher_module",
    "semantic_moe_jepa_evo_router_module",
    "broadcast_chunk_routes_module",
    "route_balance_loss_module",
    "route_selection_loss_module",
    "route_distillation_loss_module",
    "broadcast_expert_routes_module",
    "routed_attention_experts_module",
    "attentionless_decoder_module",
    "softmax_distillation_loss_module",
    "lora_linear_module",
    "nf4_linear_module",
    "masked_token_cross_entropy_module",
    "reference_forward_module",
    "sft_dataset_source_module",
    "sequence_logp_module",
    "dpo_pairwise_loss_module",
    "dpo_dataset_source_module",
    "reward_head_module",
    "preference_bce_loss_module",
    "value_head_module",
    "ppo_clipped_loss_module",
    "kl_penalty_module",
    "reward_forward_module",
    "ppo_rollout_source_module",
    "gae_compute_module",
]
