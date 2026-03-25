"""Library of common neuron functions ready to drop into a graph."""

from __future__ import annotations

import math

from .neuron import NeuronDef, neuron
from .port import Port


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
]
