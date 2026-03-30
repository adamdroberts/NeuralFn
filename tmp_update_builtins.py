import re

with open("neuralfn/builtins.py", "r") as f:
    text = f.read()

new_modules = """
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

_BUILTIN_ATTR_MAP"""

target_attr_map = "_BUILTIN_ATTR_MAP: dict[str, NeuronDef] = {"

if target_attr_map in text:
    text = text.replace(target_attr_map, new_modules + "\n\n" + target_attr_map)

mod_names = ["layer_norm_module", "dropout_module", "gelu_module", "swiglu_module", "absolute_position_embedding_module", "kv_cache_read_module", "kv_cache_write_module", "router_logits_module", "topk_route_module", "expert_dispatch_module", "expert_combine_module", "load_balance_loss_module", "aux_loss_add_module"]

for m in mod_names:
    text = text.replace('    "token_cross_entropy_module": token_cross_entropy_module,', f'    "{m}": {m},\n    "token_cross_entropy_module": token_cross_entropy_module,')

class_block = "\\n    ".join([f"{m} = {m}" for m in mod_names])
text = re.sub(r'(\s+token_cross_entropy_module = token_cross_entropy_module)', r'\\n    ' + class_block + r'\1', text, count=1)

list_block = ",\n    ".join([f'"{m}"' for m in mod_names])
text = text.replace('"token_cross_entropy_module",\n]', f'{list_block},\n    "token_cross_entropy_module",\n]')

with open("neuralfn/builtins.py", "w") as f:
    f.write(text)
