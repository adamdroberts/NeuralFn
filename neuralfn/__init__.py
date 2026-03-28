"""NeuralFn — a brain-inspired function-neuron graph framework."""

from .port import Port
from .neuron import NeuronDef, module_neuron, neuron, neuron_from_source, subgraph_neuron
from .builtins import BuiltinNeurons
from .graph import Edge, NeuronInstance, NeuronGraph
from .surrogate import SurrogateModel, probe_neuron, build_surrogates
from .trainer import SurrogateTrainer
from .evolutionary import EvolutionaryTrainer
from .hybrid import HybridConfig, HybridTrainer
from .serialization import save_graph, load_graph
from .torch_backend import TorchTrainConfig, TorchTrainer
from .torch_templates import build_gpt_root_graph, build_gpt_stage_graph, build_gpt_subgraph

__all__ = [
    "Port",
    "NeuronDef",
    "module_neuron",
    "neuron",
    "neuron_from_source",
    "subgraph_neuron",
    "BuiltinNeurons",
    "Edge",
    "NeuronInstance",
    "NeuronGraph",
    "SurrogateModel",
    "probe_neuron",
    "build_surrogates",
    "SurrogateTrainer",
    "EvolutionaryTrainer",
    "HybridConfig",
    "HybridTrainer",
    "save_graph",
    "load_graph",
    "TorchTrainConfig",
    "TorchTrainer",
    "build_gpt_root_graph",
    "build_gpt_stage_graph",
    "build_gpt_subgraph",
]
