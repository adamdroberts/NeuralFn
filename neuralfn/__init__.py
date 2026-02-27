"""NeuralFn — a brain-inspired function-neuron graph framework."""

from .port import Port
from .neuron import NeuronDef, neuron, neuron_from_source
from .graph import Edge, NeuronInstance, NeuronGraph
from .surrogate import SurrogateModel, probe_neuron, build_surrogates
from .trainer import SurrogateTrainer
from .evolutionary import EvolutionaryTrainer
from .serialization import save_graph, load_graph

__all__ = [
    "Port",
    "NeuronDef",
    "neuron",
    "neuron_from_source",
    "Edge",
    "NeuronInstance",
    "NeuronGraph",
    "SurrogateModel",
    "probe_neuron",
    "build_surrogates",
    "SurrogateTrainer",
    "EvolutionaryTrainer",
    "save_graph",
    "load_graph",
]
