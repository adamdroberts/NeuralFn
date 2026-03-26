import type { Edge as RFEdge, Node } from "@xyflow/react";
import type { EdgeData, GraphData, NeuronDefData, NodeData, PortData, TrainingMethod } from "../api/client";

export interface FlowNodeData extends Record<string, unknown> {
  label: string;
  neuronDef: NeuronDefData;
  isInput: boolean;
  isOutput: boolean;
}

export interface Breadcrumb {
  id: string;
  label: string;
}

export function createEmptyGraph(name = "Graph"): GraphData {
  return {
    name,
    training_method: "surrogate",
    surrogate_config: {},
    evo_config: {},
    nodes: {},
    edges: {},
    input_node_ids: [],
    output_node_ids: [],
  };
}

export function createCustomNeuronDef(name = "custom"): NeuronDefData {
  return {
    id: "",
    name,
    kind: "function",
    input_ports: [{ name: "x", range: [-10, 10], precision: 0.001, dtype: "float" }],
    output_ports: [{ name: "y", range: [-10, 10], precision: 0.001, dtype: "float" }],
    source_code: `def ${name}(x):\n    return x\n`,
    subgraph: null,
    input_aliases: [],
    output_aliases: [],
  };
}

const DEFAULT_INPUT_DEF: NeuronDefData = {
  id: "builtin-input",
  name: "input",
  kind: "function",
  input_ports: [{ name: "in", range: [-100, 100], precision: 0.001, dtype: "float" }],
  output_ports: [{ name: "out", range: [-100, 100], precision: 0.001, dtype: "float" }],
  source_code: "def input_node(x):\n    return x\n",
  subgraph: null,
  input_aliases: [],
  output_aliases: [],
};

const DEFAULT_OUTPUT_DEF: NeuronDefData = {
  id: "builtin-output",
  name: "output",
  kind: "function",
  input_ports: [{ name: "in", range: [-100, 100], precision: 0.001, dtype: "float" }],
  output_ports: [{ name: "out", range: [-100, 100], precision: 0.001, dtype: "float" }],
  source_code: "def output_node(x):\n    return x\n",
  subgraph: null,
  input_aliases: [],
  output_aliases: [],
};

export function createSubgraphNeuronDef(name = "subgraph"): NeuronDefData {
  const subgraph = createEmptyGraph(`${name} graph`);
  const inId = "n-in";
  const outId = "n-out";

  subgraph.nodes[inId] = {
    instance_id: inId,
    position: [50, 150],
    neuron_def: DEFAULT_INPUT_DEF,
    measured: undefined,
  };

  subgraph.nodes[outId] = {
    instance_id: outId,
    position: [350, 150],
    neuron_def: DEFAULT_OUTPUT_DEF,
    measured: undefined,
  };

  subgraph.input_node_ids = [inId];
  subgraph.output_node_ids = [outId];

  return normalizeNeuronDef({
    id: "",
    name,
    kind: "subgraph",
    input_ports: [],
    output_ports: [],
    source_code: "",
    subgraph: subgraph,
    input_aliases: ["x"],
    output_aliases: ["y"],
  });
}

export function normalizeGraph(graph: GraphData | null | undefined, fallbackName = "Graph"): GraphData {
  const base = createEmptyGraph(fallbackName);
  const raw = graph ?? base;
  const normalized: GraphData = {
    name: raw.name ?? base.name,
    training_method: (raw.training_method ?? base.training_method) as TrainingMethod,
    surrogate_config: { ...(raw.surrogate_config ?? {}) },
    evo_config: { ...(raw.evo_config ?? {}) },
    nodes: {},
    edges: {},
    input_node_ids: [],
    output_node_ids: [],
  };

  for (const [id, node] of Object.entries(raw.nodes ?? {})) {
    normalized.nodes[id] = normalizeNode(node, id);
  }

  const validIds = new Set(Object.keys(normalized.nodes));

  normalized.edges = Object.fromEntries(
    Object.entries(raw.edges ?? {}).filter(
      ([, edge]) => validIds.has(edge.src_node) && validIds.has(edge.dst_node),
    ),
  );
  normalized.input_node_ids = (raw.input_node_ids ?? []).filter((id) => validIds.has(id));
  normalized.output_node_ids = (raw.output_node_ids ?? []).filter((id) => validIds.has(id));

  return normalized;
}

export function normalizeNeuronDef(ndef: NeuronDefData): NeuronDefData {
  const kind = ndef.kind ?? "function";
  if (kind === "subgraph") {
    const subgraph = normalizeGraph(ndef.subgraph, `${ndef.name} graph`);
    const inputPorts = deriveExternalPorts(subgraph, ndef.input_aliases, "input");
    const outputPorts = deriveExternalPorts(subgraph, ndef.output_aliases, "output");
    return {
      ...ndef,
      kind: "subgraph",
      source_code: "",
      subgraph,
      input_ports: inputPorts,
      output_ports: outputPorts,
      input_aliases: inputPorts.map((port) => port.name),
      output_aliases: outputPorts.map((port) => port.name),
    };
  }

  return {
    ...createCustomNeuronDef(ndef.name || "custom"),
    ...ndef,
    kind: "function",
    subgraph: null,
    input_aliases: [],
    output_aliases: [],
  };
}

function normalizeNode(node: NodeData, fallbackId: string): NodeData {
  return {
    instance_id: node.instance_id || fallbackId,
    position: node.position ?? [0, 0],
    measured: node.measured,
    neuron_def: normalizeNeuronDef(node.neuron_def),
  };
}

function deriveExternalPorts(
  graph: GraphData,
  aliases: string[] | undefined,
  direction: "input" | "output",
): PortData[] {
  const nodeIds = direction === "input" ? graph.input_node_ids : graph.output_node_ids;
  const flattened: Array<{ defaultName: string; port: PortData }> = [];

  for (const nodeId of nodeIds) {
    const node = graph.nodes[nodeId];
    if (!node) continue;
    for (const port of node.neuron_def.output_ports) {
      flattened.push({
        defaultName: `${nodeId}.${port.name}`,
        port,
      });
    }
  }

  return flattened.map((entry, index) => ({
    ...entry.port,
    name: aliases?.[index] ?? entry.defaultName,
  }));
}

export function graphToFlowNodes(graph: GraphData): Node<FlowNodeData>[] {
  return Object.entries(graph.nodes).map(([id, node]) => ({
    id,
    type: "neuron",
    position: { x: node.position[0], y: node.position[1] },
    measured: node.measured,
    data: {
      label: node.neuron_def.name,
      neuronDef: normalizeNeuronDef(node.neuron_def),
      isInput: graph.input_node_ids.includes(id),
      isOutput: graph.output_node_ids.includes(id),
    },
  }));
}

export function graphToFlowEdges(graph: GraphData): RFEdge[] {
  return Object.values(graph.edges).map((edge) => ({
    id: edge.id,
    source: edge.src_node,
    target: edge.dst_node,
    sourceHandle: `out-${edge.src_port}`,
    targetHandle: `in-${edge.dst_port}`,
    type: "default",
    data: { weight: edge.weight, bias: edge.bias },
  })) as RFEdge[];
}

export function flowNodesToGraphNodes(
  graph: GraphData,
  nodes: Node<FlowNodeData>[],
): Record<string, NodeData> {
  const nextNodes: Record<string, NodeData> = {};
  for (const node of nodes) {
    nextNodes[node.id] = {
      instance_id: node.id,
      position: [node.position.x, node.position.y],
      measured: node.measured,
      neuron_def: normalizeNeuronDef(node.data.neuronDef),
    };
  }

  for (const [id, node] of Object.entries(graph.nodes)) {
    if (!nextNodes[id]) continue;
    nextNodes[id] = {
      ...nextNodes[id],
      measured: nextNodes[id].measured ?? node.measured,
      neuron_def: normalizeNeuronDef(nextNodes[id].neuron_def ?? node.neuron_def),
    };
  }

  return nextNodes;
}

export function flowEdgesToGraphEdges(edges: RFEdge[]): Record<string, EdgeData> {
  return Object.fromEntries(
    edges.map((edge) => [
      edge.id,
      {
        id: edge.id,
        src_node: edge.source,
        src_port: parseInt((edge.sourceHandle ?? "out-0").split("-")[1], 10) || 0,
        dst_node: edge.target,
        dst_port: parseInt((edge.targetHandle ?? "in-0").split("-")[1], 10) || 0,
        weight: ((edge.data as { weight?: number } | undefined)?.weight ?? 1.0),
        bias: ((edge.data as { bias?: number } | undefined)?.bias ?? 0.0),
      } satisfies EdgeData,
    ])
  );
}

export function getGraphAtPath(root: GraphData, path: string[]): GraphData {
  let current = root;
  for (const nodeId of path) {
    const node = current.nodes[nodeId];
    const subgraph = node?.neuron_def.subgraph;
    if (!subgraph) break;
    current = subgraph;
  }
  return current;
}

export function clampGraphPath(root: GraphData, path: string[]): string[] {
  const clamped: string[] = [];
  let current = root;

  for (const nodeId of path) {
    const node = current.nodes[nodeId];
    const subgraph = node?.neuron_def.subgraph;
    if (!node || node.neuron_def.kind !== "subgraph" || !subgraph) {
      break;
    }
    clamped.push(nodeId);
    current = subgraph;
  }

  return clamped;
}

export function updateGraphAtPath(
  root: GraphData,
  path: string[],
  updater: (graph: GraphData) => GraphData,
): GraphData {
  const normalizedRoot = normalizeGraph(root, root.name);
  const clampedPath = clampGraphPath(normalizedRoot, path);

  if (clampedPath.length === 0) {
    return normalizeGraph(updater(normalizedRoot), normalizedRoot.name);
  }

  const [nodeId, ...rest] = clampedPath;
  const node = normalizedRoot.nodes[nodeId];
  const child = node?.neuron_def.subgraph;
  if (!node || !child) {
    return normalizedRoot;
  }

  const updatedChild = updateGraphAtPath(child, rest, updater);
  const updatedNode: NodeData = {
    ...node,
    neuron_def: normalizeNeuronDef({
      ...node.neuron_def,
      subgraph: updatedChild,
    }),
  };

  return normalizeGraph(
    {
      ...normalizedRoot,
      nodes: {
        ...normalizedRoot.nodes,
        [nodeId]: updatedNode,
      },
    },
    normalizedRoot.name,
  );
}

export function breadcrumbsForPath(root: GraphData, path: string[]): Breadcrumb[] {
  const crumbs: Breadcrumb[] = [{ id: "", label: root.name || "Root" }];
  let current = root;
  for (const nodeId of clampGraphPath(root, path)) {
    const node = current.nodes[nodeId];
    if (!node || !node.neuron_def.subgraph) break;
    crumbs.push({ id: nodeId, label: node.neuron_def.name });
    current = node.neuron_def.subgraph;
  }
  return crumbs;
}

export function graphContainsSubgraphs(graph: GraphData): boolean {
  return Object.values(graph.nodes).some((node) => {
    const child = node.neuron_def.subgraph;
    return node.neuron_def.kind === "subgraph" || (child ? graphContainsSubgraphs(child) : false);
  });
}
