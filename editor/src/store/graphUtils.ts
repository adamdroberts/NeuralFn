import type { Edge as RFEdge, Node } from "@xyflow/react";
import type {
  EdgeData,
  GraphData,
  NeuronDefData,
  NodeData,
  PortData,
  TrainingMethod,
  VariantLibraryData,
  VariantRefData,
} from "../api/client";

export interface FlowNodeData extends Record<string, unknown> {
  label: string;
  neuronDef: NeuronDefData;
  isInput: boolean;
  isOutput: boolean;
}

export type GraphPathSegment =
  | { kind: "node"; nodeId: string }
  | { kind: "variant"; family: string; version: string };

export interface Breadcrumb {
  id: string;
  label: string;
  path: GraphPathSegment[];
}

function deepClone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function createVariantRef(variantRef?: VariantRefData | null): VariantRefData | null {
  if (!variantRef?.family || !variantRef?.version) {
    return null;
  }
  return {
    family: variantRef.family,
    version: variantRef.version,
  };
}

export function createEmptyGraph(name = "Graph"): GraphData {
  return {
    name,
    training_method: "surrogate",
    runtime: "scalar",
    surrogate_config: {},
    evo_config: {},
    torch_config: { device: "cuda", amp_dtype: "bfloat16" },
    variant_library: {},
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
    module_type: "",
    module_config: {},
    module_state: "",
    input_aliases: [],
    output_aliases: [],
    variant_ref: null,
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
  module_type: "",
  module_config: {},
  module_state: "",
  input_aliases: [],
  output_aliases: [],
  variant_ref: null,
};

const DEFAULT_OUTPUT_DEF: NeuronDefData = {
  id: "builtin-output",
  name: "output",
  kind: "function",
  input_ports: [{ name: "in", range: [-100, 100], precision: 0.001, dtype: "float" }],
  output_ports: [{ name: "out", range: [-100, 100], precision: 0.001, dtype: "float" }],
  source_code: "def output_node(x):\n    return x\n",
  subgraph: null,
  module_type: "",
  module_config: {},
  module_state: "",
  input_aliases: [],
  output_aliases: [],
  variant_ref: null,
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
    subgraph,
    module_type: "",
    module_config: {},
    module_state: "",
    input_aliases: ["x"],
    output_aliases: ["y"],
    variant_ref: null,
  });
}

export function createLinkedVariantNeuronDef(
  name: string,
  variantRef: VariantRefData,
  graph: GraphData,
  inputAliases?: string[],
  outputAliases?: string[],
): NeuronDefData {
  return normalizeNeuronDef({
    id: "",
    name,
    kind: "subgraph",
    input_ports: [],
    output_ports: [],
    source_code: "",
    subgraph: graph,
    module_type: "",
    module_config: {},
    module_state: "",
    input_aliases: inputAliases ?? deriveExternalPorts(graph, undefined, "input").map((port) => port.name),
    output_aliases: outputAliases ?? deriveExternalPorts(graph, undefined, "output").map((port) => port.name),
    variant_ref: variantRef,
  });
}

function normalizeVariantLibraryStructure(rawLibrary: VariantLibraryData | null | undefined): VariantLibraryData {
  const normalized: VariantLibraryData = {};
  for (const [family, versions] of Object.entries(rawLibrary ?? {})) {
    normalized[family] = {};
    for (const [version, graph] of Object.entries(versions ?? {})) {
      normalized[family][version] = normalizeGraphStructure(graph, graph?.name ?? `${family}_${version}`, false);
    }
  }
  return normalized;
}

function normalizeGraphStructure(
  graph: GraphData | null | undefined,
  fallbackName = "Graph",
  isRoot = true,
): GraphData {
  const base = createEmptyGraph(fallbackName);
  const raw = graph ?? base;
  const normalized: GraphData = {
    name: raw.name ?? base.name,
    training_method: (raw.training_method ?? base.training_method) as TrainingMethod,
    runtime: (raw.runtime ?? base.runtime) as "scalar" | "torch",
    surrogate_config: { ...(raw.surrogate_config ?? {}) },
    evo_config: { ...(raw.evo_config ?? {}) },
    torch_config: { ...(raw.torch_config ?? {}) },
    variant_library: isRoot ? normalizeVariantLibraryStructure(raw.variant_library) : {},
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

function portsCompatible(current: PortData[], target: PortData[]): boolean {
  if (current.length !== target.length) {
    return false;
  }
  return current.every((port, index) => {
    const other = target[index];
    return (
      port.name === other.name &&
      port.dtype === other.dtype &&
      port.precision === other.precision &&
      port.range[0] === other.range[0] &&
      port.range[1] === other.range[1]
    );
  });
}

function resolveVariantLibrary(root: GraphData): GraphData {
  const workingRoot = deepClone(root);
  const resolvedVariants = new Map<string, GraphData>();
  const resolving = new Set<string>();

  const resolveVariant = (family: string, version: string): GraphData => {
    const key = `${family}@${version}`;
    const cached = resolvedVariants.get(key);
    if (cached) {
      return deepClone(cached);
    }
    if (resolving.has(key)) {
      throw new Error(`Recursive variant reference detected: ${key}`);
    }
    const variant = workingRoot.variant_library[family]?.[version];
    if (!variant) {
      throw new Error(`Missing variant '${key}'`);
    }
    resolving.add(key);
    const resolved = resolveGraph(deepClone(variant));
    resolving.delete(key);
    resolvedVariants.set(key, deepClone(resolved));
    return deepClone(resolved);
  };

  const resolveGraph = (graph: GraphData): GraphData => {
    const resolvedGraph = deepClone(graph);
    for (const [nodeId, node] of Object.entries(resolvedGraph.nodes)) {
      const ndef = node.neuron_def;
      if (ndef.kind !== "subgraph") {
        continue;
      }
      if (ndef.variant_ref) {
        const linked = resolveVariant(ndef.variant_ref.family, ndef.variant_ref.version);
        const expectedInputs = deriveExternalPorts(linked, ndef.input_aliases, "input");
        const expectedOutputs = deriveExternalPorts(linked, ndef.output_aliases, "output");
        if (ndef.input_ports.length > 0 && !portsCompatible(ndef.input_ports, expectedInputs)) {
          throw new Error(
            `Variant '${ndef.variant_ref.family}@${ndef.variant_ref.version}' is incompatible with linked node '${nodeId}' inputs`,
          );
        }
        if (ndef.output_ports.length > 0 && !portsCompatible(ndef.output_ports, expectedOutputs)) {
          throw new Error(
            `Variant '${ndef.variant_ref.family}@${ndef.variant_ref.version}' is incompatible with linked node '${nodeId}' outputs`,
          );
        }
        resolvedGraph.nodes[nodeId] = {
          ...node,
          neuron_def: {
            ...ndef,
            subgraph: linked,
            input_ports: expectedInputs,
            output_ports: expectedOutputs,
            input_aliases: expectedInputs.map((port) => port.name),
            output_aliases: expectedOutputs.map((port) => port.name),
          },
        };
      } else if (ndef.subgraph) {
        const child = resolveGraph(ndef.subgraph);
        const inputPorts = deriveExternalPorts(child, ndef.input_aliases, "input");
        const outputPorts = deriveExternalPorts(child, ndef.output_aliases, "output");
        resolvedGraph.nodes[nodeId] = {
          ...node,
          neuron_def: {
            ...ndef,
            subgraph: child,
            input_ports: inputPorts,
            output_ports: outputPorts,
            input_aliases: inputPorts.map((port) => port.name),
            output_aliases: outputPorts.map((port) => port.name),
          },
        };
      }
    }
    return resolvedGraph;
  };

  const nextLibrary: VariantLibraryData = {};
  for (const family of Object.keys(workingRoot.variant_library)) {
    nextLibrary[family] = {};
    for (const version of Object.keys(workingRoot.variant_library[family])) {
      nextLibrary[family][version] = resolveVariant(family, version);
    }
  }

  const resolvedRoot = resolveGraph(workingRoot);
  resolvedRoot.variant_library = nextLibrary;
  return resolvedRoot;
}

export function normalizeGraph(graph: GraphData | null | undefined, fallbackName = "Graph"): GraphData {
  return resolveVariantLibrary(normalizeGraphStructure(graph, fallbackName, true));
}

export function normalizeNeuronDef(ndef: NeuronDefData): NeuronDefData {
  const kind = ndef.kind ?? "function";
  if (kind === "subgraph") {
    const subgraph = ndef.subgraph
      ? normalizeGraphStructure(ndef.subgraph, `${ndef.name} graph`, false)
      : null;
    const inputPorts = subgraph ? deriveExternalPorts(subgraph, ndef.input_aliases, "input") : [...(ndef.input_ports ?? [])];
    const outputPorts = subgraph ? deriveExternalPorts(subgraph, ndef.output_aliases, "output") : [...(ndef.output_ports ?? [])];
    return {
      ...createCustomNeuronDef(ndef.name || "subgraph"),
      ...ndef,
      kind: "subgraph",
      source_code: "",
      subgraph,
      module_type: "",
      module_config: {},
      module_state: "",
      input_ports: inputPorts,
      output_ports: outputPorts,
      input_aliases: inputPorts.map((port) => port.name),
      output_aliases: outputPorts.map((port) => port.name),
      variant_ref: createVariantRef(ndef.variant_ref),
    };
  }

  if (kind === "module") {
    return {
      ...createCustomNeuronDef(ndef.name || "module"),
      ...ndef,
      kind: "module",
      source_code: "",
      subgraph: null,
      module_type: ndef.module_type || ndef.name || "module",
      module_config: { ...(ndef.module_config ?? {}) },
      module_state: ndef.module_state ?? "",
      input_aliases: [],
      output_aliases: [],
      variant_ref: null,
    };
  }

  return {
    ...createCustomNeuronDef(ndef.name || "custom"),
    ...ndef,
    kind: "function",
    subgraph: null,
    module_type: "",
    module_config: {},
    module_state: "",
    input_aliases: [],
    output_aliases: [],
    variant_ref: null,
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

export function deriveExternalPorts(
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

export function areGraphInterfacesCompatible(left: GraphData, right: GraphData): boolean {
  return (
    portsCompatible(deriveExternalPorts(left, undefined, "input"), deriveExternalPorts(right, undefined, "input")) &&
    portsCompatible(deriveExternalPorts(left, undefined, "output"), deriveExternalPorts(right, undefined, "output"))
  );
}

export function mergeVariantLibraries(
  current: VariantLibraryData,
  incoming: VariantLibraryData,
): VariantLibraryData {
  const next: VariantLibraryData = deepClone(current ?? {});
  for (const [family, versions] of Object.entries(incoming ?? {})) {
    next[family] = {
      ...(next[family] ?? {}),
      ...deepClone(versions),
    };
  }
  return next;
}

export function listCompatibleVariantVersions(
  root: GraphData,
  family: string,
  graph: GraphData | null | undefined,
): string[] {
  const versions = root.variant_library[family] ?? {};
  if (!graph) {
    return Object.keys(versions).sort();
  }
  return Object.entries(versions)
    .filter(([, variantGraph]) => areGraphInterfacesCompatible(graph, variantGraph))
    .map(([version]) => version)
    .sort();
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
    ]),
  );
}

export function getGraphAtPath(root: GraphData, path: GraphPathSegment[]): GraphData {
  let current = root;
  for (const segment of path) {
    if (segment.kind === "variant") {
      const variant = root.variant_library[segment.family]?.[segment.version];
      if (!variant) {
        break;
      }
      current = variant;
      continue;
    }
    const node = current.nodes[segment.nodeId];
    const subgraph = node?.neuron_def.subgraph;
    if (!subgraph) {
      break;
    }
    current = subgraph;
  }
  return current;
}

export function clampGraphPath(root: GraphData, path: GraphPathSegment[]): GraphPathSegment[] {
  const clamped: GraphPathSegment[] = [];
  let current = root;

  for (const segment of path) {
    if (segment.kind === "variant") {
      const variant = root.variant_library[segment.family]?.[segment.version];
      if (!variant) {
        break;
      }
      clamped.push(segment);
      current = variant;
      continue;
    }
    const node = current.nodes[segment.nodeId];
    const subgraph = node?.neuron_def.subgraph;
    if (!node || node.neuron_def.kind !== "subgraph" || !subgraph) {
      break;
    }
    clamped.push(segment);
    current = subgraph;
  }

  return clamped;
}

export function updateGraphAtPath(
  root: GraphData,
  path: GraphPathSegment[],
  updater: (graph: GraphData) => GraphData,
): GraphData {
  const normalizedRoot = normalizeGraph(root, root.name);
  const clampedPath = clampGraphPath(normalizedRoot, path);

  if (clampedPath.length === 0) {
    return normalizeGraph(updater(normalizedRoot), normalizedRoot.name);
  }

  const updateNestedGraph = (
    graph: GraphData,
    nestedPath: GraphPathSegment[],
  ): GraphData => {
    if (nestedPath.length === 0) {
      return updater(graph);
    }
    const [segment, ...rest] = nestedPath;
    if (segment.kind !== "node") {
      return graph;
    }
    const node = graph.nodes[segment.nodeId];
    const child = node?.neuron_def.subgraph;
    if (!node || !child) {
      return graph;
    }
    const updatedChild = updateNestedGraph(child, rest);
    return {
      ...graph,
      nodes: {
        ...graph.nodes,
        [segment.nodeId]: {
          ...node,
          neuron_def: normalizeNeuronDef({
            ...node.neuron_def,
            subgraph: updatedChild,
          }),
        },
      },
    };
  };

  // Variants live in root.variant_library, so preceding node segments are
  // navigation context only. Slice to the last variant for data updates.
  let lastVariantIndex = -1;
  for (let i = clampedPath.length - 1; i >= 0; i--) {
    if (clampedPath[i].kind === "variant") {
      lastVariantIndex = i;
      break;
    }
  }
  const effectivePath = lastVariantIndex >= 0
    ? clampedPath.slice(lastVariantIndex)
    : clampedPath;

  const [segment, ...rest] = effectivePath;
  if (segment.kind === "variant") {
    const familyVariants = normalizedRoot.variant_library[segment.family];
    const target = familyVariants?.[segment.version];
    if (!familyVariants || !target) {
      return normalizedRoot;
    }
    return normalizeGraph(
      {
        ...normalizedRoot,
        variant_library: {
          ...normalizedRoot.variant_library,
          [segment.family]: {
            ...familyVariants,
            [segment.version]: updateNestedGraph(target, rest),
          },
        },
      },
      normalizedRoot.name,
    );
  }

  return normalizeGraph(updateNestedGraph(normalizedRoot, effectivePath), normalizedRoot.name);
}

export function breadcrumbsForPath(root: GraphData, path: GraphPathSegment[]): Breadcrumb[] {
  const crumbs: Breadcrumb[] = [{ id: "", label: root.name || "Root", path: [] }];
  const clamped = clampGraphPath(root, path);
  let current = root;

  clamped.forEach((segment, index) => {
    const nextPath = clamped.slice(0, index + 1);
    if (segment.kind === "variant") {
      crumbs.push({
        id: `${segment.family}@${segment.version}`,
        label: `${segment.family} / ${segment.version}`,
        path: nextPath,
      });
      const variant = root.variant_library[segment.family]?.[segment.version];
      if (variant) {
        current = variant;
      }
      return;
    }
    const node = current.nodes[segment.nodeId];
    if (!node?.neuron_def.subgraph) {
      return;
    }
    crumbs.push({
      id: segment.nodeId,
      label: node.neuron_def.name,
      path: nextPath,
    });
    current = node.neuron_def.subgraph;
  });

  return crumbs;
}

export function graphContainsSubgraphs(graph: GraphData): boolean {
  return Object.values(graph.nodes).some((node) => {
    const child = node.neuron_def.subgraph;
    return node.neuron_def.kind === "subgraph" || (child ? graphContainsSubgraphs(child) : false);
  });
}
