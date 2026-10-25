import type { Edge as RFEdge, Node } from "@xyflow/react";
import type {
  EdgeData,
  GraphData,
  GraphRuntime,
  NeuronDefData,
  NodeData,
  PortData,
  TrainingMethod,
  VariantLibraryData,
  VariantRefData,
} from "../api/client";

const VARIANT_FAMILY_ALIASES: Record<string, string[]> = {
  gpt2: ["transformer_block"],
  nanogpt: ["transformer_block"],
  llama: ["transformer_block"],
  attn_block: ["transformer_block"],
  transformer_block: ["attn_block"],
  mixllama: ["attn_block"],
};

export interface CoreMorphSource {
  archetypeId: string;
  neuronName: string;
  primary: string;
  secondary: string;
}

export interface FlowNodeData extends Record<string, unknown> {
  label: string;
  neuronDef: NeuronDefData;
  isInput: boolean;
  isOutput: boolean;
  subgraphPath?: GraphPathSegment[];
  instancePath?: GraphPathSegment[];
  localNodeId?: string;
  breadcrumbLabel?: string;
  spawnScale?: number;
  titleDetached?: number;
  coreOnlyMorph?: number;
  coreMorphProgress?: number;
  coreMorphFrom?: CoreMorphSource;
  shellExpandProgress?: number;
}

export interface SubgraphZone {
  id: string;
  label: string;
  breadcrumb: string;
  path: GraphPathSegment[];
  instancePath: GraphPathSegment[];
  x: number;
  y: number;
  width: number;
  height: number;
  depth: number;
  variantLabel: string | null;
  nodeCount: number;
}

export interface FullExpandedGraphResult {
  nodes: Node<FlowNodeData>[];
  edges: RFEdge[];
  zones: SubgraphZone[];
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
    input_ports: [{ name: "x", range: null, precision: null, dtype: "float" }],
    output_ports: [{ name: "y", range: null, precision: null, dtype: "float" }],
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
  input_ports: [{ name: "in", range: null, precision: null, dtype: "tensor" }],
  output_ports: [{ name: "out", range: null, precision: null, dtype: "tensor" }],
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
  input_ports: [{ name: "in", range: null, precision: null, dtype: "tensor" }],
  output_ports: [{ name: "out", range: null, precision: null, dtype: "tensor" }],
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
    runtime: (raw.runtime ?? base.runtime) as GraphRuntime,
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
    if (
      port.name !== other.name ||
      port.dtype !== other.dtype ||
      port.precision !== other.precision
    ) {
      return false;
    }
    if (port.range === null || other.range === null) {
      return port.range === other.range;
    }
    return port.range[0] === other.range[0] && port.range[1] === other.range[1];
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
    const variant =
      workingRoot.variant_library[family]?.[version] ??
      VARIANT_FAMILY_ALIASES[family]
        ?.map((alias) => workingRoot.variant_library[alias]?.[version])
        .find((candidate): candidate is GraphData => Boolean(candidate));
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
        const inputsOk = ndef.input_ports.length === 0 || portsCompatible(ndef.input_ports, expectedInputs);
        const outputsOk = ndef.output_ports.length === 0 || portsCompatible(ndef.output_ports, expectedOutputs);
        if (inputsOk && outputsOk) {
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
          // Variant library entry is incompatible (e.g. stale from a
          // different template loaded into the same session).  Fall back to
          // the node's own inline subgraph so the graph stays usable.
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
        // else: no inline subgraph and incompatible variant — leave node as-is
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

function portsEqual(a: PortData, b: PortData): boolean {
  if (a.name !== b.name || a.dtype !== b.dtype || a.precision !== b.precision) {
    return false;
  }
  if (a.range === null || b.range === null) {
    return a.range === b.range;
  }
  return a.range[0] === b.range[0] && a.range[1] === b.range[1];
}

function portListsEqual(a: PortData[], b: PortData[]): boolean {
  if (a.length !== b.length) return false;
  return a.every((port, i) => portsEqual(port, b[i]));
}

export function refreshBuiltinNodesInGraph(
  graph: GraphData,
  builtinsByName: Map<string, NeuronDefData>,
): { graph: GraphData; changed: boolean } {
  if (builtinsByName.size === 0) {
    return { graph, changed: false };
  }
  let anyChanged = false;

  const refreshGraph = (g: GraphData): GraphData => {
    let nodesChanged = false;
    const nextNodes: Record<string, NodeData> = {};
    for (const [id, node] of Object.entries(g.nodes)) {
      const ndef = node.neuron_def;
      let nextNdef = ndef;

      if (ndef.kind === "function") {
        const builtin = builtinsByName.get(ndef.name);
        if (builtin && ndef.source_code.trimStart().startsWith("@neuron(")) {
          const sourceMatches = ndef.source_code === builtin.source_code;
          const inputsMatch = portListsEqual(ndef.input_ports, builtin.input_ports);
          const outputsMatch = portListsEqual(ndef.output_ports, builtin.output_ports);
          if (!sourceMatches || !inputsMatch || !outputsMatch) {
            nextNdef = {
              ...ndef,
              source_code: builtin.source_code,
              input_ports: builtin.input_ports.map((p) => ({ ...p })),
              output_ports: builtin.output_ports.map((p) => ({ ...p })),
            };
          }
        }
      }

      if (ndef.subgraph) {
        const refreshedSub = refreshGraph(ndef.subgraph);
        if (refreshedSub !== ndef.subgraph) {
          nextNdef = { ...nextNdef, subgraph: refreshedSub };
        }
      }

      if (nextNdef !== ndef) {
        nodesChanged = true;
        nextNodes[id] = { ...node, neuron_def: nextNdef };
      } else {
        nextNodes[id] = node;
      }
    }

    let nextLibrary = g.variant_library;
    let libraryChanged = false;
    if (g.variant_library && Object.keys(g.variant_library).length > 0) {
      nextLibrary = {};
      for (const [family, versions] of Object.entries(g.variant_library)) {
        nextLibrary[family] = {};
        for (const [version, variantGraph] of Object.entries(versions)) {
          const refreshed = refreshGraph(variantGraph);
          nextLibrary[family][version] = refreshed;
          if (refreshed !== variantGraph) {
            libraryChanged = true;
          }
        }
      }
      if (!libraryChanged) {
        nextLibrary = g.variant_library;
      }
    }

    if (nodesChanged || libraryChanged) {
      anyChanged = true;
      return { ...g, nodes: nextNodes, variant_library: nextLibrary };
    }
    return g;
  };

  const result = refreshGraph(graph);
  return { graph: result, changed: anyChanged };
}

export function makeFullGraphNodeId(
  instancePath: GraphPathSegment[],
  nodeId: string,
): string {
  if (instancePath.length === 0) {
    return nodeId;
  }
  const encodedPath = instancePath
    .map((seg) =>
      seg.kind === "node"
        ? encodeURIComponent(seg.nodeId)
        : `v@${encodeURIComponent(seg.family)}:${encodeURIComponent(seg.version)}`,
    )
    .join("/");
  return `fullg::${encodedPath}::${nodeId}`;
}

export function parseFullGraphNodeId(
  fullId: string,
): { instancePath: GraphPathSegment[]; nodeId: string } | null {
  if (!fullId.startsWith("fullg::")) {
    return null;
  }
  const parts = fullId.split("::");
  if (parts.length < 3) {
    return null;
  }
  const encodedPath = parts[1];
  const nodeId = parts.slice(2).join("::");
  const instancePath: GraphPathSegment[] = encodedPath
    .split("/")
    .filter(Boolean)
    .map((part) => {
      if (part.startsWith("v@")) {
        const [family, version] = part.slice(2).split(":");
        return {
          kind: "variant",
          family: decodeURIComponent(family ?? ""),
          version: decodeURIComponent(version ?? "default"),
        };
      }
      return {
        kind: "node",
        nodeId: decodeURIComponent(part),
      };
    });
  return { instancePath, nodeId };
}

export function resolveEditableGraphPath(
  rootGraph: GraphData,
  instancePath: GraphPathSegment[],
): GraphPathSegment[] {
  const navPath: GraphPathSegment[] = [];
  let current: GraphData | undefined = rootGraph;
  for (const seg of instancePath) {
    if (!current) break;
    if (seg.kind === "variant") {
      navPath.push(seg);
      current = rootGraph.variant_library[seg.family]?.[seg.version];
      continue;
    }
    const node: NodeData | undefined = current.nodes[seg.nodeId];
    if (!node) break;
    if (node.neuron_def.variant_ref) {
      navPath.push({
        kind: "variant",
        family: node.neuron_def.variant_ref.family,
        version: node.neuron_def.variant_ref.version,
      });
    } else {
      navPath.push({ kind: "node", nodeId: seg.nodeId });
    }
    current = node.neuron_def.subgraph ?? undefined;
  }
  return navPath;
}

interface PortEndpoint {
  fullNodeId: string;
  portIdx: number;
}

interface RecursiveExpandResult {
  nodes: Node<FlowNodeData>[];
  edges: RFEdge[];
  zones: SubgraphZone[];
  width: number;
  height: number;
  externalInputTargets: PortEndpoint[][];
  externalOutputSources: PortEndpoint[][];
}

export function buildFullExpandedGraph(
  rootGraph: GraphData,
  startPath: GraphPathSegment[] = [],
): FullExpandedGraphResult {
  const LEAF_WIDTH = 226;
  const LEAF_HEIGHT = 118;
  const COL_GAP_X = 76;
  const ROW_GAP_Y = 38;
  const ZONE_PAD_X = 32;
  const ZONE_PAD_TOP = 48;
  const ZONE_PAD_BOTTOM = 28;

  const baseGraph = getGraphAtPath(rootGraph, startPath);

  const expandRecursive = (
    graph: GraphData,
    instancePath: GraphPathSegment[],
    navPath: GraphPathSegment[],
    breadcrumbs: string[],
    isRoot: boolean,
  ): RecursiveExpandResult => {
    const outgoingByNode = new Map<string, EdgeData[]>();
    const incomingByNode = new Map<string, EdgeData[]>();
    for (const edge of Object.values(graph.edges)) {
      const outList = outgoingByNode.get(edge.src_node) ?? [];
      outList.push(edge);
      outgoingByNode.set(edge.src_node, outList);

      const inList = incomingByNode.get(edge.dst_node) ?? [];
      inList.push(edge);
      incomingByNode.set(edge.dst_node, inList);
    }

    const passthroughInputIds = new Set<string>();
    const passthroughOutputIds = new Set<string>();

    if (!isRoot) {
      for (const inId of graph.input_node_ids) {
        const n = graph.nodes[inId];
        if (
          n &&
          n.neuron_def.kind === "function" &&
          n.neuron_def.name === "input" &&
          (outgoingByNode.get(inId)?.length ?? 0) > 0
        ) {
          passthroughInputIds.add(inId);
        }
      }
      for (const outId of graph.output_node_ids) {
        const n = graph.nodes[outId];
        if (
          n &&
          n.neuron_def.kind === "function" &&
          n.neuron_def.name === "output" &&
          (incomingByNode.get(outId)?.length ?? 0) > 0
        ) {
          passthroughOutputIds.add(outId);
        }
      }
      // If any edge connects a passthrough input directly to a passthrough output,
      // keep both visible so no direct wire is lost.
      for (const edge of Object.values(graph.edges)) {
        if (
          passthroughInputIds.has(edge.src_node) &&
          passthroughOutputIds.has(edge.dst_node)
        ) {
          passthroughInputIds.delete(edge.src_node);
          passthroughOutputIds.delete(edge.dst_node);
        }
      }
    }

    // Map external input/output port indices matching deriveExternalPorts
    const extInPortMap = new Map<string, number>();
    let extInCount = 0;
    for (const inId of graph.input_node_ids) {
      const n = graph.nodes[inId];
      if (!n) continue;
      const portCount = Math.max(1, n.neuron_def.output_ports.length);
      for (let p = 0; p < portCount; p++) {
        extInPortMap.set(`${inId}:${p}`, extInCount++);
      }
    }

    const extOutPortMap = new Map<string, number>();
    let extOutCount = 0;
    for (const outId of graph.output_node_ids) {
      const n = graph.nodes[outId];
      if (!n) continue;
      const portCount = Math.max(1, n.neuron_def.output_ports.length);
      for (let p = 0; p < portCount; p++) {
        extOutPortMap.set(`${outId}:${p}`, extOutCount++);
      }
    }

    type LayoutItem =
      | {
          kind: "leaf";
          nodeId: string;
          node: NodeData;
          fullId: string;
          origX: number;
          origY: number;
          width: number;
          height: number;
          inputTargets: PortEndpoint[][];
          outputSources: PortEndpoint[][];
        }
      | {
          kind: "subgraph";
          nodeId: string;
          node: NodeData;
          origX: number;
          origY: number;
          width: number;
          height: number;
          childInstancePath: GraphPathSegment[];
          childNavPath: GraphPathSegment[];
          childBreadcrumbs: string[];
          childResult: RecursiveExpandResult;
          inputTargets: PortEndpoint[][];
          outputSources: PortEndpoint[][];
        };

    const items: LayoutItem[] = [];
    const itemById = new Map<string, LayoutItem>();

    for (const [nodeId, node] of Object.entries(graph.nodes)) {
      if (passthroughInputIds.has(nodeId) || passthroughOutputIds.has(nodeId)) {
        continue;
      }

      const ndef = normalizeNeuronDef(node.neuron_def);
      const isExpandableSubgraph =
        ndef.kind === "subgraph" &&
        ndef.subgraph &&
        Object.keys(ndef.subgraph.nodes).length > 0 &&
        instancePath.length < 6;

      if (isExpandableSubgraph && ndef.subgraph) {
        const childInstancePath: GraphPathSegment[] = [
          ...instancePath,
          { kind: "node", nodeId },
        ];
        const childNavPath: GraphPathSegment[] = [
          ...navPath,
          ndef.variant_ref
            ? {
                kind: "variant",
                family: ndef.variant_ref.family,
                version: ndef.variant_ref.version,
              }
            : { kind: "node", nodeId },
        ];
        const childBreadcrumbs = [...breadcrumbs, ndef.name || nodeId];
        const childResult = expandRecursive(
          ndef.subgraph,
          childInstancePath,
          childNavPath,
          childBreadcrumbs,
          false,
        );

        if (childResult.nodes.length > 0) {
          const item: LayoutItem = {
            kind: "subgraph",
            nodeId,
            node: { ...node, neuron_def: ndef },
            origX: node.position[0] ?? 0,
            origY: node.position[1] ?? 0,
            width: childResult.width + ZONE_PAD_X * 2,
            height: childResult.height + ZONE_PAD_TOP + ZONE_PAD_BOTTOM,
            childInstancePath,
            childNavPath,
            childBreadcrumbs,
            childResult,
            inputTargets: childResult.externalInputTargets,
            outputSources: childResult.externalOutputSources,
          };
          items.push(item);
          itemById.set(nodeId, item);
          continue;
        }
      }

      const fullId = makeFullGraphNodeId(instancePath, nodeId);
      const inPortCount = Math.max(1, ndef.input_ports.length);
      const outPortCount = Math.max(1, ndef.output_ports.length);
      const inputTargets: PortEndpoint[][] = Array.from(
        { length: inPortCount },
        (_, p) => [{ fullNodeId: fullId, portIdx: p }],
      );
      const outputSources: PortEndpoint[][] = Array.from(
        { length: outPortCount },
        (_, p) => [{ fullNodeId: fullId, portIdx: p }],
      );

      const item: LayoutItem = {
        kind: "leaf",
        nodeId,
        node: { ...node, neuron_def: ndef },
        fullId,
        origX: node.position[0] ?? 0,
        origY: node.position[1] ?? 0,
        width: LEAF_WIDTH,
        height: LEAF_HEIGHT,
        inputTargets,
        outputSources,
      };
      items.push(item);
      itemById.set(nodeId, item);
    }

    // Sort items by original X, then Y, and group into non-overlapping columns
    items.sort((a, b) =>
      Math.abs(a.origX - b.origX) > 10 ? a.origX - b.origX : a.origY - b.origY,
    );

    const columns: Array<{
      anchorX: number;
      items: LayoutItem[];
      colW: number;
      colH: number;
    }> = [];

    for (const item of items) {
      const lastCol = columns[columns.length - 1];
      if (lastCol && Math.abs(item.origX - lastCol.anchorX) <= 75) {
        lastCol.items.push(item);
      } else {
        columns.push({
          anchorX: item.origX,
          items: [item],
          colW: 0,
          colH: 0,
        });
      }
    }

    for (const col of columns) {
      col.items.sort((a, b) => a.origY - b.origY);
      col.colW = Math.max(LEAF_WIDTH, ...col.items.map((it) => it.width));
      col.colH =
        col.items.reduce((acc, it) => acc + it.height, 0) +
        Math.max(0, col.items.length - 1) * ROW_GAP_Y;
    }

    const totalHeight = Math.max(LEAF_HEIGHT, ...columns.map((c) => c.colH));
    let cursorX = 0;

    const nodesOut: Node<FlowNodeData>[] = [];
    const edgesOut: RFEdge[] = [];
    const zonesOut: SubgraphZone[] = [];

    for (const col of columns) {
      let cursorY = (totalHeight - col.colH) * 0.5;
      for (const item of col.items) {
        const itemX = cursorX + (col.colW - item.width) * 0.5;
        const itemY = cursorY;
        cursorY += item.height + ROW_GAP_Y;

        if (item.kind === "leaf") {
          nodesOut.push({
            id: item.fullId,
            type: "neuron",
            position: { x: itemX, y: itemY },
            measured: item.node.measured,
            data: {
              label: item.node.neuron_def.name,
              neuronDef: item.node.neuron_def,
              isInput: isRoot && graph.input_node_ids.includes(item.nodeId),
              isOutput: isRoot && graph.output_node_ids.includes(item.nodeId),
              subgraphPath: navPath,
              instancePath,
              localNodeId: item.nodeId,
              breadcrumbLabel:
                breadcrumbs.length > 0 ? breadcrumbs.join(" ▸ ") : undefined,
            },
          });
        } else {
          const offsetX = itemX + ZONE_PAD_X;
          const offsetY = itemY + ZONE_PAD_TOP;
          for (const childNode of item.childResult.nodes) {
            nodesOut.push({
              ...childNode,
              position: {
                x: childNode.position.x + offsetX,
                y: childNode.position.y + offsetY,
              },
            });
          }
          for (const childZone of item.childResult.zones) {
            zonesOut.push({
              ...childZone,
              x: childZone.x + offsetX,
              y: childZone.y + offsetY,
            });
          }
          edgesOut.push(...item.childResult.edges);

          const vRef = item.node.neuron_def.variant_ref;
          zonesOut.push({
            id: makeFullGraphNodeId(instancePath, `zone:${item.nodeId}`),
            label: item.node.neuron_def.name || item.nodeId,
            breadcrumb: item.childBreadcrumbs.join(" ▸ "),
            path: item.childNavPath,
            instancePath: item.childInstancePath,
            x: itemX,
            y: itemY,
            width: item.width,
            height: item.height,
            depth: instancePath.length,
            variantLabel: vRef ? `${vRef.family}@${vRef.version}` : null,
            nodeCount: item.childResult.nodes.length,
          });
        }
      }
      cursorX += col.colW + COL_GAP_X;
    }

    const totalWidth = Math.max(LEAF_WIDTH, cursorX - COL_GAP_X);

    // Build external port mappings for this graph
    const externalInputTargets: PortEndpoint[][] = Array.from(
      { length: Math.max(1, extInCount) },
      () => [],
    );
    const externalOutputSources: PortEndpoint[][] = Array.from(
      { length: Math.max(1, extOutCount) },
      () => [],
    );

    // Non-passthrough input/output nodes map directly
    for (const inId of graph.input_node_ids) {
      if (passthroughInputIds.has(inId)) continue;
      const item = itemById.get(inId);
      const n = graph.nodes[inId];
      if (!item || !n) continue;
      const portCount = Math.max(1, n.neuron_def.output_ports.length);
      for (let p = 0; p < portCount; p++) {
        const extIdx = extInPortMap.get(`${inId}:${p}`);
        if (extIdx !== undefined) {
          externalInputTargets[extIdx].push(
            ...(item.inputTargets[p] ?? item.inputTargets[0] ?? []),
          );
        }
      }
    }

    for (const outId of graph.output_node_ids) {
      if (passthroughOutputIds.has(outId)) continue;
      const item = itemById.get(outId);
      const n = graph.nodes[outId];
      if (!item || !n) continue;
      const portCount = Math.max(1, n.neuron_def.output_ports.length);
      for (let p = 0; p < portCount; p++) {
        const extIdx = extOutPortMap.get(`${outId}:${p}`);
        if (extIdx !== undefined) {
          externalOutputSources[extIdx].push(
            ...(item.outputSources[p] ?? item.outputSources[0] ?? []),
          );
        }
      }
    }

    // Wire edges & populate passthrough external ports
    const seenEdgeIds = new Set<string>();
    for (const edge of Object.values(graph.edges)) {
      const srcIsPassthrough = passthroughInputIds.has(edge.src_node);
      const dstIsPassthrough = passthroughOutputIds.has(edge.dst_node);

      if (srcIsPassthrough && !dstIsPassthrough) {
        const extIdx =
          extInPortMap.get(`${edge.src_node}:${edge.src_port}`) ??
          extInPortMap.get(`${edge.src_node}:0`) ??
          0;
        const dstItem = itemById.get(edge.dst_node);
        if (dstItem) {
          const targets =
            dstItem.inputTargets[edge.dst_port] ??
            dstItem.inputTargets[0] ??
            [];
          if (!externalInputTargets[extIdx]) {
            externalInputTargets[extIdx] = [];
          }
          externalInputTargets[extIdx].push(...targets);
        }
        continue;
      }

      if (!srcIsPassthrough && dstIsPassthrough) {
        const extIdx =
          extOutPortMap.get(`${edge.dst_node}:${edge.dst_port}`) ??
          extOutPortMap.get(`${edge.dst_node}:0`) ??
          0;
        const srcItem = itemById.get(edge.src_node);
        if (srcItem) {
          const sources =
            srcItem.outputSources[edge.src_port] ??
            srcItem.outputSources[0] ??
            [];
          if (!externalOutputSources[extIdx]) {
            externalOutputSources[extIdx] = [];
          }
          externalOutputSources[extIdx].push(...sources);
        }
        continue;
      }

      const srcItem = itemById.get(edge.src_node);
      const dstItem = itemById.get(edge.dst_node);
      if (!srcItem || !dstItem) continue;

      const sources =
        srcItem.outputSources[edge.src_port] ?? srcItem.outputSources[0] ?? [];
      const targets =
        dstItem.inputTargets[edge.dst_port] ?? dstItem.inputTargets[0] ?? [];

      for (const s of sources) {
        for (const t of targets) {
          const edgeId = `fe::${s.fullNodeId}:${s.portIdx}->${t.fullNodeId}:${t.portIdx}`;
          if (seenEdgeIds.has(edgeId)) continue;
          seenEdgeIds.add(edgeId);
          edgesOut.push({
            id: edgeId,
            source: s.fullNodeId,
            target: t.fullNodeId,
            sourceHandle: `out-${s.portIdx}`,
            targetHandle: `in-${t.portIdx}`,
            type: "default",
            data: { weight: edge.weight, bias: edge.bias },
          });
        }
      }
    }

    return {
      nodes: nodesOut,
      edges: edgesOut,
      zones: zonesOut,
      width: totalWidth,
      height: totalHeight,
      externalInputTargets,
      externalOutputSources,
    };
  };

  const expanded = expandRecursive(baseGraph, startPath, startPath, [], true);
  // Sort zones so outer (shallower depth) zones render behind inner (deeper) zones
  expanded.zones.sort((a, b) => a.depth - b.depth);
  return {
    nodes: expanded.nodes,
    edges: expanded.edges,
    zones: expanded.zones,
  };
}
