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
    runtime: "scalar",
    surrogate_config: {},
    evo_config: {},
    torch_config: { device: "cuda", amp_dtype: "bfloat16" },
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
    module_type: "",
    module_config: {},
    module_state: "",
    input_aliases: ["x"],
    output_aliases: ["y"],
  });
}

function createTerminalNeuronDef(
  role: "input" | "output",
  portName: string,
  dtype: string,
): NeuronDefData {
  return normalizeNeuronDef({
    id: "",
    name: role,
    kind: "function",
    input_ports: [{ name: portName, range: [-1_000_000, 1_000_000], precision: 0.001, dtype }],
    output_ports: [{ name: portName, range: [-1_000_000, 1_000_000], precision: 0.001, dtype }],
    source_code: `def ${role}(x):\n    return x\n`,
    subgraph: null,
    module_type: "",
    module_config: {},
    module_state: "",
    input_aliases: [],
    output_aliases: [],
  });
}

function createModuleNeuronDef(
  name: string,
  moduleType: string,
  inputPorts: PortData[],
  outputPorts: PortData[],
  moduleConfig: Record<string, unknown>,
): NeuronDefData {
  return normalizeNeuronDef({
    id: "",
    name,
    kind: "module",
    input_ports: inputPorts,
    output_ports: outputPorts,
    source_code: "",
    subgraph: null,
    module_type: moduleType,
    module_config: moduleConfig,
    module_state: "",
    input_aliases: [],
    output_aliases: [],
  });
}

const DEFAULT_GPT_CONFIG = {
  vocab_size: 256,
  num_layers: 4,
  model_dim: 128,
  num_heads: 4,
  num_kv_heads: 2,
  mlp_mult: 2,
  tie_embeddings: true,
  logit_softcap: 30,
  rope_base: 10000,
  qk_gain_init: 1,
};

function createTransformerBlockGraph(
  name: string,
  config: typeof DEFAULT_GPT_CONFIG,
): GraphData {
  const graph = createEmptyGraph(name);
  graph.training_method = "torch";
  graph.runtime = "torch";

  graph.nodes["x_in"] = {
    instance_id: "x_in",
    position: [40, 140],
    neuron_def: createTerminalNeuronDef("input", "x", "tensor"),
  };
  graph.nodes["x0_in"] = {
    instance_id: "x0_in",
    position: [40, 260],
    neuron_def: createTerminalNeuronDef("input", "x0", "tensor"),
  };
  graph.nodes["resid_mix"] = {
    instance_id: "resid_mix",
    position: [220, 200],
    neuron_def: createModuleNeuronDef(
      "residual_mix",
      "residual_mix",
      [
        { name: "x", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" },
        { name: "x0", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" },
      ],
      [{ name: "mixed", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" }],
      { dim: config.model_dim, primary_init: 1, skip_init: 0 },
    ),
  };
  graph.nodes["attn_norm"] = {
    instance_id: "attn_norm",
    position: [420, 140],
    neuron_def: createModuleNeuronDef(
      "rms_norm",
      "rms_norm",
      [{ name: "x", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" }],
      [{ name: "y", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" }],
      { eps: 1e-6 },
    ),
  };
  graph.nodes["attention"] = {
    instance_id: "attention",
    position: [620, 140],
    neuron_def: createModuleNeuronDef(
      "causal_self_attention",
      "causal_self_attention",
      [{ name: "x", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" }],
      [{ name: "attn_out", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" }],
      {
        model_dim: config.model_dim,
        num_heads: config.num_heads,
        num_kv_heads: config.num_kv_heads,
        rope_base: config.rope_base,
        qk_gain_init: config.qk_gain_init,
      },
    ),
  };
  graph.nodes["attn_residual"] = {
    instance_id: "attn_residual",
    position: [820, 140],
    neuron_def: createModuleNeuronDef(
      "residual_add",
      "residual_add",
      [
        { name: "residual", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" },
        { name: "delta", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" },
      ],
      [{ name: "y", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" }],
      { dim: config.model_dim, init_scale: 1 },
    ),
  };
  graph.nodes["mlp_norm"] = {
    instance_id: "mlp_norm",
    position: [1020, 140],
    neuron_def: createModuleNeuronDef(
      "rms_norm",
      "rms_norm",
      [{ name: "x", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" }],
      [{ name: "y", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" }],
      { eps: 1e-6 },
    ),
  };
  graph.nodes["mlp"] = {
    instance_id: "mlp",
    position: [1220, 140],
    neuron_def: createModuleNeuronDef(
      "mlp_relu2",
      "mlp_relu2",
      [{ name: "x", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" }],
      [{ name: "y", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" }],
      { model_dim: config.model_dim, mlp_mult: config.mlp_mult },
    ),
  };
  graph.nodes["mlp_residual"] = {
    instance_id: "mlp_residual",
    position: [1420, 140],
    neuron_def: createModuleNeuronDef(
      "residual_add",
      "residual_add",
      [
        { name: "residual", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" },
        { name: "delta", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" },
      ],
      [{ name: "y", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" }],
      { dim: config.model_dim, init_scale: 1 },
    ),
  };
  graph.nodes["x_out"] = {
    instance_id: "x_out",
    position: [1620, 140],
    neuron_def: createTerminalNeuronDef("output", "x", "tensor"),
  };

  const edges: EdgeData[] = [
    { id: "e_x_mix", src_node: "x_in", src_port: 0, dst_node: "resid_mix", dst_port: 0, weight: 1, bias: 0 },
    { id: "e_x0_mix", src_node: "x0_in", src_port: 0, dst_node: "resid_mix", dst_port: 1, weight: 1, bias: 0 },
    { id: "e_mix_norm", src_node: "resid_mix", src_port: 0, dst_node: "attn_norm", dst_port: 0, weight: 1, bias: 0 },
    { id: "e_norm_attn", src_node: "attn_norm", src_port: 0, dst_node: "attention", dst_port: 0, weight: 1, bias: 0 },
    { id: "e_mix_attn_res", src_node: "resid_mix", src_port: 0, dst_node: "attn_residual", dst_port: 0, weight: 1, bias: 0 },
    { id: "e_attn_attn_res", src_node: "attention", src_port: 0, dst_node: "attn_residual", dst_port: 1, weight: 1, bias: 0 },
    { id: "e_attnres_mlpnorm", src_node: "attn_residual", src_port: 0, dst_node: "mlp_norm", dst_port: 0, weight: 1, bias: 0 },
    { id: "e_mlpnorm_mlp", src_node: "mlp_norm", src_port: 0, dst_node: "mlp", dst_port: 0, weight: 1, bias: 0 },
    { id: "e_attnres_mlpres", src_node: "attn_residual", src_port: 0, dst_node: "mlp_residual", dst_port: 0, weight: 1, bias: 0 },
    { id: "e_mlp_mlpres", src_node: "mlp", src_port: 0, dst_node: "mlp_residual", dst_port: 1, weight: 1, bias: 0 },
    { id: "e_mlpres_out", src_node: "mlp_residual", src_port: 0, dst_node: "x_out", dst_port: 0, weight: 1, bias: 0 },
  ];
  graph.edges = Object.fromEntries(edges.map((edge) => [edge.id, edge]));
  graph.input_node_ids = ["x_in", "x0_in"];
  graph.output_node_ids = ["x_out"];
  return graph;
}

export function createGPTNeuronDef(name = "gpt"): NeuronDefData {
  const config = { ...DEFAULT_GPT_CONFIG };
  const graph = createEmptyGraph(`${name} graph`);
  graph.training_method = "torch";
  graph.runtime = "torch";

  graph.nodes["tokens_in"] = {
    instance_id: "tokens_in",
    position: [40, 140],
    neuron_def: createTerminalNeuronDef("input", "tokens", "tokens"),
  };
  graph.nodes["targets_in"] = {
    instance_id: "targets_in",
    position: [40, 360],
    neuron_def: createTerminalNeuronDef("input", "targets", "tokens"),
  };
  graph.nodes["token_embedding"] = {
    instance_id: "token_embedding",
    position: [240, 140],
    neuron_def: createModuleNeuronDef(
      "token_embedding",
      "token_embedding",
      [{ name: "token_ids", range: [0, 65535], precision: 1, dtype: "tokens" }],
      [
        { name: "hidden", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" },
        { name: "weight", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" },
      ],
      { vocab_size: config.vocab_size, model_dim: config.model_dim },
    ),
  };
  graph.nodes["embed_norm"] = {
    instance_id: "embed_norm",
    position: [460, 140],
    neuron_def: createModuleNeuronDef(
      "rms_norm",
      "rms_norm",
      [{ name: "x", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" }],
      [{ name: "y", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" }],
      { eps: 1e-6 },
    ),
  };

  const edges: EdgeData[] = [
    { id: "e_tokens_embed", src_node: "tokens_in", src_port: 0, dst_node: "token_embedding", dst_port: 0, weight: 1, bias: 0 },
    { id: "e_embed_hidden", src_node: "token_embedding", src_port: 0, dst_node: "embed_norm", dst_port: 0, weight: 1, bias: 0 },
  ];

  let currentNode = "embed_norm";
  const skipNodes: string[] = [];
  const encoderCount = Math.floor(config.num_layers / 2);
  const decoderCount = config.num_layers - encoderCount;

  for (let idx = 0; idx < encoderCount; idx += 1) {
    const blockId = `encoder_block_${idx}`;
    graph.nodes[blockId] = {
      instance_id: blockId,
      position: [680 + idx * 220, 100],
      neuron_def: normalizeNeuronDef({
        id: "",
        name: blockId,
        kind: "subgraph",
        input_ports: [],
        output_ports: [],
        source_code: "",
        subgraph: createTransformerBlockGraph(blockId, config),
        module_type: "",
        module_config: {},
        module_state: "",
        input_aliases: ["x", "x0"],
        output_aliases: ["x"],
      }),
    };
    edges.push(
      { id: `e_${blockId}_x`, src_node: currentNode, src_port: 0, dst_node: blockId, dst_port: 0, weight: 1, bias: 0 },
      { id: `e_${blockId}_x0`, src_node: "embed_norm", src_port: 0, dst_node: blockId, dst_port: 1, weight: 1, bias: 0 },
    );
    currentNode = blockId;
    skipNodes.push(blockId);
  }

  for (let idx = 0; idx < decoderCount; idx += 1) {
    if (skipNodes.length > 0) {
      const skipSrc = skipNodes.pop()!;
      const skipAddId = `skip_add_${idx}`;
      graph.nodes[skipAddId] = {
        instance_id: skipAddId,
        position: [680 + (encoderCount + idx) * 220, 260],
        neuron_def: createModuleNeuronDef(
          "residual_add",
          "residual_add",
          [
            { name: "residual", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" },
            { name: "delta", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" },
          ],
          [{ name: "y", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" }],
          { dim: config.model_dim, init_scale: 1 },
        ),
      };
      edges.push(
        { id: `e_${skipAddId}_current`, src_node: currentNode, src_port: 0, dst_node: skipAddId, dst_port: 0, weight: 1, bias: 0 },
        { id: `e_${skipAddId}_skip`, src_node: skipSrc, src_port: 0, dst_node: skipAddId, dst_port: 1, weight: 1, bias: 0 },
      );
      currentNode = skipAddId;
    }

    const blockId = `decoder_block_${idx}`;
    graph.nodes[blockId] = {
      instance_id: blockId,
      position: [900 + (encoderCount + idx) * 220, 100],
      neuron_def: normalizeNeuronDef({
        id: "",
        name: blockId,
        kind: "subgraph",
        input_ports: [],
        output_ports: [],
        source_code: "",
        subgraph: createTransformerBlockGraph(blockId, config),
        module_type: "",
        module_config: {},
        module_state: "",
        input_aliases: ["x", "x0"],
        output_aliases: ["x"],
      }),
    };
    edges.push(
      { id: `e_${blockId}_x`, src_node: currentNode, src_port: 0, dst_node: blockId, dst_port: 0, weight: 1, bias: 0 },
      { id: `e_${blockId}_x0`, src_node: "embed_norm", src_port: 0, dst_node: blockId, dst_port: 1, weight: 1, bias: 0 },
    );
    currentNode = blockId;
  }

  graph.nodes["final_norm"] = {
    instance_id: "final_norm",
    position: [1240 + config.num_layers * 220, 140],
    neuron_def: createModuleNeuronDef(
      "rms_norm",
      "rms_norm",
      [{ name: "x", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" }],
      [{ name: "y", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" }],
      { eps: 1e-6 },
    ),
  };
  edges.push({ id: "e_final_norm", src_node: currentNode, src_port: 0, dst_node: "final_norm", dst_port: 0, weight: 1, bias: 0 });

  const headId = config.tie_embeddings ? "tied_lm_head" : "lm_head";
  graph.nodes[headId] = {
    instance_id: headId,
    position: [1460 + config.num_layers * 220, 140],
    neuron_def: config.tie_embeddings
      ? createModuleNeuronDef(
          "tied_lm_head",
          "tied_lm_head",
          [
            { name: "hidden", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" },
            { name: "weight", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" },
          ],
          [{ name: "logits", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" }],
          {},
        )
      : createModuleNeuronDef(
          "lm_head",
          "lm_head",
          [{ name: "hidden", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" }],
          [{ name: "logits", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" }],
          { model_dim: config.model_dim, vocab_size: config.vocab_size },
        ),
  };
  edges.push({ id: "e_final_head", src_node: "final_norm", src_port: 0, dst_node: headId, dst_port: 0, weight: 1, bias: 0 });
  if (config.tie_embeddings) {
    edges.push({ id: "e_embed_weight_head", src_node: "token_embedding", src_port: 1, dst_node: headId, dst_port: 1, weight: 1, bias: 0 });
  }

  graph.nodes["logit_softcap"] = {
    instance_id: "logit_softcap",
    position: [1680 + config.num_layers * 220, 140],
    neuron_def: createModuleNeuronDef(
      "logit_softcap",
      "logit_softcap",
      [{ name: "logits", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" }],
      [{ name: "softcapped", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" }],
      { softcap: config.logit_softcap },
    ),
  };
  graph.nodes["token_cross_entropy"] = {
    instance_id: "token_cross_entropy",
    position: [1900 + config.num_layers * 220, 220],
    neuron_def: createModuleNeuronDef(
      "token_cross_entropy",
      "token_cross_entropy",
      [
        { name: "logits", range: [-1_000_000, 1_000_000], precision: 0.001, dtype: "tensor" },
        { name: "targets", range: [0, 65535], precision: 1, dtype: "tokens" },
      ],
      [{ name: "loss", range: [0, 100], precision: 0.0001, dtype: "loss" }],
      {},
    ),
  };
  graph.nodes["loss_out"] = {
    instance_id: "loss_out",
    position: [2120 + config.num_layers * 220, 220],
    neuron_def: createTerminalNeuronDef("output", "loss", "loss"),
  };
  edges.push(
    { id: "e_head_softcap", src_node: headId, src_port: 0, dst_node: "logit_softcap", dst_port: 0, weight: 1, bias: 0 },
    { id: "e_softcap_ce", src_node: "logit_softcap", src_port: 0, dst_node: "token_cross_entropy", dst_port: 0, weight: 1, bias: 0 },
    { id: "e_targets_ce", src_node: "targets_in", src_port: 0, dst_node: "token_cross_entropy", dst_port: 1, weight: 1, bias: 0 },
    { id: "e_ce_out", src_node: "token_cross_entropy", src_port: 0, dst_node: "loss_out", dst_port: 0, weight: 1, bias: 0 },
  );

  graph.edges = Object.fromEntries(edges.map((edge) => [edge.id, edge]));
  graph.input_node_ids = ["tokens_in", "targets_in"];
  graph.output_node_ids = ["loss_out"];

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
    input_aliases: ["tokens", "targets"],
    output_aliases: ["loss"],
  });
}

export function normalizeGraph(graph: GraphData | null | undefined, fallbackName = "Graph"): GraphData {
  const base = createEmptyGraph(fallbackName);
  const raw = graph ?? base;
  const normalized: GraphData = {
    name: raw.name ?? base.name,
    training_method: (raw.training_method ?? base.training_method) as TrainingMethod,
    runtime: (raw.runtime ?? base.runtime) as "scalar" | "torch",
    surrogate_config: { ...(raw.surrogate_config ?? {}) },
    evo_config: { ...(raw.evo_config ?? {}) },
    torch_config: { ...(raw.torch_config ?? {}) },
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
      module_type: "",
      module_config: {},
      module_state: "",
      input_ports: inputPorts,
      output_ports: outputPorts,
      input_aliases: inputPorts.map((port) => port.name),
      output_aliases: outputPorts.map((port) => port.name),
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
