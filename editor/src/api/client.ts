const BASE = "/api";

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + url, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export interface PortData {
  name: string;
  range: [number, number];
  precision: number;
  dtype: string;
}

export interface NeuronDefData {
  id: string;
  name: string;
  input_ports: PortData[];
  output_ports: PortData[];
  source_code: string;
}

export interface NodeData {
  instance_id: string;
  neuron_def: NeuronDefData;
  position: [number, number];
}

export interface EdgeData {
  id: string;
  src_node: string;
  src_port: number;
  dst_node: string;
  dst_port: number;
  weight: number;
  bias: number;
}

export interface GraphData {
  nodes: Record<string, NodeData>;
  edges: Record<string, EdgeData>;
  input_node_ids: string[];
  output_node_ids: string[];
}

export const api = {
  getGraph: () => json<GraphData>("/graph"),

  putGraph: (g: GraphData) =>
    json<GraphData>("/graph", { method: "PUT", body: JSON.stringify(g) }),

  addNode: (node: Partial<NodeData>) =>
    json<NodeData>("/nodes", { method: "POST", body: JSON.stringify(node) }),

  deleteNode: (id: string) =>
    json<{ status: string }>(`/nodes/${id}`, { method: "DELETE" }),

  addEdge: (edge: Partial<EdgeData>) =>
    json<EdgeData>("/edges", { method: "POST", body: JSON.stringify(edge) }),

  deleteEdge: (id: string) =>
    json<{ status: string }>(`/edges/${id}`, { method: "DELETE" }),

  execute: (inputs: Record<string, number[]>) =>
    json<Record<string, number[]>>("/execute", {
      method: "POST",
      body: JSON.stringify({ inputs }),
    }),

  getBuiltins: () => json<NeuronDefData[]>("/builtins"),

  probe: (nodeId: string, nSamples = 1000) =>
    json<{ inputs: number[][]; outputs: number[][] }>(
      `/probe/${nodeId}?n_samples=${nSamples}`,
      { method: "POST" }
    ),

  setIO: (inputIds: string[], outputIds: string[]) =>
    json<{ input_node_ids: string[]; output_node_ids: string[] }>(
      "/graph/io",
      {
        method: "PUT",
        body: JSON.stringify({ input_ids: inputIds, output_ids: outputIds }),
      }
    ),

  startTraining: (
    body: {
      method: string;
      train_inputs: number[][];
      train_targets: number[][];
      epochs?: number;
      learning_rate?: number;
      population_size?: number;
      generations?: number;
    },
    onMessage: (data: { step?: number; loss?: number; done?: boolean }) => void
  ) => {
    const ctrl = new AbortController();
    fetch(BASE + "/train/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    }).then(async (res) => {
      const reader = res.body?.getReader();
      if (!reader) return;
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n\n");
        buf = lines.pop() ?? "";
        for (const line of lines) {
          const match = line.match(/^data:\s*(.*)/);
          if (match) {
            try {
              onMessage(JSON.parse(match[1]));
            } catch {}
          }
        }
      }
    });
    return ctrl;
  },

  stopTraining: () =>
    json<{ status: string }>("/train/stop", { method: "POST" }),
};
