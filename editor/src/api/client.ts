const BASE = "/api";

export class ApiError extends Error {
  status: number;
  payload: unknown;

  constructor(status: number, message: string, payload: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

async function parseError(res: Response): Promise<never> {
  const text = await res.text();
  let payload: unknown = text;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch {
    payload = text;
  }
  const message =
    typeof payload === "object" && payload !== null && "detail" in payload
      ? String((payload as { detail: unknown }).detail)
      : text || `Request failed with status ${res.status}`;
  throw new ApiError(res.status, message, payload);
}

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers ?? {});
  if (!headers.has("Content-Type") && init?.body !== undefined && !(init.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  const res = await fetch(BASE + url, {
    credentials: "include",
    ...init,
    headers,
  });
  if (!res.ok) {
    await parseError(res);
  }
  return res.json();
}

export interface PortData {
  name: string;
  range: [number, number];
  precision: number;
  dtype: string;
}

export type TrainingMethod = "surrogate" | "evolutionary" | "frozen" | "torch";

export interface VariantRefData {
  family: string;
  version: string;
}

export interface NeuronDefData {
  id: string;
  name: string;
  kind: "function" | "subgraph" | "module";
  input_ports: PortData[];
  output_ports: PortData[];
  source_code: string;
  subgraph: GraphData | null;
  module_type: string;
  module_config: Record<string, unknown>;
  module_state: string;
  input_aliases: string[];
  output_aliases: string[];
  variant_ref: VariantRefData | null;
}

export interface NodeData {
  instance_id: string;
  neuron_def: NeuronDefData;
  position: [number, number];
  measured?: { width?: number; height?: number };
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
  name: string;
  training_method: TrainingMethod;
  runtime: "scalar" | "torch";
  surrogate_config: Record<string, unknown>;
  evo_config: Record<string, unknown>;
  torch_config: Record<string, unknown>;
  variant_library: VariantLibraryData;
  nodes: Record<string, NodeData>;
  edges: Record<string, EdgeData>;
  input_node_ids: string[];
  output_node_ids: string[];
}

export type VariantLibraryData = Record<string, Record<string, GraphData>>;

export interface TrainingMessage {
  event_id?: number;
  step?: number;
  loss?: number;
  done?: boolean;
  graph_path?: string[];
  graph_name?: string;
  method?: string;
  round?: number;
  local_step?: number;
  error?: string;
}

export interface TorchTraceStat {
  shape?: number[];
  dtype?: string;
  mean?: number;
  std?: number;
  min?: number;
  max?: number;
  kind?: string;
  preview?: Array<number | string>;
  preview_shape?: number[];
}

export interface TorchTraceResponse {
  source: "manual" | "dataset";
  sample_inputs: Record<string, Array<number | string>>;
  trace: Record<string, TorchTraceStat[]>;
}

export interface GPTTemplateResponse {
  node_def: NeuronDefData;
  variant_library: VariantLibraryData;
  graph_settings: Pick<GraphData, "training_method" | "runtime" | "torch_config">;
}

export interface DatasetInfo {
  name: string;
  source: string;
  hf_path: string | null;
  hf_split: string | null;
  text_column: string;
  num_tokens: number | null;
  num_rows: number | null;
  variant?: string | null;
  train_shards?: number | null;
  val_shards?: number | null;
  data_format?: string | null;
  repo_id?: string | null;
  remote_root_prefix?: string | null;
  project_ids?: string[];
}

export interface UserData {
  id: string;
  email: string;
  display_name: string;
  is_admin: boolean;
}

export interface ProjectSummary {
  id: string;
  slug: string;
  name: string;
  description: string | null;
  role: string;
  created_at: string;
  updated_at: string;
}

export interface SessionSummary {
  id: string;
  project_id: string;
  name: string;
  description: string | null;
  branch_name: string;
  latest_revision: number;
  created_at: string;
  updated_at: string;
}

export interface SessionDetail {
  project: ProjectSummary;
  session: SessionSummary;
  graph: GraphData;
  revision: number;
}

export interface ProjectCreateResponse {
  project: ProjectSummary;
  default_session: SessionSummary;
}

export interface MembershipInfo {
  id: string;
  project_id: string;
  user: UserData;
  role: string;
  created_at: string;
}

export interface ProjectAnalytics {
  project: ProjectSummary;
  session_count: number;
  recent_run_count: number;
  status_counts: Record<string, number>;
  latest_runs: Array<{
    id: string;
    session_id: string;
    status: string;
    resolved_method: string | null;
    last_loss: number | null;
    started_at: string;
    completed_at: string | null;
  }>;
}

export interface RunSnapshot {
  run_id?: string;
  status?: string;
  running?: boolean;
  done?: boolean;
  method?: string | null;
  requested_method?: string | null;
  graph_name?: string | null;
  dataset_names?: string[];
  seq_len?: number | null;
  event_id?: number;
  history_length?: number;
  last_loss?: number | null;
  last_step?: number | null;
  stop_requested?: boolean;
  error?: string | null;
  started_at?: number | null;
  updated_at?: number | null;
  completed_at?: number | null;
  thread_alive?: boolean;
  events?: Array<Record<string, unknown>>;
}

export interface BootstrapResponse {
  requires_setup: boolean;
  authenticated: boolean;
  user: UserData | null;
  projects: ProjectSummary[];
  active_project_id: string | null;
  active_session_id: string | null;
  active_session: SessionDetail | null;
}

const sessionBase = (projectId: string, sessionId: string) =>
  `/projects/${projectId}/sessions/${sessionId}`;

const projectBase = (projectId: string) => `/projects/${projectId}`;

export const api = {
  getBootstrap: () => json<BootstrapResponse>("/bootstrap"),

  bootstrapAdmin: (body: { email: string; password: string; display_name: string }) =>
    json<{ user: UserData; project: ProjectSummary; session: SessionSummary }>("/auth/bootstrap-admin", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  login: (body: { email: string; password: string }) =>
    json<{ user: UserData }>("/auth/login", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  logout: () => json<{ status: string }>("/auth/logout", { method: "POST" }),

  me: () => json<UserData>("/auth/me"),

  setActiveSession: (projectId: string | null, sessionId: string | null) =>
    json<{ project_id: string | null; session_id: string | null }>("/auth/active-session", {
      method: "PUT",
      body: JSON.stringify({ project_id: projectId, session_id: sessionId }),
    }),

  listProjects: () => json<ProjectSummary[]>("/projects"),

  createProject: (body: { name: string; description?: string | null }) =>
    json<ProjectCreateResponse>("/projects", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  getProjectAnalytics: (projectId: string) =>
    json<ProjectAnalytics>(`${projectBase(projectId)}/analytics`),

  listUsers: () => json<UserData[]>("/admin/users"),

  createUser: (body: { email: string; password: string; display_name: string; is_admin?: boolean }) =>
    json<UserData>("/admin/users", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  listMemberships: (projectId: string) =>
    json<MembershipInfo[]>(`/admin/projects/${projectId}/memberships`),

  addMembership: (projectId: string, body: { user_id?: string; email?: string; role?: string }) =>
    json<MembershipInfo>(`/admin/projects/${projectId}/memberships`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  listSessions: (projectId: string) =>
    json<SessionSummary[]>(`${projectBase(projectId)}/sessions`),

  createSession: (projectId: string, body: { name: string; description?: string | null }) =>
    json<SessionSummary>(`${projectBase(projectId)}/sessions`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  getSession: (projectId: string, sessionId: string) =>
    json<SessionDetail>(`${sessionBase(projectId, sessionId)}`),

  getSessionGraph: (projectId: string, sessionId: string) =>
    json<{ graph: GraphData; revision: number }>(`${sessionBase(projectId, sessionId)}/graph`),

  putSessionGraph: (
    projectId: string,
    sessionId: string,
    body: {
      graph: GraphData;
      expected_revision?: number | null;
      persist_snapshot?: boolean;
      snapshot_reason?: string;
    },
  ) =>
    json<SessionDetail>(`${sessionBase(projectId, sessionId)}/graph`, {
      method: "PUT",
      body: JSON.stringify(body),
    }),

  execute: (projectId: string, sessionId: string, inputs: Record<string, number[]>) =>
    json<Record<string, number[]>>(`${sessionBase(projectId, sessionId)}/execute`, {
      method: "POST",
      body: JSON.stringify({ inputs }),
    }),

  executeTrace: (projectId: string, sessionId: string, inputs: Record<string, number[]>) =>
    json<Record<string, number[]>>(`${sessionBase(projectId, sessionId)}/execute-trace`, {
      method: "POST",
      body: JSON.stringify({ inputs }),
    }),

  traceTorchPreview: (
    projectId: string,
    sessionId: string,
    body: {
      inputs?: Record<string, Array<number | string> | Array<Array<number | string>>>;
      dataset_names?: string[];
      seq_len?: number;
      preview_batch_size?: number;
    },
  ) =>
    json<TorchTraceResponse>(`${sessionBase(projectId, sessionId)}/trace/torch`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  getBuiltins: () => json<NeuronDefData[]>("/builtins"),

  buildGPTTemplate: (body?: { name?: string; config?: Record<string, unknown> }) =>
    json<GPTTemplateResponse>("/templates/gpt", {
      method: "POST",
      body: JSON.stringify(body ?? {}),
    }),

  getAgentStatus: (projectId: string, sessionId: string) =>
    json<{ active: boolean }>(`${sessionBase(projectId, sessionId)}/agent/status`),

  startTraining: (
    projectId: string,
    sessionId: string,
    body: {
      method?: string | null;
      train_inputs?: Array<Array<number | string>>;
      train_targets?: Array<Array<number | string>>;
      dataset_names?: string[];
      seq_len?: number;
      outer_rounds?: number;
      loss_fn?: string;
      epochs?: number;
      learning_rate?: number;
      population_size?: number;
      generations?: number;
      batch_size?: number;
      weight_decay?: number;
    },
    onMessage: (data: TrainingMessage) => void,
  ) => {
    const ctrl = new AbortController();
    fetch(BASE + `${sessionBase(projectId, sessionId)}/runs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      credentials: "include",
      signal: ctrl.signal,
    }).then(async (res) => {
      if (!res.ok) {
        await parseError(res);
      }
      const reader = res.body?.getReader();
      if (!reader) return;
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const chunks = buffer.split("\n\n");
        buffer = chunks.pop() ?? "";
        for (const chunk of chunks) {
          const match = chunk.match(/^data:\s*(.*)/);
          if (match) {
            try {
              onMessage(JSON.parse(match[1]));
            } catch {
              // Ignore malformed SSE payloads.
            }
          }
        }
      }
    });
    return ctrl;
  },

  getActiveRun: (projectId: string, sessionId: string) =>
    json<RunSnapshot>(`${sessionBase(projectId, sessionId)}/runs/active`),

  listRuns: (projectId: string, sessionId: string) =>
    json<Array<Record<string, unknown>>>(`${sessionBase(projectId, sessionId)}/runs`),

  stopTraining: (projectId: string, sessionId: string, runId: string) =>
    json<{ status: string }>(`${sessionBase(projectId, sessionId)}/runs/${runId}/stop`, { method: "POST" }),

  getDatasets: (projectId: string) =>
    json<DatasetInfo[]>(`${projectBase(projectId)}/datasets`),

  downloadDataset: (
    projectId: string,
    body: {
      hf_path: string;
      hf_split?: string;
      text_column?: string;
      max_rows?: number | null;
      alias?: string | null;
      variant?: string | null;
      train_shards?: number | null;
      skip_manifest?: boolean;
      with_docs?: boolean;
      repo_id?: string | null;
      remote_root_prefix?: string;
      project_ids?: string[];
    },
  ) =>
    json<DatasetInfo>(`${projectBase(projectId)}/datasets/download`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  uploadDataset: async (
    projectId: string,
    name: string,
    file: File,
    projectIds?: string[],
  ): Promise<DatasetInfo> => {
    const form = new FormData();
    form.append("name", name);
    form.append("file", file);
    if (projectIds && projectIds.length > 0) {
      form.append("project_ids", JSON.stringify(projectIds));
    }
    const res = await fetch(BASE + `${projectBase(projectId)}/datasets/upload`, {
      method: "POST",
      body: form,
      credentials: "include",
    });
    if (!res.ok) {
      await parseError(res);
    }
    return res.json();
  },

  setDatasetAccess: (projectId: string, name: string, body: { project_ids: string[] }) =>
    json<DatasetInfo>(`${projectBase(projectId)}/datasets/${name}/access`, {
      method: "PUT",
      body: JSON.stringify(body),
    }),

  deleteDataset: (projectId: string, name: string) =>
    json<{ status: string }>(`${projectBase(projectId)}/datasets/${name}`, { method: "DELETE" }),
};
