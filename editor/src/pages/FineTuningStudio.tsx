import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";
import { useGraphStore } from "../store/graphStore";

type TrainingMode = "pretrain" | "sft" | "dpo" | "ppo" | "reward_model";
type AdapterType = "none" | "lora" | "qlora" | "randmap";

type LossPoint = { step: number; loss: number; phase?: string };

/**
 * Unsloth-style Fine-Tuning Studio.
 *
 * A dedicated top-level page for parameter-efficient fine-tuning and RLHF.
 * It re-uses the existing runs/{id}/generate training pipeline but wraps
 * it in a task-focused UI: pick an objective, pick adapters, point at a
 * base checkpoint, wire a dataset, kick off training, watch loss live, and
 * save either the full state dict or an adapter-only artifact.
 */
export default function FineTuningStudio() {
  const params = useParams<{ projectId: string; sessionId: string }>();
  const projectId = params.projectId ?? "";
  const sessionId = params.sessionId ?? "";
  const hydrateSession = useGraphStore((state) => state.hydrateSession);
  const rootGraph = useGraphStore((state) => state.rootGraph);
  const storeSession = useGraphStore((state) => state.sessionId);

  // ── Training state ──────────────────────────────────────────────────
  const [trainingMode, setTrainingMode] = useState<TrainingMode>("sft");
  const [adapterType, setAdapterType] = useState<AdapterType>("lora");
  const [loraRank, setLoraRank] = useState(8);
  const [loraAlpha, setLoraAlpha] = useState(16);
  const [loraDropout, setLoraDropout] = useState(0.0);
  const [loraTargets, setLoraTargets] = useState("q_proj,v_proj");
  const [qloraGroupSize, setQloraGroupSize] = useState(64);
  const [qloraDtype, setQloraDtype] = useState("bf16");
  const [baseCheckpoint, setBaseCheckpoint] = useState("");
  const [refCheckpoint, setRefCheckpoint] = useState("");
  const [rewardCheckpoint, setRewardCheckpoint] = useState("");
  const [adapterOnlySave, setAdapterOnlySave] = useState(true);
  const [learningRate, setLearningRate] = useState(0.0003);
  const [epochs, setEpochs] = useState(5);
  const [batchSize, setBatchSize] = useState(4);
  const [weightDecay, setWeightDecay] = useState(0.01);
  const [dpoBeta, setDpoBeta] = useState(0.1);
  const [dpoLossType, setDpoLossType] = useState("sigmoid");
  const [klCoef, setKlCoef] = useState(0.1);
  const [ppoClip, setPpoClip] = useState(0.2);
  const [rolloutLength, setRolloutLength] = useState(64);
  const [isTraining, setIsTraining] = useState(false);
  const [loss, setLoss] = useState<LossPoint[]>([]);
  const [statusMessage, setStatusMessage] = useState<string>("Idle — configure and click Launch");
  const ctrlRef = useRef<AbortController | null>(null);

  // Hydrate store session on mount so startTraining has a graph context.
  useEffect(() => {
    if (!projectId || !sessionId || storeSession === sessionId) return;
    let cancelled = false;
    api
      .getSession(projectId, sessionId)
      .then((detail) => {
        if (cancelled) return;
        hydrateSession({
          projectId,
          sessionId,
          graph: detail.graph,
          revision: detail.revision,
        });
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [projectId, sessionId, hydrateSession, storeSession]);

  const attachedDatasets = useMemo(() => {
    const results: string[] = [];
    const visit = (g: any) => {
      for (const node of Object.values(g?.nodes ?? {}) as any[]) {
        const cfg = node?.neuron_def?.module_config;
        if (cfg?.dataset_names && Array.isArray(cfg.dataset_names)) {
          for (const d of cfg.dataset_names) results.push(String(d));
        }
        if (node?.neuron_def?.kind === "subgraph" && node.neuron_def.subgraph) {
          visit(node.neuron_def.subgraph);
        }
      }
    };
    visit(rootGraph);
    return Array.from(new Set(results));
  }, [rootGraph]);

  const isPPO = trainingMode === "ppo";
  const isDPO = trainingMode === "dpo";
  const isSFT = trainingMode === "sft";
  const isRM = trainingMode === "reward_model";
  const usesLora = adapterType === "lora" || adapterType === "qlora";

  const trainableFraction = useMemo(() => {
    if (adapterType === "none") return "100%";
    if (adapterType === "randmap") return "~10%";
    // LoRA trainable params: sum over targeted linears of rank*(in+out).
    // For a rough indicator: r*(2*dim) ÷ dim² ≈ 2r/dim. Assume dim≈512 average.
    const frac = (2 * loraRank) / 512;
    return `${(frac * 100).toFixed(2)}%`;
  }, [adapterType, loraRank]);

  const launch = useCallback(async () => {
    if (!projectId || !sessionId) return;
    setLoss([]);
    setIsTraining(true);
    setStatusMessage("Starting…");
    ctrlRef.current = api.startTraining(
      projectId,
      sessionId,
      {
        method: "torch",
        loss_fn: "cross_entropy",
        epochs,
        learning_rate: learningRate,
        batch_size: batchSize,
        weight_decay: weightDecay,
        training_mode: trainingMode,
        base_checkpoint_path: baseCheckpoint || undefined,
        ref_checkpoint_path: refCheckpoint || undefined,
        reward_checkpoint_path: rewardCheckpoint || undefined,
        adapter_only_save: adapterOnlySave,
        finetune_config: {
          adapter_type: adapterType,
          lora_rank: loraRank,
          lora_alpha: loraAlpha,
          lora_dropout: loraDropout,
          lora_targets: loraTargets
            .split(",")
            .map((t) => t.trim())
            .filter(Boolean),
          qlora_group_size: qloraGroupSize,
          qlora_compute_dtype: qloraDtype,
          beta: dpoBeta,
          dpo_loss_type: dpoLossType,
          kl_coef: klCoef,
          ppo_clip: ppoClip,
          rollout_length: rolloutLength,
        },
      },
      (msg) => {
        if (msg.done) {
          setIsTraining(false);
          setStatusMessage(msg.error ? `Error: ${msg.error}` : "Training complete");
        } else if (msg.loss !== undefined) {
          const step = Number(msg.local_step ?? msg.step ?? 0);
          setLoss((prev) => [...prev, { step, loss: msg.loss as number, phase: (msg as any).phase }]);
          setStatusMessage(`Step ${step} • loss ${Number(msg.loss).toFixed(4)}`);
        } else if ((msg as any).message) {
          setStatusMessage(String((msg as any).message));
        } else if ((msg as any).phase === "rollout") {
          setStatusMessage(`Rollout step ${(msg as any).step ?? "?"}`);
        }
      },
    );
  }, [
    projectId,
    sessionId,
    epochs,
    learningRate,
    batchSize,
    weightDecay,
    trainingMode,
    baseCheckpoint,
    refCheckpoint,
    rewardCheckpoint,
    adapterOnlySave,
    adapterType,
    loraRank,
    loraAlpha,
    loraDropout,
    loraTargets,
    qloraGroupSize,
    qloraDtype,
    dpoBeta,
    dpoLossType,
    klCoef,
    ppoClip,
    rolloutLength,
  ]);

  const stop = useCallback(async () => {
    if (!projectId || !sessionId) return;
    ctrlRef.current?.abort();
    try {
      const active = await api.getActiveRun(projectId, sessionId);
      if (active.run_id) {
        await api.stopTraining(projectId, sessionId, active.run_id);
      }
    } catch {
      /* ignore */
    }
    setIsTraining(false);
    setStatusMessage("Stopped");
  }, [projectId, sessionId]);

  if (!projectId || !sessionId) {
    return (
      <div className="p-6 text-sm text-gray-400">
        Select a project and session to open Fine-Tuning Studio.
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-1 flex-col overflow-hidden bg-gray-950 text-gray-100">
      {/* ── Header ───────────────────────────────────────────────────── */}
      <div className="border-b border-gray-800 bg-gradient-to-r from-indigo-950/60 via-gray-900 to-rose-950/40 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-xl font-bold text-indigo-200 flex items-center gap-2">
              <span>🧪 Fine-Tuning Studio</span>
              <span className="text-[11px] font-mono uppercase tracking-wider text-indigo-400/70">
                SFT · DPO · RLHF · LoRA · qLoRA
              </span>
            </div>
            <div className="text-[12px] text-gray-500 mt-0.5">
              Parameter-efficient fine-tuning and preference training on your session graph.
            </div>
          </div>
          <div className="flex items-center gap-2">
            {!isTraining ? (
              <button
                onClick={() => void launch()}
                className="rounded bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-1.5 text-sm font-semibold shadow"
              >
                🚀 Launch Run
              </button>
            ) : (
              <button
                onClick={() => void stop()}
                className="rounded bg-rose-600 hover:bg-rose-500 text-white px-4 py-1.5 text-sm font-semibold shadow animate-pulse"
              >
                ■ Stop Run
              </button>
            )}
          </div>
        </div>
      </div>

      {/* ── Body ─────────────────────────────────────────────────────── */}
      <div className="flex-1 grid grid-cols-12 gap-4 p-5 overflow-auto">
        {/* Objective + adapter */}
        <section className="col-span-12 lg:col-span-4 space-y-4">
          <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-4">
            <div className="text-sm font-bold text-amber-200 mb-2">1. Objective</div>
            <div className="grid grid-cols-5 gap-1">
              {([
                ["pretrain", "Pretrain"],
                ["sft", "SFT"],
                ["dpo", "DPO"],
                ["ppo", "RLHF"],
                ["reward_model", "Reward"],
              ] as [TrainingMode, string][]).map(([mode, label]) => (
                <button
                  key={mode}
                  onClick={() => setTrainingMode(mode)}
                  className={`rounded text-[11px] py-1.5 border transition ${
                    trainingMode === mode
                      ? "bg-indigo-700 border-indigo-400 text-white font-semibold"
                      : "bg-gray-800 border-gray-700 text-gray-300 hover:bg-gray-700"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
            <div className="mt-2 text-[10px] text-gray-500 leading-relaxed">
              {trainingMode === "pretrain" && "Full next-token prediction from scratch."}
              {trainingMode === "sft" && "Supervised fine-tuning — loss only on response tokens."}
              {trainingMode === "dpo" && "Direct Preference Optimization — pairwise (chosen, rejected)."}
              {trainingMode === "ppo" && "RLHF: rollout → reward → clipped policy/value loss."}
              {trainingMode === "reward_model" && "Bradley-Terry preference reward model."}
            </div>
          </div>

          <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-4">
            <div className="text-sm font-bold text-amber-200 mb-2">2. Adapter</div>
            <div className="grid grid-cols-4 gap-1">
              {(["none", "lora", "qlora", "randmap"] as AdapterType[]).map((t) => (
                <button
                  key={t}
                  onClick={() => setAdapterType(t)}
                  className={`rounded text-[11px] py-1.5 border transition ${
                    adapterType === t
                      ? "bg-pink-700 border-pink-400 text-white font-semibold"
                      : "bg-gray-800 border-gray-700 text-gray-300 hover:bg-gray-700"
                  }`}
                >
                  {t === "none" ? "Full" : t.toUpperCase()}
                </button>
              ))}
            </div>
            {usesLora && (
              <div className="mt-3 space-y-2 text-[11px] text-gray-400">
                <div className="flex items-center gap-2">
                  <label className="w-20">Rank</label>
                  <input
                    type="number"
                    min={1}
                    max={256}
                    value={loraRank}
                    onChange={(e) => setLoraRank(parseInt(e.target.value) || 8)}
                    className="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200"
                  />
                </div>
                <div className="flex items-center gap-2">
                  <label className="w-20">Alpha</label>
                  <input
                    type="number"
                    value={loraAlpha}
                    onChange={(e) => setLoraAlpha(parseFloat(e.target.value) || 16)}
                    className="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200"
                  />
                </div>
                <div className="flex items-center gap-2">
                  <label className="w-20">Dropout</label>
                  <input
                    type="number"
                    step={0.01}
                    value={loraDropout}
                    onChange={(e) => setLoraDropout(parseFloat(e.target.value) || 0)}
                    className="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200"
                  />
                </div>
                <div className="flex items-center gap-2">
                  <label className="w-20">Targets</label>
                  <input
                    type="text"
                    value={loraTargets}
                    onChange={(e) => setLoraTargets(e.target.value)}
                    placeholder="q_proj,v_proj"
                    className="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200 font-mono"
                  />
                </div>
                {adapterType === "qlora" && (
                  <>
                    <div className="flex items-center gap-2">
                      <label className="w-20">Group</label>
                      <input
                        type="number"
                        value={qloraGroupSize}
                        onChange={(e) => setQloraGroupSize(parseInt(e.target.value) || 64)}
                        className="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200"
                      />
                    </div>
                    <div className="flex items-center gap-2">
                      <label className="w-20">Dtype</label>
                      <select
                        value={qloraDtype}
                        onChange={(e) => setQloraDtype(e.target.value)}
                        className="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200"
                      >
                        <option value="bf16">bf16</option>
                        <option value="fp16">fp16</option>
                        <option value="fp32">fp32</option>
                      </select>
                    </div>
                  </>
                )}
              </div>
            )}
            <div className="mt-3 flex items-center justify-between text-[10px]">
              <span className="text-gray-500">Trainable params (est.)</span>
              <span className="rounded bg-emerald-950 text-emerald-300 font-mono px-2 py-0.5">
                {trainableFraction}
              </span>
            </div>
          </div>

          <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-4">
            <div className="text-sm font-bold text-amber-200 mb-2">3. Checkpoints</div>
            <div className="space-y-2 text-[11px]">
              <div>
                <label className="text-gray-400">Base model</label>
                <input
                  type="text"
                  value={baseCheckpoint}
                  onChange={(e) => setBaseCheckpoint(e.target.value)}
                  placeholder="artifacts/base.pt"
                  className="mt-0.5 w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200 font-mono"
                />
              </div>
              {(isDPO || isPPO) && (
                <div>
                  <label className="text-gray-400">Reference model (frozen)</label>
                  <input
                    type="text"
                    value={refCheckpoint}
                    onChange={(e) => setRefCheckpoint(e.target.value)}
                    placeholder="artifacts/ref.pt"
                    className="mt-0.5 w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200 font-mono"
                  />
                </div>
              )}
              {isPPO && (
                <div>
                  <label className="text-gray-400">Reward model (frozen)</label>
                  <input
                    type="text"
                    value={rewardCheckpoint}
                    onChange={(e) => setRewardCheckpoint(e.target.value)}
                    placeholder="artifacts/reward.pt"
                    className="mt-0.5 w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200 font-mono"
                  />
                </div>
              )}
              <label className="mt-1 inline-flex items-center gap-2 text-gray-400">
                <input
                  type="checkbox"
                  checked={adapterOnlySave}
                  onChange={(e) => setAdapterOnlySave(e.target.checked)}
                />
                Save adapter only (small artifact)
              </label>
            </div>
          </div>
        </section>

        {/* Hyperparameters + monitoring */}
        <section className="col-span-12 lg:col-span-8 space-y-4">
          <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-4">
            <div className="text-sm font-bold text-amber-200 mb-2">4. Hyperparameters</div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-[11px] text-gray-400">
              <label>
                Epochs
                <input
                  type="number"
                  value={epochs}
                  onChange={(e) => setEpochs(parseInt(e.target.value) || 1)}
                  className="mt-0.5 w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200"
                />
              </label>
              <label>
                Learning rate
                <input
                  type="number"
                  step={0.00001}
                  value={learningRate}
                  onChange={(e) => setLearningRate(parseFloat(e.target.value) || 3e-4)}
                  className="mt-0.5 w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200"
                />
              </label>
              <label>
                Batch size
                <input
                  type="number"
                  value={batchSize}
                  onChange={(e) => setBatchSize(parseInt(e.target.value) || 1)}
                  className="mt-0.5 w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200"
                />
              </label>
              <label>
                Weight decay
                <input
                  type="number"
                  step={0.001}
                  value={weightDecay}
                  onChange={(e) => setWeightDecay(parseFloat(e.target.value) || 0)}
                  className="mt-0.5 w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200"
                />
              </label>
              {isDPO && (
                <>
                  <label>
                    DPO β
                    <input
                      type="number"
                      step={0.05}
                      value={dpoBeta}
                      onChange={(e) => setDpoBeta(parseFloat(e.target.value) || 0.1)}
                      className="mt-0.5 w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200"
                    />
                  </label>
                  <label>
                    DPO loss
                    <select
                      value={dpoLossType}
                      onChange={(e) => setDpoLossType(e.target.value)}
                      className="mt-0.5 w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200"
                    >
                      <option value="sigmoid">sigmoid</option>
                      <option value="hinge">hinge</option>
                      <option value="ipo">ipo</option>
                    </select>
                  </label>
                </>
              )}
              {isPPO && (
                <>
                  <label>
                    KL coef
                    <input
                      type="number"
                      step={0.01}
                      value={klCoef}
                      onChange={(e) => setKlCoef(parseFloat(e.target.value) || 0.1)}
                      className="mt-0.5 w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200"
                    />
                  </label>
                  <label>
                    PPO clip
                    <input
                      type="number"
                      step={0.01}
                      value={ppoClip}
                      onChange={(e) => setPpoClip(parseFloat(e.target.value) || 0.2)}
                      className="mt-0.5 w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200"
                    />
                  </label>
                  <label>
                    Rollout len
                    <input
                      type="number"
                      value={rolloutLength}
                      onChange={(e) => setRolloutLength(parseInt(e.target.value) || 64)}
                      className="mt-0.5 w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200"
                    />
                  </label>
                </>
              )}
            </div>
          </div>

          <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-4">
            <div className="flex items-center justify-between">
              <div className="text-sm font-bold text-amber-200">5. Live metrics</div>
              <div className="text-[11px] font-mono text-gray-400">{statusMessage}</div>
            </div>
            <div className="h-64 mt-2">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={loss} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis dataKey="step" stroke="#6b7280" style={{ fontSize: 10 }} />
                  <YAxis stroke="#6b7280" style={{ fontSize: 10 }} />
                  <Tooltip
                    contentStyle={{
                      background: "#111827",
                      border: "1px solid #374151",
                      fontSize: 11,
                    }}
                    labelStyle={{ color: "#d1d5db" }}
                  />
                  <Line
                    type="monotone"
                    dataKey="loss"
                    stroke="#818cf8"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-4">
              <div className="text-sm font-bold text-amber-200 mb-2">Session graph</div>
              <div className="text-[11px] text-gray-400 space-y-1">
                <div>
                  <span className="text-gray-500">Name:</span>{" "}
                  <span className="font-mono text-gray-200">{rootGraph.name || "(unnamed)"}</span>
                </div>
                <div>
                  <span className="text-gray-500">Nodes:</span>{" "}
                  <span className="font-mono">{Object.keys(rootGraph.nodes ?? {}).length}</span>
                </div>
                <div>
                  <span className="text-gray-500">Edges:</span>{" "}
                  <span className="font-mono">{Object.keys(rootGraph.edges ?? {}).length}</span>
                </div>
                <div>
                  <span className="text-gray-500">Method:</span>{" "}
                  <span className="font-mono">{rootGraph.training_method}</span>
                </div>
                <div>
                  <span className="text-gray-500">Datasets:</span>{" "}
                  <span className="font-mono">
                    {attachedDatasets.length > 0 ? attachedDatasets.join(", ") : "(none attached)"}
                  </span>
                </div>
              </div>
            </div>
            <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-4">
              <div className="text-sm font-bold text-amber-200 mb-2">Recipe summary</div>
              <pre className="text-[11px] font-mono text-gray-300 leading-relaxed whitespace-pre-wrap">
{`objective:   ${trainingMode}
adapter:     ${adapterType}${usesLora ? ` (r=${loraRank}, α=${loraAlpha}, targets=${loraTargets})` : ""}
base ckpt:   ${baseCheckpoint || "(none)"}
${isDPO || isPPO ? `ref ckpt:    ${refCheckpoint || "(none)"}\n` : ""}${isPPO ? `reward ckpt: ${rewardCheckpoint || "(none)"}\n` : ""}epochs:      ${epochs}
lr:          ${learningRate}
batch:       ${batchSize}
save_mode:   ${adapterOnlySave ? "adapter-only" : "full state"}
${isDPO ? `dpo:         β=${dpoBeta}, loss=${dpoLossType}\n` : ""}${isPPO ? `ppo:         clip=${ppoClip}, kl=${klCoef}, rollout=${rolloutLength}\n` : ""}`}
              </pre>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
