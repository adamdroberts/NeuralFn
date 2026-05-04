import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { useGraphStore } from "../store/graphStore";
import { api } from "../api/client";
import { graphContainsSubgraphs } from "../store/graphUtils";
import { syncActiveSessionGraph } from "../routes/sessionSync";

function stripGraphForTrace(graph: any): any {
  return {
    ...graph,
    nodes: Object.fromEntries(
      Object.entries(graph.nodes ?? {}).map(([id, node]: [string, any]) => [
        id,
        {
          ...node,
          position: [0, 0],
          measured: undefined,
          neuron_def: node.neuron_def?.subgraph
            ? {
                ...node.neuron_def,
                subgraph: stripGraphForTrace(node.neuron_def.subgraph),
              }
            : node.neuron_def,
        },
      ]),
    ),
    variant_library: Object.fromEntries(
      Object.entries(graph.variant_library ?? {}).map(([family, versions]: [string, any]) => [
        family,
        Object.fromEntries(
          Object.entries(versions ?? {}).map(([version, variantGraph]: [string, any]) => [
            version,
            stripGraphForTrace(variantGraph),
          ]),
        ),
      ]),
    ),
  };
}

function findTraceDatasetConfig(graph: any): { datasetNames: string[]; seqLen: number } | null {
  const search = (
    current: any,
    preferredType: "dataset_source" | "semantic_data_source",
  ): { datasetNames: string[]; seqLen: number } | null => {
    for (const node of Object.values(current.nodes ?? {}) as any[]) {
      const neuronDef = node?.neuron_def;
      if (neuronDef?.module_type === preferredType) {
        const datasetNames = Array.isArray(neuronDef.module_config?.dataset_names)
          ? neuronDef.module_config.dataset_names.filter(Boolean)
          : [];
        if (datasetNames.length > 0 || neuronDef?.module_type === "semantic_data_source") {
          return {
            datasetNames: datasetNames.length > 0 ? datasetNames : ["__semantic_builtin__"],
            seqLen: Number(neuronDef.module_config?.seq_len) || 9,
          };
        }
      }
      if (neuronDef?.kind === "subgraph" && neuronDef.subgraph) {
        const nested = search(neuronDef.subgraph, preferredType);
        if (nested) return nested;
      }
    }
    return null;
  };

  return search(graph, "dataset_source") ?? search(graph, "semantic_data_source");
}

export default function TrainingPanel() {
  const {
    projectId,
    sessionId,
    lossHistory,
    isTraining,
    addLossPoint,
    clearLoss,
    setTraining,
    rootGraph,
    updateEdgeTelemetry,
    torchTrace,
    updateTorchTrace,
  } = useGraphStore();

  const hasNestedGraphs = graphContainsSubgraphs(rootGraph);
  const usesTorch = rootGraph.training_method === "torch" || rootGraph.runtime === "torch";
  const [outerRounds, setOuterRounds] = useState(3);
  const [epochs, setEpochs] = useState(200);
  const [lr, setLr] = useState(0.001);
  const [batchSize, setBatchSize] = useState(8);
  const [weightDecay, setWeightDecay] = useState(0.01);
  const [popSize, setPopSize] = useState(50);
  const [dataInput, setDataInput] = useState("[[0,1,2,3],[1,2,3,4],[2,3,4,5],[3,4,5,6]]");
  const [dataTarget, setDataTarget] = useState("[[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]]");
  const [isTorchTraceMinimized, setIsTorchTraceMinimized] = useState(false);
  const ctrlRef = useRef<AbortController | null>(null);
  // ── Fine-tuning controls ──────────────────────────────────────────
  const [trainingMode, setTrainingMode] = useState<"pretrain" | "sft" | "dpo" | "ppo" | "reward_model">("pretrain");
  const [adapterType, setAdapterType] = useState<"none" | "lora" | "qlora" | "randmap">("none");
  const [loraRank, setLoraRank] = useState(8);
  const [loraAlpha, setLoraAlpha] = useState(16);
  const [loraTargets, setLoraTargets] = useState("q_proj,v_proj");
  const [baseCheckpoint, setBaseCheckpoint] = useState("");
  const [refCheckpoint, setRefCheckpoint] = useState("");
  const [rewardCheckpoint, setRewardCheckpoint] = useState("");
  const [dpoBeta, setDpoBeta] = useState(0.1);
  const [ppoClip, setPpoClip] = useState(0.2);
  const [rolloutLength, setRolloutLength] = useState(64);
  const [adapterOnlySave, setAdapterOnlySave] = useState(true);

  const traceGraphSignature = useMemo(
    () => JSON.stringify(stripGraphForTrace(rootGraph)),
    [rootGraph],
  );

  const graphDatasetConfig = useMemo(() => findTraceDatasetConfig(rootGraph), [rootGraph]);
  const graphDatasetConfigKey = useMemo(
    () =>
      graphDatasetConfig
        ? `${graphDatasetConfig.datasetNames.join("|")}::${graphDatasetConfig.seqLen}`
        : "",
    [graphDatasetConfig],
  );

  // ── Training ───────────────────────────────────────────────────────

  const start = useCallback(async () => {
    if (!projectId || !sessionId) {
      return;
    }
    const usingGraphDatasets = Boolean(graphDatasetConfig?.datasetNames.length);

    let inputs: number[][] = [];
    let targets: number[][] = [];
    if (!usingGraphDatasets) {
      try {
        inputs = JSON.parse(dataInput);
        targets = JSON.parse(dataTarget);
      } catch {
        alert("Invalid JSON in training data fields");
        return;
      }
    }

    clearLoss();
    setTraining(true);

    try {
      await syncActiveSessionGraph({ skipIfClean: true });
    } catch (err) {
      console.error("Failed to sync graph", err);
    }

    ctrlRef.current = api.startTraining(
      projectId,
      sessionId,
      {
        method: hasNestedGraphs ? null : rootGraph.training_method,
        ...(usingGraphDatasets ? {} : { train_inputs: inputs, train_targets: targets }),
        outer_rounds: outerRounds,
        loss_fn: usesTorch ? "cross_entropy" : "mse",
        epochs,
        learning_rate: lr,
        population_size: popSize,
        generations: epochs,
        batch_size: batchSize,
        weight_decay: weightDecay,
        // ── Fine-tuning ───────────────────────────────────────────────
        training_mode: trainingMode,
        base_checkpoint_path: baseCheckpoint || undefined,
        ref_checkpoint_path: refCheckpoint || undefined,
        reward_checkpoint_path: rewardCheckpoint || undefined,
        adapter_only_save: adapterOnlySave,
        finetune_config:
          trainingMode === "pretrain" && adapterType === "none"
            ? undefined
            : {
                adapter_type: adapterType,
                lora_rank: loraRank,
                lora_alpha: loraAlpha,
                lora_targets: loraTargets
                  .split(",")
                  .map((t) => t.trim())
                  .filter(Boolean),
                beta: dpoBeta,
                ppo_clip: ppoClip,
                rollout_length: rolloutLength,
              },
      },
      (msg) => {
        if (msg.done) {
          setTraining(false);
        } else if (msg.loss !== undefined) {
          addLossPoint({
            step: msg.local_step ?? msg.step ?? lossHistory.length,
            loss: msg.loss,
            graphName: msg.graph_name,
            method: msg.method,
          });
        }
      }
    );
  }, [
    projectId,
    sessionId,
    rootGraph,
    hasNestedGraphs,
    usesTorch,
    outerRounds,
    epochs,
    lr,
    batchSize,
    weightDecay,
    popSize,
    dataInput,
    dataTarget,
    graphDatasetConfigKey,
    clearLoss,
    setTraining,
    addLossPoint,
    lossHistory.length,
  ]);

  const stop = useCallback(async () => {
    if (!projectId || !sessionId) {
      return;
    }
    ctrlRef.current?.abort();
    const activeRun = await api.getActiveRun(projectId, sessionId);
    if (activeRun.run_id) {
      await api.stopTraining(projectId, sessionId, activeRun.run_id);
    }
    setTraining(false);
  }, [projectId, sessionId, setTraining]);

  // Fetch graph telemetry continuously when inputs change
  useEffect(() => {
    if (!projectId || !sessionId) {
      updateEdgeTelemetry({});
      updateTorchTrace({}, null);
      return;
    }
    if (usesTorch) {
      updateEdgeTelemetry({});
      const previewDatasets = graphDatasetConfig?.datasetNames.length ? graphDatasetConfig : null;

      const syncAndTrace = async () => {
        await syncActiveSessionGraph({ skipIfClean: true });
        if (previewDatasets) {
          const response = await api.traceTorchPreview(projectId, sessionId, { preview_batch_size: 1 });
          updateTorchTrace(response.trace, response.source);
          return;
        }

        try {
          const inputs = JSON.parse(dataInput);
          const targets = JSON.parse(dataTarget);
          if (Array.isArray(inputs) && Array.isArray(targets)) {
            const response = await api.traceTorchPreview(projectId, sessionId, {
              inputs: {
                [rootGraph.input_node_ids[0] ?? "tokens_in"]: inputs,
                [rootGraph.input_node_ids[1] ?? "targets_in"]: targets,
              },
            });
            updateTorchTrace(response.trace, response.source);
            return;
          }
        } catch {
          // ignore parse errors while typing
        }

        updateTorchTrace({}, null);
      };

      syncAndTrace().catch(() => updateTorchTrace({}, null));
      return;
    }
    updateTorchTrace({}, null);
    try {
      const inputs = JSON.parse(dataInput);
      if (Array.isArray(inputs) && inputs.length > 0 && Array.isArray(inputs[0])) {
        const payload: Record<string, number[]> = {};
        rootGraph.input_node_ids.forEach((id, colIdx) => {
          payload[id] = inputs.map((row: any[]) => Number(row[colIdx]) || 0);
        });
        syncActiveSessionGraph({ skipIfClean: true }).then(() => {
          api.executeTrace(projectId, sessionId, payload).then(res => {
            updateEdgeTelemetry(res);
          }).catch(() => {});
        }).catch(() => {});
      }
    } catch (e) {
      // ignore parse errors while typing
    }
  }, [
    dataInput,
    dataTarget,
    graphDatasetConfigKey,
    projectId,
    sessionId,
    traceGraphSignature,
    updateEdgeTelemetry,
    updateTorchTrace,
    usesTorch,
  ]);

  return (
    <div className="border-t border-gray-800 bg-gray-900 p-3">
      <div className="flex items-center gap-3 flex-wrap">
        <span className="text-xs text-gray-300">
          Root method: <span className="font-mono">{rootGraph.training_method}</span>
        </span>

        {hasNestedGraphs && (
          <label className="text-[10px] text-gray-400">
            Outer Rounds
            <input
              type="number"
              value={outerRounds}
              onChange={(e) => setOuterRounds(parseInt(e.target.value, 10) || 1)}
              className="ml-1 w-16 bg-gray-800 border border-gray-700 rounded px-1 py-0.5 text-gray-200 text-xs"
            />
          </label>
        )}

        <label className="text-[10px] text-gray-400">
          Epochs
          <input
            type="number"
            value={epochs}
            onChange={(e) => setEpochs(parseInt(e.target.value) || 100)}
            className="ml-1 w-16 bg-gray-800 border border-gray-700 rounded px-1 py-0.5 text-gray-200 text-xs"
          />
        </label>

        <label className="text-[10px] text-gray-400">
          LR
          <input
            type="number"
            value={lr}
            step={0.0001}
            onChange={(e) => setLr(parseFloat(e.target.value) || 0.001)}
            className="ml-1 w-20 bg-gray-800 border border-gray-700 rounded px-1 py-0.5 text-gray-200 text-xs"
          />
        </label>

        {usesTorch && (
          <>
            <label className="text-[10px] text-gray-400">
              Batch
              <input
                type="number"
                value={batchSize}
                onChange={(e) => setBatchSize(parseInt(e.target.value, 10) || 1)}
                className="ml-1 w-16 bg-gray-800 border border-gray-700 rounded px-1 py-0.5 text-gray-200 text-xs"
              />
            </label>

            <label className="text-[10px] text-gray-400">
              WD
              <input
                type="number"
                value={weightDecay}
                step={0.001}
                onChange={(e) => setWeightDecay(parseFloat(e.target.value) || 0)}
                className="ml-1 w-20 bg-gray-800 border border-gray-700 rounded px-1 py-0.5 text-gray-200 text-xs"
              />
            </label>
          </>
        )}

        <label className="text-[10px] text-gray-400">
          Pop
          <input
            type="number"
            value={popSize}
            onChange={(e) => setPopSize(parseInt(e.target.value) || 50)}
            className="ml-1 w-16 bg-gray-800 border border-gray-700 rounded px-1 py-0.5 text-gray-200 text-xs"
          />
        </label>

        <label className="text-[10px] text-gray-400">
          Mode
          <select
            value={trainingMode}
            onChange={(e) => setTrainingMode(e.target.value as any)}
            className="ml-1 bg-gray-800 border border-gray-700 rounded px-1 py-0.5 text-gray-200 text-xs"
          >
            <option value="pretrain">pretrain</option>
            <option value="sft">sft</option>
            <option value="dpo">dpo</option>
            <option value="ppo">ppo</option>
            <option value="reward_model">reward_model</option>
          </select>
        </label>

        <label className="text-[10px] text-gray-400">
          Adapter
          <select
            value={adapterType}
            onChange={(e) => setAdapterType(e.target.value as any)}
            className="ml-1 bg-gray-800 border border-gray-700 rounded px-1 py-0.5 text-gray-200 text-xs"
          >
            <option value="none">none</option>
            <option value="lora">lora</option>
            <option value="qlora">qlora</option>
            <option value="randmap">randmap</option>
          </select>
        </label>

        {!isTraining ? (
          <button
            onClick={start}
            className="bg-blue-600 hover:bg-blue-500 text-white text-xs px-3 py-1 rounded font-medium"
          >
            Train
          </button>
        ) : (
          <button
            onClick={stop}
            className="bg-red-600 hover:bg-red-500 text-white text-xs px-3 py-1 rounded font-medium"
          >
            Stop
          </button>
        )}

        {lossHistory.length > 0 && (
          <span className="text-[10px] text-gray-400 ml-auto">
            Loss: {lossHistory[lossHistory.length - 1].loss.toFixed(6)}
          </span>
        )}
      </div>

      {hasNestedGraphs && (
        <div className="mt-2 text-[10px] text-gray-500">
          Nested networks use each graph&apos;s own training method. Set root and subgraph methods from the side panel.
        </div>
      )}

      {trainingMode !== "pretrain" && (
        <div className="mt-2 flex flex-wrap items-center gap-2 bg-gray-900/50 rounded p-2 border border-gray-800">
          <span className="text-[10px] font-bold text-amber-300">Fine-tune</span>
          <label className="text-[10px] text-gray-400">
            Base ckpt
            <input
              type="text"
              value={baseCheckpoint}
              onChange={(e) => setBaseCheckpoint(e.target.value)}
              placeholder="artifacts/base.pt"
              className="ml-1 w-40 bg-gray-800 border border-gray-700 rounded px-1 py-0.5 text-gray-200 text-[10px]"
            />
          </label>
          {(trainingMode === "dpo" || trainingMode === "ppo") && (
            <label className="text-[10px] text-gray-400">
              Ref ckpt
              <input
                type="text"
                value={refCheckpoint}
                onChange={(e) => setRefCheckpoint(e.target.value)}
                placeholder="artifacts/ref.pt"
                className="ml-1 w-40 bg-gray-800 border border-gray-700 rounded px-1 py-0.5 text-gray-200 text-[10px]"
              />
            </label>
          )}
          {trainingMode === "ppo" && (
            <label className="text-[10px] text-gray-400">
              Reward ckpt
              <input
                type="text"
                value={rewardCheckpoint}
                onChange={(e) => setRewardCheckpoint(e.target.value)}
                placeholder="artifacts/reward.pt"
                className="ml-1 w-40 bg-gray-800 border border-gray-700 rounded px-1 py-0.5 text-gray-200 text-[10px]"
              />
            </label>
          )}
          {(adapterType === "lora" || adapterType === "qlora") && (
            <>
              <label className="text-[10px] text-gray-400">
                Rank
                <input
                  type="number"
                  value={loraRank}
                  onChange={(e) => setLoraRank(parseInt(e.target.value) || 8)}
                  className="ml-1 w-16 bg-gray-800 border border-gray-700 rounded px-1 py-0.5 text-gray-200 text-[10px]"
                />
              </label>
              <label className="text-[10px] text-gray-400">
                Alpha
                <input
                  type="number"
                  value={loraAlpha}
                  onChange={(e) => setLoraAlpha(parseFloat(e.target.value) || 16)}
                  className="ml-1 w-16 bg-gray-800 border border-gray-700 rounded px-1 py-0.5 text-gray-200 text-[10px]"
                />
              </label>
              <label className="text-[10px] text-gray-400">
                Targets
                <input
                  type="text"
                  value={loraTargets}
                  onChange={(e) => setLoraTargets(e.target.value)}
                  className="ml-1 w-40 bg-gray-800 border border-gray-700 rounded px-1 py-0.5 text-gray-200 text-[10px]"
                />
              </label>
            </>
          )}
          {trainingMode === "dpo" && (
            <label className="text-[10px] text-gray-400">
              β
              <input
                type="number"
                value={dpoBeta}
                step={0.01}
                onChange={(e) => setDpoBeta(parseFloat(e.target.value) || 0.1)}
                className="ml-1 w-20 bg-gray-800 border border-gray-700 rounded px-1 py-0.5 text-gray-200 text-[10px]"
              />
            </label>
          )}
          {trainingMode === "ppo" && (
            <>
              <label className="text-[10px] text-gray-400">
                Clip
                <input
                  type="number"
                  value={ppoClip}
                  step={0.01}
                  onChange={(e) => setPpoClip(parseFloat(e.target.value) || 0.2)}
                  className="ml-1 w-20 bg-gray-800 border border-gray-700 rounded px-1 py-0.5 text-gray-200 text-[10px]"
                />
              </label>
              <label className="text-[10px] text-gray-400">
                Rollout
                <input
                  type="number"
                  value={rolloutLength}
                  onChange={(e) => setRolloutLength(parseInt(e.target.value) || 64)}
                  className="ml-1 w-20 bg-gray-800 border border-gray-700 rounded px-1 py-0.5 text-gray-200 text-[10px]"
                />
              </label>
            </>
          )}
          <label className="text-[10px] text-gray-400">
            <input
              type="checkbox"
              checked={adapterOnlySave}
              onChange={(e) => setAdapterOnlySave(e.target.checked)}
              className="mr-1"
            />
            Adapter-only save
          </label>
        </div>
      )}

      {usesTorch && (
        <div className="mt-2 text-[10px] text-gray-500">
          Torch graphs expect integer token IDs shaped as `[batch, seq_len]` for both inputs and shifted targets.
        </div>
      )}

      {graphDatasetConfig ? (
        <div className="mt-3 rounded border border-blue-900/70 bg-blue-950/20 px-3 py-3">
          <div className="text-[11px] font-medium text-blue-200">
            Dataset-backed training is configured from the graph&apos;s `dataset_source` node.
          </div>
          <div className="mt-1 text-[10px] text-gray-400">
            {graphDatasetConfig.datasetNames.length} dataset
            {graphDatasetConfig.datasetNames.length > 1 ? "s" : ""} selected with seq_len{" "}
            {graphDatasetConfig.seqLen}. Manage project datasets from the Datasets tab and pick them
            from the dataset_source node panel.
          </div>
        </div>
      ) : (
        <div className="mt-3 flex gap-2">
          <label className="text-[10px] text-gray-400 flex-1">
            Inputs (JSON)
            <textarea
              value={dataInput}
              onChange={(e) => setDataInput(e.target.value)}
              className="block w-full mt-0.5 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200 text-xs font-mono h-12 resize-none"
            />
          </label>
          <label className="text-[10px] text-gray-400 flex-1">
            Targets (JSON)
            <textarea
              value={dataTarget}
              onChange={(e) => setDataTarget(e.target.value)}
              className="block w-full mt-0.5 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200 text-xs font-mono h-12 resize-none"
            />
          </label>
        </div>
      )}

      {lossHistory.length > 0 && (
        <div className="mt-2 h-28">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={lossHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="step" stroke="#6b7280" tick={{ fontSize: 10 }} />
              <YAxis stroke="#6b7280" tick={{ fontSize: 10 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1f2937",
                  border: "1px solid #374151",
                  fontSize: 11,
                }}
              />
              <Line
                type="monotone"
                dataKey="loss"
                stroke="#3b82f6"
                dot={false}
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {usesTorch && Object.keys(torchTrace).length > 0 && (
        <div className="mt-3 border border-gray-800 rounded bg-gray-950/70">
          <div className="px-3 py-2 flex items-center justify-between text-[10px] font-bold text-gray-400 uppercase tracking-wider border-b border-gray-800">
            <span>Torch Trace</span>
            <button
              onClick={() => setIsTorchTraceMinimized((prev) => !prev)}
              className="text-[10px] normal-case font-medium text-gray-400 hover:text-gray-200"
            >
              {isTorchTraceMinimized ? "Expand" : "Minimize"}
            </button>
          </div>
          {isTorchTraceMinimized ? (
            <div className="px-3 py-2 text-[10px] text-gray-500">
              {Object.keys(torchTrace).length} traced node{Object.keys(torchTrace).length === 1 ? "" : "s"}
            </div>
          ) : (
            <div className="max-h-48 overflow-auto divide-y divide-gray-900">
              {Object.entries(torchTrace).map(([nodeId, stats]) => (
                <div key={nodeId} className="px-3 py-2 text-[10px] text-gray-300 font-mono">
                  <div className="text-emerald-300 mb-1">{nodeId}</div>
                  {stats.map((stat, idx) => (
                    <div key={`${nodeId}-${idx}`} className="text-gray-400">
                      {stat.kind
                        ? stat.kind
                        : `${JSON.stringify(stat.shape)}${stat.dtype ? ` ${stat.dtype}` : ""} mean=${stat.mean?.toFixed(4)} std=${stat.std?.toFixed(4)} min=${stat.min?.toFixed(4)} max=${stat.max?.toFixed(4)}${Array.isArray(stat.preview) && stat.preview.length > 0 ? ` preview=${JSON.stringify(stat.preview)}` : ""}`}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
