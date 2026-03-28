import React, { useCallback, useRef, useState } from "react";
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
import { api, type TorchTraceStat } from "../api/client";
import { graphContainsSubgraphs } from "../store/graphUtils";

export default function TrainingPanel() {
  const {
    lossHistory,
    isTraining,
    addLossPoint,
    clearLoss,
    setTraining,
    rootGraph,
    updateEdgeTelemetry,
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
  const [torchTrace, setTorchTrace] = useState<Record<string, TorchTraceStat[]>>({});
  const ctrlRef = useRef<AbortController | null>(null);

  const start = useCallback(async () => {
    let inputs: number[][];
    let targets: number[][];
    try {
      inputs = JSON.parse(dataInput);
      targets = JSON.parse(dataTarget);
    } catch {
      alert("Invalid JSON in training data fields");
      return;
    }

    clearLoss();
    setTraining(true);

    try {
      await api.putGraph(rootGraph);
    } catch (err) {
      console.error("Failed to sync graph", err);
    }

    ctrlRef.current = api.startTraining(
      {
        method: hasNestedGraphs ? null : rootGraph.training_method,
        train_inputs: inputs,
        train_targets: targets,
        outer_rounds: outerRounds,
        loss_fn: usesTorch ? "cross_entropy" : "mse",
        epochs,
        learning_rate: lr,
        population_size: popSize,
        generations: epochs,
        batch_size: batchSize,
        weight_decay: weightDecay,
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
    clearLoss,
    setTraining,
    addLossPoint,
    lossHistory.length,
  ]);

  const stop = useCallback(async () => {
    ctrlRef.current?.abort();
    await api.stopTraining();
    setTraining(false);
  }, [setTraining]);

  // Fetch graph telemetry continuously when inputs change
  React.useEffect(() => {
    if (usesTorch) {
      updateEdgeTelemetry({});
      try {
        const inputs = JSON.parse(dataInput);
        const targets = JSON.parse(dataTarget);
        if (Array.isArray(inputs) && Array.isArray(targets)) {
          api
            .putGraph(rootGraph)
            .then(() =>
              api.traceTorch({
                [rootGraph.input_node_ids[0] ?? "tokens_in"]: inputs,
                [rootGraph.input_node_ids[1] ?? "targets_in"]: targets,
              })
            )
            .then(setTorchTrace)
            .catch(() => setTorchTrace({}));
        }
      } catch {
        setTorchTrace({});
      }
      return;
    }
    try {
      const inputs = JSON.parse(dataInput);
      if (Array.isArray(inputs) && inputs.length > 0 && Array.isArray(inputs[0])) {
        const payload: Record<string, number[]> = {};
        rootGraph.input_node_ids.forEach((id, colIdx) => {
          payload[id] = inputs.map((row: any[]) => Number(row[colIdx]) || 0);
        });
        api.putGraph(rootGraph).then(() => {
          api.executeTrace(payload).then(res => {
            updateEdgeTelemetry(res);
          }).catch(() => {});
        }).catch(() => {});
      }
    } catch (e) {
      // ignore parse errors while typing
    }
  }, [dataInput, dataTarget, rootGraph, updateEdgeTelemetry, usesTorch]);

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

      {usesTorch && (
        <div className="mt-2 text-[10px] text-gray-500">
          Torch graphs expect integer token IDs shaped as `[batch, seq_len]` for both inputs and shifted targets.
        </div>
      )}

      <div className="flex gap-2 mt-2">
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
          <div className="px-3 py-2 text-[10px] font-bold text-gray-400 uppercase tracking-wider border-b border-gray-800">
            Torch Trace
          </div>
          <div className="max-h-48 overflow-auto divide-y divide-gray-900">
            {Object.entries(torchTrace).map(([nodeId, stats]) => (
              <div key={nodeId} className="px-3 py-2 text-[10px] text-gray-300 font-mono">
                <div className="text-emerald-300 mb-1">{nodeId}</div>
                {stats.map((stat, idx) => (
                  <div key={`${nodeId}-${idx}`} className="text-gray-400">
                    {stat.kind
                      ? stat.kind
                      : `${JSON.stringify(stat.shape)} mean=${stat.mean?.toFixed(4)} std=${stat.std?.toFixed(4)} min=${stat.min?.toFixed(4)} max=${stat.max?.toFixed(4)}`}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
