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
import { api } from "../api/client";

export default function TrainingPanel() {
  const {
    lossHistory,
    isTraining,
    addLossPoint,
    clearLoss,
    setTraining,
    nodes,
    edges,
  } = useGraphStore();

  const [method, setMethod] = useState<"surrogate" | "evolutionary">("surrogate");
  const [epochs, setEpochs] = useState(200);
  const [lr, setLr] = useState(0.001);
  const [popSize, setPopSize] = useState(50);
  const [dataInput, setDataInput] = useState("[[0,0],[0,1],[1,0],[1,1]]");
  const [dataTarget, setDataTarget] = useState("[[0],[1],[1],[0]]");
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

    const inputIds = nodes.filter((n) => n.data.isInput).map((n) => n.id);
    const outputIds = nodes.filter((n) => n.data.isOutput).map((n) => n.id);

    try {
      await api.setIO(inputIds, outputIds);
    } catch {}

    const graphData = {
      nodes: Object.fromEntries(
        nodes.map((n) => [
          n.id,
          {
            instance_id: n.id,
            neuron_def: n.data.neuronDef,
            position: [n.position.x, n.position.y] as [number, number],
          },
        ])
      ),
      edges: Object.fromEntries(
        edges.map((e) => [
          e.id,
          {
            id: e.id,
            src_node: e.source,
            src_port: parseInt((e.sourceHandle ?? "out-0").split("-")[1]) || 0,
            dst_node: e.target,
            dst_port: parseInt((e.targetHandle ?? "in-0").split("-")[1]) || 0,
            weight: (e.data as any)?.weight ?? 1.0,
            bias: (e.data as any)?.bias ?? 0.0,
          },
        ])
      ),
      input_node_ids: inputIds,
      output_node_ids: outputIds,
    };

    try {
      await api.putGraph(graphData);
    } catch (err) {
      console.error("Failed to sync graph", err);
    }

    ctrlRef.current = api.startTraining(
      {
        method,
        train_inputs: inputs,
        train_targets: targets,
        epochs,
        learning_rate: lr,
        population_size: popSize,
        generations: epochs,
      },
      (msg) => {
        if (msg.done) {
          setTraining(false);
        } else if (msg.step !== undefined && msg.loss !== undefined) {
          addLossPoint({ step: msg.step, loss: msg.loss });
        }
      }
    );
  }, [
    method, epochs, lr, popSize, dataInput, dataTarget,
    nodes, edges, clearLoss, setTraining, addLossPoint,
  ]);

  const stop = useCallback(async () => {
    ctrlRef.current?.abort();
    await api.stopTraining();
    setTraining(false);
  }, [setTraining]);

  return (
    <div className="border-t border-gray-800 bg-gray-900 p-3">
      <div className="flex items-center gap-3 flex-wrap">
        <select
          value={method}
          onChange={(e) => setMethod(e.target.value as any)}
          className="bg-gray-800 border border-gray-700 rounded text-xs px-2 py-1 text-gray-200"
        >
          <option value="surrogate">Surrogate (gradient)</option>
          <option value="evolutionary">Evolutionary</option>
        </select>

        <label className="text-[10px] text-gray-400">
          Epochs
          <input
            type="number"
            value={epochs}
            onChange={(e) => setEpochs(parseInt(e.target.value) || 100)}
            className="ml-1 w-16 bg-gray-800 border border-gray-700 rounded px-1 py-0.5 text-gray-200 text-xs"
          />
        </label>

        {method === "surrogate" ? (
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
        ) : (
          <label className="text-[10px] text-gray-400">
            Pop
            <input
              type="number"
              value={popSize}
              onChange={(e) => setPopSize(parseInt(e.target.value) || 50)}
              className="ml-1 w-16 bg-gray-800 border border-gray-700 rounded px-1 py-0.5 text-gray-200 text-xs"
            />
          </label>
        )}

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
    </div>
  );
}
