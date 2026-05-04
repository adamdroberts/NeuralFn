import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import { useGraphStore } from "../store/graphStore";

type ChatMessage = {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  pending?: boolean;
  tokenCount?: number;
};

const randomId = () => `m_${Math.random().toString(36).slice(2, 9)}_${Date.now()}`;

/**
 * Interactive chat drawer for the graph editor.
 *
 * The panel loads the current session's graph, optionally applies a base
 * checkpoint + adapter checkpoint before generation, and streams completions
 * back in a typical chat-transcript UI. This is the "load a model + chat"
 * surface that sits alongside the graph canvas.
 */
export default function ChatPanel() {
  const projectId = useGraphStore((state) => state.projectId);
  const sessionId = useGraphStore((state) => state.sessionId);

  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [baseCheckpoint, setBaseCheckpoint] = useState("");
  const [adapterCheckpoint, setAdapterCheckpoint] = useState("");
  const [maxNewTokens, setMaxNewTokens] = useState(64);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(32);
  const [error, setError] = useState<string | null>(null);
  const scrollerRef = useRef<HTMLDivElement>(null);

  const canChat = Boolean(projectId && sessionId);

  useEffect(() => {
    if (scrollerRef.current) {
      scrollerRef.current.scrollTop = scrollerRef.current.scrollHeight;
    }
  }, [messages, open]);

  const send = useCallback(async () => {
    if (!projectId || !sessionId) return;
    const text = input.trim();
    if (!text || busy) return;
    setError(null);

    const userMsg: ChatMessage = { id: randomId(), role: "user", content: text };
    const pendingAssistant: ChatMessage = {
      id: randomId(),
      role: "assistant",
      content: "",
      pending: true,
    };
    setMessages((m) => [...m, userMsg, pendingAssistant]);
    setInput("");
    setBusy(true);
    try {
      const resp = await api.chatGenerate(projectId, sessionId, {
        prompt: text,
        max_new_tokens: maxNewTokens,
        temperature,
        top_k: topK > 0 ? topK : null,
        base_checkpoint: baseCheckpoint || undefined,
        adapter_checkpoint: adapterCheckpoint || undefined,
      });
      setMessages((m) =>
        m.map((msg) =>
          msg.id === pendingAssistant.id
            ? { ...msg, content: resp.generated || "(empty completion)", pending: false, tokenCount: resp.tokens.length }
            : msg,
        ),
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : "Generation failed";
      setError(message);
      setMessages((m) =>
        m.map((msg) =>
          msg.id === pendingAssistant.id
            ? { ...msg, content: `⚠️ ${message}`, pending: false }
            : msg,
        ),
      );
    } finally {
      setBusy(false);
    }
  }, [projectId, sessionId, input, busy, maxNewTokens, temperature, topK, baseCheckpoint, adapterCheckpoint]);

  const reset = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  const hasAdapter = useMemo(() => adapterCheckpoint.trim().length > 0, [adapterCheckpoint]);
  const hasBase = useMemo(() => baseCheckpoint.trim().length > 0, [baseCheckpoint]);

  return (
    <>
      <button
        onClick={() => setOpen((v) => !v)}
        className="absolute right-3 top-2 z-20 rounded bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold px-3 py-1 shadow-md"
        title="Open chat panel"
      >
        {open ? "✕ Close Chat" : "💬 Chat with model"}
      </button>

      {open && (
        <aside className="absolute right-0 top-0 bottom-0 z-10 w-[420px] max-w-[90vw] flex flex-col border-l border-gray-800 bg-gray-950/95 shadow-2xl">
          <div className="border-b border-gray-800 px-3 py-2">
            <div className="text-sm font-bold text-indigo-300">Model Chat</div>
            <div className="text-[11px] text-gray-500">
              Run the current session's graph as an autoregressive model.
              Optionally load a pretrained base and a LoRA adapter on top.
            </div>
          </div>

          {/* Checkpoint loader */}
          <div className="border-b border-gray-800 bg-gray-900/60 p-3 space-y-2 text-[11px]">
            <div className="flex items-center gap-2">
              <span className="w-16 text-gray-400">Base</span>
              <input
                type="text"
                placeholder="artifacts/base.pt"
                value={baseCheckpoint}
                onChange={(e) => setBaseCheckpoint(e.target.value)}
                className="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200 font-mono"
              />
              {hasBase && (
                <span className="rounded bg-emerald-950 text-emerald-300 px-1 text-[10px]">loaded</span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <span className="w-16 text-gray-400">Adapter</span>
              <input
                type="text"
                placeholder="artifacts/lora_adapter.pt"
                value={adapterCheckpoint}
                onChange={(e) => setAdapterCheckpoint(e.target.value)}
                className="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200 font-mono"
              />
              {hasAdapter && (
                <span className="rounded bg-pink-950 text-pink-300 px-1 text-[10px]">LoRA</span>
              )}
            </div>
            <div className="flex items-center gap-2 text-[10px] text-gray-400">
              <label>
                Max tokens
                <input
                  type="number"
                  value={maxNewTokens}
                  onChange={(e) => setMaxNewTokens(parseInt(e.target.value) || 64)}
                  className="ml-1 w-16 bg-gray-800 border border-gray-700 rounded px-1 py-0.5 text-gray-200"
                />
              </label>
              <label>
                Temp
                <input
                  type="number"
                  step={0.1}
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value) || 0.8)}
                  className="ml-1 w-16 bg-gray-800 border border-gray-700 rounded px-1 py-0.5 text-gray-200"
                />
              </label>
              <label>
                Top-k
                <input
                  type="number"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value) || 0)}
                  className="ml-1 w-16 bg-gray-800 border border-gray-700 rounded px-1 py-0.5 text-gray-200"
                />
              </label>
              <button
                onClick={reset}
                className="ml-auto rounded bg-gray-800 hover:bg-gray-700 px-2 py-0.5 text-gray-200"
              >
                Clear
              </button>
            </div>
          </div>

          {/* Message transcript */}
          <div
            ref={scrollerRef}
            className="flex-1 overflow-y-auto p-3 space-y-2 bg-gray-950"
          >
            {messages.length === 0 ? (
              <div className="text-[11px] text-gray-500 italic">
                {canChat
                  ? "Type a prompt below to chat with the current graph. The session graph is used as an autoregressive model; byte-level tokenization is assumed."
                  : "Open a session to start chatting."}
              </div>
            ) : (
              messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`rounded px-2 py-1.5 text-xs leading-relaxed whitespace-pre-wrap font-mono ${
                    msg.role === "user"
                      ? "bg-blue-950/60 border border-blue-900/40 text-blue-100"
                      : msg.role === "assistant"
                      ? `bg-emerald-950/40 border border-emerald-900/40 text-emerald-100 ${
                          msg.pending ? "opacity-60 animate-pulse" : ""
                        }`
                      : "bg-gray-900 text-gray-400"
                  }`}
                >
                  <div className="text-[9px] uppercase tracking-wider text-gray-500 mb-0.5">
                    {msg.role}
                    {msg.tokenCount !== undefined && (
                      <span className="ml-2 text-gray-600">{msg.tokenCount} toks</span>
                    )}
                  </div>
                  {msg.content}
                </div>
              ))
            )}
          </div>

          {/* Input */}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void send();
            }}
            className="border-t border-gray-800 bg-gray-900 p-2"
          >
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void send();
                }
              }}
              placeholder="Type a prompt… (Enter to send, Shift+Enter for newline)"
              rows={3}
              disabled={!canChat || busy}
              className="w-full bg-gray-950 border border-gray-700 rounded px-2 py-1 text-xs text-gray-100 font-mono resize-none"
            />
            <div className="mt-1 flex items-center justify-between">
              <span className="text-[10px] text-gray-500">
                {error
                  ? `⚠️ ${error}`
                  : hasAdapter
                  ? `Base + LoRA adapter active`
                  : hasBase
                  ? `Base checkpoint loaded`
                  : "Using in-memory graph weights"}
              </span>
              <button
                type="submit"
                disabled={!canChat || busy || !input.trim()}
                className="rounded bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-700 text-white text-xs font-semibold px-3 py-1"
              >
                {busy ? "Generating…" : "Send"}
              </button>
            </div>
          </form>
        </aside>
      )}
    </>
  );
}
