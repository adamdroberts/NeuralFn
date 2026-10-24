import React from "react";
import { useGraphStore } from "../store/graphStore";
import { useSettingsStore } from "../store/settingsStore";

export default function LibraryPanel() {
  const variantLibrary = useGraphStore((state) => state.rootGraph.variant_library);
  const openVariant = useGraphStore((state) => state.openVariant);
  const addVariantNode = useGraphStore((state) => state.addVariantNode);
  const particleGraphEditorEnabled = useSettingsStore(
    (state) => state.particleGraphEditorEnabled,
  );

  const families = Object.entries(variantLibrary);

  return (
    <aside
      className={`w-64 min-h-0 p-3 overflow-auto transition-all duration-300 ${
        particleGraphEditorEnabled
          ? "cyber-panel-left z-10"
          : "border-r border-gray-800 bg-gray-950/80"
      }`}
    >
      <div className="mb-3">
        <div
          className={`text-sm font-bold ${
            particleGraphEditorEnabled
              ? "font-mono uppercase tracking-wider text-amber-300 drop-shadow-[0_0_8px_rgba(251,191,36,0.4)]"
              : "text-amber-200"
          }`}
        >
          {particleGraphEditorEnabled ? "◈ Variant Matrix" : "Variant Library"}
        </div>
        <div className="text-[11px] text-gray-400">
          Saved subgraph families and swappable versions for this file.
        </div>
      </div>

      {families.length === 0 && (
        <div
          className={
            particleGraphEditorEnabled
              ? "rounded-lg border border-dashed border-cyan-500/30 bg-slate-950/70 px-3 py-4 font-mono text-[11px] text-cyan-200/60"
              : "rounded border border-dashed border-gray-800 px-3 py-4 text-[11px] text-gray-500"
          }
        >
          No saved variants yet. Save a subgraph as a named family/version or add a GPT template to seed the library.
        </div>
      )}

      <div className="space-y-3">
        {families.map(([family, versions]) => (
          <div
            key={family}
            className={
              particleGraphEditorEnabled
                ? "cyber-card-3d rounded-lg p-2.5"
                : "rounded border border-gray-800 bg-gray-900/70 p-2"
            }
          >
            <div className="mb-2 flex items-center justify-between text-xs font-bold font-mono text-amber-300">
              <span>{family}</span>
              <span className="text-[9px] text-cyan-400/70">
                {Object.keys(versions).length}v
              </span>
            </div>
            <div className="space-y-2">
              {Object.keys(versions)
                .sort()
                .map((version) => (
                  <div
                    key={`${family}-${version}`}
                    className={
                      particleGraphEditorEnabled
                        ? "rounded border border-cyan-500/20 bg-slate-950/85 px-2.5 py-2"
                        : "rounded bg-gray-950/70 px-2 py-2"
                    }
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[11px] font-mono text-gray-200">{version}</span>
                      <div className="flex gap-1">
                        <button
                          onClick={() => openVariant(family, version)}
                          className={
                            particleGraphEditorEnabled
                              ? "rounded border border-cyan-500/40 bg-cyan-950/60 px-2 py-1 font-mono text-[10px] text-cyan-200 hover:bg-cyan-900/70"
                              : "rounded bg-gray-800 px-2 py-1 text-[10px] text-gray-200 hover:bg-gray-700"
                          }
                        >
                          Open
                        </button>
                        <button
                          onClick={() => addVariantNode(family, version)}
                          className={
                            particleGraphEditorEnabled
                              ? "rounded border border-amber-500/50 bg-amber-950/60 px-2 py-1 font-mono text-[10px] text-amber-200 hover:bg-amber-900/70"
                              : "rounded bg-amber-900 px-2 py-1 text-[10px] text-amber-100 hover:bg-amber-800"
                          }
                        >
                          Insert
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
            </div>
          </div>
        ))}
      </div>
    </aside>
  );
}
