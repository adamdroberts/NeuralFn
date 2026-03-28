import React from "react";
import { useGraphStore } from "../store/graphStore";

export default function LibraryPanel() {
  const variantLibrary = useGraphStore((state) => state.rootGraph.variant_library);
  const openVariant = useGraphStore((state) => state.openVariant);
  const addVariantNode = useGraphStore((state) => state.addVariantNode);

  const families = Object.entries(variantLibrary);

  return (
    <aside className="w-64 border-r border-gray-800 bg-gray-950/80 p-3 overflow-auto">
      <div className="mb-3">
        <div className="text-sm font-bold text-amber-200">Variant Library</div>
        <div className="text-[11px] text-gray-500">
          Saved subgraph families and swappable versions for this file.
        </div>
      </div>

      {families.length === 0 && (
        <div className="rounded border border-dashed border-gray-800 px-3 py-4 text-[11px] text-gray-500">
          No saved variants yet. Save a subgraph as a named family/version or add a GPT template to seed the library.
        </div>
      )}

      <div className="space-y-3">
        {families.map(([family, versions]) => (
          <div key={family} className="rounded border border-gray-800 bg-gray-900/70 p-2">
            <div className="mb-2 text-xs font-bold text-amber-300">{family}</div>
            <div className="space-y-2">
              {Object.keys(versions).sort().map((version) => (
                <div key={`${family}-${version}`} className="rounded bg-gray-950/70 px-2 py-2">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[11px] font-mono text-gray-200">{version}</span>
                    <div className="flex gap-1">
                      <button
                        onClick={() => openVariant(family, version)}
                        className="rounded bg-gray-800 px-2 py-1 text-[10px] text-gray-200 hover:bg-gray-700"
                      >
                        Open
                      </button>
                      <button
                        onClick={() => addVariantNode(family, version)}
                        className="rounded bg-amber-900 px-2 py-1 text-[10px] text-amber-100 hover:bg-amber-800"
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
