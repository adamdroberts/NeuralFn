---
name: neuralfn-editor
description: >-
  Work on the NeuralFn web editor (editor/ React + Vite + React Flow + Zustand
  app): the graph canvas, Volumetric 3D Cyber mode, View Full Graph mode,
  subgraph expand/collapse animations, panel dialogs, settings store, toolbar,
  node picker, and template preset dropdown. Use when the user asks to change,
  debug, or extend anything under editor/src, or asks how the editor renders or
  edits graphs. Do not use for Python SDK, REST, MCP, or CLI work.
---

# NeuralFn Editor

The editor is the React single-page app in `editor/`. It talks to the FastAPI backend through `editor/src/api/client.ts` and keeps all graph state in Zustand stores. Read the docs before editing code:

- [Editor overview](../../../docs/editor/README.md) -- tech stack, directory layout, page index.
- [Store](../../../docs/editor/store.md) -- `useGraphStore` (graph document) and `useSettingsStore` (feature flags, cyber-mode settings, panel dialogs).
- [Graph utilities](../../../docs/editor/graph-utils.md) -- `FlowNodeData`, path types, variant resolution, `buildFullExpandedGraph`.
- [Components](../../../docs/editor/components.md) -- canvas, nodes, edges, toolbar, panels, modals.
- [Cyber mode, Full Graph view, panel dialogs](../../../docs/editor/cyber-mode.md) -- how the 3D canvas, settings, full-graph mode, and expansion animations fit together.
- [Pages and routing](../../../docs/editor/pages.md) -- `EditorPage`, routes, session sync.

Full documentation index: [docs/README.md](../../../docs/README.md). Single-file bundle: [llms-full.txt](../../../llms-full.txt).

## Run and verify

```bash
./start-servers.sh                 # backend :8000 + editor :5173 together
npm --prefix editor run dev        # editor only
npm --prefix editor run build      # tsc -b && vite build -- run this before claiming an editor change works
```

There is no frontend unit-test suite; `npm --prefix editor run build` is the required verification for every editor change. Python-side preset coverage lives in `tests/test_template_presets.py`.

## Repo rules that are easy to break

1. **Template presets are hardcoded in the toolbar.** A new backend preset is invisible until you add `<option value="preset_id">Label</option>` to the GPT template `<select>` in `editor/src/components/Toolbar.tsx`, and append it to `PRESETS` in `tests/test_template_presets.py`.
2. **Never turn variant-library port mismatches into hard errors.** `resolveVariantLibrary` in `editor/src/store/graphUtils.ts` (and `resolve_variant_library` in `neuralfn/graph.py`) must fall back to the node's inline subgraph when the library entry's ports do not match. Presets share one flat family namespace (`attention`, `mlp`, `attn_block`, ...) and loading two presets back-to-back must not break.
3. **Two stores, two jobs.** `useGraphStore` holds the server-synced graph document; `useSettingsStore` holds client-only UI state persisted in `localStorage` under `neuralfn:editor_settings:v1`. Never put graph data in the settings store or UI flags in the graph document.
4. **Canvas selection.** `GraphCanvas` renders `ParticleGraphCanvas` when `particleGraphEditorEnabled` is true, else `StandardGraphCanvas`. Any feature that must work in both modes (View Full Graph, subgraph expansion, zones) is implemented once in `SubgraphExpansionOverlay.tsx` / `graphUtils.ts` and wired into both canvases.
5. **Scoped node ids.** In View Full Graph mode nested leaves use ids of the form `fullg::<path>::<nodeId>` (`makeFullGraphNodeId` / `parseFullGraphNodeId`). Store actions that take a node id must keep accepting both plain and scoped ids.
6. **Keep heavy panels mounted.** `EditorPage` hides (not unmounts) the bottom `TrainingPanel` and right `CodePanel` / `ChatPanel` dialogs so training runs, trace polling, Monaco state, and chat transcripts survive closing the dialog. Do not switch them to conditional rendering.
7. **Node archetypes.** Cyber-mode visuals are keyed by `resolveNodeArchetype(label, neuronDef, isInput, isOutput)` in `editor/src/components/nodeArchetypes.ts`, which matches substrings of the node label, neuron name, and `module_type`. When adding a builtin neuron family, extend the matcher (and `PICKER_CATEGORIES` in `NodePickerModal.tsx`) or it falls through to `signal_modulator`.
8. **Docs are part of the change.** Editor behaviour changes must update `docs/editor/*.md`, `README.md`, `CHANGELOG.md`, and regenerate `llms-full.txt` (see `AGENTS.md`).

## Quick reference

| Need | Where |
|------|-------|
| Add a toolbar action | `editor/src/components/Toolbar.tsx` |
| Add a settings toggle | `settingsStore.ts` (field + default + load validation + action) then `SettingsModal.tsx` |
| Change how nodes look in standard mode | `NeuronNode.tsx`, `InteractiveEdge.tsx` |
| Change how nodes look in cyber mode | `ParticleNeuronNode.tsx`, `VolumetricNodeCore.tsx`, `ParticleEdge.tsx`, `nodeArchetypes.ts` |
| Change the full-graph layout | `buildFullExpandedGraph` in `graphUtils.ts` (constants at the top of the function) |
| Change the expand/collapse choreography | `useSubgraphExpansion` in `SubgraphExpansionOverlay.tsx` (`DURATION_MS = 2000`, stage boundaries 0.34 / 0.82) |
| Add a builtin node category to the picker | `PICKER_CATEGORIES` in `NodePickerModal.tsx` |
| Server calls / DTOs | `editor/src/api/client.ts` ([API client](../../../docs/editor/api-client.md)) |
