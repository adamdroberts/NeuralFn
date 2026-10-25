# NeuralFn Editor

The NeuralFn editor is a **React** single-page application for building, wiring, and training neural-network graphs in the browser.

## Tech Stack

| Technology | Purpose |
|------------|---------|
| React 18 | UI framework |
| Vite 5 | Dev server and bundler |
| TypeScript | Type safety |
| React Flow (`@xyflow/react`) | Graph canvas with drag-and-drop nodes and edges |
| WebGL + Canvas2D | Cyber-mode background shader, wireframe cages, and 60 FPS particle layers (`WebGLCyberScene`, `ParticleGraphCanvas`, `VolumetricNodeCore`) |
| Zustand | Lightweight state management |
| React Router | Client-side routing |
| Monaco Editor | Code editing for custom neurons |
| Recharts | Loss curves and analytics charts |

## Directory Layout

```
editor/src/
  api/          API client and DTO types
  store/        Zustand store, graph utilities, selectors
  components/   Reusable UI components (canvas, panels, toolbar, modals)
  components/shell/  Layout wrapper (AppShell) and SettingsModal mount
  pages/        Route-level page components
  routes/       Routing config, AppState context, session sync
```

Two Zustand stores live in `store/`: `graphStore.ts` (the server-synced graph document) and `settingsStore.ts` (client-only feature flags, cyber-mode visuals, View Full Graph, and panel-dialog state, persisted in `localStorage`). Graph helpers, including the View Full Graph flattener, are in `graphUtils.ts`.

## Sub-pages

- [API Client](api-client.md) -- REST client, DTO interfaces, SSE streaming.
- [Store](store.md) -- Zustand graph store and settings store: state shape, actions, selectors, persistence.
- [Graph Utilities](graph-utils.md) -- graph normalization, variant resolution, Flow conversion, full-graph expansion.
- [Components](components.md) -- canvases, node renderers, toolbar, dialogs, modals, archetype table.
- [Cyber Mode, Full Graph View, and Panel Dialogs](cyber-mode.md) -- how the 3D cyber canvas, settings, View Full Graph, the expand/collapse animation, the node picker, and the panel dialogs work together (with diagrams).
- [Pages and Routing](pages.md) -- route structure, app state context, session sync.
- [Native CUDA Training](native-training.md) -- runtime selection, compatibility preflight, artifacts, and current fail-closed limits.
