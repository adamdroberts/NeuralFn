# NeuralFn Editor

The NeuralFn editor is a **React** single-page application for building, wiring, and training neural-network graphs in the browser.

## Tech Stack

| Technology | Purpose |
|------------|---------|
| React 18 | UI framework |
| Vite 5 | Dev server and bundler |
| TypeScript | Type safety |
| React Flow (`@xyflow/react`) | Graph canvas with drag-and-drop nodes and edges |
| Zustand | Lightweight state management |
| React Router | Client-side routing |
| Monaco Editor | Code editing for custom neurons |
| Recharts | Loss curves and analytics charts |

## Directory Layout

```
editor/src/
  api/          API client and DTO types
  store/        Zustand store, graph utilities, selectors
  components/   Reusable UI components (canvas, panels, toolbar)
  pages/        Route-level page components
  routes/       App shell, routing config, session sync
  shell/        Layout wrapper (AppShell)
```

## Sub-pages

- [API Client](api-client.md) -- REST client, DTO interfaces, SSE streaming.
- [Store](store.md) -- Zustand graph store, state shape, actions, selectors.
- [Graph Utilities](graph-utils.md) -- graph normalization, variant resolution, Flow conversion.
- [Components](components.md) -- canvas, panels, toolbar, node renderers.
- [Pages and Routing](pages.md) -- route structure, app state context, session sync.
