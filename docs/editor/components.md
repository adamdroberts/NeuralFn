# React Components

## GraphCanvas

**Default export** from the canvas module.

The main graph editing surface, built on React Flow (`@xyflow/react`). Renders the active subgraph from `useGraphStore` and handles:

- Custom node types (via `NeuronNode`) and edge types (via `InteractiveEdge`).
- Drag-and-drop node insertion from the library panel.
- Node selection, multi-select, and deletion.
- Edge creation via port-to-port dragging.
- Panning and zooming.
- Wires node/edge change events back to the Zustand store actions (`applyActiveNodeChanges`, `applyActiveEdgeChanges`, `connectActiveGraph`).

## NeuronNode

**Named export** from the canvas module.

Custom React Flow node renderer. Displays:

- The neuron's name and kind icon (via `NeuronIcon`).
- Input and output port handles with labels.
- Visual indicators for subgraph nodes (expandable) and variant-linked nodes.
- Selection highlight and input/output designation badges.

## InteractiveEdge

**Default export** from the edge module.

Custom React Flow edge renderer. Shows:

- The edge path between source and target ports.
- Weight and bias values as inline labels (editable on click).
- Telemetry coloring when `edgeTelemetry` data is available (activation magnitude mapped to color intensity).

## CodePanel

**Default export.**

A Monaco Editor panel for writing and editing custom neuron source code. Appears when a custom neuron node is selected. Features:

- Python syntax highlighting and IntelliSense.
- Live updates pushed to the selected node's `neuron_def.source_code` via `updateNodeData`.

## LibraryPanel

**Default export.**

The builtin neuron palette displayed alongside the canvas. Lists all available builtin neuron definitions grouped by category. Supports:

- Drag-and-drop onto the canvas to insert a builtin node.
- Click-to-insert at the preferred insert position.
- Search/filter by neuron name.

## Toolbar

**Default export.**

Top-of-canvas toolbar providing:

- Template preset selector (GPT, NanoGPT, Llama, MoE, etc.) with apply action.
- Node insertion buttons (custom, subgraph, variant).
- Graph-level actions: save, execute, trace, clear.
- Breadcrumb navigation bar for subgraph depth.
- Training method and runtime selectors.

## TrainingPanel

**Default export.**

Right-side panel for training controls and visualization:

- Training configuration form (method, epochs, learning rate, batch size, etc.).
- Start/stop training buttons.
- Live loss chart (Recharts line chart fed from `lossHistory`).
- Torch trace display showing per-layer timing and shape statistics.
- Error display for failed runs.

## PortConfig

**Default export.**

Port editor panel for the selected node. Allows adding, removing, and renaming input and output ports. Changes are applied via `updateNodeData`.

## NeuronIcon

**Default export.**

SVG icon component that renders a distinct icon based on the neuron's `kind` field (e.g. builtin, custom, subgraph, input, output, module).

## DatasetSourcePanel

**Default export.**

Panel for configuring dataset sources on the graph:

- Dataset selection from the project's granted datasets.
- Sequence length and column configuration.
- Load/refresh actions that wire dataset tokens into the graph's input nodes.

## AppShell

**Default export** from `shell/`.

Layout wrapper component that provides:

- Top navigation bar with project and session selectors.
- Sidebar navigation links (Editor, Datasets, Runs, Analytics, Admin).
- React Router `<Outlet />` for rendering the active page.
- User menu with logout.
