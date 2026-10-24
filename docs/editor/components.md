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
- Automatically switches to `ParticleGraphCanvas` when the `particleGraphEditorEnabled` feature flag is enabled in `useSettingsStore`.

## ParticleGraphCanvas & WebGLCyberScene

**Default exports** from `ParticleGraphCanvas.tsx` and `WebGLCyberScene.tsx`.

A 60 FPS 3D volumetric WebGL and particle cybernetic graph editor, toggled via the `particleGraphEditorEnabled` feature flag in `useSettingsStore`. Combines a hardware-accelerated WebGL (`webgl2` / `webgl`) GLSL shader & 3D wireframe layer (`WebGLCyberScene`), a 60 FPS HTML5 Canvas2D particle simulation, and interactive React Flow editing:

- **WebGL 3D Shader & Wireframe Layer (`WebGLCyberScene`)**: Uses a single shared WebGL context synchronized with the React Flow camera to render a 3D perspective cybernetic grid floor and rotating/twisting 3D geodesic wireframe energy cages around every node.
- **60 FPS Multi-Layer Particle Engine**: Simulates ambient curl-noise field particles, semantic 3D orbital node swarms (`tri_prism`, `lissajous_wave`, `vortex_funnel`, `satellite_clusters`, `matrix_lattice`, `stabilizer_rings`, `cylinder_stack`, `radial_pulsar`, `harmonic_orbit`, `hex_shield`, `emitter_beam`), double-helix synaptic photon packets along cubic Bézier edges, and supernova burst/shockwave rings.
- **Smooth Vortex Morphing Transitions**: Re-targets orbital particles using damped spring physics when navigating subgraphs (`currentPath`), loading GPT templates, or switching projection modes.
- **Projection Modes**: Supports both `hybrid_hud` (multi-plane CSS 3D extruded node chassis via `ParticleNeuronNode` + orbital particle halos) and `particle_nebula` (floating 3D rotating & twisting sculptures with laser callout banners).
- **View Full Graph Mode & 3D Core Clone-and-Morph Subgraph Transitions (`SubgraphExpansionOverlay.tsx`, `ParticleNeuronNode.tsx`)**:
  - **View Full Graph Toggle (`viewFullGraph`)**: Recursively expands all nested subgraphs into a single unified full-architecture graph view via `buildFullExpandedGraph`, bridging external ports directly to internal components and rendering interactive nested subgraph zone containers (`FullGraphZonesLayer`) with `⌖ Fit` and `⤢ Isolate` actions.
  - **Radial Burst Emission, Simultaneous 3D Vector Wireframe Morphing & Outer-Edge Box/Label Growth (`useSubgraphExpansion`, `SemanticNodeGlyph`, `VolumetricNodeCore`, `ParticleNeuronNode`)**: When not in Full Graph mode, clicking on any subgraph node detaches the node's title/name plate (`titleDetached`) and executes a `500ms` 3-stage transition:
    1. **Radial Burst Emission Around Parent**: Clones of the parent node's central animated 3D core (`SemanticNodeGlyph` / `VolumetricNodeCore`) are pulled out into a circular halo ring all around the parent node (`coreMorphProgress = 0`, `shellExpandProgress = 0`).
    2. **Simultaneous Transition & True 3D Vector Wireframe Metamorphosis**: All emitted 3D cores glide into their target locations at once while `VolumetricNodeCore` interpolates its 3D rotating wireframe vertices and edges (`WireframeSegment3D[]`) in 3D `(x, y, z)` space from the parent's 3D wireframe into each child node's new 3D vector wireframe (`coreMorphProgress: 0 -> 1`).
    3. **Outer-Edge Box & Label Growth**: Once in place with their new 3D vector wireframes formed, each node's shaped outer box, ports, and labels grow outward from the outer edges of its central animated 3D core (`shellExpandProgress: 0 -> 1`). Clicking a parent/ancestor breadcrumb plays the exact reverse: all child nodes retract their labels and outer boxes into their central 3D cores, all bare 3D cores transition into a ring around the parent while morphing their 3D vector wireframes back into the parent's 3D wireframe, pull into the parent center, and the parent node's box and labels grow outward from its central core's outer edges.
- **Interactive HUD Chrome & Tactical Radar**: Includes top telemetry/mode controls (`FULL GRAPH`, `3D TILT`, `PULSE`, color matrix cycler) and a bottom-right circular radar sweep scanner with click-to-warp node blips.

## ParticleNeuronNode, VolumetricNodeCore & ParticleEdge

**Exports** from `ParticleNeuronNode.tsx`, `VolumetricNodeCore.tsx`, `nodeArchetypes.tsx`, and `ParticleEdge.tsx`.

- **14 Semantic Node Archetypes (`nodeArchetypes.ts`)**: Maps every node label and `NeuronDefData` to one of 14 distinct visual archetypes (`attention_prism`, `sequence_wave`, `latent_tesseract`, `moe_router`, `tensor_matrix`, `lora_adapter`, `norm_stabilizer`, `residual_combiner`, `memory_vault`, `objective_beacon`, `signal_modulator`, `citadel_subgraph`, `portal_input`, `portal_output`), each with unique polygon silhouettes, neon color pairs, and particle swarm geometries.
- **60 FPS 3D Volumetric Rotating & Twisting Cores (`VolumetricNodeCore.tsx`)**: Renders depth-sorted, normal-lit 3D meshes inside each node—including counter-rotating gimbal rings + octahedral crystals for Attention, twisting 3D double-helices for Mamba/SSM, 4D-to-3D stereographic rotating tesseracts for JEPA/Latent nodes, orbiting 3D expert satellites with laser tethers for MoE, torsional twisting 3D tensor cubes for Linear/MLP, counter-twisting hyperbolic hourglasses for LoRA, 3-axis gyroscopic armillary spheres for Norms, stacked counter-rotating octagonal platters for Embeddings/KV Memory, dual counter-rotating Merkaba tetrahedra for Losses/Objectives, and 3D extruded twisting ribbons for Activations.
- **Multi-Plane CSS 3D Extruded Chassis (`ParticleNeuronNode.tsx`)**: Uses `transform-style: preserve-3d` with 5 distinct `translateZ` depth planes (`-20px` to `+22px`) and interactive cursor-tilt parallax.

## NodePickerModal

**Default export** from `NodePickerModal.tsx`.

Categorized 3-column **Holographic Node Arsenal** modal launched via the **`✦ + Add Node`** button in `Toolbar.tsx` (replacing the inline 120-button toolbar wall). Groups all builtin neurons into 9 semantic subsystems with staggered 3D fan-out card entry animations (`cyber-node-card-enter`), live search filtering, quick-spawn buttons for structural constructs (`+ Custom Python Node`, `+ Scalar Override`, `+ Subgraph Citadel`), multi-drop mode, and a right-hand 3D Blueprint Inspector.

## SettingsModal

**Default export** from `SettingsModal.tsx`.

Global modal dialog for configuring platform feature flags and holographic visual engine settings (`particleGraphEditorEnabled`, `particleVisualMode`, `particleDensity`, `colorPalette`, `holoPerspective3D`, `motionIntensity`, `hudTelemetryOverlay`). Accessible from the `AppShell` header, `Toolbar`, and `CodePanel` active graph settings.

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

## EditorPage Dialog Panels (`LibraryPanel`, `TrainingPanel`, `CodePanel`, `ChatPanel`)

To maximize graph canvas visibility, `EditorPage.tsx` dedicates 100% of the stage below `Toolbar.tsx` to `GraphCanvas` and houses the left, bottom, and right panels inside on-demand modal dialogs (`activePanelDialog` in `useSettingsStore`), launched from floating edge buttons (left, bottom, right) or from the top `Toolbar`:

- **Left Dialog (`◈ Variants` — `LibraryPanel`)**: Displays saved variant families and swappable versions in a responsive multi-column grid, automatically closing when a variant is opened or inserted.
- **Bottom Dialog (`⚡ Training & Trace` — `TrainingPanel`)**: Houses all training configuration, fine-tuning controls, dataset/JSON inputs, Recharts loss curve, and Torch Trace inside a modal dialog while keeping `TrainingPanel` mounted so active training runs and background trace polling continue uninterrupted when the dialog is closed.
- **Right Dialog (`◧ Inspector` / `💬 Model Chat` — `CodePanel` & `ChatPanel`)**: Houses both the Node & Graph Inspector / Monaco Code Editor (`CodePanel`) and the interactive `ChatPanel` inside a tabbed modal dialog.

## Toolbar

**Default export.**

Top-of-canvas toolbar providing:

- Template preset selector (GPT, NanoGPT, Llama, MoE, etc.) with apply action.
- Node insertion buttons (custom, subgraph, variant).
- Graph-level actions: save, execute, trace, clear.
- Breadcrumb navigation bar for subgraph depth.
- Training method selection; runtime selection lives in CodePanel graph settings.

## TrainingPanel

**Default export.**

Right-side panel for training controls and visualization:

- Training configuration form (method, epochs, learning rate, batch size, etc.).
- Start/stop training buttons.
- Live loss chart (Recharts line chart fed from `lossHistory`).
- Torch trace display showing per-layer timing and shape statistics.
- Error display for failed runs.
- Native CUDA project-dataset selector and pre-start Native IR compatibility report with exact node paths.
- Native Stop is disabled because the current compiled trainer ABI does not expose cooperative cancellation.

## ChatPanel

**Default export.**

Interactive generation drawer for the current session graph. It accepts an
optional base checkpoint and adapter, generation length, temperature, and
top-k controls. The temperature field preserves literal zero and rejects
negative or non-finite input. At zero it shows a `strict` indicator, sends the
unchanged numeric value to the server, and displays the returned compute-policy
badge on the assistant message so users can distinguish strict deterministic
execution from ordinary optimized or `top_k=1` greedy selection.

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
