# React Components

Component-level reference for `editor/src/components/`. For the workflow view of the cyber canvas, View Full Graph mode, the expansion animation, and the panel dialogs, read [Cyber Mode, Full Graph View, and Panel Dialogs](cyber-mode.md) first.

## GraphCanvas

**Default export** from the canvas module.

The main graph editing surface, built on React Flow (`@xyflow/react`). Renders the active subgraph from `useGraphStore` and handles:

- Custom node types (via `NeuronNode`) and edge types (via `InteractiveEdge`).
- Drag-and-drop node insertion from the library panel.
- Node selection, multi-select, and deletion.
- Edge creation via port-to-port dragging.
- Panning and zooming.
- Wires node/edge change events back to the Zustand store actions (`applyActiveNodeChanges`, `applyActiveEdgeChanges`, `connectActiveGraph`).
- `GraphCanvas` itself is only a switch: it renders `ParticleGraphCanvas` when `particleGraphEditorEnabled` is true in `useSettingsStore`, otherwise the internal `StandardGraphCanvas`.
- `StandardGraphCanvas` also hosts the shared full-graph and expansion machinery: it memoizes `buildFullExpandedGraph(rootGraph)` while `viewFullGraph` is on (forwarding only selection changes to the store and calling `fitView` on toggle), mounts `FullGraphZonesLayer` (`⌖ Fit` animates the viewport to a zone, `⤢ Isolate` turns the mode off and `setPath`s into the zone), and runs `useSubgraphExpansion` so a click or double-click on a subgraph node plays the expansion animation and a breadcrumb jump plays the collapse. See [cyber-mode.md](cyber-mode.md#subgraph-expand-and-collapse-animation).

## ParticleGraphCanvas & WebGLCyberScene

**Default exports** from `ParticleGraphCanvas.tsx` and `WebGLCyberScene.tsx`.

A 60 FPS 3D volumetric WebGL and particle cybernetic graph editor, toggled via the `particleGraphEditorEnabled` feature flag in `useSettingsStore`. Combines a hardware-accelerated WebGL (`webgl2` / `webgl`) GLSL shader & 3D wireframe layer (`WebGLCyberScene`), a 60 FPS HTML5 Canvas2D particle simulation, and interactive React Flow editing:

- **WebGL 3D Shader & Wireframe Layer (`WebGLCyberScene`)**: Uses a single shared WebGL context synchronized with the React Flow camera to render a 3D perspective cybernetic grid floor and rotating/twisting 3D geodesic wireframe energy cages around every node.
- **60 FPS Multi-Layer Particle Engine**: Simulates ambient curl-noise field particles, semantic 3D orbital node swarms (`tri_prism`, `lissajous_wave`, `vortex_funnel`, `satellite_clusters`, `matrix_lattice`, `stabilizer_rings`, `cylinder_stack`, `radial_pulsar`, `harmonic_orbit`, `hex_shield`, `emitter_beam`), double-helix synaptic photon packets along cubic Bézier edges, and supernova burst/shockwave rings.
- **Smooth Vortex Morphing Transitions**: Re-targets orbital particles using damped spring physics when navigating subgraphs (`currentPath`), loading GPT templates, or switching projection modes.
- **Projection Modes**: Supports both `hybrid_hud` (multi-plane CSS 3D extruded node chassis via `ParticleNeuronNode` + orbital particle halos) and `particle_nebula` (floating 3D rotating & twisting sculptures with laser callout banners).
- **View Full Graph Mode & 3D Core Clone-and-Morph Subgraph Transitions (`SubgraphExpansionOverlay.tsx`, `ParticleNeuronNode.tsx`)**:
  - **View Full Graph Toggle (`viewFullGraph`)**: Recursively expands all nested subgraphs into a single unified full-architecture graph view via `buildFullExpandedGraph`, bridging external ports directly to internal components and rendering interactive nested subgraph zone containers (`FullGraphZonesLayer`) with `⌖ Fit` and `⤢ Isolate` actions.
  - **Radial Burst Emission, Simultaneous 3D Vector Wireframe Morphing & Outer-Edge Box/Label Growth (`useSubgraphExpansion`, `SemanticNodeGlyph`, `VolumetricNodeCore`, `ParticleNeuronNode`)**: When not in Full Graph mode, clicking on any subgraph node detaches the node's title/name plate (`titleDetached`) and executes a `2000ms` (`2s`) 3-stage transition:
    1. **Radial Burst Emission Around Parent**: Clones of the parent node's central animated 3D core (`SemanticNodeGlyph` / `VolumetricNodeCore`) are pulled out into a circular halo ring all around the parent node (`coreMorphProgress = 0`, `shellExpandProgress = 0`).
    2. **Simultaneous Transition & True 3D Vector Wireframe Metamorphosis**: All emitted 3D cores glide into their target locations at once while `VolumetricNodeCore` interpolates its 3D rotating wireframe vertices and edges (`WireframeSegment3D[]`) in 3D `(x, y, z)` space from the parent's 3D wireframe into each child node's new 3D vector wireframe (`coreMorphProgress: 0 -> 1`).
    3. **Outer-Edge Box & Label Growth**: Once in place with their new 3D vector wireframes formed, each node's shaped outer box, ports, and labels grow outward from the outer edges of its central animated 3D core (`shellExpandProgress: 0 -> 1`). Clicking a parent/ancestor breadcrumb plays the reverse: `GraphCanvas` delegates cleanly to `ParticleGraphCanvas` or `StandardGraphCanvas` so only the active canvas drives the viewport, and `startSubgraphCollapse` tracks the live DOM bounding rect of the parent node's `<canvas>` core so all child nodes converge 100% inside the parent node's exact screen coordinates and core dimensions (`singularityCoreSize`). During morphing, all glowing circular halos, radial ambient fills, shimmer rings, shockwave bursts, and additive canvas blending are disabled so the child nodes keep their own colors, smoothly transition their 3D vector wireframes into the parent's 3D wireframe directly inside the parent node, and alpha-fade to nothing before the parent node's outer box and labels grow outward.
- **Interactive HUD Chrome & Tactical Radar**: Includes top telemetry/mode controls (`FULL GRAPH`, `3D TILT`, `PULSE`, color matrix cycler) and a bottom-right circular radar sweep scanner with click-to-warp node blips.

## ParticleNeuronNode, VolumetricNodeCore & ParticleEdge

**Exports** from `ParticleNeuronNode.tsx` (`ParticleNeuronNode`, `SemanticNodeGlyph`), `VolumetricNodeCore.tsx` (`VolumetricNodeCore`, `VolumetricNodeCoreProps`, `VolumetricMorphSource`), `nodeArchetypes.ts` (`NODE_ARCHETYPES`, `resolveNodeArchetype`, `NodeArchetypeId`, `ParticleSwarmGeometry`, `NodeArchetypeSpec`), and `ParticleEdge.tsx`.

- **14 Semantic Node Archetypes (`nodeArchetypes.ts`)**: Maps every node label and `NeuronDefData` to one of 14 distinct visual archetypes (`attention_prism`, `sequence_wave`, `latent_tesseract`, `moe_router`, `tensor_matrix`, `lora_adapter`, `norm_stabilizer`, `residual_combiner`, `memory_vault`, `objective_beacon`, `signal_modulator`, `citadel_subgraph`, `portal_input`, `portal_output`), each with unique polygon silhouettes, neon color pairs, and particle swarm geometries.
- **60 FPS 3D Volumetric Rotating & Twisting Cores (`VolumetricNodeCore.tsx`)**: Renders depth-sorted, normal-lit 3D meshes inside each node—including counter-rotating gimbal rings + octahedral crystals for Attention, twisting 3D double-helices for Mamba/SSM, 4D-to-3D stereographic rotating tesseracts for JEPA/Latent nodes, orbiting 3D expert satellites with laser tethers for MoE, torsional twisting 3D tensor cubes for Linear/MLP, counter-twisting hyperbolic hourglasses for LoRA, 3-axis gyroscopic armillary spheres for Norms, stacked counter-rotating octagonal platters for Embeddings/KV Memory, dual counter-rotating Merkaba tetrahedra for Losses/Objectives, and 3D extruded twisting ribbons for Activations.
- **Multi-Plane CSS 3D Extruded Chassis (`ParticleNeuronNode.tsx`)**: Uses `transform-style: preserve-3d` with 5 distinct `translateZ` depth planes (`-20px` to `+22px`) and interactive cursor-tilt parallax.


### Archetype table

`resolveNodeArchetype(label, neuronDef, isInput?, isOutput?)` returns one `NodeArchetypeSpec` (`id`, `categoryId`, `categoryLabel`, `badgeText`, `primary` / `secondary` colors with `[r, g, b]` pairs, `clipPath`, `borderRadius`, `swarmGeometry`, `minWidth`). Rules are evaluated in this order; the first match wins.

| Archetype | Matched when | Badge | Swarm geometry |
|-----------|--------------|-------|----------------|
| `citadel_subgraph` | `neuronDef.kind === "subgraph"` | `MACRO // CITADEL` | `hex_shield` |
| `portal_input` | `isInput`, name `input`, or key contains `dataset_source` / `rollout_source` / `data_source` | `SOURCE // EMITTER` | `emitter_beam` |
| `portal_output` | `isOutput`, name `output`, or key contains `lm_head` / `denoise_head` | `SINK // NEXUS` | `emitter_beam` |
| `sequence_wave` | `mamba`, `ssm`, `causal_chunk_state`, `ttt_`, `universal_transformer`, `act_halt`, `act_weighted` | `SSM // WAVEGUIDE` | `lissajous_wave` |
| `attention_prism` | `attention`, `sdpa`, `reshape_heads`, `merge_heads`, `repeat_kv`, `qk_gain` | `ATTN // PRISM` | `tri_prism` |
| `latent_tesseract` | `latent`, `jepa`, `semantic_projector`, `semantic_hasher`, `semantic_chunk`, `byte_patch`, `mask_scheduler`, `random_timesteps`, `muse_glimmer` | `LATENT // TESSERACT` | `vortex_funnel` |
| `moe_router` | `moe`, `router`, `topk_route`, `expert_`, `auxfree`, `broadcast_chunk_routes`, `broadcast_expert_routes` | `MOE // MULTIPLEX` | `satellite_clusters` |
| `lora_adapter` | `lora`, `nf4`, `randmap` | `ADAPTER // LOW-RANK` | `matrix_lattice` |
| `tensor_matrix` | `linear`, `mlp`, `swiglu`, `geglu`, `reglu`, `solu` | `TENSOR // LATTICE` | `matrix_lattice` |
| `residual_combiner` | `residual`, `manifold_hyper`, `pair_batch` | `STREAM // JUNCTION` | `stabilizer_rings` |
| `norm_stabilizer` | `norm`, `dyt`, `softcap`, `dropout`, `tensor_scale`, `loss_scale` | `NORM // GYRO-GATE` | `stabilizer_rings` |
| `memory_vault` | `embedding`, `rotary`, `kv_cache`, `kv_pca`, `kv_quant` | `MEM // VAULT-BANK` | `cylinder_stack` |
| `objective_beacon` | `loss`, `cross_entropy`, `dpo`, `ppo`, `kl_penalty`, `reward`, `value_head`, `policy_logits`, `gae`, `logp`, `reference_forward` | `OBJ // CRITIC-CORE` | `radial_pulsar` |
| `signal_modulator` | anything else | `ACT // WAVE-SHAPER` | `harmonic_orbit` |

"key" is the lower-cased concatenation of the node label, `neuronDef.name`, and `neuronDef.module_type`. `VolumetricNodeCore` accepts `archetypeId`, `neuronName`, `primaryColor`, `secondaryColor`, optional `size`, `speedMultiplier`, `selected`, and the morph pair `morphFrom` (`VolumetricMorphSource`) / `morphProgress` used by the expansion animation.

## SubgraphExpansionOverlay

**Named exports** from `SubgraphExpansionOverlay.tsx`: `useSubgraphExpansion`, `SubgraphExpansionOverlay`, `FullGraphZonesLayer`, and the state types `SubgraphExpansionAnimState`, `SiblingGhost`, `CollapsingChildGhost`, `CollapsingEdgeGhost`, `EmergingNodeTether`, `ScreenRect`.

- `useSubgraphExpansion({ containerRef, rootGraph, currentPath, activeNodes, activeEdges, viewFullGraph, openSubgraph, setViewport, getViewport, onViewportSync?, onSpawnBurst? })` returns `{ expansionState, isExpanding, expansionMode, triggerSubgraphExpansion }`. `triggerSubgraphExpansion(node)` returns `false` for non-subgraph nodes; otherwise it snapshots the current view, calls `openSubgraph(node.id)` immediately, and animates for 2000 ms. A `useLayoutEffect` watches `currentPath`: when it becomes a strict prefix of the previous snapshot's path (a breadcrumb jump) and `viewFullGraph` is off, it starts the collapse animation automatically, so callers never invoke collapse directly.
- `expansionState.animatedNodes` / `animatedEdges` are what the canvas feeds to `<ReactFlow>` while an animation runs; the other fields (`singularityX/Y`, `singularityCoreSize`, `titleAnchorX/Y`, `portalPulse`, ghost arrays, `progress`, `spawnedCount`, `totalCount`) drive the overlay.
- `SubgraphExpansionOverlay({ expansionState, cyberMode })` draws the non-node parts of the choreography: the detached title plate and portal at the parent position, sibling ghosts, collapsing child ghosts (rendered with `SemanticNodeGlyph`), fading edge ghosts, and emerging-node tethers. `cyberMode` picks neon or plain styling.
- `FullGraphZonesLayer({ zones, viewport, cyberMode?, onIsolateZone, onFitZone })` renders the `SubgraphZone` frames in flow space (translated and scaled by the viewport) with per-depth colours, the zone name, breadcrumb, `family@version` badge, node count, and the `⌖ Fit` / `⤢ Isolate` buttons. Renders nothing when `zones` is empty.

## NodePickerModal

**Default export** from `NodePickerModal.tsx`.

Categorized 3-column **Holographic Node Arsenal** modal launched via the **`✦ + Add Node`** button in `Toolbar.tsx` (replacing the inline 120-button toolbar wall). Exports `PICKER_CATEGORIES: CategoryMeta[]` (`id`, `title`, `subtitle`, `accent`, `archetypeId`) with nine entries: `attention`, `linear_mlp`, `moe`, `jepa_latent`, `embeddings_kv`, `norms_residuals`, `rlhf_losses`, `activations`, `terminals_custom`. Groups all builtin neurons from `useGraphStore.builtins` into those semantic subsystems with staggered 3D fan-out card entry animations (`cyber-node-card-enter`), live search filtering, quick-spawn buttons for structural constructs (`+ Custom Python Node`, `+ Scalar Override`, `+ Subgraph Citadel`), multi-drop mode, and a right-hand 3D Blueprint Inspector.

## SettingsModal

**Default export** from `SettingsModal.tsx`.

Global modal dialog for the persisted fields of `useSettingsStore`: the cyber-mode flag (`particleGraphEditorEnabled`), visual mode, palette cards built from `PARTICLE_PALETTES`, particle density, motion intensity, the `holoPerspective3D` and `hudTelemetryOverlay` checkboxes, and the `⊞ View Full Graph (Expand All Subgraphs)` checkbox, plus a reset button that calls `resetSettings()`. Selecting a visual mode while cyber mode is off also enables cyber mode. Opened with `setSettingsOpen(true)` from the `AppShell` header (`⚙ Settings`), the `Toolbar` `⚙` button, the cyber HUD, and `CodePanel`'s graph settings; `AppShell` mounts the modal once for the whole app. Field reference: [store.md](store.md#settings-store-usesettingsstore).

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

- **Left Dialog (`◈ Variants` — `LibraryPanel`)**: Displays saved variant families and swappable versions from `rootGraph.variant_library` in a responsive multi-column grid (`openVariant` to navigate into a version, `addVariantNode` to insert one). Accepts an optional `onDone` callback, which `EditorPage` uses to close the dialog after either action. Builtin neurons are no longer listed here; they live in `NodePickerModal`.
- **Bottom Dialog (`⚡ Training & Trace` — `TrainingPanel`)**: Houses all training configuration, fine-tuning controls, dataset/JSON inputs, Recharts loss curve, and Torch Trace inside a modal dialog while keeping `TrainingPanel` mounted so active training runs and background trace polling continue uninterrupted when the dialog is closed.
- **Right Dialog (`◧ Inspector` / `💬 Model Chat` — `CodePanel` & `ChatPanel`)**: Houses both the Node & Graph Inspector / Monaco Code Editor (`CodePanel`) and the interactive `ChatPanel` inside a tabbed modal dialog (`rightPanelTab`).

### CodePanel

**Default export.**

The inspector tab of the right dialog. With no node selected it shows the active graph's settings: name, training method (`surrogate` / `evolutionary` / `frozen` / `torch`), runtime (`scalar` / `torch` / `native-cuda`, with native precision options when `native-cuda` is chosen), the per-subgraph training-method table, and a feature-flag block with a cyber-mode toggle and the `⚙ Configure Holographic & Particle Settings...` button. Selecting native CUDA does not imply compatibility; the Training panel owns preflight. With a node selected it shows the node inspector: Monaco editor for custom neuron `source_code` (Python highlighting, live `updateNodeData` writes), `PortConfig`, variant controls, and `DatasetSourcePanel` for `dataset_source` nodes. It works on nested leaves selected in View Full Graph mode because `selectSelectedNode` resolves scoped ids.

## Toolbar

**Default export.**

Top-of-canvas toolbar. In cyber mode it takes the neon cockpit styling; the controls are the same in both modes:

- `✦ + Add Node` opens `NodePickerModal`; `+ Custom Node`, `+ Override`, `+ Subgraph` quick-spawn a custom Python node, a scalar override node, and an empty subgraph.
- Template preset `<select>` plus apply action (`load_gpt_template`). The option list is hardcoded here, so every new backend preset must be added as an `<option>`.
- Breadcrumb bar for subgraph depth. Clicking a parent crumb calls `setViewFullGraph(false)` then `setPath(crumb.path)`; the active canvas turns that path change into the collapse animation.
- Panel launchers that mirror the canvas-edge buttons: `◈ Variants` (left dialog), `⚡ Training` (bottom dialog), `◧ Inspector` and `💬 Chat` (right dialog tabs).
- `⊞ View Full Graph` / `⊞ Full Graph: ON` (`toggleViewFullGraph`), `✦ Cyber 3D` / `✦ Cyber 3D: ON` (`toggleParticleGraphEditor`), and `⚙` (`setSettingsOpen(true)`).
- Save and Load actions (`onSave` / `onLoad`) for the session graph. Execution, tracing, and training controls live in `TrainingPanel`; runtime selection lives in `CodePanel` graph settings.

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
- `⚙ Settings` button (`setSettingsOpen(true)`) and the single app-wide `<SettingsModal />` mount. When `particleGraphEditorEnabled` is on, the shell and header switch to the `cyber-cockpit-shell` / `cyber-header-deck` styling from `index.css`.
