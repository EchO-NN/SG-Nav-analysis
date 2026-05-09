# SparseGNN-Nav Current Method Snapshot

This note synchronizes the repository with the current paper draft `main.pdf`
and the implementation guide `sparsegnn_nav_codex_iteration_master_plan.md`.

## Paper Story

SparseGNN-Nav treats SG-Nav's online graph as a learnable decision state rather
than a graph-to-text prompt. The first implemented system focuses on frontier
value prediction over a Sparse Goal-Conditioned Decision Graph:

- object nodes from SG-Nav object memory
- room nodes from the fixed/observed room set
- frontier nodes from clustered FBE frontier cells
- one goal node from the language goal
- sparse typed edges among object, room, frontier, and goal nodes

The intended final story also includes candidate-goal credibility, uncertainty
fallback, and autonomous replay learning. Those are second-wave modules unless
explicitly marked implemented below.

## Implemented

- Raw SG-Nav replay logging under `--collect_gnn_data`.
- `raw_sgnav_step_v1` snapshots with metadata, goal, agent, frontier, teacher,
  scenegraph summary, maps, optional oracle field, and debug metadata.
- Teacher labels for sanity checks.
- Saved online-oracle label consumption via `simulator_geodesic` /
  `oracle_online_saved`.
- Final-map / hindsight goal label entry points when a goal map coordinate is
  available.
- Hindsight all-objects pseudo-goal relabeling from confident object centers.
- Raw-to-graph conversion into `gnn_graph_step_v1` samples.
- Custom PyTorch heterogeneous GNN frontier scorer.
- Offline training with soft CE, optional teacher KL, and optional ranking loss.
- Offline baselines: random, distance-only, teacher, and GNN.
- Online `--use_gnn_nav` frontier scoring path with distance fallback.
- Raw and graph visualization helpers.

## Not Yet Implemented

- Fully validated Habitat map-to-world geodesic oracle.
- Candidate-goal node features, candidate edges, candidate credibility head, and
  BCE loss.
- Uncertainty-triggered online VLM/LLM fallback integration.
- Sparse runtime mode that disables online LLM/VLM edge reasoning.
- Runtime call/timing counters for all modules in the paper tables.
- Periodic continual fine-tuning from fallback-assisted successes.
- HM3D/RoboTHOR data collection and evaluation scripts.

## Claim Boundary

Supported wording:

- online LLM-free graph reasoning for the GNN frontier scoring branch
- open-vocabulary goal-conditioned graph policy through frozen text embeddings
- human-annotation-free labels when using simulator/final-map/hindsight labels
- teacher scores are available only as debugging/distillation signals

Do not claim yet:

- pure zero-shot learned policy
- full replacement of all perception or VLM calls
- validated continual self-improvement
- candidate-goal credibility improvements
- final SR/SPL/latency gains

