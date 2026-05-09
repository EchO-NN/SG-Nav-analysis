# SparseGNN-Nav Experiment Status

All paper tables must remain placeholders until filled from real logs or
confirmed CSV files in `paper_results/`.

## Required Offline Checks

- [ ] Raw sample inspection passes on real SG-Nav replay.
- [ ] Teacher-only sanity model overfits a tiny debug set.
- [ ] Simulator/final-map hindsight labels are visualized and plausible.
- [ ] Label entropy is not always uniform.
- [ ] Teacher-vs-hindsight agreement is measured.
- [ ] GNN top3 beats random under hindsight/oracle cost.
- [ ] GNN cost_ratio matches or beats distance-only baseline.

## Required Online Checks

- [ ] Original SG-Nav runs unchanged with no GNN flags.
- [ ] `--collect_gnn_data` does not change selected frontier.
- [ ] `--use_gnn_nav --gnn_ckpt none` runs with distance fallback.
- [ ] `--use_gnn_nav --gnn_ckpt <path>` loads checkpoint and scores frontiers.
- [ ] Runtime profile measures graph build and GNN inference latency.
- [ ] LLM/VLM call counts are logged before claiming call reduction.

## Paper Tables

- `paper_results/main_results.csv`: SR/SPL/SoftSPL by benchmark.
- `paper_results/ablation.csv`: ablation variants from the paper draft.
- `paper_results/runtime.csv`: module timing and large-model call counts.
- `paper_results/offline_metrics.csv`: offline ranking/cost metrics.
- `paper_results/fallback.csv`: fallback trigger and success statistics.

Values marked `--` are placeholders, not measurements.

