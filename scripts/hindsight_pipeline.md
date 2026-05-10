# Hindsight SparseGNN-Nav Pipeline

This file is the formal command checklist for raw SG-Nav replay logging,
episode hindsight summaries, strict relabeling, graph conversion, offline
training, and online GNN navigation smoke tests.

## 1. Collect raw step replay + episode summaries

```bash
python SG_Nav.py \
  --collect_gnn_data \
  --collect_episode_summary \
  --gnn_raw_log_dir data/gnn_raw/train50 \
  --gnn_episode_summary_dir data/gnn_episode_summary/train50 \
  --gnn_data_tag train50_hindsight \
  --gnn_save_maps \
  --gnn_strict_logging \
  --gnn_logging_error_path data/logs/gnn_logging_errors_train50.jsonl
```

After completion:

```bash
find data/gnn_raw/train50 -name "*.pt" | wc -l
find data/gnn_episode_summary/train50 -name "*.pt" | wc -l
python -m gnn_data.inspect_episode_summary --dir data/gnn_episode_summary/train50
```

Strict schema check:

```bash
python -m gnn_data.inspect_episode_summary --dir data/gnn_episode_summary/train50 --strict
```

Failure conditions:

- raw step count is `0`
- episode summary count is `0`
- `inspect_episode_summary` reports missing `final_maps.free_map`
- `inspect_episode_summary` reports missing `discovered_objects`

## 2. Strict final-map target-goal hindsight labels

```bash
python -m gnn_data.label_frontiers \
  --input_dir data/gnn_raw/train50 \
  --output_dir data/gnn_labeled/train50_final_map \
  --episode_summary_dir data/gnn_episode_summary/train50 \
  --label_mode final_map_hindsight_strict \
  --strict_labels \
  --forbid_teacher_fallback \
  --forbid_approx_fallback \
  --require_hindsight_or_oracle \
  --forbid_sim_gt_debug \
  --require_final_free_map \
  --min_label_frontiers 2 \
  --skip_unlabeled \
  --write_label_report data/reports/train50_final_map_label_report.json
```

Expected report fields:

- `label_type_counts`
- `skip_reason_counts`
- `target_goal_source_counts`
- `final_map_source_counts`
- `forbid_sim_gt_debug: true`
- `require_final_free_map: true`

Failure conditions:

- `label_type_counts.teacher_debug_fallback > 0`
- `label_type_counts.oracle_approx_map_fallback > 0`
- `target_goal_source_counts.sim_gt_debug > 0`
- `final_map_source_counts.final_full_map_debug > 0`
- `num_labeled_samples == 0`

## 3. Strict hindsight-all-objects pseudo-goal labels

```bash
python -m gnn_data.label_frontiers \
  --input_dir data/gnn_raw/train50 \
  --output_dir data/gnn_labeled/train50_all_objects \
  --episode_summary_dir data/gnn_episode_summary/train50 \
  --label_mode hindsight_all_objects_strict \
  --strict_labels \
  --forbid_teacher_fallback \
  --forbid_approx_fallback \
  --require_hindsight_or_oracle \
  --forbid_sim_gt_debug \
  --require_final_free_map \
  --pseudo_goal_min_confidence 0.7 \
  --pseudo_goal_min_observed_count 3 \
  --pseudo_goal_min_lifetime_steps 5 \
  --pseudo_goal_max_per_category 5 \
  --pseudo_goal_exclude_unknown \
  --pseudo_goal_exclude_rejected_candidates \
  --pseudo_goal_balance_categories \
  --min_label_frontiers 2 \
  --skip_unlabeled \
  --write_label_report data/reports/train50_all_objects_label_report.json
```

Failure conditions are the same as strict final-map labeling. The report should
also include non-empty `pseudo_goal_category_counts` unless no stable objects
were discovered in the collected episodes.

## 4. Convert labeled raw samples to graph samples

```bash
python -m gnn_train.convert_raw_to_graph \
  --input_dir data/gnn_labeled/train50_all_objects \
  --output_dir data/gnn_graph/train50_all_objects \
  --max_frontier_clusters 32 \
  --min_cluster_size 3 \
  --strict_frontier_clusters \
  --strict_label_aggregation \
  --forbid_distance_label_fallback \
  --write_conversion_report data/reports/train50_all_objects_conversion_report.json
```

The conversion report must satisfy:

```text
cluster_fallback_to_points_count == 0
cluster_label_missing_count == 0
distance_only_label_count == 0
num_converted_samples > 0
```

## 5. Train GNN

```bash
python -m gnn_train.train_frontier_gnn \
  --train_dir data/gnn_graph/train50_all_objects/train \
  --val_dir data/gnn_graph/train50_all_objects/val \
  --output_dir checkpoints/frontier_gnn_hindsight_all_objects_v1 \
  --epochs 20 \
  --lr 1e-4 \
  --hidden_dim 256 \
  --num_layers 3 \
  --lambda_teacher 0.0 \
  --lambda_rank 0.0 \
  --batch_size 1
```

Expected validation metrics:

```text
pred_cost_ratio < random_cost_ratio
pred_cost_ratio <= distance_cost_ratio, or explain why distance baseline is stronger
label_type_counts contains only strict hindsight labels
```

The trainer must print:

```text
pred_label_top1
pred_teacher_agreement
teacher_label_agreement
distance_cost_ratio
random_cost_ratio
```

## 6. Online GNN navigation smoke test

```bash
python SG_Nav.py \
  --use_gnn_nav \
  --gnn_ckpt checkpoints/frontier_gnn_hindsight_all_objects_v1/best.pt \
  --gnn_enable_fallback \
  --gnn_fallback_alpha 1.0 \
  --gnn_fallback_mode original_sgnav_score \
  --gnn_log_fallback \
  --debug_gnn \
  --collect_episode_summary \
  --gnn_episode_summary_dir data/gnn_episode_summary/online_smoke \
  --gnn_data_tag online_smoke
```

Expected debug output includes:

```text
[GNN] score_shape:
[GNN] selected:
[GNN] fallback_used:
```

Expected episode summary fields:

```text
fallback.num_fallback_calls
fallback.fallback_records
fallback.num_fallback_decisions
fallback.fallback_decision_records
```

Failure conditions:

- no `[GNN] score_shape` line appears when `--use_gnn_nav` is enabled
- frontier selection does not enter `fbe_gnn`
- `fallback_records` is absent when `--gnn_enable_fallback --gnn_log_fallback` is enabled
- original SG-Nav behavior changes when `--use_gnn_nav` is absent

## 7. Optional sparse-only low-cost mode

Use only after online GNN navigation works:

```bash
python SG_Nav.py \
  --use_gnn_nav \
  --gnn_ckpt checkpoints/frontier_gnn_hindsight_all_objects_v1/best.pt \
  --disable_llm_edges \
  --sparse_graph_only \
  --gnn_keyframe_update_k 5 \
  --debug_sgnav
```

Expected runtime counters include:

```text
online_llm_calls
online_vlm_calls
scenegraph_update_time
edge_update_time
sparse_graph_build_time
gnn_forward_time
fallback_score_time
```

## 8. Later candidate-goal GNN head

Do not start this before the frontier GNN and fallback path are working online.
The later extension should add candidate-goal nodes, candidate edges,
candidate logits, candidate labels, BCE loss, and an online candidate
accept/reject policy.
