#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUNNER="${RUNNER:-./run_sg_nav.sh}"
RUN_COLLECT="${RUN_COLLECT:-0}"

LABEL_MODE="${LABEL_MODE:-final_map_hindsight_strict}"
case "$LABEL_MODE" in
  final_map_hindsight_strict|hindsight_all_objects_strict)
    ;;
  *)
    echo "LABEL_MODE must be final_map_hindsight_strict or hindsight_all_objects_strict, got: $LABEL_MODE" >&2
    exit 2
    ;;
esac

DATA_TAG="${DATA_TAG:-train50_hindsight_strict}"
CONFIG="${CONFIG:-configs/challenge_objectnav2021.local.rgbd.gnn_unseen50_s0_e2.yaml}"
NUM_EPISODES="${NUM_EPISODES:-100}"
SPLIT_L="${SPLIT_L:-0}"
SPLIT_R="${SPLIT_R:-50}"
SHOW_NAV_STEPS="${SHOW_NAV_STEPS:-0}"
NAV_STEP_LOG_INTERVAL="${NAV_STEP_LOG_INTERVAL:-1}"
SHUFFLE_SCENES="${SHUFFLE_SCENES:-0}"

RAW_ROOT="${RAW_ROOT:-data/gnn_raw/${DATA_TAG}}"
SUMMARY_ROOT="${SUMMARY_ROOT:-data/gnn_episode_summary/${DATA_TAG}}"
LABELED_ROOT="${LABELED_ROOT:-data/gnn_labeled/${DATA_TAG}_${LABEL_MODE}}"
GRAPH_ROOT="${GRAPH_ROOT:-data/gnn_graph/${DATA_TAG}_${LABEL_MODE}}"
REPORT_DIR="${REPORT_DIR:-data/reports/${DATA_TAG}_${LABEL_MODE}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/frontier_gnn_${DATA_TAG}_${LABEL_MODE}}"

make_abs_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    echo "$path"
  else
    echo "$ROOT_DIR/$path"
  fi
}

REPORT_DIR="$(make_abs_path "$REPORT_DIR")"

SPLITS="${SPLITS:-train val}"
TRAIN_RAW_DIR="${TRAIN_RAW_DIR:-}"
VAL_RAW_DIR="${VAL_RAW_DIR:-}"
TRAIN_SUMMARY_DIR="${TRAIN_SUMMARY_DIR:-}"
VAL_SUMMARY_DIR="${VAL_SUMMARY_DIR:-}"

MIN_LABEL_FRONTIERS="${MIN_LABEL_FRONTIERS:-2}"
TAU="${TAU:-2.0}"
LAMBDA_GOAL="${LAMBDA_GOAL:-1.0}"
PSEUDO_GOAL_MIN_CONFIDENCE="${PSEUDO_GOAL_MIN_CONFIDENCE:-0.7}"
PSEUDO_GOAL_MIN_OBSERVED_COUNT="${PSEUDO_GOAL_MIN_OBSERVED_COUNT:-3}"
PSEUDO_GOAL_MIN_LIFETIME_STEPS="${PSEUDO_GOAL_MIN_LIFETIME_STEPS:-5}"
PSEUDO_GOAL_MAX_PER_CATEGORY="${PSEUDO_GOAL_MAX_PER_CATEGORY:-5}"

MAX_FRONTIER_CLUSTERS="${MAX_FRONTIER_CLUSTERS:-32}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-3}"
TEXT_CACHE="${TEXT_CACHE:-data/gnn/text_embeddings.pt}"
TEXT_DIM="${TEXT_DIM:-384}"
TEXT_BACKEND="${TEXT_BACKEND:-auto}"

EPOCHS="${EPOCHS:-20}"
LR="${LR:-1e-4}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_LAYERS="${NUM_LAYERS:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"

mkdir -p "$REPORT_DIR"

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

has_pt_files() {
  local dir="$1"
  [[ -d "$dir" ]] && find "$dir" -name '*.pt' -print -quit | grep -q .
}

raw_dir_for_split() {
  local split="$1"
  local override=""
  case "$split" in
    train) override="$TRAIN_RAW_DIR" ;;
    val) override="$VAL_RAW_DIR" ;;
  esac
  if [[ -n "$override" ]]; then
    echo "$override"
    return
  fi
  if has_pt_files "$RAW_ROOT/$split"; then
    echo "$RAW_ROOT/$split"
    return
  fi
  if [[ "$split" == "train" ]] && has_pt_files "$RAW_ROOT"; then
    echo "$RAW_ROOT"
    return
  fi
  echo "$RAW_ROOT/$split"
}

summary_dir_for_split() {
  local split="$1"
  local override=""
  case "$split" in
    train) override="$TRAIN_SUMMARY_DIR" ;;
    val) override="$VAL_SUMMARY_DIR" ;;
  esac
  if [[ -n "$override" ]]; then
    echo "$override"
    return
  fi
  if has_pt_files "$SUMMARY_ROOT/$split"; then
    echo "$SUMMARY_ROOT/$split"
    return
  fi
  if [[ "$split" == "train" ]] && has_pt_files "$SUMMARY_ROOT"; then
    echo "$SUMMARY_ROOT"
    return
  fi
  echo "$SUMMARY_ROOT/$split"
}

if [[ "$RUN_COLLECT" == "1" ]]; then
  collect_cmd=(
    "$RUNNER"
    --config "$CONFIG" \
    --collect_gnn_data \
    --collect_episode_summary \
    --gnn_raw_log_dir "$RAW_ROOT/train" \
    --gnn_episode_summary_dir "$SUMMARY_ROOT/train" \
    --gnn_collect_every_k_fbe 1 \
    --gnn_save_maps \
    --gnn_data_tag "$DATA_TAG" \
    --gnn_strict_logging \
    --gnn_logging_error_path "$REPORT_DIR/gnn_logging_errors.jsonl" \
    --num_episodes "$NUM_EPISODES" \
    --split_l "$SPLIT_L" \
    --split_r "$SPLIT_R"
  )
  if [[ "$SHUFFLE_SCENES" == "1" ]]; then
    collect_cmd+=(--shuffle_scenes)
  fi
  if [[ "$SHOW_NAV_STEPS" == "1" ]]; then
    collect_cmd+=(--show_nav_steps --nav_step_log_interval "$NAV_STEP_LOG_INTERVAL")
  fi
  run_cmd "${collect_cmd[@]}"
fi

converted_splits=()
for split in $SPLITS; do
  raw_dir="$(raw_dir_for_split "$split")"
  summary_dir="$(summary_dir_for_split "$split")"
  labeled_dir="$LABELED_ROOT/$split"
  graph_dir="$GRAPH_ROOT/$split"

  if ! has_pt_files "$raw_dir"; then
    if [[ "$split" == "train" ]]; then
      echo "No raw .pt files found for required train split: $raw_dir" >&2
      exit 3
    fi
    echo "Skipping split '$split': no raw .pt files in $raw_dir"
    continue
  fi
  if ! has_pt_files "$summary_dir"; then
    if [[ "$split" == "train" ]]; then
      echo "No episode summary .pt files found for required train split: $summary_dir" >&2
      exit 4
    fi
    echo "Skipping split '$split': no episode summary .pt files in $summary_dir"
    continue
  fi

  run_cmd "$PYTHON_BIN" -m gnn_data.inspect_episode_summary \
    --dir "$summary_dir" \
    --strict \
    --max_print 1

  label_cmd=(
    "$PYTHON_BIN" -m gnn_data.label_frontiers
    --input_dir "$raw_dir"
    --output_dir "$labeled_dir"
    --episode_summary_dir "$summary_dir"
    --label_mode "$LABEL_MODE"
    --strict_labels
    --forbid_teacher_fallback
    --forbid_approx_fallback
    --require_hindsight_or_oracle
    --forbid_sim_gt_debug
    --require_final_free_map
    --min_label_frontiers "$MIN_LABEL_FRONTIERS"
    --skip_unlabeled
    --tau "$TAU"
    --lambda_goal "$LAMBDA_GOAL"
    --write_label_report "$REPORT_DIR/${split}_label_report.json"
  )
  if [[ "$LABEL_MODE" == "hindsight_all_objects_strict" ]]; then
    label_cmd+=(
      --pseudo_goal_min_confidence "$PSEUDO_GOAL_MIN_CONFIDENCE"
      --pseudo_goal_min_observed_count "$PSEUDO_GOAL_MIN_OBSERVED_COUNT"
      --pseudo_goal_min_lifetime_steps "$PSEUDO_GOAL_MIN_LIFETIME_STEPS"
      --pseudo_goal_max_per_category "$PSEUDO_GOAL_MAX_PER_CATEGORY"
      --pseudo_goal_exclude_unknown
      --pseudo_goal_exclude_rejected_candidates
      --pseudo_goal_balance_categories
    )
  fi
  run_cmd "${label_cmd[@]}"

  run_cmd "$PYTHON_BIN" -m gnn_train.convert_raw_to_graph \
    --input_dir "$labeled_dir" \
    --output_dir "$graph_dir" \
    --max_frontier_clusters "$MAX_FRONTIER_CLUSTERS" \
    --min_cluster_size "$MIN_CLUSTER_SIZE" \
    --text_cache "$TEXT_CACHE" \
    --text_dim "$TEXT_DIM" \
    --text_backend "$TEXT_BACKEND" \
    --strict_frontier_clusters \
    --strict_label_aggregation \
    --forbid_distance_label_fallback \
    --write_conversion_report "$REPORT_DIR/${split}_conversion_report.json"

  converted_splits+=("$split")
done

if [[ ! -d "$GRAPH_ROOT/train" ]]; then
  echo "Missing converted train graph directory: $GRAPH_ROOT/train" >&2
  exit 5
fi

train_cmd=(
  "$PYTHON_BIN" -m gnn_train.train_frontier_gnn
  --train_dir "$GRAPH_ROOT/train"
  --output_dir "$CHECKPOINT_DIR"
  --epochs "$EPOCHS"
  --lr "$LR"
  --hidden_dim "$HIDDEN_DIM"
  --num_layers "$NUM_LAYERS"
  --lambda_teacher 0.0
  --lambda_rank 0.0
  --batch_size "$BATCH_SIZE"
)
if [[ -d "$GRAPH_ROOT/val" ]]; then
  train_cmd+=(--val_dir "$GRAPH_ROOT/val")
fi
run_cmd "${train_cmd[@]}"

echo "Strict hindsight pipeline complete."
echo "label_mode=$LABEL_MODE"
echo "graph_root=$GRAPH_ROOT"
echo "checkpoint_dir=$CHECKPOINT_DIR"
echo "reports=$REPORT_DIR"
