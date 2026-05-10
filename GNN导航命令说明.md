# GNN 导航命令说明

本文档说明 SG-Nav 中新增的 GNN 导航相关命令分别做什么，以及推荐的使用顺序。

## 推荐主流程：raw -> label -> graph -> train

这一流程对应论文初稿和 `sparsegnn_nav_codex_iteration_master_plan.md` 的设计：先采集原始 SG-Nav frontier 决策快照，再用 oracle / hindsight 几何标签作为主监督。teacher 分数只用于 sanity check 或蒸馏实验。

论文同步文件：

```text
paper_notes/current_method.md
paper_notes/experiment_status.md
paper_results/*.csv
```

`paper_results` 里的 `--` 都是占位，不是实测结果。

### 1. 采集原始 SG-Nav replay

如果使用 `gnn_unseen50` 这类完整 split，初始化 dataset 时会读取每个场景的大量 episode。
例如 50 个场景、每个场景约 50000 条 episode 时，初始化会非常慢甚至像“卡住”。
正式采集 GNN 数据前，建议先离线采样一个小 split，让每个场景只保留少量 episode：

```bash
python scripts/sample_objectnav_split.py \
  --source_split gnn_unseen50 \
  --output_split gnn_unseen50_s0_e2 \
  --episodes_per_scene 2 \
  --strategy balanced_categories \
  --seed 0 \
  --write_config configs/challenge_objectnav2021.local.rgbd.gnn_unseen50_s0_e2.yaml
```

说明：

- 同一个 `--seed` 加同一份输入，每次采样和打乱结果都会一样，这是为了可复现。
- 想采集新的 episode 组合，就换 `--seed`，同时换 `--output_split` 和 `--gnn_data_tag`。
- `balanced_categories` 会尽量让同一场景内采到不同目标类别，避免连续 episode 都是同一个目标。
- `--episodes_per_scene` / `--shuffle_scenes` 只能控制运行时迭代顺序，不能减少 dataset 初始化时读取的 episode 数量；小 split 才能解决初始化过慢。

然后用生成的小 split 配置采集：

```bash
./run_sg_nav.sh \
  --config configs/challenge_objectnav2021.local.rgbd.gnn_unseen50_s0_e2.yaml \
  --collect_gnn_data \
  --gnn_raw_log_dir data/gnn_raw/mp3d/train \
  --gnn_collect_every_k_fbe 1 \
  --gnn_save_maps \
  --gnn_data_tag sgnav_teacher_s0_e2 \
  --gnn_strict_logging \
  --num_episodes 100 \
  --shuffle_scenes
```

可选在线 oracle 标注：

```bash
./run_sg_nav.sh \
  --collect_gnn_data \
  --gnn_raw_log_dir data/gnn_raw/mp3d/train \
  --gnn_save_maps \
  --gnn_compute_oracle_online
```

注意：`--gnn_compute_oracle_online` 的 map/world 坐标变换必须先用可视化检查，确认 oracle best frontier 合理后再用于正式训练。

### 2. 检查 raw 样本

```bash
python -m gnn_data.inspect_raw_sample \
  --path data/gnn_raw/mp3d/train/某个样本.pt

python -m gnn_data.inspect_episode_summary \
  --path data/gnn_episode_summary/mp3d/train/某个summary.pt \
  --strict
```

### 3. 生成标签

正式训练建议优先使用 episode summary + final-map hindsight 标注，不要把 teacher 分数当作主监督。
新流程是先采集 raw step 和 episode summary：

```bash
./run_sg_nav.sh \
  --config configs/challenge_objectnav2021.local.rgbd.gnn_unseen50_s0_e2.yaml \
  --collect_gnn_data \
  --collect_episode_summary \
  --gnn_raw_log_dir data/gnn_raw/mp3d/train50 \
  --gnn_episode_summary_dir data/gnn_episode_summary/mp3d/train50 \
  --gnn_collect_every_k_fbe 1 \
  --gnn_save_maps \
  --gnn_data_tag train50_sgnav \
  --gnn_strict_logging \
  --num_episodes 100 \
  --split_l 0 \
  --split_r 50 \
  --shuffle_scenes
```

然后用成功 episode 的 final map 和确认目标位置生成严格 hindsight 标签：

```bash
python -m gnn_data.label_frontiers \
  --input_dir data/gnn_raw/mp3d/train50 \
  --episode_summary_dir data/gnn_episode_summary/mp3d/train50 \
  --output_dir data/gnn_labeled_hindsight/mp3d/train50 \
  --label_mode final_map_hindsight_strict \
  --strict_labels \
  --forbid_teacher_fallback \
  --forbid_approx_fallback \
  --require_hindsight_or_oracle \
  --min_label_frontiers 2 \
  --skip_unlabeled \
  --tau 2.0 \
  --lambda_goal 1.0 \
  --write_label_report
```

检查 hindsight 标签可视化：

```bash
python -m gnn_data.visualize_hindsight_sample \
  --labeled_dir data/gnn_labeled_hindsight/mp3d/train50 \
  --save_dir data/debug_hindsight_vis/train50 \
  --num_samples 50
```

严格模式会跳过没有成功、没有确认目标位置、没有 final map、frontier 太少或距离全无效的样本。
`label_report.json` 里必须确认 `teacher_debug_fallback` 和 `oracle_approx_map_fallback` 为 0。

如果 raw 中已经保存在线 oracle：

```bash
python -m gnn_data.label_frontiers \
  --input_dir data/gnn_raw/mp3d/train \
  --output_dir data/gnn_labeled/mp3d/train \
  --label_mode simulator_geodesic
```

如果 raw 中包含已发现目标的地图坐标，例如 `goal_rc` / `found_goal_rc`：

```bash
python -m gnn_data.label_frontiers \
  --input_dir data/gnn_raw/mp3d/train \
  --output_dir data/gnn_labeled_hindsight/mp3d/train \
  --label_mode hindsight_goal
```

使用所有已发现物体做 pseudo-goal 数据扩增：

```bash
python -m gnn_data.label_frontiers \
  --input_dir data/gnn_raw/mp3d/train50 \
  --output_dir data/gnn_labeled_all_objects/mp3d/train50 \
  --episode_summary_dir data/gnn_episode_summary/mp3d/train50 \
  --label_mode hindsight_all_objects_strict \
  --strict_labels \
  --forbid_teacher_fallback \
  --forbid_approx_fallback \
  --require_hindsight_or_oracle \
  --pseudo_goal_min_confidence 0.7 \
  --pseudo_goal_min_observed_count 2 \
  --pseudo_goal_min_lifetime_steps 5 \
  --pseudo_goal_max_per_category 5 \
  --pseudo_goal_exclude_unknown \
  --pseudo_goal_exclude_rejected_candidates \
  --skip_unlabeled \
  --write_label_report
```

只做 teacher sanity check：

```bash
python -m gnn_data.label_frontiers \
  --input_dir data/gnn_raw/mp3d/train \
  --output_dir data/gnn_labeled_teacher_debug/mp3d/train \
  --label_mode teacher
```

混合模式会优先使用保存的 oracle，其次使用 hindsight goal 坐标，最后才退到 teacher/debug：

```bash
python -m gnn_data.label_frontiers \
  --input_dir data/gnn_raw/mp3d/train \
  --output_dir data/gnn_labeled_hybrid/mp3d/train \
  --label_mode hybrid
```

最终训练不要直接用上面的非严格 `hybrid`。如果只是检查 raw 是否具备 hindsight/oracle 标签，可以用严格混合模式：

```bash
python -m gnn_data.label_frontiers \
  --input_dir data/gnn_raw/mp3d/train50 \
  --output_dir data/gnn_labeled_strict_check/mp3d/train50 \
  --label_mode hybrid_hindsight_first_strict \
  --strict_labels \
  --skip_unlabeled \
  --write_label_report
```

### 4. 转成稀疏决策图

```bash
python -m gnn_train.convert_raw_to_graph \
  --input_dir data/gnn_labeled/mp3d/train \
  --output_dir data/gnn_graph/mp3d/train \
  --max_frontier_clusters 32 \
  --strict_frontier_clusters \
  --strict_label_aggregation \
  --forbid_distance_label_fallback \
  --write_conversion_report
```

检查转换报告：

```bash
python -m gnn_train.analyze_conversion_report \
  --path data/gnn_graph/mp3d/train/conversion_report.json \
  --strict
```

### 5. 训练 Frontier GNN

```bash
python -m gnn_train.train_frontier_gnn \
  --train_dir data/gnn_graph/mp3d/train \
  --val_dir data/gnn_graph/mp3d/val \
  --output_dir checkpoints/frontier_gnn_oracle_v1 \
  --epochs 20 \
  --lr 1e-4 \
  --lambda_teacher 0.0 \
  --lambda_rank 0.0
```

输出：

```text
checkpoints/frontier_gnn_oracle_v1/frontier_gnn.pt
checkpoints/frontier_gnn_oracle_v1/gnn_scorer.pt
```

### 6. 离线评估 baseline

```bash
python -m gnn_train.eval_offline \
  --data_dir data/gnn_graph/mp3d/val \
  --ckpt checkpoints/frontier_gnn_oracle_v1/frontier_gnn.pt
```

会同时报告 random、distance、teacher 和 GNN 的 top1、top3、chosen_cost、best_cost、cost_ratio。
重点看 `summary.pred_label_top1`、`summary.pred_label_top3`、`summary.cost_ratio`、
`summary.random_cost_ratio`、`summary.distance_cost_ratio`、`summary.teacher_cost_ratio`、
`summary.pred_teacher_agreement` 和 `summary.teacher_label_agreement`。

训练后也可以检查 `metrics.jsonl`：

```bash
python -m gnn_train.analyze_training_metrics \
  --path checkpoints/frontier_gnn_oracle_v1/metrics.jsonl \
  --split val
```

检查 scene-disjoint split：

```bash
python -m gnn_data.check_scene_splits \
  --train_dir data/gnn_graph/mp3d/train \
  --val_dir data/gnn_graph/mp3d/val \
  --test_dir data/gnn_graph/mp3d/test
```

### 7. 可视化检查

```bash
python -m gnn_data.visualize_raw_sample \
  --path data/gnn_raw/mp3d/train/某个样本.pt \
  --output data/gnn_vis/raw_sample.png
```

```bash
python -m gnn_data.visualize_hindsight_label \
  --labeled_dir data/gnn_labeled_hindsight/mp3d/train50 \
  --save_dir data/gnn_vis/hindsight_train50 \
  --num_samples 20
```

```bash
python -m gnn_train.visualize_graph_sample \
  --path data/gnn_graph/mp3d/train/某个样本.pt \
  --output data/gnn_vis/graph_sample.png \
  --ckpt checkpoints/frontier_gnn_oracle_v1/frontier_gnn.pt
```

## 兼容调试流程：直接保存 graph replay

```bash
./run_sg_nav.sh --gnn_log --gnn_log_dir data/gnn_replay/debug
```

这个命令运行原始 SG-Nav 导航逻辑，同时在每次 frontier 决策时额外保存一份已经构建好的稀疏图样本。它适合快速调试在线分支，但不是推荐的正式训练数据格式。

保存目录：

```text
data/gnn_replay/debug
```

用途：

- 保持原始 SG-Nav 行为基本不变。
- 记录 object、room、frontier、goal 节点组成的稀疏图。
- 为后续 GNN 训练准备原始 replay 数据。

## 2. 测试 GNN 在线分支

```bash
./run_sg_nav.sh --use_gnn_nav --gnn_ckpt none --debug_gnn
```

这个命令启用新的 GNN frontier 选择分支，但不加载训练好的模型。

因为指定了：

```bash
--gnn_ckpt none
```

所以系统会使用距离 fallback 分数，也就是用 frontier 的距离启发式分数代替真正的 GNN 输出。

用途：

- 测试 `--use_gnn_nav` 分支能不能完整跑通。
- 不需要提前训练 checkpoint。
- `--debug_gnn` 会打印 frontier cluster 数量、分数和最终选择结果。

## 3. 给采集样本添加训练标签

```bash
python -m gnn_nav.label_frontiers \
  --input_dir data/gnn_replay/debug \
  --output_dir data/gnn_replay_labeled/debug \
  --label_mode teacher
```

这个命令会读取已经保存的 raw graph 样本，并给每个样本添加 frontier 训练标签。

输入目录：

```text
data/gnn_replay/debug
```

输出目录：

```text
data/gnn_replay_labeled/debug
```

`--label_mode teacher` 表示使用原始 SG-Nav 的 frontier 分数作为 teacher label。

输出样本中会新增：

```text
labels/frontier_cost
labels/frontier_y_soft
labels/frontier_best_idx
```

## 4. 训练 GNN Frontier Scorer

```bash
python -m gnn_nav.train_gnn \
  --train_dir data/gnn_replay_labeled/debug \
  --output_dir checkpoints/gnn_nav/frontier_v1
```

这个命令会用标注好的 replay 数据训练 GNN frontier scorer。

训练完成后会生成：

```text
checkpoints/gnn_nav/frontier_v1/gnn_scorer.pt
```

这个文件就是在线导航时可以加载的 GNN checkpoint。

## 5. 使用训练好的 GNN 在线导航

```bash
./run_sg_nav.sh \
  --use_gnn_nav \
  --gnn_ckpt checkpoints/gnn_nav/frontier_v1/gnn_scorer.pt \
  --debug_gnn
```

这个命令会真正使用训练好的 GNN 模型替代原来的 frontier reasoning 分数。

用途：

- 在线构建稀疏决策图。
- 加载训练好的 GNN checkpoint。
- 输出每个 frontier 的 GNN 分数。
- 根据 GNN 分数选择下一个 frontier。

带不确定性 fallback 的在线测试：

```bash
./run_sg_nav.sh \
  --use_gnn_nav \
  --gnn_ckpt checkpoints/gnn_nav/frontier_v1/gnn_scorer.pt \
  --gnn_use_fallback \
  --gnn_fallback_mode sgnav_score \
  --gnn_fallback_alpha 1.0 \
  --gnn_fallback_max_prob_threshold 0.45 \
  --gnn_fallback_margin_threshold 0.10 \
  --gnn_fallback_entropy_threshold 1.50 \
  --gnn_fallback_min_object_nodes 1 \
  --gnn_log_fallback \
  --debug_gnn
```

`--gnn_log_fallback` 会把 fallback 触发原因、GNN 分数、fallback 分数和最终选择写入 replay 样本 debug 字段，并汇总到 episode summary 的 `fallback.fallback_records`。
如果需要低延迟稀疏模式，可显式加 `--disable_llm_edges` 或 `--sparse_graph_only`，此时 scenegraph 不再调用在线 LLM/VLM 关系边生成，object memory 和地图仍会更新。
`--sparse_graph_only --gnn_keyframe_update_k 5` 会把昂贵的目标检测/物体记忆更新降到每 5 个导航步一次；疑似目标或卡住时会强制更新。

分析 fallback 触发率：

```bash
python -m gnn_nav.analyze_fallback_log \
  --dir data/gnn_episode_summary/mp3d/train50
```

## 推荐流程

按顺序执行：

```bash
./run_sg_nav.sh --gnn_log --gnn_log_dir data/gnn_replay/debug
```

```bash
python -m gnn_nav.label_frontiers \
  --input_dir data/gnn_replay/debug \
  --output_dir data/gnn_replay_labeled/debug \
  --label_mode teacher
```

```bash
python -m gnn_nav.train_gnn \
  --train_dir data/gnn_replay_labeled/debug \
  --output_dir checkpoints/gnn_nav/frontier_v1
```

```bash
./run_sg_nav.sh \
  --use_gnn_nav \
  --gnn_ckpt checkpoints/gnn_nav/frontier_v1/gnn_scorer.pt \
  --debug_gnn
```

## 说明

如果只是想确认新分支能不能运行，可以先不用训练，直接运行：

```bash
./run_sg_nav.sh --use_gnn_nav --gnn_ckpt none --debug_gnn
```

这个模式不会使用训练好的 GNN，而是使用距离 fallback 分数。

## 6. 使用未见 50 个 Matterport3D 场景采集数据

已经准备好的未见场景 split：

```text
data/MatterPort3D/objectnav/mp3d/v1/gnn_unseen50
```

对应配置文件：

```text
configs/challenge_objectnav2021.local.rgbd.gnn_unseen50.yaml
```

50 个 scene 列表：

```text
data/MatterPort3D/objectnav/mp3d/v1/gnn_unseen50/scenes.txt
```

这些 scene 和当前原始 val split 的 11 个 scene 没有交集。

### 6.1 采集 SG-Nav Teacher 数据

```bash
./run_sg_nav.sh \
  --config configs/challenge_objectnav2021.local.rgbd.gnn_unseen50.yaml \
  --gnn_log \
  --gnn_log_dir data/gnn_replay/mp3d/unseen50_sgnav \
  --num_episodes 500 \
  --split_l 0 \
  --split_r 50
```

这会使用原始 SG-Nav 策略导航，同时保存 GNN replay 样本和 teacher frontier scores。

### 6.2 采集 Distance Policy 数据

```bash
./run_sg_nav.sh \
  --config configs/challenge_objectnav2021.local.rgbd.gnn_unseen50.yaml \
  --use_gnn_nav \
  --gnn_ckpt none \
  --gnn_data_policy distance \
  --gnn_log \
  --gnn_log_dir data/gnn_replay/mp3d/unseen50_distance \
  --num_episodes 300 \
  --split_l 0 \
  --split_r 50
```

这个模式使用距离 fallback 选择 frontier，用于增加不同状态分布的数据。

### 6.3 采集 Random Policy 数据

```bash
./run_sg_nav.sh \
  --config configs/challenge_objectnav2021.local.rgbd.gnn_unseen50.yaml \
  --use_gnn_nav \
  --gnn_ckpt none \
  --gnn_data_policy random \
  --gnn_log \
  --gnn_log_dir data/gnn_replay/mp3d/unseen50_random \
  --num_episodes 200 \
  --split_l 0 \
  --split_r 50
```

这个模式随机选择 frontier，用于增加探索多样性。

### 6.4 标注与训练

先分别标注：

```bash
python -m gnn_nav.label_frontiers \
  --input_dir data/gnn_replay/mp3d/unseen50_sgnav \
  --output_dir data/gnn_replay_labeled/mp3d/unseen50_sgnav \
  --label_mode teacher
```

```bash
python -m gnn_nav.label_frontiers \
  --input_dir data/gnn_replay/mp3d/unseen50_distance \
  --output_dir data/gnn_replay_labeled/mp3d/unseen50_distance \
  --label_mode teacher
```

```bash
python -m gnn_nav.label_frontiers \
  --input_dir data/gnn_replay/mp3d/unseen50_random \
  --output_dir data/gnn_replay_labeled/mp3d/unseen50_random \
  --label_mode teacher
```

可以把三个 labeled 目录合并到一个训练目录，也可以先只用 `unseen50_sgnav` 训练：

```bash
python -m gnn_nav.train_gnn \
  --train_dir data/gnn_replay_labeled/mp3d/unseen50_sgnav \
  --output_dir checkpoints/gnn_nav/frontier_unseen50_v1 \
  --epochs 20 \
  --lr 1e-4
```

训练完成后在线测试：

```bash
./run_sg_nav.sh \
  --config configs/challenge_objectnav2021.local.rgbd.gnn_unseen50.yaml \
  --use_gnn_nav \
  --gnn_ckpt checkpoints/gnn_nav/frontier_unseen50_v1/gnn_scorer.pt \
  --debug_gnn \
  --num_episodes 100 \
  --split_l 0 \
  --split_r 50
```

注意：完整 SG-Nav 仍需要 CUDA 和 vLLM 服务。当前机器如果 `torch.cuda.is_available()` 为 `False`，会在 GLIP 初始化阶段报：

```text
RuntimeError: No CUDA GPUs are available
```
