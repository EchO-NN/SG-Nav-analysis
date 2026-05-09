# SG-Nav Mamba Environments

This repository uses two local mamba environments under the repo root:

- `.mamba/envs/sg-nav`: Habitat, SG-Nav, GLIP, GroundingDINO, SAM, mapping, and navigation.
- `.mamba/envs/sg-nav-vllm`: vLLM OpenAI-compatible server for Qwen/VLM calls.

The run scripts already expect these paths:

- `./run_sg_nav.sh`
- `./run_vllm.sh`

## Files

- `envs/sg-nav.yml`: conda/mamba packages for the main SG-Nav runtime.
- `envs/sg-nav-pip.txt`: pip packages plus editable local installs.
- `envs/sg-nav-vllm.yml`: conda/mamba packages for the vLLM runtime.
- `envs/sg-nav-vllm-pip.txt`: vLLM pip package.
- `scripts/setup_mamba_envs.sh`: creates or updates both local environments.

## System Packages

On Ubuntu 22.04, install the common native libraries first:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential git curl wget ffmpeg \
  libgl1 libegl1 libgles2 libglib2.0-0 libglvnd0 libgomp1 \
  libopenblas0 libosmesa6 libsm6 libx11-6 libxcursor1 libxext6 \
  libxfixes3 libxi6 libxinerama1 libxrandr2 libxrender1
```

GPU runs also need a working NVIDIA driver. Check it with:

```bash
nvidia-smi
```

## Create Environments

From the repository root:

```bash
cd /home/echo/SG-Nav
./scripts/setup_mamba_envs.sh
```

If `mamba` is not on `PATH`, point the script at it:

```bash
MAMBA_BIN=/home/echo/miniforge3/bin/mamba ./scripts/setup_mamba_envs.sh
```

The script creates:

```text
.mamba/envs/sg-nav
.mamba/envs/sg-nav-vllm
```

## Manual Commands

Main environment:

```bash
mamba env create -p ./.mamba/envs/sg-nav -f envs/sg-nav.yml
./.mamba/envs/sg-nav/bin/pip install -r envs/sg-nav-pip.txt
```

vLLM environment:

```bash
mamba env create -p ./.mamba/envs/sg-nav-vllm -f envs/sg-nav-vllm.yml
./.mamba/envs/sg-nav-vllm/bin/pip install -r envs/sg-nav-vllm-pip.txt
```

If an environment already exists, use update instead:

```bash
mamba env update -p ./.mamba/envs/sg-nav -f envs/sg-nav.yml --prune
mamba env update -p ./.mamba/envs/sg-nav-vllm -f envs/sg-nav-vllm.yml --prune
```

Then rerun the matching `pip install -r ...` command.

## Assets

The environments do not include large model/data assets. Required paths are checked by:

```bash
./.mamba/envs/sg-nav/bin/python check_setup.py
```

Important paths:

```text
GLIP/MODEL/glip_large_model.pth
data/models/sam_vit_h_4b8939.pth
data/models/groundingdino_swint_ogc.pth
MatterPort3D/
data/MatterPort3D -> ../MatterPort3D
.cache/huggingface/
.cache/nltk_data/
```

## Run

Terminal 1:

```bash
./run_vllm.sh
```

Terminal 2:

```bash
./run_sg_nav.sh \
  --split_l 0 \
  --split_r 1 \
  --num_episodes 1 \
  --debug_sgnav \
  --debug_sgnav_dir data/debug_sgnav/mamba_check \
  --reperception_min_observations 4 \
  --reperception_threshold 1.2 \
  --reperception_max_steps 12 \
  --found_goal_stop_distance_m 0.35
```

## Offline Or Server Migration

For a server where the envs already work on this machine, the most reliable transfer is to copy these directories with the repo:

```text
.mamba/envs/sg-nav
.mamba/envs/sg-nav-vllm
.cache/huggingface
.cache/nltk_data
GLIP/MODEL
data/models
MatterPort3D
```

If recreating from scratch on the server, run `scripts/setup_mamba_envs.sh` after installing system packages and NVIDIA drivers.

## Notes

- The main environment is pinned around Python 3.9, Habitat-Sim 0.2.4, Torch 1.9.1 CUDA 11.1 wheels, and conda CUDA 11.8 runtime packages because that matches the currently working local setup.
- The vLLM environment is intentionally separate because it uses Python 3.11 and modern Torch/CUDA packages.
- Keep the two environments separate. Mixing vLLM packages into `sg-nav` will likely break the Habitat/GLIP stack.
