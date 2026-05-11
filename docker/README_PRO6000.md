# SG-Nav Pro6000 Docker Bundle

This bundle is meant for moving the current working SG-Nav setup to an NVIDIA Pro6000 server with minimal reinstall work.

The image copies the repo into `/home/echo/SG-Nav` and keeps the existing local environments at the same path:

- `.mamba/envs/sg-nav`
- `.mamba/envs/sg-nav-vllm`
- `.cache/huggingface`
- `.cache/nltk_data`
- `GLIP/MODEL`
- `data/models`
- `MatterPort3D`

Large generated outputs are excluded from the image: `data/results`, `data/debug_sgnav`, and `data/visualization`.

## Build On A Machine With Docker

```bash
cd /home/echo/SG-Nav
./docker/build_image.sh
```

If your current shell has not picked up Docker group membership yet, prefix helper scripts with:

```bash
DOCKER_BIN="sudo docker" ./docker/build_image.sh
```

If you build with a custom Docker daemon that has no bridge network, add:

```bash
BUILD_NETWORK=host DOCKER_BIN="sudo docker -H unix:///path/to/docker.sock" ./docker/build_image.sh
```

The default base image is pulled from NVIDIA NGC (`nvcr.io`) because Docker Hub can be slow or rate-limited. It uses the smaller CUDA `base` image because the bundled `.mamba` environments already carry the Python-side CUDA/cuDNN libraries. If that image is not available on your server, override it:

```bash
CUDA_IMAGE=nvidia/cuda:12.8.1-base-ubuntu22.04 ./docker/build_image.sh
```

## Export For Transfer

```bash
./docker/save_image.sh
```

This writes:

```text
dist/sgnav-pro6000-image.tar.gz
```

Copy it to the Pro6000 server:

```bash
scp dist/sgnav-pro6000-image.tar.gz user@pro6000:/path/to/
```

## Load On The Pro6000 Server

The server needs Docker plus NVIDIA Container Toolkit.

```bash
gunzip -c sgnav-pro6000-image.tar.gz | docker load
docker image ls | grep sgnav-pro6000
```

Quick GPU sanity check:

```bash
docker run --rm --gpus all sgnav-pro6000:latest nvidia-smi
```

## Run With Two Terminals

Terminal 1, start vLLM:

```bash
docker run --gpus all --ipc=host --shm-size=16g --rm -it \
  --name sgnav-vllm \
  -p 8000:8000 \
  -e VLLM_HOST=0.0.0.0 \
  sgnav-pro6000:latest vllm
```

Terminal 2, run SG-Nav:

```bash
mkdir -p docker_outputs/results docker_outputs/debug_sgnav docker_outputs/visualization

docker run --gpus all --ipc=host --shm-size=16g --rm -it --network host \
  -e VLLM_BASE_URL=http://127.0.0.1:8000/v1 \
  -v "$PWD/docker_outputs/results:/home/echo/SG-Nav/data/results" \
  -v "$PWD/docker_outputs/debug_sgnav:/home/echo/SG-Nav/data/debug_sgnav" \
  -v "$PWD/docker_outputs/visualization:/home/echo/SG-Nav/data/visualization" \
  sgnav-pro6000:latest sg-nav \
  --split_l 0 --split_r 1 --num_episodes 1 \
  --debug_sgnav --debug_sgnav_dir data/debug_sgnav/pro6000 \
  --reperception_min_observations 4 \
  --reperception_threshold 1.2 \
  --reperception_max_steps 12 \
  --found_goal_stop_distance_m 0.35
```

The helper scripts do the same thing:

```bash
./docker/run_vllm_container.sh
./docker/run_sg_nav_container.sh --split_l 0 --split_r 1 --num_episodes 1 --debug_sgnav
```

## Run With Compose

```bash
docker compose -f docker-compose.pro6000.yml up vllm
docker compose -f docker-compose.pro6000.yml run --rm sg-nav
```

## Notes

- The image is intentionally large because it preserves the working local Python environments and model caches.
- `data/results`, `data/debug_sgnav`, and `data/visualization` are mounted out to `docker_outputs/`.
- If the target server already has MatterPort3D or model weights in a different location, you can mount them over the bundled paths with `-v /host/path:/home/echo/SG-Nav/MatterPort3D:ro` or the corresponding model directories.
