ARG CUDA_IMAGE=nvcr.io/nvidia/cuda:12.8.1-base-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG UID=1000
ARG GID=1000

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libegl1 \
    libgl1 \
    libgles2 \
    libglib2.0-0 \
    libglvnd0 \
    libgomp1 \
    libopenblas0 \
    libosmesa6 \
    libsm6 \
    libx11-6 \
    libxcursor1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxinerama1 \
    libxrandr2 \
    libxrender1 \
    tini \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g "${GID}" echo \
    && useradd -m -u "${UID}" -g "${GID}" -s /bin/bash echo

WORKDIR /home/echo/SG-Nav

COPY --chown=echo:echo . /home/echo/SG-Nav

RUN mkdir -p \
    /home/echo/SG-Nav/data/results \
    /home/echo/SG-Nav/data/debug_sgnav \
    /home/echo/SG-Nav/data/visualization \
    /home/echo/SG-Nav/.cache/matplotlib \
    /home/echo/SG-Nav/.cache/torch \
    && ln -sfnT ../MatterPort3D /home/echo/SG-Nav/data/MatterPort3D \
    && chmod +x \
    /home/echo/SG-Nav/run_sg_nav.sh \
    /home/echo/SG-Nav/run_vllm.sh \
    /home/echo/SG-Nav/docker/entrypoint.sh

ENV HF_HOME=/home/echo/SG-Nav/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/echo/SG-Nav/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    NLTK_DATA=/home/echo/SG-Nav/.cache/nltk_data \
    MPLCONFIGDIR=/home/echo/SG-Nav/.cache/matplotlib \
    PYTHONPATH=/home/echo/SG-Nav/GLIP \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

USER echo

ENTRYPOINT ["/usr/bin/tini", "--", "/home/echo/SG-Nav/docker/entrypoint.sh"]
CMD ["bash"]
