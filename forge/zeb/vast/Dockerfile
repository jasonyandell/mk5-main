# Vast.ai worker image for Crystal Forge self-play
# Pre-installs deps so onstart is just: git pull && start worker
#
# Build:  docker build -t jasonyandell/zeb-worker:latest forge/zeb/vast/
# Push:   docker push jasonyandell/zeb-worker:latest
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir huggingface_hub

RUN git clone --depth 1 --branch forge \
    https://github.com/jasonyandell/mk5-main.git /root/code

WORKDIR /root/code
