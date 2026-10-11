FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS gpu-base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv python3-dev build-essential git \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && pip install --upgrade pip

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src


FROM gpu-base AS generate
RUN pip install ".[generate]"
COPY configs ./configs
ENTRYPOINT ["najdi"]


FROM gpu-base AS train
RUN pip install ".[train]"
COPY configs ./configs
ENTRYPOINT ["najdi"]


FROM gpu-base AS serve
RUN pip install ".[serve]"
COPY configs ./configs
EXPOSE 7860
ENTRYPOINT ["najdi"]
CMD ["serve", "--config", "configs/serve.yaml"]


FROM python:3.12-slim AS cpu
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install ".[mix]"
COPY configs ./configs
ENTRYPOINT ["najdi"]
