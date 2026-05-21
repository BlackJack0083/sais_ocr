ARG DOCKER_PLATFORM=linux/amd64
ARG BASE_IMAGE=nvcr.io/nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04
FROM --platform=${DOCKER_PLATFORM} ${BASE_IMAGE}

ENV TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
    PIP_TRUSTED_HOST=mirrors.aliyun.com \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN mkdir -p /app /saisresult

RUN set -eux; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i \
            -e 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
            -e 's|http://security.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
            -e 's|https://archive.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
            -e 's|https://security.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
            /etc/apt/sources.list; \
    fi; \
    find /etc/apt/sources.list.d -type f \( -name '*.list' -o -name '*.sources' \) -exec sed -i \
        -e 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
        -e 's|http://security.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
        -e 's|https://archive.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
        -e 's|https://security.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
        {} +; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        tini \
        bash \
        ca-certificates \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxrender1 \
        libxext6; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

ENV PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=10 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN set -eux; \
    python3 -m pip install --upgrade "pip<25" setuptools wheel

RUN set -eux; \
    python3 -m pip install --prefer-binary -r /app/requirements.txt

COPY src/ /app/src/
COPY models/ /app/models/
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

ENTRYPOINT ["/usr/bin/tini", "--", "bash", "/app/run.sh"]
