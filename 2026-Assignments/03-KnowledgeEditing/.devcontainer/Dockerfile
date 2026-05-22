FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces/EasyEdit-Reborn

COPY requirements.txt /tmp/

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip config set global.trusted-host mirrors.aliyun.com && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=1000 -r /tmp/requirements.txt && \
    pip install --no-cache-dir -U torch torchvision torchaudio && \
    rm /tmp/requirements.txt
