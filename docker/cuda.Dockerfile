FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        g++ \
        git \
        libgtest-dev \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

ENV HOME=/tmp
WORKDIR /src
