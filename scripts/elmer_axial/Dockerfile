# Minimal Elmer FEM image from the official CSC Ubuntu PPA.
# Built because no official prebuilt Elmer image is pullable:
# ghcr.io/elmercsc/* denies anonymous access and the Docker Hub
# community images (eperera/elmerfem, jkatajacsc/elmerfem) date from
# 2018-2019. The PPA ships current official CSC binaries.
# Build:  docker build -t elmer-axial:local scripts/elmer_axial
FROM ubuntu:24.04
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates gnupg \
    && add-apt-repository -y ppa:elmer-csc-ubuntu/elmer-csc-ppa \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        elmerfem-csc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
WORKDIR /work
