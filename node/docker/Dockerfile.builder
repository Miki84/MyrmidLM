# Build environment for node/docker-build.sh: NVIDIA's CUDA image plus the
# tools build.sh needs. Only for compiling; the programs run on the host.
# Ubuntu 22.04 inside (glibc 2.35): the binaries also run on older hosts
# (Debian 12, Ubuntu 22.04) and on every newer one.
ARG BASE=nvidia/cuda:13.0.1-devel-ubuntu22.04
FROM ${BASE}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake git python3 ca-certificates pkg-config \
 && rm -rf /var/lib/apt/lists/* \
 && git config --system --add safe.directory '*'
