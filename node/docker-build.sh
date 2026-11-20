#!/usr/bin/env bash
# build.sh inside a CUDA build container: no CUDA toolkit on the host. The
# programs it builds run on the host as usual (the NVIDIA driver is enough);
# the CUDA libraries they need (cudart, cuBLAS) are copied next to them.
#
#   ./docker-build.sh --install /opt/myrmid/releases/2.1.0
#   ./docker-build.sh --install DIR --image nvidia/cuda:12.9.1-devel-ubuntu22.04
#
# Without --image the CUDA version follows the driver and the oldest GPU:
# CUDA 13 for Turing and newer with a driver that supports it, else 12.9
# (CUDA 13 dropped Maxwell, Pascal and Volta).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
BACKENDS="cuda" INSTALL_DIR="" IMAGE="${BUILD_IMAGE:-}" JOBS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backends) BACKENDS="$2"; shift 2 ;;
        --install) INSTALL_DIR="$2"; shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        -j) JOBS="$2"; shift 2 ;;
        -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$INSTALL_DIR" ]] || { echo "--install DIR is required" >&2; exit 2; }
[[ "$BACKENDS" == cuda || "$BACKENDS" == cpu ]] || { echo "docker-build.sh builds cuda or cpu, not $BACKENDS" >&2; exit 2; }
command -v docker >/dev/null || { echo "docker is not installed" >&2; exit 1; }

# The GPUs' architectures (8.6 -> 86): nvcc in the container can't see them.
# Older chips also from their lspci names (GV100 -> 70): a driver newer than
# 580 doesn't list them in nvidia-smi, the build would then skip them.
old_archs() {
    command -v lspci >/dev/null || return 0
    local c
    for c in $(lspci -d 10de: 2>/dev/null | grep -E 'VGA|3D|Display' | grep -oE '\b(GM|GP|GV)[0-9]+' || true); do
        case "$c" in
            GM10*) echo 50 ;; GM20*) echo 52 ;; GP100) echo 60 ;; GP10*) echo 61 ;; GV*) echo 70 ;;
        esac
    done
}
CAPS="$( { command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
    | tr -d '. '; old_archs; } | grep -E '^[0-9]+$' | sort -un | paste -sd';' || true)"
if [[ -z "$IMAGE" ]]; then
    cuda_max="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: *[0-9]+' | grep -oE '[0-9]+$' | head -1 || true)"
    old=0
    for c in ${CAPS//;/ }; do (( c < 75 )) && old=1; done
    if (( ${cuda_max:-0} >= 13 && old == 0 )); then
        IMAGE="nvidia/cuda:13.0.1-devel-ubuntu22.04"
    else
        IMAGE="nvidia/cuda:12.9.1-devel-ubuntu22.04"
    fi
fi
TAG="myrmid-builder:$(echo "${IMAGE#*:}" | tr -c 'a-zA-Z0-9.\n' '-')"
echo "==> Build container: $IMAGE (GPU architectures: ${CAPS:-default set})"
docker build -q -t "$TAG" --build-arg BASE="$IMAGE" -f "$HERE/docker/Dockerfile.builder" "$HERE/docker" >/dev/null

mkdir -p "$INSTALL_DIR"
OUT="$(cd "$(dirname "$INSTALL_DIR")" && pwd)" NAME="$(basename "$INSTALL_DIR")"
# Its own build tree: CMake's cache remembers paths, the host's native build has others.
docker run --rm --network host \
    -e CUDA_ARCHS="$CAPS" -e LLAMA_CPP_REPO="${LLAMA_CPP_REPO:-}" -e BUILD_DIR=/src/node/build-docker \
    -v "$ROOT:/src" -v "$OUT:/out" -w /src/node "$TAG" bash -euo pipefail -c '
        [[ -n "$LLAMA_CPP_REPO" ]] || unset LLAMA_CPP_REPO
        bash build.sh --backends "$1" --install "/out/$2" ${3:+-j "$3"}
        # The CUDA libraries from the image (the driver library libcuda comes from the host).
        for so in /out/"$2"/bin/*.so* /out/"$2"/bin/llama-server; do ldd "$so" 2>/dev/null || true; done \
            | awk "\$3 ~ /cuda/ && \$3 !~ /stubs/ {print \$3}" | sort -u | while read -r lib; do
                cp -L "$lib" "/out/$2/bin/"
            done
    ' _ "$BACKENDS" "$NAME" "$JOBS"
echo "==> Built in a container: $INSTALL_DIR"
