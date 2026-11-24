#!/usr/bin/env bash
# Builds stable-diffusion.cpp's sd-server for a host's image model.
# MyrmidLM runs this itself (dashboard: host card, "Bilder"); by hand:
#
#   bash build-sdcpp.sh --dest ~/.cache/myrmid/sdcpp
#
# Backend: CUDA if nvcc and an NVIDIA GPU are there, else Vulkan if its headers
# and glslc are there, else CPU. Without the web UI (no Node needed): the API is enough.
set -euo pipefail

DEST=""
REPO="${SD_CPP_REPO:-https://github.com/leejet/stable-diffusion.cpp.git}"
REF="${SD_CPP_REF:-master}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest) DEST="$2"; shift 2 ;;
        -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$DEST" ]] || { echo "--dest fehlt" >&2; exit 2; }
SRC="$DEST/src"
BUILD="$DEST/build"
mkdir -p "$DEST"
have() { command -v "$1" >/dev/null 2>&1; }

if ! have nvcc && [[ -x /usr/local/cuda/bin/nvcc ]]; then export PATH="/usr/local/cuda/bin:$PATH"; fi
backend=cpu
if have nvcc && have nvidia-smi && nvidia-smi -L >/dev/null 2>&1; then
    backend=cuda
elif have glslc && [[ -f /usr/include/vulkan/vulkan.h ]]; then
    backend=vulkan
fi
echo "==> Backend: $backend"

for t in git cmake g++; do
    have "$t" || { echo "$t fehlt (wird auch für llama.cpp gebraucht): bitte installieren" >&2; exit 1; }
done

if [[ ! -d "$SRC/.git" ]]; then
    echo "==> Hole stable-diffusion.cpp ($REF)"
    git clone -q --depth 1 --branch "$REF" --recursive --shallow-submodules "$REPO" "$SRC"
else
    echo "==> Aktualisiere stable-diffusion.cpp ($REF)"
    git -C "$SRC" fetch -q --depth 1 origin "$REF"
    git -C "$SRC" reset -q --hard FETCH_HEAD
    git -C "$SRC" submodule update -q --init --recursive --depth 1
fi

flags=(-DCMAKE_BUILD_TYPE=Release -DSD_SERVER_BUILD_FRONTEND=OFF -DGGML_NATIVE=ON)
case "$backend" in
    cuda)
        archs="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d '. ' \
            | grep -E '^[0-9]+$' | sort -un | paste -sd';' || true)"
        nvcc_major="$(nvcc --version 2>/dev/null | grep -oE 'release [0-9]+' | grep -oE '[0-9]+$' || true)"
        if [[ -n "$archs" && -n "$nvcc_major" ]] && (( nvcc_major >= 13 )); then
            keep=""
            for a in ${archs//;/ }; do (( a >= 75 )) && keep+="${keep:+;}$a"; done
            archs="$keep"
        fi
        echo "==> CUDA-Architekturen: ${archs:-75;80;86;89}"
        flags+=(-DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${archs:-75;80;86;89}") ;;
    vulkan) flags+=(-DSD_VULKAN=ON) ;;
esac

echo "==> Konfiguriere"
cmake -S "$SRC" -B "$BUILD" "${flags[@]}"
echo "==> Baue sd-server (CUDA dauert 10 bis 30 Minuten)"
cmake --build "$BUILD" -j "$(nproc)" --target sd-server
test -x "$BUILD/bin/sd-server"
echo "$backend" > "$DEST/backend"
git -C "$SRC" rev-parse --short HEAD > "$DEST/commit"
echo "==> Fertig: $BUILD/bin/sd-server ($backend)"
