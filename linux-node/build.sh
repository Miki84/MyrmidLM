#!/usr/bin/env bash
# Builds the PrivateLM Linux node: rpc-server (GPU worker for phone masters),
# llama-server (PC master / API server) and privatelm-probe, from the exact
# llama.cpp commit the app vendors, so phones and PC speak the same RPC
# protocol.
#
#   ./build.sh                       auto-detect GPU backends
#   ./build.sh --backends cuda,vulkan
#   ./build.sh --backends rocm --rocm-targets gfx1100
#   ./build.sh --backends sycl       (needs Intel oneAPI)
#   ./build.sh --backends cpu        no GPU
#   ./build.sh --install /opt/privatelm-node
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMIT="$(tr -d '[:space:]' < "$HERE/LLAMA_CPP_COMMIT")"
SRC="${LLAMA_CPP_SRC:-$HERE/.llama.cpp}"
BUILD="${BUILD_DIR:-$HERE/build}"
BACKENDS="auto"
ROCM_TARGETS=""
INSTALL_DIR=""
JOBS="$(nproc)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backends) BACKENDS="$2"; shift 2 ;;
        --rocm-targets) ROCM_TARGETS="$2"; shift 2 ;;
        --install) INSTALL_DIR="$2"; shift 2 ;;
        -j) JOBS="$2"; shift 2 ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

have() { command -v "$1" >/dev/null 2>&1; }

if [[ "$BACKENDS" == "auto" ]]; then
    found=()
    if have nvcc || [[ -x /usr/local/cuda/bin/nvcc ]]; then found+=(cuda); fi
    if have hipcc || [[ -x /opt/rocm/bin/hipcc ]]; then found+=(rocm); fi
    if have glslc && { pkg-config --exists vulkan 2>/dev/null || [[ -f /usr/include/vulkan/vulkan.h ]]; }; then
        found+=(vulkan)
    fi
    if [[ ${#found[@]} -eq 0 ]]; then
        echo "No GPU toolkit found (nvcc, hipcc, or Vulkan headers + glslc): building CPU only."
        echo "Intel GPUs: install Vulkan dev packages (libvulkan-dev glslc) or use --backends sycl."
        found=(cpu)
    fi
    BACKENDS="$(IFS=,; echo "${found[*]}")"
fi
echo "==> Backends: $BACKENDS"

flags=()
IFS=',' read -ra wanted <<< "$BACKENDS"
for b in "${wanted[@]}"; do
    case "$b" in
        cuda)
            [[ -x /usr/local/cuda/bin/nvcc ]] && export PATH="/usr/local/cuda/bin:$PATH"
            # Native arches of every GPU in the box, else a broad default set.
            flags+=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS:-75;80;86;89}") ;;
        rocm|hip)
            export PATH="/opt/rocm/bin:$PATH"
            export HIPCXX="${HIPCXX:-$(hipconfig -l 2>/dev/null)/clang}"
            export HIP_PATH="${HIP_PATH:-$(hipconfig -R 2>/dev/null || echo /opt/rocm)}"
            if [[ -z "$ROCM_TARGETS" ]] && have rocminfo; then
                ROCM_TARGETS="$(rocminfo | grep -o 'gfx[0-9a-f]*' | sort -u | paste -sd';')"
            fi
            flags+=(-DGGML_HIP=ON)
            [[ -n "$ROCM_TARGETS" ]] && flags+=(-DGPU_TARGETS="$ROCM_TARGETS") ;;
        vulkan)
            flags+=(-DGGML_VULKAN=ON) ;;
        sycl)
            if [[ -z "${ONEAPI_ROOT:-}" ]]; then
                # shellcheck disable=SC1091
                source /opt/intel/oneapi/setvars.sh >/dev/null || {
                    echo "SYCL needs Intel oneAPI (/opt/intel/oneapi/setvars.sh)" >&2; exit 1; }
            fi
            flags+=(-DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx) ;;
        cpu) ;;
        *) echo "Unknown backend: $b (cuda, rocm, vulkan, sycl, cpu)" >&2; exit 2 ;;
    esac
done

# llama.cpp at the pinned commit + the app's RPC logging patch.
if [[ ! -d "$SRC/.git" ]]; then
    echo "==> Fetching llama.cpp $COMMIT"
    git init -q "$SRC"
    git -C "$SRC" remote add origin https://github.com/ggml-org/llama.cpp.git
fi
if [[ "$(git -C "$SRC" rev-parse HEAD 2>/dev/null)" != "$COMMIT" ]]; then
    git -C "$SRC" fetch -q --depth 1 origin "$COMMIT"
    git -C "$SRC" checkout -q --force FETCH_HEAD
fi
git -C "$SRC" checkout -q -- ggml/src/ggml-rpc
for p in "$HERE"/patches/*.patch; do
    git -C "$SRC" apply "$p"
done

echo "==> Configuring"
cmake -S "$HERE" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CPP_DIR="$SRC" ${flags[@]+"${flags[@]}"}

echo "==> Building with $JOBS jobs (CUDA/ROCm builds take a while)"
cmake --build "$BUILD" -j "$JOBS" --target ggml-rpc-server llama-server privatelm-probe

echo "==> Built in $BUILD/bin"
"$BUILD/bin/privatelm-probe" --no-bench || true

if [[ -n "$INSTALL_DIR" ]]; then
    mkdir -p "$INSTALL_DIR/bin"
    cp -a "$BUILD"/bin/*.so* "$BUILD"/bin/ggml-rpc-server "$BUILD"/bin/llama-server \
          "$BUILD"/bin/privatelm-probe "$INSTALL_DIR/bin/"
    install -m 755 "$HERE/privatelm_node.py" "$INSTALL_DIR/bin/privatelm-node"
    echo "==> Installed to $INSTALL_DIR (run $INSTALL_DIR/bin/privatelm-node --help)"
fi
