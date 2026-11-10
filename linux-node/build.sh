#!/usr/bin/env bash
# Builds the PrivateLM Linux node: rpc-server (GPU worker for phone masters),
# llama-server (PC master / API server) and privatelm-probe, from the exact
# llama.cpp commit the app vendors, so phones and PC speak the same RPC
# protocol.
#
#   ./build.sh                       auto-detect (NVIDIA: CUDA, AMD/Intel: Vulkan)
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
JOBS=""

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

# GPU vendors on the PCI bus: 0x10de NVIDIA, 0x1002 AMD, 0x8086 Intel.
gpu_vendors() {
    local v
    for f in /sys/class/drm/card[0-9]*/device/vendor; do
        [[ -r "$f" ]] || continue
        v="$(cat "$f")"
        case "$v" in
            0x10de) echo nvidia ;;
            0x1002) echo amd ;;
            0x8086) echo intel ;;
        esac
    done | sort -u
}

need_vulkan_dev() {
    local spirv=""
    for d in /usr/share/cmake/SPIRV-Headers /usr/lib/cmake/SPIRV-Headers /usr/lib/x86_64-linux-gnu/cmake/SPIRV-Headers \
             /usr/lib64/cmake/SPIRV-Headers /usr/local/share/cmake/SPIRV-Headers; do
        [[ -f "$d/SPIRV-HeadersConfig.cmake" ]] && spirv="$d"
    done
    if ! have glslc || [[ -z "$spirv" ]] || ! { pkg-config --exists vulkan 2>/dev/null || [[ -f /usr/include/vulkan/vulkan.h ]]; }; then
        echo "Vulkan backend needs: sudo apt install libvulkan-dev glslc spirv-headers mesa-vulkan-drivers" >&2
        echo "(Fedora: vulkan-loader-devel glslc spirv-headers-devel mesa-vulkan-drivers;" >&2
        echo " Arch: vulkan-icd-loader vulkan-headers shaderc spirv-headers)" >&2
        exit 1
    fi
}

if [[ "$BACKENDS" == "auto" ]]; then
    found=()
    vendors="$(gpu_vendors)"
    echo "==> GPUs found: ${vendors:-none}" | tr '\n' ' '; echo
    if grep -q nvidia <<< "$vendors"; then
        if have nvcc || [[ -x /usr/local/cuda/bin/nvcc ]]; then
            found+=(cuda)
        else
            echo "NVIDIA GPU without CUDA toolkit: using Vulkan (install CUDA for the best speed)."
            found+=(vulkan)
        fi
    fi
    # AMD: Vulkan by default. It runs on every AMD card with a Mesa driver,
    # including older ones ROCm dropped (Polaris RX 470/480/570/580, gfx803),
    # and is close to ROCm for token generation. --backends rocm to force it.
    if grep -qE 'amd|intel' <<< "$vendors"; then
        found+=(vulkan)
    fi
    if [[ ${#found[@]} -eq 0 ]]; then
        echo "No GPU found: building CPU only."
        found=(cpu)
    fi
    BACKENDS="$(printf '%s\n' "${found[@]}" | awk '!seen[$0]++' | paste -sd,)"
fi
echo "==> Backends: $BACKENDS"

flags=()
IFS=',' read -ra wanted <<< "$BACKENDS"
for b in "${wanted[@]}"; do
    case "$b" in
        cuda)
            [[ -x /usr/local/cuda/bin/nvcc ]] && export PATH="/usr/local/cuda/bin:$PATH"
            # Native arches of every GPU in the box, else a broad default set.
            # Architectures of the GPUs in this PC (8.6 -> 86), else a broad set.
            if [[ -z "${CUDA_ARCHS:-}" ]] && have nvidia-smi; then
                CUDA_ARCHS="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
                    | tr -d '. ' | sort -u | paste -sd';')"
            fi
            echo "==> CUDA architectures: ${CUDA_ARCHS:-75;80;86;89}"
            flags+=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS:-75;80;86;89}") ;;
        rocm|hip)
            export PATH="/opt/rocm/bin:$PATH"
            if have rocminfo && rocminfo 2>/dev/null | grep -q 'gfx8'; then
                echo "Warning: this PC has a gfx8 (Polaris) GPU, which ROCm no longer supports." >&2
                echo "         Use --backends vulkan for it." >&2
            fi
            export HIPCXX="${HIPCXX:-$(hipconfig -l 2>/dev/null)/clang}"
            export HIP_PATH="${HIP_PATH:-$(hipconfig -R 2>/dev/null || echo /opt/rocm)}"
            if [[ -z "$ROCM_TARGETS" ]] && have rocminfo; then
                ROCM_TARGETS="$(rocminfo | grep -o 'gfx[0-9a-f]*' | sort -u | paste -sd';')"
            fi
            flags+=(-DGGML_HIP=ON)
            [[ -n "$ROCM_TARGETS" ]] && flags+=(-DGPU_TARGETS="$ROCM_TARGETS") ;;
        vulkan)
            need_vulkan_dev
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
# Parallel jobs by free memory, not just cores: single generated Vulkan/CUDA
# files (mul_mm.comp.cpp) need 2-3 GB each to compile, and running out kills
# the compiler ("Killed signal terminated program cc1plus").
if [[ -z "$JOBS" ]]; then
    avail_mb=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 4096)
    JOBS=$(( avail_mb / 2500 ))
    (( JOBS < 1 )) && JOBS=1
    (( JOBS > $(nproc) )) && JOBS=$(nproc)
    echo "==> $JOBS parallel jobs (${avail_mb} MB free memory, $(nproc) cores; -j N to override)"
fi
targets=(ggml-rpc-server llama-server privatelm-probe)
if ! cmake --build "$BUILD" -j "$JOBS" --target "${targets[@]}"; then
    # The usual reason is memory: try once more, one file at a time.
    if (( JOBS > 1 )); then
        echo "==> Build failed with $JOBS jobs; trying again with one (slower, needs less memory)"
        cmake --build "$BUILD" -j 1 --target "${targets[@]}"
    else
        exit 1
    fi
fi

# Architectures this llama.cpp knows, for 'privatelm-node models'.
grep -oE '\{ LLM_ARCH_[A-Z0-9_]+, *"[^"]+"' "$SRC/src/llama-arch.cpp" | sed -E 's/.*"([^"]+)"/\1/' \
    | sort -u > "$BUILD/bin/supported-archs.txt"
# llama.cpp's MoE cache tuner, for 'privatelm-node tune-moe'.
cp "$SRC/scripts/moe-cache-tune.py" "$BUILD/bin/" && chmod 755 "$BUILD/bin/moe-cache-tune.py"
echo "==> Built in $BUILD/bin"
"$BUILD/bin/privatelm-probe" --no-bench || true

if [[ -n "$INSTALL_DIR" ]]; then
    mkdir -p "$INSTALL_DIR/bin"
    # --remove-destination: a running service keeps its old files (writing
    # into a busy binary fails with "Text file busy", into a loaded .so
    # crashes the process); the installer restarts the services afterwards.
    cp -a --remove-destination "$BUILD"/bin/*.so* "$BUILD"/bin/ggml-rpc-server "$BUILD"/bin/llama-server \
          "$BUILD"/bin/privatelm-probe "$BUILD"/bin/moe-cache-tune.py "$INSTALL_DIR/bin/"
    install -m 755 "$HERE/privatelm_node.py" "$INSTALL_DIR/bin/privatelm-node"
    install -m 644 "$HERE/dashboard.html" "$INSTALL_DIR/bin/dashboard.html"
    install -m 644 "$BUILD/bin/supported-archs.txt" "$INSTALL_DIR/bin/supported-archs.txt"
    echo "==> Installed to $INSTALL_DIR (run $INSTALL_DIR/bin/privatelm-node --help)"
fi
