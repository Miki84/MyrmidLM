#!/usr/bin/env bash
# Installs ComfyUI for MyrmidLM's image models: source, its own Python venv
# and PyTorch (CUDA if nvidia-smi sees a GPU, else CPU). Run by the host when
# the image engine "ComfyUI" is switched on; again to update.
#   build-comfyui.sh --dest DIR      (DIR/src, DIR/venv, DIR/backend)
# COMFY_REF selects a branch or tag (default: master), PYTHON the interpreter.
set -euo pipefail

DEST=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dest) DEST="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[ -n "$DEST" ] || { echo "usage: $0 --dest DIR" >&2; exit 2; }
REF="${COMFY_REF:-master}"
PY="${PYTHON:-python3}"
mkdir -p "$DEST"

command -v git >/dev/null || { echo "git fehlt: sudo apt install git" >&2; exit 3; }
command -v "$PY" >/dev/null || { echo "$PY fehlt: sudo apt install python3 python3-venv" >&2; exit 3; }
"$PY" - <<'PYEOF' || { echo "ComfyUI braucht Python 3.10 oder neuer" >&2; exit 3; }
import sys
sys.exit(0 if sys.version_info >= (3, 10) else 1)
PYEOF

echo "==> ComfyUI holen ($REF)"
if [ -d "$DEST/src/.git" ]; then
  git -C "$DEST/src" fetch -q --depth 1 origin "$REF"
  git -C "$DEST/src" reset -q --hard FETCH_HEAD
else
  rm -rf "$DEST/src"
  git clone -q --depth 1 --branch "$REF" https://github.com/Comfy-Org/ComfyUI "$DEST/src"
fi

echo "==> Python-Umgebung"
if [ ! -x "$DEST/venv/bin/python" ]; then
  "$PY" -m venv "$DEST/venv" || { echo "python3-venv fehlt: sudo apt install python3-venv" >&2; exit 3; }
fi
PIP=("$DEST/venv/bin/python" -m pip)
"${PIP[@]}" install -q --upgrade pip

backend=cpu
index=https://download.pytorch.org/whl/cpu
if command -v nvidia-smi >/dev/null && nvidia-smi -L 2>/dev/null | grep -q GPU; then
  ver=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: *[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' || true)
  major=${ver%%.*}; minor=${ver##*.}
  tag=""
  if [ -n "$ver" ]; then
    if [ "$major" -ge 13 ]; then tag=cu130
    elif [ "$major" -eq 12 ] && [ "$minor" -ge 8 ]; then tag=cu128
    elif [ "$major" -eq 12 ] && [ "$minor" -ge 6 ]; then tag=cu126
    fi
  fi
  if [ -n "$tag" ]; then
    backend=cuda
    index="https://download.pytorch.org/whl/$tag"
  else
    echo "==> NVIDIA-Treiber zu alt für aktuelles PyTorch (CUDA $ver, nötig: 12.6 oder neuer): nur CPU"
  fi
fi

echo "==> PyTorch ($backend)"
"${PIP[@]}" install --progress-bar off torch torchvision --index-url "$index"

echo "==> ComfyUI-Abhängigkeiten"
"${PIP[@]}" install --progress-bar off -r "$DEST/src/requirements.txt"

echo "$backend" > "$DEST/backend"
git -C "$DEST/src" rev-parse --short HEAD > "$DEST/commit"
echo "==> Fertig: ComfyUI $(cat "$DEST/commit") ($backend)"
