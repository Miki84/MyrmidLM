#!/usr/bin/env bash
# Neue MyrmidLM-Version: setzt VERSION und die App-Version (pubspec.yaml),
# committet und taggt. Danach: git push && git push --tags
#
#   scripts/release.sh 1.2.0
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ver="${1:-}"
[[ "$ver" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "Aufruf: $0 X.Y.Z" >&2; exit 2; }
if ! git diff --quiet || ! git diff --cached --quiet; then echo "Erst alle Änderungen committen." >&2; exit 1; fi
git rev-parse -q --verify "refs/tags/v$ver" >/dev/null && { echo "v$ver gibt es schon." >&2; exit 1; }
old="$(cat VERSION)"
python3 - "$old" "$ver" <<'PY'
import sys
def key(v): return tuple(int(x) for x in v.split("."))
if key(sys.argv[2]) <= key(sys.argv[1]):
    sys.exit(f"{sys.argv[2]} ist nicht neuer als {sys.argv[1]}")
PY

echo "$ver" > VERSION
# App: versionName = ver, versionCode + 1 (Android braucht steigende Nummern)
code="$(sed -n 's/^version: [0-9.]*+\([0-9]*\)$/\1/p' pubspec.yaml)"
sed -i "s/^version: .*/version: $ver+$(( ${code:-0} + 1 ))/" pubspec.yaml

python3 -m py_compile node/myrmid.py
bash -n install.sh node/build.sh

git add VERSION pubspec.yaml
git commit -q -m "MyrmidLM $ver"
git tag -a "v$ver" -m "MyrmidLM $ver"
echo "v$ver ist bereit. Veröffentlichen mit: git push && git push --tags"
echo "RPC-Protokoll dieser Version: $(sed -n 's/^RPC_VERSION = "\(.*\)"/\1/p' node/myrmid.py)"
