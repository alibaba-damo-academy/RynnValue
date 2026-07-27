#!/bin/bash
# Install pyzed (ZED SDK Python bindings) into the client venv.
#
# Why this is a separate step:
#   The pyzed cp311 wheel declares `numpy>=2.0` in its metadata, but DROID's
#   stack pins older deps that resolve to numpy 1.x. The binary actually works
#   fine against numpy 1.26 — only the metadata is overly conservative. So we
#   bypass the resolver with `uv pip install --no-deps`.
#
# Prerequisites:
#   - ZED SDK installed at /usr/local/zed (https://www.stereolabs.com/developers/release/)
#   - client/.venv already created via `uv sync`
#
# Run from the repo root: bash client/install_pyzed.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="$REPO_ROOT/client/.venv/bin/python"
ZED_API_SCRIPT="/usr/local/zed/get_python_api.py"

if [[ ! -x "$VENV_PY" ]]; then
    echo "ERROR: $VENV_PY not found. Run 'cd client && uv sync' first." >&2
    exit 1
fi
if [[ ! -f "$ZED_API_SCRIPT" ]]; then
    echo "ERROR: $ZED_API_SCRIPT not found. Install the ZED SDK first." >&2
    exit 1
fi

# get_python_api.py downloads the matching cp<py_ver> wheel into CWD.
# We run it from a temp dir so the wheel doesn't pollute the repo, then
# install with --no-deps to bypass the conservative numpy>=2.0 metadata.
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

cd "$TMPDIR"
# The script exits non-zero because the venv has no pip; that's expected.
# We only need it to download the wheel for our Python version.
"$VENV_PY" "$ZED_API_SCRIPT" || true

WHEEL="$(ls "$TMPDIR"/pyzed-*.whl 2>/dev/null | head -1)"
if [[ -z "$WHEEL" ]]; then
    echo "ERROR: get_python_api.py did not download a wheel for this Python version." >&2
    exit 1
fi

echo "Installing $WHEEL into client venv..."
uv pip install --python "$VENV_PY" --no-deps "$WHEEL"

echo
echo "Verifying..."
"$VENV_PY" -c "import pyzed.sl as sl; print('pyzed OK, SDK version:', sl.Camera.get_sdk_version())"
