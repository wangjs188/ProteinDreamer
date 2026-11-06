#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
PY="${PY:-python}"
"${PY}" -B script/release/verify_release.py
if [[ "${STRICT:-1}" == "1" ]]; then
  "${PY}" -B script/setup/verify_external_assets.py
  if [[ "${DEVICE:-cuda}" == cuda* ]]; then
    "${PY}" -B script/setup/check_runtime.py --require-cuda
  else
    "${PY}" -B script/setup/check_runtime.py
  fi
fi
