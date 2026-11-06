#!/usr/bin/env bash
set -euo pipefail
# Compatibility entry point: the historical evaluator is now included.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PROTEINGYM_ROOT="${1:-${REPO_ROOT}/external/ProteinGym-main}"
"${PY:-python}" -B script/setup/verify_external_assets.py --evaluator-only
