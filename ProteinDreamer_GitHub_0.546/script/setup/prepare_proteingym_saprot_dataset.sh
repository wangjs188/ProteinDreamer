#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
"${PY:-python}" -B "${REPO_ROOT}/script/setup/prepare_proteingym_raw.py" \
  --raw-dir "${1:-${REPO_ROOT}/data/proteingym_downloads/raw}" \
  --dataset "${2:-${REPO_ROOT}/data/ProteinGym_v1_SaProt_AF2_v2}" \
  --flat-dms "${3:-${REPO_ROOT}/data/proteingym_local/DMS_ProteinGym_substitutions_v2}" \
  --reference "${REF_CSV:-${REPO_ROOT}/data/proteingym_local/reference/DMS_substitutions.csv}"
