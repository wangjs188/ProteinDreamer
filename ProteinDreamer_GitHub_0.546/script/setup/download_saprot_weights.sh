#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export MODEL_ROOT="${1:-${REPO_ROOT}/model}"
PY="${PY:-python}"
"${PY}" - <<'PYDOWNLOAD'
import json, os
from pathlib import Path
from huggingface_hub import snapshot_download
lock = json.loads(Path('model/SAPROT_LOCK.json').read_text())
for family, info in lock['models'].items():
    snapshot_download(repo_id=info['repo_id'], revision=info['revision'],
                      allow_patterns=list(info['sha256']),
                      local_dir=str(Path(os.environ['MODEL_ROOT']) / family))
PYDOWNLOAD
"${PY}" -B script/setup/verify_external_assets.py --models-only
