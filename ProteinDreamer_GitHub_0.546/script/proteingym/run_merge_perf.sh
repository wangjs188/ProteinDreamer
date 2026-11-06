#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROTEINGYM_ROOT="${PROTEINGYM_ROOT:-${REPO_ROOT}/external/ProteinGym-main}"

PY="${PY:-python}"

DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/ProteinGym_v1_SaProt_AF2_v2}"
FLAT_DMS_DIR="${FLAT_DMS_DIR:-${REPO_ROOT}/data/proteingym_local/DMS_ProteinGym_substitutions_v2}"
SCORES_ROOT="${SCORES_ROOT:-${REPO_ROOT}/data/proteingym_local/zero_shot_substitutions_scores}"
MERGED_SCORES_DIR="${MERGED_SCORES_DIR:-merged_scores/subs}"

CONFIG_JSON="${CONFIG_JSON:-${REPO_ROOT}/script/proteingym/protein_dreamer_proteingym_config_best_0546.json}"
REF_CSV="${REF_CSV:-${REPO_ROOT}/data/proteingym_local/reference/DMS_substitutions.csv}"

EVAL_OUT="${EVAL_OUT:-${REPO_ROOT}/result/proteingym_eval}"

abspath() {
  local p="$1"
  if [[ -z "${p}" ]]; then
    echo ""
  elif [[ "${p}" = /* ]]; then
    echo "${p}"
  else
    echo "${REPO_ROOT}/${p}"
  fi
}

DATASET_ROOT="$(abspath "${DATASET_ROOT}")"
FLAT_DMS_DIR="$(abspath "${FLAT_DMS_DIR}")"
SCORES_ROOT="$(abspath "${SCORES_ROOT}")"
CONFIG_JSON="$(abspath "${CONFIG_JSON}")"
EVAL_OUT="$(abspath "${EVAL_OUT}")"

if [[ "${MERGED_SCORES_DIR}" = /* ]]; then
  echo "MERGED_SCORES_DIR must be relative to SCORES_ROOT, got: ${MERGED_SCORES_DIR}"
  exit 2
fi

MERGED_DIR="${SCORES_ROOT}/${MERGED_SCORES_DIR}"

mkdir -p "${SCORES_ROOT}"
mkdir -p "${EVAL_OUT}"

echo "[run_merge_perf] REPO_ROOT=${REPO_ROOT}"
echo "[run_merge_perf] PROTEINGYM_ROOT=${PROTEINGYM_ROOT}"
echo "[run_merge_perf] DATASET_ROOT=${DATASET_ROOT}"
echo "[run_merge_perf] FLAT_DMS_DIR=${FLAT_DMS_DIR}"
echo "[run_merge_perf] SCORES_ROOT=${SCORES_ROOT}"
echo "[run_merge_perf] MERGED_SCORES_DIR=${MERGED_SCORES_DIR}"
echo "[run_merge_perf] MERGED_DIR=${MERGED_DIR}"
echo "[run_merge_perf] CONFIG_JSON=${CONFIG_JSON}"
echo "[run_merge_perf] REF_CSV=${REF_CSV}"
echo "[run_merge_perf] EVAL_OUT=${EVAL_OUT}"

if [[ ! -d "${DATASET_ROOT}/DATASET" ]]; then
  echo "Missing dataset: ${DATASET_ROOT}/DATASET"
  echo "Run: bash ${REPO_ROOT}/script/setup/prepare_proteingym_saprot_dataset.sh"
  exit 1
fi
if [[ ! -d "${FLAT_DMS_DIR}" ]]; then
  echo "Missing flat DMS dir: ${FLAT_DMS_DIR}"
  echo "Run: bash ${REPO_ROOT}/script/setup/prepare_proteingym_saprot_dataset.sh"
  exit 1
fi
if [[ ! -f "${CONFIG_JSON}" ]]; then
  echo "Missing config: ${CONFIG_JSON}"
  exit 1
fi
if [[ ! -f "${REF_CSV}" ]]; then
  echo "Missing reference csv: ${REF_CSV}"
  exit 1
fi

mkdir -p "${MERGED_DIR}"

echo "[run_merge_perf] sanity-check score file coverage"
${PY} - <<PY
import json, os
import pandas as pd

cfg_path = "${CONFIG_JSON}"
ref_path = "${REF_CSV}"
scores_root = "${SCORES_ROOT}"

cfg = json.load(open(cfg_path))
models = cfg["model_list_zero_shot_substitutions_DMS"]
ref = pd.read_csv(ref_path)
dms_ids = ref["DMS_id"].tolist()

missing = []
for m, info in models.items():
    loc = info["location"]
    for dms_id in dms_ids:
        p = os.path.join(scores_root, loc, f"{dms_id}.csv")
        if not os.path.exists(p):
            missing.append((m, dms_id, p))

if missing:
    print(f"[run_merge_perf] ERROR: missing {len(missing)} score files. Showing first 30:")
    for m, dms_id, p in missing[:30]:
        print("  ", m, dms_id, "->", p)
    raise SystemExit(2)
print(f"[run_merge_perf] OK: all score files present for {len(models)} models x {len(dms_ids)} assays")
PY

echo "[run_merge_perf] merge.py"
(
  cd "${PROTEINGYM_ROOT}"
  "${PY}" proteingym/merge.py \
    --DMS_assays_location "${FLAT_DMS_DIR}" \
    --model_scores_location "${SCORES_ROOT}" \
    --merged_scores_dir "${MERGED_SCORES_DIR}" \
    --mutation_type "substitutions" \
    --dataset "DMS" \
    --DMS_reference_file "${REF_CSV}" \
    --config_file "${CONFIG_JSON}"
)

echo "[run_merge_perf] sanity-check merged file coverage"
${PY} - <<PY
import os
import pandas as pd

ref_path = "${REF_CSV}"
merged_dir = "${MERGED_DIR}"

ref = pd.read_csv(ref_path)
dms_ids = ref["DMS_id"].tolist()

missing = [dms_id for dms_id in dms_ids if not os.path.exists(os.path.join(merged_dir, f"{dms_id}.csv"))]
if missing:
    print(f"[run_merge_perf] ERROR: missing {len(missing)} merged score files. Showing first 30:")
    for dms_id in missing[:30]:
        print("  ", dms_id, "->", os.path.join(merged_dir, f"{dms_id}.csv"))
    raise SystemExit(3)

sample = dms_ids[0]
df = pd.read_csv(os.path.join(merged_dir, f"{sample}.csv"), nrows=5)
print(f"[run_merge_perf] OK: merged files present for {len(dms_ids)} assays; sample={sample}; ncols={len(df.columns)}")
PY

echo "[run_merge_perf] performance_DMS_benchmarks.py"
(
  cd "${PROTEINGYM_ROOT}"
  "${PY}" proteingym/performance_DMS_benchmarks.py \
    --input_scoring_files_folder "${MERGED_DIR}" \
    --output_performance_file_folder "${EVAL_OUT}" \
    --DMS_reference_file_path "${REF_CSV}" \
    --DMS_data_folder "${FLAT_DMS_DIR}" \
    --performance_by_depth \
    --config_file "${CONFIG_JSON}"
)

echo "[run_merge_perf] done"
