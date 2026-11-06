#!/usr/bin/env bash
set -euo pipefail

# Full 0.546 scoring pipeline; prepare the inputs listed in README.md first.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PY="${PY:-python}"
DEVICE="${DEVICE:-cuda}"
PLL_BATCH_SIZE="${PLL_BATCH_SIZE:-1}"
FOLDSEEK_BIN="${FOLDSEEK_BIN:-foldseek}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/ProteinGym_v1_SaProt_AF2_v2}"
FOLDS_DIR="${FOLDS_DIR:-${REPO_ROOT}/data/proteingym_local/folds/PG_AF2_by_protein_k5_seed3407}"
MSA_CACHE_DIR="${MSA_CACHE_DIR:-${REPO_ROOT}/data/proteingym_local/msa_cache_dms_v1}"
TEACHER_CACHE_DIR="${TEACHER_CACHE_DIR:-${REPO_ROOT}/data/proteingym_local/teacher_siterm_dms_s4096_e100_q64_reg05_v2t1}"
FLAT_DMS_DIR="${FLAT_DMS_DIR:-${REPO_ROOT}/data/proteingym_local/DMS_ProteinGym_substitutions_v2}"
PROTEINGYM_ROOT="${PROTEINGYM_ROOT:-${REPO_ROOT}/external/ProteinGym-main}"
REF_CSV="${REF_CSV:-${REPO_ROOT}/data/proteingym_local/reference/DMS_substitutions.csv}"
SCORES_ROOT="${SCORES_ROOT:-${REPO_ROOT}/data/proteingym_local/zero_shot_substitutions_scores}"
EVAL_OUT="${EVAL_OUT:-${REPO_ROOT}/result/proteingym_eval_best_0546}"

CKPT_650="${CKPT_650:-${REPO_ROOT}/model/world_model_zeroshot/wmz_saprot650m_retrain_v2_sd3407_e60_rs1024_pp16384_cache512}"
CKPT_13="${CKPT_13:-${REPO_ROOT}/model/world_model_zeroshot/wmz_saprot1p3b_retrain_v2_sd3407_e60_rs1024_pp16384_cache512}"
PLM_650="${REPO_ROOT}/model/SaProt_650M_AF2"
PLM_13="${REPO_ROOT}/model/SaProt_1.3B_AFDB_OMG_NCBI"

BUNDLE_650_NAME="ProteinDreamer_WMZ_SiteRM_bundle_wmz_saprot650m_retrain_v2_sd3407_e60_rs1024_pp16384_cache512"
BUNDLE_13_NAME="ProteinDreamer_WMZ_SiteRM_bundle_wmz_saprot1p3b_retrain_v2_sd3407_e60_rs1024_pp16384_cache512"
FUSION_NAME="ProteinDreamer_WMZ_SiteRM_fusion_neffgate_wmz_saprot_plmens2_sd3407_e60_rs1024_pp16384_cache512"
BASELINE_NAME="ProteinDreamer_SaProt_650M1p3B_AF2only"
ADD_NAME="ProteinDreamer_WMZ_SiteRM_SaProt_add_beta_sweep_plmens2_AF2only_sd3407_e60_rs1024_pp16384_cache512"

export PY DEVICE FOLDSEEK_BIN DATASET_ROOT FOLDS_DIR MSA_CACHE_DIR TEACHER_CACHE_DIR
export FLAT_DMS_DIR PROTEINGYM_ROOT REF_CSV
STRICT=1 bash script/setup/verify_best_0546_ready.sh
# Require fresh output directories to avoid mixing runs.
for name in "${BUNDLE_650_NAME}" "${BUNDLE_13_NAME}" \
            "${FUSION_NAME}" "${BASELINE_NAME}" "${ADD_NAME}"; do
  if [[ -e "${SCORES_ROOT}/${name}" ]]; then
    echo "Output exists: ${SCORES_ROOT}/${name}; select a new SCORES_ROOT." >&2
    exit 2
  fi
done
if [[ -e "${EVAL_OUT}" ]]; then
  echo "Output exists: ${EVAL_OUT}; select a new EVAL_OUT." >&2
  exit 2
fi
mkdir -p "${SCORES_ROOT}" "${EVAL_OUT}"

score_family() {
  local plm="$1"
  local ckpt="$2"
  local bundle_name="$3"
  local fold
  for fold in 0 1 2 3 4; do
    "${PY}" script/proteingym/score_wmz_siterm_fusion_cv5.py \
      --seed 3407 \
      --device "${DEVICE}" \
      --dataset_root "${DATASET_ROOT}" \
      --struct_root "${DATASET_ROOT}" \
      --folds_dir "${FOLDS_DIR}" \
      --fold "${fold}" \
      --ckpt_dir "${ckpt}" \
      --plm "${plm}" \
      --backbone saprot \
      --saprot_scoring fast \
      --foldseek_bin "${FOLDSEEK_BIN}" \
      --teacher_cache_dir "${TEACHER_CACHE_DIR}" \
      --alphas 0.10 \
      --msa_cache_dir "${MSA_CACHE_DIR}" \
      --msa_feat_mode full \
      --msa_gate_hidden 64 \
      --msa_gate_init_bias -10.0 \
      --msa_gate_dropout 0.0 \
      --chunk_size 32768 \
      --out_dir "${SCORES_ROOT}/${bundle_name}" \
      --location_name "${bundle_name}"
  done
}

echo "[1/7] score held-out CV5 WMZ/SiteRM bundles"
score_family "${PLM_650}" "${CKPT_650}" "${BUNDLE_650_NAME}"
score_family "${PLM_13}" "${CKPT_13}" "${BUNDLE_13_NAME}"

echo "[2/7] fuse each WMZ branch with SiteRM before averaging"
"${PY}" script/proteingym/build_b130_fusion.py \
  --bundle650 "${SCORES_ROOT}/${BUNDLE_650_NAME}" \
  --bundle13 "${SCORES_ROOT}/${BUNDLE_13_NAME}" \
  --reference "${REF_CSV}" --out "${SCORES_ROOT}/${FUSION_NAME}"
echo "[3/7] historical fused-score ensemble complete"

echo "[4/7] score the external SaProt PLL ensemble"
"${PY}" zeroshot_predict.py \
  --device "${DEVICE}" \
  --backbone saprot \
  --saprot_scoring pll \
  --ensemble_dtype float32 \
  --pll_batch_size "${PLL_BATCH_SIZE}" \
  --foldseek_bin "${FOLDSEEK_BIN}" \
  --saprot_models "${PLM_650}" "${PLM_13}" \
  --mutant_dataset_dir "${DATASET_ROOT}" \
  --mutant_score_col DMS_score \
  --result_dir "${SCORES_ROOT}/${BASELINE_NAME}"

echo "[5/7] add beta=0.13 SaProt residual"
"${PY}" script/proteingym/build_wmz_siterm_baseline_add_sweep_from_dirs.py \
  --wmz_dir "${SCORES_ROOT}/${FUSION_NAME}" \
  --baseline_dir "${SCORES_ROOT}/${BASELINE_NAME}" \
  --out_dir "${SCORES_ROOT}/${ADD_NAME}" \
  --ref_csv "${REF_CSV}" \
  --wmz_col ProteinDreamer_WMZ_SiteRM_fusion_raw_neffgate_la350_ma300_ha250_gneg050 \
  --baseline_col SaProt_ensemble \
  --betas 0.13 \
  --out_prefix ProteinDreamer_WMZ_SiteRM_SaProt_add_neffgate_wmz_saprot_plmens2_AF2only \
  --overwrite

echo "[6/7] run official ProteinGym evaluation"
"${PY}" script/setup/verify_score_coverage.py \
  --reference "${REF_CSV}" --flat-dms "${FLAT_DMS_DIR}" \
  --scores-root "${SCORES_ROOT}" \
  --config "${REPO_ROOT}/script/proteingym/protein_dreamer_proteingym_config_best_0546.json"
PROTEINGYM_ROOT="${PROTEINGYM_ROOT}" \
REF_CSV="${REF_CSV}" \
DATASET_ROOT="${DATASET_ROOT}" \
FLAT_DMS_DIR="${FLAT_DMS_DIR}" \
SCORES_ROOT="${SCORES_ROOT}" \
MERGED_SCORES_DIR="merged_scores/best_0546" \
CONFIG_JSON="${REPO_ROOT}/script/proteingym/protein_dreamer_proteingym_config_best_0546.json" \
EVAL_OUT="${EVAL_OUT}" \
PY="${PY}" \
bash script/proteingym/run_merge_perf.sh

echo "[7/7] paired bootstrap against the SaProt baseline"
"${PY}" script/proteingym/paired_bootstrap_official.py \
  --dms_level_csv "${EVAL_OUT}/Spearman/DMS_substitutions_Spearman_DMS_level.csv" \
  --model_a ProteinDreamer_final_lite_b130 \
  --model_b ProteinDreamer_SaProt_650M1p3B_AF2only \
  --bootstrap_n 20000 \
  --seed 0 \
  --out_csv "${EVAL_OUT}/paired_bootstrap_final_vs_baseline.csv"

echo "[done] summary: ${EVAL_OUT}/Spearman/Summary_performance_DMS_substitutions_Spearman.csv"
