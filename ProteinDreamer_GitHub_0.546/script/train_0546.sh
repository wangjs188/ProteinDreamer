#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PY="${PY:-python}"
DEVICE="${DEVICE:-cuda}"
FOLDSEEK_BIN="${FOLDSEEK_BIN:-foldseek}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/ProteinGym_v1_SaProt_AF2_v2}"
FOLDS_DIR="${FOLDS_DIR:-${REPO_ROOT}/data/proteingym_local/folds/PG_AF2_by_protein_k5_seed3407}"
MSA_CACHE_DIR="${MSA_CACHE_DIR:-${REPO_ROOT}/data/proteingym_local/msa_cache_dms_v1}"
TEACHER_CACHE_DIR="${TEACHER_CACHE_DIR:-${REPO_ROOT}/data/proteingym_local/teacher_siterm_dms_s4096_e100_q64_reg05_v2t1}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/model/retrained_0546}"
TRAIN_MODELS="${TRAIN_MODELS:-650m 1p3b}"
FOLDS="${FOLDS:-0 1 2 3 4}"
RUN_EVAL="${RUN_EVAL:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${RUN_EVAL}" == "1" && ("${TRAIN_MODELS}" != "650m 1p3b" || "${FOLDS}" != "0 1 2 3 4") ]]; then
  echo "RUN_EVAL=1 requires both model families and all five folds." >&2
  exit 2
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  test -d "${DATASET_ROOT}/DATASET"
  test -d "${FOLDS_DIR}"
  test -d "${MSA_CACHE_DIR}"
  command -v "${FOLDSEEK_BIN}" >/dev/null
  "${PY}" -B script/setup/verify_benchmark_inputs.py \
    --dataset "${DATASET_ROOT}" \
    --msa-cache "${MSA_CACHE_DIR}" --teacher-cache "${TEACHER_CACHE_DIR}"
  mkdir -p "${OUT_ROOT}"
fi

for family in ${TRAIN_MODELS}; do
  case "${family}" in
    650m)
      plm="${REPO_ROOT}/model/SaProt_650M_AF2"
      tag="wmz_saprot650m_retrain_v2_sd3407_e60_rs1024_pp16384_cache512"
      ;;
    1p3b)
      plm="${REPO_ROOT}/model/SaProt_1.3B_AFDB_OMG_NCBI"
      tag="wmz_saprot1p3b_retrain_v2_sd3407_e60_rs1024_pp16384_cache512"
      ;;
    *)
      echo "Unsupported model family: ${family}; choose 650m and/or 1p3b." >&2
      exit 2
      ;;
  esac
  if [[ "${DRY_RUN}" != "1" ]]; then
    test -f "${plm}/config.json"
    test -f "${plm}/vocab.txt"
  fi
  for fold in ${FOLDS}; do
    command=(
      "${PY}" -B train_wmz_saprot.py
      --dataset-root "${DATASET_ROOT}"
      --folds-dir "${FOLDS_DIR}"
      --fold "${fold}"
      --plm "${plm}"
      --msa-cache-dir "${MSA_CACHE_DIR}"
      --out-dir "${OUT_ROOT}/${tag}"
      --foldseek-bin "${FOLDSEEK_BIN}"
      --struct-root "${DATASET_ROOT}"
      --struct-chain-id A
      --plddt-mask auto
      --plddt-threshold 70.0
      --device "${DEVICE}"
      --seed 3407
      --epochs 60
      --proteins-per-epoch 999999
      --reward-samples-per-protein 1024
      --pairwise-pairs 16384
      --wt-cache-size 512
      --max-rows-per-protein 50000
      --max-eval-rows 0
      --eval-every 1
      --lr 0.0002
      --weight-decay 0.01
      --dropout 0.1
    )
    printf '[train] family=%s fold=%s\n' "${family}" "${fold}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf '%q ' "${command[@]}"
      printf '\n'
    else
      "${command[@]}"
    fi
  done
done

if [[ "${RUN_EVAL}" == "1" && "${DRY_RUN}" != "1" ]]; then
  CKPT_650="${OUT_ROOT}/wmz_saprot650m_retrain_v2_sd3407_e60_rs1024_pp16384_cache512" \
  CKPT_13="${OUT_ROOT}/wmz_saprot1p3b_retrain_v2_sd3407_e60_rs1024_pp16384_cache512" \
  DATASET_ROOT="${DATASET_ROOT}" FOLDS_DIR="${FOLDS_DIR}" \
  MSA_CACHE_DIR="${MSA_CACHE_DIR}" TEACHER_CACHE_DIR="${TEACHER_CACHE_DIR}" \
  SCORES_ROOT="${SCORES_ROOT:-${REPO_ROOT}/data/proteingym_local/retrained_0546_scores}" \
  EVAL_OUT="${EVAL_OUT:-${REPO_ROOT}/result/retrained_0546}" \
  PY="${PY}" DEVICE="${DEVICE}" FOLDSEEK_BIN="${FOLDSEEK_BIN}" \
    bash script/proteingym/reproduce_best_0546.sh
fi

printf '[done] checkpoints: %s\n' "${OUT_ROOT}"
