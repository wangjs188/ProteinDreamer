#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="${1:-data/proteingym_downloads}"
BASE_URL="${BASE_URL:-https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3}"
mkdir -p "${OUT_DIR}/raw"
archives=(DMS_ProteinGym_substitutions.zip ProteinGym_AF2_structures.zip)
if [[ "${DOWNLOAD_MSA:-0}" == "1" ]]; then
  archives+=(DMS_msa_files.zip DMS_msa_weights.zip)
fi
for name in "${archives[@]}"; do
  archive="${OUT_DIR}/${name}"
  if [[ ! -f "${archive}" ]]; then
    # Failed HTTP responses must never be saved as successful archives.
    curl --fail --location --retry 3 --connect-timeout 30 \
      "${BASE_URL}/${name}" -o "${archive}.part"
    unzip -tq "${archive}.part"
    mv -n "${archive}.part" "${archive}"
  fi
  unzip -tq "${archive}"
  unzip -q -n "${archive}" -d "${OUT_DIR}/raw"
done
echo "Downloaded official v1.3 resources to ${OUT_DIR}/raw."
echo "Next: bash script/setup/prepare_proteingym_saprot_dataset.sh ${OUT_DIR}/raw"
