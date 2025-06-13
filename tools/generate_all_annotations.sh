#!/usr/bin/env bash
set -e

# 스크립트 디렉터리 (/home/jeongwoo/AUE8088/tools)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# 프로젝트 루트 (/home/jeongwoo/AUE8088)
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# 각종 경로 설정
TXT_DIR="${BASE_DIR}/datasets/kaist-rgbt"
XML_DIR="${TXT_DIR}/train/labels-xml"
OUT_DIR="${BASE_DIR}/datasets/kaist-rgbt/annotations"
GEN_SCRIPT="${SCRIPT_DIR}/../utils/eval/generate_kaist_ann_json.py"

# 출력 디렉터리 생성
mkdir -p "${OUT_DIR}"

for fold in {1..5}; do
  TEXT_FILE="${TXT_DIR}/val_fold${fold}.txt"
  JSON_FILE="${OUT_DIR}/KAIST_val_fold${fold}_annotation.json"

  echo ">>> Generating annotation for fold ${fold}"
  python "${GEN_SCRIPT}" \
    --textListFile "${TEXT_FILE}" \
    --xmlAnnDir "${XML_DIR}" \
    --jsonAnnFile "${JSON_FILE}"
done

echo "✅ All folds annotation JSON generated in ${OUT_DIR}"
