#!/usr/bin/env bash
set -e

# 1) 스크립트 디렉터리 (tools)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# 2) 프로젝트 루트
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# 3) 경로 설정
TXT_FILE="${BASE_DIR}/datasets/kaist-rgbt/val.txt"
XML_DIR="${BASE_DIR}/datasets/kaist-rgbt/train/labels-xml"
OUT_DIR="${BASE_DIR}/datasets/kaist-rgbt/annotations"
GEN_SCRIPT="${SCRIPT_DIR}/../utils/eval/generate_kaist_ann_json.py"

# 4) 출력 디렉터리 생성
mkdir -p "${OUT_DIR}"

# 5) JSON 생성
echo ">>> Generating set05 annotation JSON from ${TXT_FILE}"
python "${GEN_SCRIPT}" \
  --textListFile "${TXT_FILE}" \
  --xmlAnnDir "${XML_DIR}" \
  --jsonAnnFile "${OUT_DIR}/KAIST_val_set05_annotation.json"

# 6) 완료 메시지
echo "✅ Generated ${OUT_DIR}/KAIST_val_set05_annotation.json"