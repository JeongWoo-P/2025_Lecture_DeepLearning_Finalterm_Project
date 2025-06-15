#!/usr/bin/env bash
set -euo pipefail

# 프로젝트 루트로 이동
cd /home/jeongwoo/AUE8088

# folds 1~5, 그리고 두 가지 체크포인트 버전(best, last)을 순차 실행
for F in {1..5}; do
  for CKPT in best last; do
    NAME="val_train_ver6_fold${F}_${CKPT}"
    WEIGHTS="/home/jeongwoo/AUE8088/runs/train/yolov5n-rgbt-fold${F}_ver6/weights/${CKPT}.pt"
    echo ">>> Running fold ${F} (${CKPT}) → --name ${NAME}"
    python val_custom.py \
      --img 640 \
      --batch-size 32 \
      --data data/kaist-rgbt.yaml \
      --cfg models/yolov5n_kaist-rgbt.yaml \
      --workers 4 \
      --name "${NAME}" \
      --entity "$WANDB_ENTITY" \
      --rgbt \
      --single-cls \
      --weights "${WEIGHTS}"
  done
done
