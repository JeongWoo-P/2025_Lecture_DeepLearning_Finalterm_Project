#!/usr/bin/env bash
set -e

BASE_YAML=/home/jeongwoo/AUE8088/data/kaist-rgbt.yaml
DST_DIR=/home/jeongwoo/AUE8088/data
DATASET_PATH=/home/jeongwoo/AUE8088/datasets/kaist-rgbt

for F in {1..5}; do
  OUT_YAML=${DST_DIR}/kaist-rgbt-fold${F}.yaml

  # 1) head(메타)과 classes/blabla 부분은 그대로, 
  # 2) train: 항목만 train_foldF.txt 로, val: 항목만 val_foldF.txt 로 변경
  awk '
    /^train:/ { print "train: [" ENVIRON["DATASET_PATH"] "/train_fold" ENVIRON["F"] ".txt]"; next }
    /^val:/   { print "val:   [" ENVIRON["DATASET_PATH"] "/val_fold" ENVIRON["F"]   ".txt]"; next }
    { print }
  ' DATASET_PATH="$DATASET_PATH" F="$F" "$BASE_YAML" > "$OUT_YAML"

  echo "Generated $OUT_YAML"
done
