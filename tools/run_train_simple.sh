#!/usr/bin/env bash
# Single-split training script for KAIST-RGBT (set00-04 train, set05 val)

# 1) 프로젝트 루트로 이동
cd /home/jeongwoo/AUE8088

# 2) 검증 annotation을 가리키는 심볼릭 링크 갱신
#    val.txt 기준으로 생성한 JSON 파일 경로에 맞춰 수정하세요
ln -sf "../datasets/kaist-rgbt/annotations/KAIST_annotation_updated.json" \
       "utils/eval/KAIST_val-A_annotation.json"

# 3) train_simple.py 실행
python train_simple.py \
  --img 640 \
  --batch-size 4 \
  --epochs 30 \
  --data data/kaist-rgbt-split.yaml \
  --cfg models/yolov5s_kaist-rgbt.yaml \
  --weights yolov5s.pt \
  --workers 2 \
  --name yolov5s-rgbt-set05_hyp \
  --entity "$WANDB_ENTITY" \
  --rgbt \
  --single-cls