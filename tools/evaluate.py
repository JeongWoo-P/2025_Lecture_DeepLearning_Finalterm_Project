#!/usr/bin/env python3
# ~/AUE8088/tools/evaluate_with_gt.py
"""
KAIST Val ↔ YOLO 예측 결과 평가 스크립트  (mAP@0.50)
─────────────────────────────────────────────────────
Val  : /home/jeongwoo/AUE8088/datasets/kaist-rgbt/annotations/KAIST_annotation.json
PRED: /home/jeongwoo/AUE8088/runs/train/val_train_ver3_foldX_last/epochNone_predictions.json
"""

import json
import numpy as np
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 1. Val 경로 (한 번만 로드)
Val_PATH = Path("/home/jeongwoo/AUE8088/datasets/kaist-rgbt/annotations/KAIST_val_set_annotation.json")

# --------------------------------------------------------------------------- #
# Val 로드 + 누락 필드 보정
# --------------------------------------------------------------------------- #
def load_and_fix_gt(gt_path: Path) -> COCO:
    print(f"[INFO] Loading Val  : {gt_path}")
    coco = COCO(str(gt_path))
    anns = coco.dataset["annotations"]
    modified = False

    for ann in anns:
        if "iscrowd" not in ann:
            ann["iscrowd"] = 0
            modified = True
        if "area" not in ann:
            _, _, w, h = ann["bbox"]
            ann["area"] = float(w * h)
            modified = True

    if modified:
        coco.createIndex()
        print("[INFO] Val ‘iscrowd’/‘area’ 필드 보정 및 인덱스 재생성 완료 ✔")

    return coco

# --------------------------------------------------------------------------- #
# PRED 로드 & 포맷 변환
# --------------------------------------------------------------------------- #
def load_preds_as_coco_results(pred_path: Path) -> list:
    print(f"[INFO] Loading Pred: {pred_path}")
    with open(pred_path, "r") as f:
        preds = json.load(f)

    # KAIST pedestrian: Val class-id = 1  / YOLO 예측 = 0 → 1 매핑
    for det in preds:
        det["category_id"] = 1
        det["bbox"]        = [float(v) for v in det["bbox"]]
        det["score"]       = float(det["score"])
    return preds

# --------------------------------------------------------------------------- #
# 평가 함수 (IoU 단일값 = 0.50)
# --------------------------------------------------------------------------- #
def evaluate(coco_gt: COCO, coco_dt, iou_thr: float = 0.50):
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.iouThrs = np.array([iou_thr])
    coco_eval.params.maxDets = [1, 10, 100]
    coco_eval.params.useCats = 1

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

# --------------------------------------------------------------------------- #
# 메인: fold 1~5 반복
# --------------------------------------------------------------------------- #
def main():
    coco_gt = load_and_fix_gt(Val_PATH)

    for fold in range(1, 6):
        pred_path = Path(f"/home/jeongwoo/AUE8088/runs/train/val_train_ver6_fold{fold}_last/epochNone_predictions.json")
        if not pred_path.exists():
            print(f"[WARN] Predictions not found for fold {fold}: {pred_path}")
            continue

        print(f"\n===== Fold {fold} 평가 시작 =====")
        preds = load_preds_as_coco_results(pred_path)
        coco_dt = coco_gt.loadRes(preds)
        evaluate(coco_gt, coco_dt, iou_thr=0.50)

    for fold in range(1, 6):
        pred_path = Path(f"/home/jeongwoo/AUE8088/runs/train/val_train_ver6_fold{fold}_best/epochNone_predictions.json")
        if not pred_path.exists():
            print(f"[WARN] Predictions not found for fold {fold}: {pred_path}")
            continue

        print(f"\n===== Fold {fold} 평가 시작 =====")
        preds = load_preds_as_coco_results(pred_path)
        coco_dt = coco_gt.loadRes(preds)
        evaluate(coco_gt, coco_dt, iou_thr=0.50)
        
    # 2) 고정 Best 경로 평가
    best_path = Path("/home/jeongwoo/AUE8088/runs/train/val_train_ver3_fold1_best/ctr_Adam.json")
    print(f"\n===== 고정 Best 평가 시작 =====")
    if not best_path.exists():
        print(f"[ERROR] Best predictions not found: {best_path}")
    else:
        preds = load_preds_as_coco_results(best_path)
        coco_dt = coco_gt.loadRes(preds)
        evaluate(coco_gt, coco_dt, iou_thr=0.50)

if __name__ == "__main__":
    main()
