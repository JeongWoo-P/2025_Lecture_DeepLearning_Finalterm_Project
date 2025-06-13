#!/usr/bin/env python3
# ~/AUE8088/tools/evaluate_with_gt.py
"""
KAIST GT ↔ YOLO 예측 결과 평가 스크립트  (mAP@0.50)
─────────────────────────────────────────────────────
GT  : /home/jeongwoo/AUE8088/datasets/kaist-rgbt/annotations/KAIST_annotation.json
PRED: /home/jeongwoo/AUE8088/runs/train/val_train5_yolo_s_iou_down_best.pt/epochNone_predictions.json
"""

import json
import numpy as np
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


GT_PATH   = Path("/home/jeongwoo/AUE8088/datasets/kaist-rgbt/annotations/KAIST_annotation.json")
PRED_PATH = Path("/home/jeongwoo/AUE8088/runs/train/val_train_fold2_last/epochNone_predictions.json")


# --------------------------------------------------------------------------- #
# 1. GT 로드 + 누락 필드 보정
# --------------------------------------------------------------------------- #
def load_and_fix_gt(gt_path: Path) -> COCO:
    print(f"[INFO] Loading GT  : {gt_path}")
    coco = COCO(str(gt_path))                  # ➊ 일단 로드
    anns = coco.dataset["annotations"]
    modified = False

    for ann in anns:
        # (1) iscrowd 없으면 기본값 0
        if "iscrowd" not in ann:
            ann["iscrowd"] = 0
            modified = True
        # (2) area 없으면 bbox[2]*bbox[3]로 계산
        if "area" not in ann:
            _, _, w, h = ann["bbox"]
            ann["area"] = float(w * h)
            modified = True

    # 수정했다면 인덱스 재생성
    if modified:
        coco.createIndex()
        print("[INFO] GT ‘iscrowd’/‘area’ 필드 보정 및 인덱스 재생성 완료 ✔")

    return coco


# --------------------------------------------------------------------------- #
# 2. 예측 JSON  →  COCOeval 포맷
# --------------------------------------------------------------------------- #
def load_preds_as_coco_results(pred_path: Path) -> list:
    print(f"[INFO] Loading Pred: {pred_path}")
    with open(pred_path, "r") as f:
        preds = json.load(f)

    # KAIST pedestrian: GT class-id = 1  / YOLO 예측 = 0 → 1 매핑
    for det in preds:
        det["category_id"] = 1
        det["bbox"]  = [float(v) for v in det["bbox"]]
        det["score"] = float(det["score"])
    return preds


# --------------------------------------------------------------------------- #
# 3. 평가 함수 (IoU 단일값 = 0.50)
# --------------------------------------------------------------------------- #
def evaluate(coco_gt: COCO, coco_dt, iou_thr: float = 0.50):
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.iouThrs = np.array([iou_thr])
    coco_eval.params.maxDets = [1, 10, 100]
    coco_eval.params.useCats = 1

    coco_eval.evaluate()
    coco_eval.accumulate()

    print(f"\n===== KAIST Pedestrian 평가 결과 (IoU ≥ {iou_thr:.2f}) =====")
    coco_eval.summarize()


# --------------------------------------------------------------------------- #
# 4. 메인
# --------------------------------------------------------------------------- #
def main():
    coco_gt = load_and_fix_gt(GT_PATH)
    preds   = load_preds_as_coco_results(PRED_PATH)
    coco_dt = coco_gt.loadRes(preds)
    evaluate(coco_gt, coco_dt, iou_thr=0.50)


if __name__ == "__main__":
    main()
