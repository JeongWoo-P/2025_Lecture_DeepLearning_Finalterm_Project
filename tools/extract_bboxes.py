#!/usr/bin/env python3
import os
import json
import pandas as pd

# 1) annotations 폴더 경로
ANNOT_DIR = "/home/jeongwoo/AUE8088/datasets/kaist-rgbt/annotations"

# 2) 결과를 쌓을 리스트
records = []

# 3) 모든 .json 파일 순회
for fname in os.listdir(ANNOT_DIR):
    if not fname.endswith(".json"):
        continue
    path = os.path.join(ANNOT_DIR, fname)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 이미지 ID → 파일명 매핑
    id2name = {img['id']: img['im_name'] for img in data.get('images', [])}
    # annotations 순회
    for ann in data.get('annotations', []):
        img_id = ann['image_id']
        im_name = id2name.get(img_id, "")
        x, y, w, h = ann['bbox']
        records.append({
            "json_file": fname,
            "image_id": img_id,
            "im_name": im_name,
            "category_id": ann.get('category_id'),
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": w,
            "bbox_h": h,
            "occlusion": ann.get('occlusion'),
            "ignore": ann.get('ignore')
        })

# 4) DataFrame 생성 및 CSV 저장
df = pd.DataFrame.from_records(records)
OUT_CSV = "kaist_rgbt_bboxes.csv"
df.to_csv(OUT_CSV, index=False)
print(f"Saved {len(df)} bboxes to {OUT_CSV}")
