#!/usr/bin/env python3
import glob
import os
from pathlib import Path

# 1) 설정
LABEL_DIR = "datasets/kaist-rgbt/train/labels"   # 라벨 폴더
OUT_DIR   = "datasets/kaist-rgbt"                # TXT 파일 저장 폴더
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_TXT = Path(OUT_DIR) / "train.txt"
VAL_TXT   = Path(OUT_DIR) / "val.txt"

# 2) 모든 라벨 파일 순회
train_files = []
val_files   = []

for label_path in glob.glob(f"{LABEL_DIR}/*.txt"):
    fname = os.path.basename(label_path)[:-4]   # e.g. "set00_V000_I00003"
    set_id, *_ = fname.split("_")              # "set00"

    # 3) literal '{}' placeholder 사용
    img_path = f"datasets/kaist-rgbt/train/images/{{}}/{fname}.jpg"

    # 4) set05 는 val, 나머지는 train
    if set_id == "set05":
        val_files.append(img_path)
    else:
        train_files.append(img_path)

# 5) 파일 쓰기
with open(TRAIN_TXT, "w") as ft:
    for p in sorted(train_files):
        ft.write(p + "\n")

with open(VAL_TXT, "w") as fv:
    for p in sorted(val_files):
        fv.write(p + "\n")

# 6) 요약 출력
print(f"Train images: {len(train_files)}  -> {TRAIN_TXT}")
print(f"  Val images: {len(val_files)}  -> {VAL_TXT}")
