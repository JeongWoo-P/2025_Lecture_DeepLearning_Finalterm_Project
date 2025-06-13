#!/usr/bin/env python3
import yaml
from pathlib import Path

# 원본 YAML
SRC = Path("data/kaist-rgbt.yaml")
assert SRC.exists(), f"{SRC} not found"

# 출력 디렉터리
DST_DIR = Path("data")
DST_DIR.mkdir(exist_ok=True)

# 기본 설정 로드
base_cfg = yaml.safe_load(SRC.read_text())
base = base_cfg["path"]    # "datasets/kaist-rgbt"
names = base_cfg["names"]  # 클래스 dict

# train.txt / val.txt 분할 결과에 맞춰 단일 YAML 생성
cfg = {
    "path":  base,
    "train": [f"{base}/train.txt"],
    "val":   [f"{base}/val.txt"],
    "test":  base_cfg.get("test", []),
    "nc":    len(names),
    "names": names
}

out = DST_DIR / "kaist-rgbt-split.yaml"
out.write_text(yaml.dump(cfg, sort_keys=False))
print(f"→ created {out}")
