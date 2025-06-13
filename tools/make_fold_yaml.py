#!/usr/bin/env python3
import yaml
from pathlib import Path

SRC = Path("data/kaist-rgbt.yaml")
assert SRC.exists(), f"{SRC} not found"

base_cfg = yaml.safe_load(SRC.read_text())
base = base_cfg["path"]    # "datasets/kaist-rgbt"
names = base_cfg["names"]  # your classes dict

for fold in range(1,6):
    cfg = {
        "path": base,
        "train": [f"train_fold{fold}.txt"],
        "val":   [f"val_fold{fold}.txt"],
        "test":  base_cfg.get("test", []),
        "nc":    len(names),
        "names": names
    }
    out = Path("data") / f"kaist-rgbt-fold{fold}.yaml"
    out.write_text(yaml.dump(cfg, sort_keys=False))
    print(f"→ created {out}")
