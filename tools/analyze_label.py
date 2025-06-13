#!/usr/bin/env python3
import os
import glob
import pandas as pd

# 1) Settings
LABEL_DIR = "/home/jeongwoo/AUE8088/datasets/kaist-rgbt/train/labels"
OUTPUT_DIR = "/home/jeongwoo/AUE8088/analysis_labels"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2) Counters
from collections import Counter
set_counter = Counter()
v_counter   = Counter()
i_counter   = Counter()
total_person = 0

# 3) Scan files and count only person (class 0)
for path in glob.glob(os.path.join(LABEL_DIR, "*.txt")):
    filename = os.path.basename(path)[:-4]  # remove ".txt"
    try:
        set_id, v_id, i_id = filename.split("_")
    except ValueError:
        continue
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(parts[0])
            if cls == 0:  # person
                total_person += 1
                set_counter[set_id] += 1
                v_counter[v_id]     += 1
                i_counter[i_id]     += 1

# 4) Build DataFrames
def build_df(counter, key_name):
    df = pd.DataFrame.from_records(
        [(k, v) for k, v in counter.items()],
        columns=[key_name, "person_count"]
    ).sort_values(key_name)
    df["percentage"] = df["person_count"] / total_person * 100
    return df

df_set = build_df(set_counter, "set")
df_v   = build_df(v_counter,   "V")
df_i   = build_df(i_counter,   "I")

# 5) Save to CSV
df_set.to_csv(os.path.join(OUTPUT_DIR, "set_person_distribution.csv"), index=False)
df_v  .to_csv(os.path.join(OUTPUT_DIR, "V_person_distribution.csv"),   index=False)
df_i  .to_csv(os.path.join(OUTPUT_DIR, "I_person_distribution.csv"),   index=False)

# 6) Print summary
print(f"Total person annotations: {total_person}\n")
print("Person annotations by setXX:")
print(df_set.to_string(index=False))
print("\nPerson annotations by VXXX:")
print(df_v.to_string(index=False))
print("\nPerson annotations by IXXXXX (first 10):")
print(df_i.head(10).to_string(index=False))
print(f"\nSaved CSVs to {OUTPUT_DIR}/")
