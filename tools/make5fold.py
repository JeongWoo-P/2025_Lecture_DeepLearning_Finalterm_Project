#!/usr/bin/env python3
import re
from collections import defaultdict

# 파라미터
INPUT_LIST = "/home/jeongwoo/AUE8088/datasets/kaist-rgbt/train-all-04.txt"  # 원본 리스트
NUM_FOLDS = 5

# 1) 파일 읽어들여 그룹별로 묶기
pattern = re.compile(r'(set\d{2}_V\d{3})')
group_to_lines = defaultdict(list)
with open(INPUT_LIST, 'r') as f:
    for line in f:
        line = line.strip()
        m = pattern.search(line)
        if m:
            group = m.group(1)
            group_to_lines[group].append(line)

# 2) 그룹별 개수 확인 & 그룹 리스트 정렬 (큰 것부터)
groups = list(group_to_lines.items())  # [(group, [lines]), ...]
groups.sort(key=lambda x: len(x[1]), reverse=True)

# 3) Greedy 로 그룹을 fold 에 배정 (각 fold 총 이미지 수를 균등하게)
fold_sums = [0] * NUM_FOLDS
fold_groups = [[] for _ in range(NUM_FOLDS)]
for group, lines in groups:
    # 최소 sum 을 가진 fold 에 배정
    i_min = min(range(NUM_FOLDS), key=lambda i: fold_sums[i])
    fold_groups[i_min].append(group)
    fold_sums[i_min] += len(lines)

# 4) 각 fold 별로 train/val 텍스트 파일 생성
for i in range(NUM_FOLDS):
    val_groups = set(fold_groups[i])
    with open(f"val_fold{i+1}.txt", "w") as fv, open(f"train_fold{i+1}.txt", "w") as ft:
        for group, lines in group_to_lines.items():
            # 해당 그룹이 이번 fold 의 validation 에 들어가는지 체크
            target = fv if group in val_groups else ft
            for path in lines:
                target.write(path + "\n")

# 5) 요약 출력
print("=== Fold assignment summary ===")
for i in range(NUM_FOLDS):
    print(f"Fold {i+1}:")
    print("  val groups:", fold_groups[i])
    print("  val images:", sum(len(group_to_lines[g]) for g in fold_groups[i]))
    print("  train images:", sum(len(lines) for grp, lines in group_to_lines.items() if grp not in fold_groups[i]))
    print()
