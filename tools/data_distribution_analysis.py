#!/usr/bin/env python3
import re
from collections import Counter

# 1. 모을 패턴 정의: set00_V000, set01_V123 같은 형태
pattern = re.compile(r'set\d{2}_V\d{3}')

# 2. Counter 객체 생성
counter = Counter()

# 3. 파일 한 줄씩 읽으면서 패턴 검색 후 카운트
with open('./datasets/kaist-rgbt/train-all-04.txt', 'r') as f:
    for line in f:
        line = line.strip()
        m = pattern.search(line)
        if m:
            counter[m.group()] += 1

# 4. 결과 출력
print("=== setXX_VYYY 별 이미지 개수 ===")
for key, cnt in sorted(counter.items()):
    print(f"{key}: {cnt}")
