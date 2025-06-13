# 툴 디렉터리(프로젝트 루트 기준)로 이동
cd /home/jeongwoo/AUE8088

for F in {1..5}; do
  # 1) utils/eval/KAIST_val-A_annotation.json 을
  #    각 fold의 JSON으로 가리키는 심볼릭 링크로 갱신
  ln -sf "../../datasets/kaist-rgbt/annotations/KAIST_val_fold${F}_annotation.json" \
        "utils/eval/KAIST_val-A_annotation.json"

  # 2) 그 다음 train_simple 돌리기
  python train_simple.py \
    --img 640 --batch-size 4 --epochs 20 \
    --data data/kaist-rgbt-fold${F}.yaml \
    --cfg models/yolov5s_kaist-rgbt.yaml \
    --weights yolov5s.pt \
    --workers 2 \
    --name "yolov5s-rgbt-fold${F}_ver4" \
    --entity "$WANDB_ENTITY" \
    --rgbt --single-cls
done
