#!/bin/bash
# Полный конвейер: от train.parquet до итогового сабмита.
#
# Данные и последовательности удалены при уборке репозитория (85 ГБ), поэтому
# первый этап обязателен, если нужно что-то переобучить. Если задача — только
# воспроизвести итоговое решение, ничего этого не нужно:
#     python rebuild_final.py
#
# Времена указаны для RTX 3060 (12 ГБ) и 32 ГБ оперативной памяти.
set -euo pipefail
cd "$(dirname "$0")"

# Табличная часть работает на Python 3.14, сети — на 3.11 с CUDA-сборкой torch.
PY=python
PY_GPU="${PY_GPU:-/c/Users/$USER/AppData/Local/Programs/Python/Python311/python.exe}"

export LONG_WINDOWS=1 FEATURE_VERSION=v5

echo "=== 1. Признаки (~35 мин, 31 ГБ) ==="
# Окна 270 и 365 включает LONG_WINDOWS: данных 409 дней, а окна кончались на 180.
$PY features.py --stride 7

echo "=== 2. Последовательности и панель (~40 мин, 54 ГБ) ==="
$PY seq_data.py --stride 7 --n-anchors 25   # ВАЖНО: дефолты (9 якорей, шаг 14) — остаток ранних экспериментов
$PY seq_hist.py --stride 7          # 90 подневных + 45 недельных, охват 405 дней
$PY panel.py                        # плотная панель: все якоря в 1 ГБ

echo "=== 3. Бустинги (~1.5 ч) ==="
# Порядок важен: valid подбирает число итераций, final его использует.
$PY ensemble.py --mode valid --models two_stage catboost
$PY ensemble.py --mode final --models two_stage catboost

echo "=== 4. Сети (~2 ч на модель, valid + final) ==="
# Каждая модель бленда. Флаги — в таблице производителей в README.
for mode in valid final; do
  extra=""; [ "$mode" = final ] && extra="--stop-epoch 7"
  $PY_GPU seq_model.py --mode $mode --stride 7 --n-anchors 22 --epochs 12 --seeds 3 \
      --dist --n-bins 32 --hidden 128 $extra --tag d32v5
  $PY_GPU seq_model.py --mode $mode --stride 7 --n-anchors 22 --epochs 12 --seeds 3 \
      --hidden 128 $extra --tag seqnet_big
  $PY_GPU seq_model.py --mode $mode --stride 7 --n-anchors 22 --epochs 12 --seeds 3 \
      --seq-len 90 --hidden 128 $extra --tag seqnet_90
  $PY_GPU seq_model.py --mode $mode --stride 7 --n-anchors 22 --epochs 12 --seeds 3 \
      --dist --n-bins 32 --hidden 128 --hist $extra --tag hist_dist
  $PY_GPU seq_model.py --mode $mode --stride 7 --n-anchors 22 --epochs 12 --seeds 3 \
      --two-stage --hidden 128 $extra --tag ts_clean
done
$PY_GPU panel_model.py --mode valid --anchor-stride 1 --seq-len 180 --steps 4000 \
    --epochs 10 --seeds 3 --hidden 192 --tag panel180
$PY_GPU panel_model.py --mode final --anchor-stride 1 --seq-len 180 --steps 4000 \
    --epochs 10 --stop-epoch 6 --seeds 3 --hidden 192 --tag panel180

echo "=== 5. Бленд и сабмит ==="
# Уровень 2.3292 и масштаб измерены по лидерборду, а не подобраны на валидации.
# ВАЖНО: при смене состава масштаб пересчитать (METHOD.md, раздел про масштаб).
$PY blend.py --models ts_clean seqnet_big d32v5 hist_dist panel180 seqnet_90 \
    --shrink-weights 1.0 --scale 1.0341 --level 2.3292 \
    --write-submission --name blend_new

echo "=== готово ==="
