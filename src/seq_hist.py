"""Последовательности со полным охватом истории: 90 дней подневно + 45 недель.

Сеть сейчас видит 180 дней из доступных 409, то есть меньше половины истории.
Растянуть подневное окно мешает память, но полное разрешение глубже и не нужно:
вариант с 90 подневными днями (`seqnet_90`, 1.66715) практически равен варианту
с 180 (`seqnet_big`, 1.66696). Значит глубину лучше добирать агрегатами.

Раскладка (135 шагов вместо 180, то есть памяти даже меньше):
  шаги 0..44    — недельные суммы за дни 90..404 до якоря, от старых к свежим;
  шаги 45..134  — подневные значения за последние 90 дней.

Недельные суммы считаются по сырым величинам и логарифмируются после сложения:
сумма log1p не равна log1p суммы, и агрегировать уже сжатые значения было бы
ошибкой.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta

import numpy as np
import polars as pl

from config import (PREDICT_ANCHOR, SEQ_CHANNELS, SEQ_DIR, SEQ_LOG_CHANNELS, SEQ_SCALE,
                    TRAIN_PARQUET, VALID_ANCHOR, final_anchors, train_anchors)

N_DAILY = 90
N_WEEKLY = 45
HIST_LEN = N_DAILY + N_WEEKLY          # 135
SPAN = N_DAILY + N_WEEKLY * 7          # 405 дней истории


def seq_path(anchor: date) -> "object":
    # число каналов в имени обязательно: без него пересборка на другом наборе
    # молча затирает файлы и ломает уже обученные модели
    suffix = "" if len(SEQ_CHANNELS) == 11 else f"c{len(SEQ_CHANNELS)}_"
    return SEQ_DIR / f"seqh{HIST_LEN}_{suffix}{anchor.isoformat()}.npy"


def _channel_values(d: pl.DataFrame) -> dict[str, np.ndarray]:
    """Сырые значения каналов (без логарифма — он применяется после агрегации)."""
    out = {}
    for name in SEQ_CHANNELS:
        out[name] = (np.ones(d.height, np.float32) if name == "row"
                     else d[name].to_numpy().astype(np.float32))
    return out


def build_anchor(df: pl.DataFrame, anchor: date, users: np.ndarray) -> np.ndarray:
    lo = anchor - timedelta(days=SPAN - 1)
    d = df.filter(pl.col("event_date").is_between(lo, anchor))

    ui = np.searchsorted(users, d["user_id"].to_numpy())
    db = (np.datetime64(anchor) - d["event_date"].to_numpy().astype("datetime64[D]")
          ).astype(np.int32)                      # дней до якоря, 0 = день якоря

    # подневная часть занимает хвост тензора, недельная — начало
    daily = db < N_DAILY
    ti = np.empty(len(db), np.int32)
    ti[daily] = HIST_LEN - 1 - db[daily]
    wk = (db[~daily] - N_DAILY) // 7             # 0 = самая свежая неделя
    ti[~daily] = N_WEEKLY - 1 - np.clip(wk, 0, N_WEEKLY - 1)

    vals = _channel_values(d)
    out = np.zeros((len(users), HIST_LEN, len(SEQ_CHANNELS)), np.float32)
    for c, name in enumerate(SEQ_CHANNELS):
        # накопление именно суммой: в недельных корзинах несколько дней
        np.add.at(out[:, :, c], (ui, ti), vals[name])

    for c, name in enumerate(SEQ_CHANNELS):
        if name in SEQ_LOG_CHANNELS:
            out[:, :, c] = np.log1p(out[:, :, c])
    return np.clip(np.rint(out * SEQ_SCALE), 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description="Последовательности с полной историей")
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    df = pl.read_parquet(TRAIN_PARQUET)
    users = np.sort(df["user_id"].unique().to_numpy())
    anchors = sorted(set(train_anchors(stride=args.stride))
                     | set(final_anchors(stride=args.stride))
                     | {VALID_ANCHOR, PREDICT_ANCHOR})
    print(f"якорей {len(anchors)}, раскладка {N_DAILY} подневно + {N_WEEKLY} недель "
          f"= {HIST_LEN} шагов, охват {SPAN} дней")
    for a in anchors:
        p = seq_path(a)
        if p.exists() and not args.force:
            continue
        arr = build_anchor(df, a, users)
        np.save(p, arr)
        wk_filled = float((arr[:, :N_WEEKLY, SEQ_CHANNELS.index("row")] > 0).mean())
        print(f"built {a}  {arr.nbytes / 2**30:.2f} ГБ  заполнено недельных {wk_filled:.3f}")
        del arr


if __name__ == "__main__":
    main()
