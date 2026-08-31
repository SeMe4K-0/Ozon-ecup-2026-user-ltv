"""Подневные последовательности поведения пользователя для нейросети.

Табличная модель видит только агрегаты по окнам, и все её варианты упёрлись в
плато с корреляцией предсказаний 0.997. Сеть получает сырой подневный ряд —
это единственный источник по-настоящему другой модели, а значит и единственный
способ получить реальный выигрыш от ансамбля.

Для каждого якоря сохраняется тензор (n_users, SEQ_LEN, n_channels) в float16:
последний день по времени = день якоря. Дни без строк остаются нулями, а факт
визита без действий отмечает отдельный канал `row`.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta

import numpy as np
import polars as pl

from config import (PREDICT_ANCHOR, SEQ_CHANNEL_SET, SEQ_CHANNELS, SEQ_DIR, SEQ_LEN,
                    SEQ_LOG_CHANNELS, SEQ_SCALE, TRAIN_PARQUET, VALID_ANCHOR, train_anchors)


def seq_path(anchor: date, quantized: bool = True) -> "object":
    prefix = f"seq8{SEQ_CHANNEL_SET}" if quantized else f"seq{SEQ_CHANNEL_SET}"
    return SEQ_DIR / f"{prefix}_{anchor.isoformat()}.npy"


def build_anchor(df: pl.DataFrame, anchor: date, users: np.ndarray,
                 quantized: bool = True) -> np.ndarray:
    lo = anchor - timedelta(days=SEQ_LEN - 1)
    d = df.filter(pl.col("event_date").is_between(lo, anchor))

    ui = np.searchsorted(users, d["user_id"].to_numpy())
    ti = (d["event_date"].to_numpy().astype("datetime64[D]")
          - np.datetime64(lo)).astype(np.int32)

    dtype = np.uint8 if quantized else np.float16
    out = np.zeros((len(users), SEQ_LEN, len(SEQ_CHANNELS)), dtype)
    for c, name in enumerate(SEQ_CHANNELS):
        if name == "row":
            v = np.ones(d.height, np.float32)
        else:
            v = d[name].to_numpy().astype(np.float32)
            if name in SEQ_LOG_CHANNELS:
                v = np.log1p(v)
        out[ui, ti, c] = (np.clip(np.rint(v * SEQ_SCALE), 0, 255).astype(np.uint8)
                          if quantized else v.astype(np.float16))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Сборка подневных последовательностей")
    ap.add_argument("--n-anchors", type=int, default=9,
                    help="сколько последних обучающих якорей брать")
    ap.add_argument("--stride", type=int, default=14)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    df = pl.read_parquet(TRAIN_PARQUET)
    users = np.sort(df["user_id"].unique().to_numpy())
    anchors = train_anchors(stride=args.stride)[-args.n_anchors:] + [VALID_ANCHOR, PREDICT_ANCHOR]
    print(f"якорей: {len(anchors)} ({anchors[0]} .. {anchors[-1]}), "
          f"форма тензора ({len(users)}, {SEQ_LEN}, {len(SEQ_CHANNELS)})")

    for a in anchors:
        p = seq_path(a)
        if p.exists() and not args.force:
            print(f"skip  {a}")
            continue
        arr = build_anchor(df, a, users)
        np.save(p, arr)
        nz = float((arr[:, :, SEQ_CHANNELS.index("row")] > 0).mean())
        print(f"built {a}  {arr.nbytes / 2**30:.2f} ГБ  доля активных дней {nz:.3f}")
        del arr
    np.save(SEQ_DIR / "users.npy", users)


if __name__ == "__main__":
    main()
