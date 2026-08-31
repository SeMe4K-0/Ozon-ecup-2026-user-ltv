"""Плотная подневная панель: одна матрица вместо перекрывающихся окон.

Раньше последовательности хранились отдельным файлом на каждый якорь. Но окна
соседних якорей перекрываются почти полностью, и одни и те же дни лежали на
диске по двадцать раз: 22 файла по 495 МБ, 11 ГБ в памяти.

Вся панель целиком — это (250000 пользователей x 409 дней x 11 каналов) в uint8,
то есть **1.05 ГБ**. Именно память диктовала выбор 22 якорей с шагом 7 и мешала
поднять ёмкость модели. Плотное хранение снимает ограничение полностью: окно для
любого якоря нарезается на лету, и доступны все ~200 якорей с шагом 1.

Хранится также подневный gmv в float32 — по нему считается таргет для любого
якоря без обращения к исходному parquet.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta

import numpy as np
import polars as pl

from config import (DATA_END, DATA_START, HORIZON, MIN_HISTORY_DAYS, SEQ_CHANNELS,
                    SEQ_DIR, SEQ_LOG_CHANNELS, SEQ_SCALE, TRAIN_PARQUET)

N_DAYS = (DATA_END - DATA_START).days + 1          # 409
PANEL_PATH = SEQ_DIR / f"panel_u8_{len(SEQ_CHANNELS)}ch.npy"
RAW_PATH = SEQ_DIR / f"panel_raw_{len(SEQ_CHANNELS)}ch.npy"
GMV_PATH = SEQ_DIR / "panel_gmv.npy"
USERS_PATH = SEQ_DIR / "panel_users.npy"


def day_index(d: date) -> int:
    return (d - DATA_START).days


def build() -> None:
    df = pl.read_parquet(TRAIN_PARQUET)
    users = np.sort(df["user_id"].unique().to_numpy())
    ui = np.searchsorted(users, df["user_id"].to_numpy())
    di = (df["event_date"].to_numpy().astype("datetime64[D]")
          - np.datetime64(DATA_START)).astype(np.int32)

    panel = np.zeros((len(users), N_DAYS, len(SEQ_CHANNELS)), np.uint8)
    for c, name in enumerate(SEQ_CHANNELS):
        v = (np.ones(df.height, np.float32) if name == "row"
             else df[name].to_numpy().astype(np.float32))
        if name in SEQ_LOG_CHANNELS:
            v = np.log1p(v)
        # одна строка на пару (пользователь, день), поэтому присваивание, не сложение
        panel[ui, di, c] = np.clip(np.rint(v * SEQ_SCALE), 0, 255).astype(np.uint8)

    gmv = np.zeros((len(users), N_DAYS), np.float32)
    gmv[ui, di] = df["gmv"].to_numpy().astype(np.float32)

    # Сырые величины нужны недельным агрегатам: сумма log1p не равна log1p суммы,
    # поэтому складывать надо до логарифмирования, а панель хранит уже логарифмы.
    raw = np.zeros((len(users), N_DAYS, len(SEQ_CHANNELS)), np.float16)
    for c, name in enumerate(SEQ_CHANNELS):
        v = (np.ones(df.height, np.float32) if name == "row"
             else df[name].to_numpy().astype(np.float32))
        raw[ui, di, c] = v
    np.save(RAW_PATH, raw)
    print(f"сырая  {raw.shape} = {raw.nbytes / 2**30:.2f} ГБ")

    np.save(PANEL_PATH, panel)
    np.save(GMV_PATH, gmv)
    np.save(USERS_PATH, users)
    print(f"панель {panel.shape} = {panel.nbytes / 2**30:.2f} ГБ")
    print(f"gmv    {gmv.shape} = {gmv.nbytes / 2**30:.2f} ГБ")


def load(mmap: bool = True):
    mode = "r" if mmap else None
    return (np.load(PANEL_PATH, mmap_mode=mode),
            np.load(GMV_PATH, mmap_mode=mode),
            np.load(USERS_PATH))


def window(panel, anchor_day: int, rows: np.ndarray, seq_len: int) -> np.ndarray:
    """Окно [anchor_day - seq_len + 1, anchor_day] для выбранных строк."""
    lo = anchor_day - seq_len + 1
    if lo < 0:
        raise ValueError(f"якорь {anchor_day} короче окна {seq_len}")
    return panel[rows, lo:anchor_day + 1, :]


def window_hist(panel, raw, anchor_day: int, rows: np.ndarray,
                n_daily: int = 90, n_weekly: int = 45) -> np.ndarray:
    """Подневные последние n_daily дней + недельные суммы за более раннее.

    Приём взят у `seq_hist`: подневная глубина сверх 90 дней почти не помогает
    (seqnet_90 1.66715 против seqnet_big 1.66696), а агрегаты дают охват 405 дней
    за 135 шагов вместо 180. Здесь он соединён с панелью: у `hist_dist` было
    22 якоря из parquet-сетки, тут доступны все.

    Окно свободно уходит левее начала данных и добивается нулями — состав
    пользователей зафиксирован с DATA_START, поэтому отсутствие истории
    это сведение о пользователе, а не пропуск.
    """
    n_ch = panel.shape[2]
    out = np.zeros((len(rows), n_weekly + n_daily, n_ch), np.float32)

    d_lo = anchor_day - n_daily + 1
    a, b = max(d_lo, 0), anchor_day + 1
    if b > a:
        out[:, n_weekly + (a - d_lo):n_weekly + (b - d_lo)] = (
            panel[rows, a:b, :].astype(np.float32) / SEQ_SCALE)

    for k in range(n_weekly):                      # k=0 — самая старая неделя
        hi = d_lo - (n_weekly - 1 - k) * 7
        lo = hi - 7
        a, b = max(lo, 0), max(min(hi, N_DAYS), 0)
        if b > a:
            out[:, k] = np.log1p(raw[rows, a:b, :].astype(np.float32).sum(1))
    return out


def target(gmv, anchor_day: int, rows: np.ndarray, horizon: int = HORIZON) -> np.ndarray:
    """Сумма gmv за [anchor_day + 1, anchor_day + horizon]."""
    return gmv[rows, anchor_day + 1:anchor_day + 1 + horizon].sum(1)


def valid_anchor_days(seq_len: int, horizon: int = HORIZON,
                      min_history: int = MIN_HISTORY_DAYS) -> np.ndarray:
    """Якоря, у которых хватает истории слева и наблюдаемого таргета справа."""
    lo = max(seq_len, min_history) - 1
    hi = N_DAYS - 1 - horizon
    return np.arange(lo, hi + 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Сборка плотной подневной панели")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if PANEL_PATH.exists() and not args.force:
        p, g, u = load()
        print(f"уже собрана: {p.shape}, {p.nbytes / 2**30:.2f} ГБ")
    else:
        build()
    days = valid_anchor_days(180)
    print(f"доступных якорей с шагом 1: {len(days)} "
          f"({DATA_START + timedelta(days=int(days[0]))} .. "
          f"{DATA_START + timedelta(days=int(days[-1]))})")


if __name__ == "__main__":
    main()
