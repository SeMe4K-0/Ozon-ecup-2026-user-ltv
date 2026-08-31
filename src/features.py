"""Построение признаков и таргета для одного якоря (anchor date).

Правило без исключений: признаки для якоря A используют только строки с
event_date <= A. Таргет — сумма gmv за [A + 1, A + HORIZON].
"""
from __future__ import annotations

import argparse
import operator
from datetime import date, timedelta
from functools import reduce

import polars as pl

from config import (FEATURE_VERSION, FEATURES_DIR, HORIZON, PREDICT_ANCHOR, SUM_COLS,
                    TRAIN_ANCHOR_STRIDE, TRAIN_PARQUET, VALID_ANCHOR, WINDOWS, train_anchors)

EPS = 1e-6

# понедельные срезы: дают модели форму ряда, а не только суммы по окнам
N_WEEK_SLICES = 12
WEEK_SLICE_COLS = ["gmv", "to_ord", "searches"]


def load_raw() -> pl.DataFrame:
    return pl.read_parquet(TRAIN_PARQUET)


def _all_users(df: pl.DataFrame) -> pl.DataFrame:
    return df.select("user_id").unique().sort("user_id")


def _window_exprs() -> list[pl.Expr]:
    """Агрегаты по скользящим окнам, отсчитанным назад от якоря."""
    exprs: list[pl.Expr] = []
    for w in WINDOWS:
        m = pl.col("db") < w
        for c in SUM_COLS:
            exprs.append(pl.when(m).then(pl.col(c)).otherwise(0).sum().alias(f"{c}_s{w}"))
        exprs += [
            m.sum().alias(f"days_active_{w}"),
            (m & (pl.col("to_ord") > 0)).sum().alias(f"days_ord_{w}"),
            (m & (pl.col("to_cart") > 0)).sum().alias(f"days_cart_{w}"),
            (m & (pl.col("searches") > 0)).sum().alias(f"days_srch_{w}"),
            pl.when(m).then(pl.col("gmv")).otherwise(None).max().alias(f"gmv_max_{w}"),
            pl.when(m).then(pl.col("gmv")).otherwise(None).std().alias(f"gmv_std_{w}"),
            pl.when(m & (pl.col("gmv") > 0)).then(pl.col("gmv")).otherwise(None)
              .mean().alias(f"gmv_mean_pos_{w}"),
            pl.when(m).then(pl.col("searches")).otherwise(None).max().alias(f"srch_max_{w}"),
        ]
    # форма распределения дневных чеков: сейчас есть только max/std/среднее,
    # а квантили отличают «один крупный заказ» от «много мелких»
    for w in (90, 180):
        e = pl.when((pl.col("db") < w) & (pl.col("gmv") > 0)).then(pl.col("gmv"))
        exprs += [
            e.otherwise(None).quantile(0.5).alias(f"gmv_q50_{w}"),
            e.otherwise(None).quantile(0.9).alias(f"gmv_q90_{w}"),
            e.otherwise(None).min().alias(f"gmv_min_pos_{w}"),
        ]
    for k in range(N_WEEK_SLICES):
        m = (pl.col("db") >= k * 7) & (pl.col("db") < (k + 1) * 7)
        for c in WEEK_SLICE_COLS:
            exprs.append(pl.when(m).then(pl.col(c)).otherwise(0).sum().alias(f"{c}_wk{k}"))
        exprs.append(m.sum().alias(f"days_active_wk{k}"))
    # выходные против будней: у части пользователей активность смещена, и это
    # меняет ожидаемое число покупок в окне прогноза
    we = (pl.col("db") < 180) & (pl.col("wd") >= 6)
    wd = (pl.col("db") < 180) & (pl.col("wd") < 6)
    for c in ("gmv", "to_ord"):
        exprs += [
            pl.when(we).then(pl.col(c)).otherwise(0).sum().alias(f"{c}_weekend"),
            pl.when(wd).then(pl.col(c)).otherwise(0).sum().alias(f"{c}_weekday"),
        ]
    return exprs


def _slope_expr(prefix: str, n: int = N_WEEK_SLICES) -> pl.Expr:
    """Наклон линейной регрессии по недельным срезам.

    Срез 0 — самая свежая неделя, поэтому положительный наклон означает,
    что активность выше в прошлом, то есть пользователь угасает.
    """
    kbar = (n - 1) / 2
    denom = sum((k - kbar) ** 2 for k in range(n))
    num = reduce(operator.add,
                 ((k - kbar) * pl.col(f"{prefix}_wk{k}") for k in range(n)))
    return (num / denom).alias(f"slope_{prefix}")


def _gap_features(hist: pl.DataFrame, users: pl.DataFrame) -> pl.DataFrame:
    """BTYD-подобные признаки: статистики интервалов между событиями.

    `db` — число дней до якоря, поэтому сортировка по возрастанию идёт вглубь
    прошлого, а diff даёт положительные интервалы между соседними событиями.
    """
    specs = {
        "ord": pl.col("to_ord") > 0,
        "cart": pl.col("to_cart") > 0,
        "act": pl.lit(True),
    }
    out = users
    for name, cond in specs.items():
        ev = (hist.filter(cond).select("user_id", "db").sort(["user_id", "db"])
                  .with_columns(pl.col("db").diff().over("user_id").alias("gap")))
        st = ev.group_by("user_id").agg([
            pl.col("gap").mean().alias(f"gap_{name}_mean"),
            pl.col("gap").std().alias(f"gap_{name}_std"),
            pl.col("gap").median().alias(f"gap_{name}_med"),
            pl.col("gap").max().alias(f"gap_{name}_max"),
            pl.col("gap").head(3).mean().alias(f"gap_{name}_recent3"),
        ])
        out = out.join(st, on="user_id", how="left")
    return out


def _history_exprs() -> list[pl.Expr]:
    """Агрегаты по всей доступной истории пользователя + recency."""
    return [
        pl.col("db").max().alias("tenure_days"),          # дней с первой активности
        pl.col("db").min().alias("rec_any"),              # дней с последней активности
        pl.when(pl.col("to_ord") > 0).then(pl.col("db")).otherwise(None).min().alias("rec_ord"),
        pl.when(pl.col("to_cart") > 0).then(pl.col("db")).otherwise(None).min().alias("rec_cart"),
        pl.when(pl.col("searches") > 0).then(pl.col("db")).otherwise(None).min().alias("rec_srch"),
        pl.when(pl.col("gmv") > 0).then(pl.col("db")).otherwise(None).max().alias("first_ord_db"),
        pl.len().alias("days_active_all"),
        (pl.col("to_ord") > 0).sum().alias("days_ord_all"),
        pl.sum("gmv").alias("gmv_s_all"),
        pl.sum("to_ord").alias("to_ord_s_all"),
        pl.sum("to_cart").alias("to_cart_s_all"),
        pl.sum("searches").alias("searches_s_all"),
        pl.max("gmv").alias("gmv_max_all"),
        pl.when(pl.col("gmv") > 0).then(pl.col("gmv")).otherwise(None).mean().alias("aov_all"),
    ]


# Признаки, которые несопоставимы между якорями в сыром виде: суммы за всю
# историю зависят от её длины (181 день у раннего якоря против 409 у финального),
# а абсолютные уровни растут вместе с площадкой. Ранг внутри своего якоря
# сопоставим по построению и снимает обе проблемы.
RANK_COLS = [
    "gmv_s30", "gmv_s90", "gmv_s180", "gmv_s_all",
    "to_ord_s30", "to_ord_s90", "to_ord_s180", "to_ord_s_all",
    "to_cart_s30", "searches_s30", "searches_s180",
    "days_ord_90", "days_ord_180", "days_ord_all", "days_active_180",
    "gmv_per_day_all", "to_ord_per_day_all", "aov_90",
    "rec_ord", "rec_any", "tenure_days",
]


def _rank_exprs() -> list[pl.Expr]:
    return [(pl.col(c).rank() / pl.len()).cast(pl.Float32).alias(f"rk_{c}")
            for c in RANK_COLS]


def _safe_div(a: str, b: str, name: str) -> pl.Expr:
    return (pl.col(a) / (pl.col(b) + EPS)).alias(name)


def _derived_exprs() -> list[pl.Expr]:
    """Отношения, конверсии и тренды — считаются поверх оконных сумм."""
    e: list[pl.Expr] = []
    for w in WINDOWS:
        e += [
            _safe_div(f"gmv_s{w}", f"to_ord_s{w}", f"aov_{w}"),              # средний чек
            _safe_div(f"to_ord_s{w}", f"to_cart_s{w}", f"cart2ord_{w}"),     # конверсия корзина->заказ
            _safe_div(f"to_cart_s{w}", f"searches_s{w}", f"srch2cart_{w}"),  # конверсия поиск->корзина
            _safe_div(f"gmv_search_s{w}", f"gmv_s{w}", f"share_gmv_search_{w}"),
            _safe_div(f"to_ord_s{w}", f"days_active_{w}", f"ord_per_active_{w}"),
            _safe_div(f"gmv_s{w}", f"days_active_{w}", f"gmv_per_active_{w}"),
            (pl.col(f"days_active_{w}") / w).alias(f"active_rate_{w}"),
            (pl.col(f"days_ord_{w}") / w).alias(f"ord_rate_{w}"),
        ]
    # тренды: короткое окно против длинного, приведённые к одной длине
    for short, long in [(7, 30), (14, 60), (30, 90), (30, 180), (60, 180), (90, 180)]:
        k = long / short
        for c in ("gmv", "to_ord", "searches", "to_cart"):
            e.append(((pl.col(f"{c}_s{short}") * k) / (pl.col(f"{c}_s{long}") + EPS))
                     .alias(f"trend_{c}_{short}_{long}"))
        e.append(((pl.col(f"days_active_{short}") * k) / (pl.col(f"days_active_{long}") + EPS))
                 .alias(f"trend_days_{short}_{long}"))
    # активность за всю историю, нормированная на длину истории
    for c in ("gmv", "to_ord", "searches", "to_cart"):
        e.append(_safe_div(f"{c}_s_all", "tenure_days", f"{c}_per_day_all"))
    e += [
        _safe_div("days_active_all", "tenure_days", "active_rate_all"),
        _safe_div("days_ord_all", "days_active_all", "ord_day_share_all"),
        (pl.col("gmv_s30") / (pl.col("gmv_s_all") + EPS)).alias("gmv_share_last30_all"),
        (pl.col("first_ord_db") - pl.col("rec_ord")).alias("ord_lifespan"),
        _slope_expr("gmv"), _slope_expr("to_ord"), _slope_expr("searches"),
        _slope_expr("days_active"),
        _safe_div("gmv_weekend", "gmv_s180", "share_gmv_weekend"),
        _safe_div("to_ord_weekend", "to_ord_s180", "share_ord_weekend"),
    ]
    # "просрочка": сколько типичных интервалов прошло с последнего события —
    # ключевой сигнал оттока в BTYD-моделях
    for name, rec in (("ord", "rec_ord"), ("cart", "rec_cart"), ("act", "rec_any")):
        e += [
            (pl.col(rec) / (pl.col(f"gap_{name}_mean") + EPS)).alias(f"overdue_{name}"),
            (pl.col(rec) / (pl.col(f"gap_{name}_med") + EPS)).alias(f"overdue_{name}_med"),
            (pl.col(f"gap_{name}_std") / (pl.col(f"gap_{name}_mean") + EPS)).alias(f"gap_{name}_cv"),
            (pl.col(f"gap_{name}_recent3") / (pl.col(f"gap_{name}_mean") + EPS))
                .alias(f"gap_{name}_accel"),
        ]
    return e


def build_anchor_frame(df: pl.DataFrame, anchor: date, *,
                       with_target: bool = True,
                       all_users: pl.DataFrame | None = None) -> pl.DataFrame:
    """Признаки (+ таргет) всех пользователей для одного якоря."""
    users = all_users if all_users is not None else _all_users(df)
    max_w = max(WINDOWS)

    hist = (df.filter(pl.col("event_date") <= anchor)
              .with_columns(((pl.lit(anchor) - pl.col("event_date"))
                             .dt.total_days()).cast(pl.Int32).alias("db"),
                            pl.col("event_date").dt.weekday().alias("wd")))

    win = (hist.filter(pl.col("db") < max_w)
               .group_by("user_id").agg(_window_exprs()))
    his = hist.group_by("user_id").agg(_history_exprs())

    gaps = _gap_features(hist, users)
    out = (users.join(win, on="user_id", how="left")
                .join(his, on="user_id", how="left")
                .join(gaps, on="user_id", how="left"))

    # Пользователь без активности в окне: суммы и счётчики честно равны нулю.
    # А вот "средний чек" или "макс. дневной gmv" при отсутствии событий не
    # определены — там оставляем null, LightGBM обработает его отдельной веткой.
    keep_null = (tuple(f"{p}{w}" for p in ("gmv_max_", "gmv_std_", "gmv_mean_pos_", "srch_max_")
                       for w in WINDOWS)
                 + tuple(f"{p}{w}" for p in ("gmv_q50_", "gmv_q90_", "gmv_min_pos_")
                         for w in (90, 180))
                 + ("gmv_max_all", "aov_all"))
    zero_cols = [c for c in (win.columns + his.columns)
                 if c != "user_id" and c not in keep_null
                 and not c.startswith("rec_") and c not in ("tenure_days", "first_ord_db")]
    out = out.with_columns([pl.col(c).fill_null(0) for c in zero_cols])
    far = max((anchor - df["event_date"].min()).days, 400)
    out = out.with_columns([pl.col(c).fill_null(far) for c in
                            ("tenure_days", "rec_any", "rec_ord", "rec_cart", "rec_srch")])
    out = out.with_columns(pl.col("first_ord_db").fill_null(-1))

    out = out.with_columns(_derived_exprs())
    out = out.with_columns(_rank_exprs())
    out = out.with_columns(pl.lit(anchor).alias("anchor_date"))

    if with_target:
        t0, t1 = anchor + timedelta(days=1), anchor + timedelta(days=HORIZON)
        tgt = (df.filter(pl.col("event_date").is_between(t0, t1))
                 .group_by("user_id").agg(pl.sum("gmv").alias("target")))
        out = (out.join(tgt, on="user_id", how="left")
                  .with_columns(pl.col("target").fill_null(0.0)))
    return out


def anchor_path(anchor: date, version: str = FEATURE_VERSION) -> "object":
    return FEATURES_DIR / f"{version}_anchor_{anchor.isoformat()}.parquet"


def build_all(anchors: list[date], *, force: bool = False) -> None:
    df = load_raw()
    users = _all_users(df)
    for a in anchors:
        p = anchor_path(a)
        if p.exists() and not force:
            print(f"skip  {a} (есть кэш)")
            continue
        with_target = a + timedelta(days=HORIZON) <= df["event_date"].max()
        frame = build_anchor_frame(df, a, with_target=with_target, all_users=users)
        frame.write_parquet(p)
        print(f"built {a}  rows={frame.height}  cols={frame.width}  target={with_target}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Сборка признаков по якорям")
    ap.add_argument("--force", action="store_true", help="пересобрать даже при наличии кэша")
    ap.add_argument("--stride", type=int, default=TRAIN_ANCHOR_STRIDE)
    args = ap.parse_args()

    anchors = train_anchors(stride=args.stride) + [VALID_ANCHOR, PREDICT_ANCHOR]
    print(f"якорей: {len(anchors)}  ({anchors[0]} .. {anchors[-1]})")
    build_all(anchors, force=args.force)


if __name__ == "__main__":
    main()
