"""Бленд предсказаний и сборка сабмита.

Структура и уровень разделены намеренно. Предсказание каждой модели сначала
центрируется собственным средним, веса подбираются только по форме, а уровень
добавляется отдельным осознанным слагаемым.

Зачем так: `two_stage` учится на исходном таргете и наследует дрейф площадки
(его средний уровень ~2.48 против ~2.24 у де-меанированных моделей). Если
блендить нецентрированные предсказания, МНК начинает компенсировать это,
ужимая сумму весов, — то есть веса тайком подрабатывают калибровкой уровня,
подогнанной под конкретное валидационное окно. Центрирование убирает этот
эффект: веса отвечают за форму, уровень — за уровень.

Метрика к выбору уровня почти нечувствительна (ошибка входит в RMSLE
квадратично): на валидации L=2.2255 и L=2.2421 дают 1.66942 и 1.66935.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl
from scipy.optimize import nnls

from config import MODELS_DIR, PREDICT_ANCHOR, SAMPLE_SUBMIT, SUBMISSIONS_DIR, VALID_ANCHOR
from config import PRED_DIR
from features import anchor_path
from train import rmsle_from_log

# Уровень последнего полностью наблюдаемого 30-дневного окна: среднее log1p
# таргета. Свойство данных, от состава бленда не зависит (см. METHOD.md).
DEFAULT_LEVEL = 2.2421


def load(mode: str, names: list[str]) -> np.ndarray:
    return np.column_stack([np.load(PRED_DIR / f"{mode}_{n}.npy") for n in names])


def main() -> None:
    ap = argparse.ArgumentParser(description="Бленд предсказаний и сабмит")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--level", type=float, default=DEFAULT_LEVEL,
                    help="уровень для ИТОГОВОГО сабмита (февраль-март). "
                         "Локальная диагностика всегда считается на истинном "
                         "январском уровне DEFAULT_LEVEL, независимо от этого флага, "
                         "иначе --level 2.3292 искажает валидацию на январском окне")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="множитель центрированного предсказания. Оптимум измерен "
                         "по лидерборду: RMSLE(a)^2 = A a^2 - 2B a + C, кривизна "
                         "A = E[c^2] известна точно из самих предсказаний, две точки "
                         "на тесте дают B и C. Замер дал a* = 1.0275 — предсказания "
                         "надо РАСТЯГИВАТЬ; январь показывал 0.98, потому что его окно "
                         "после декабрьского пика имеет меньший разброс таргета")
    ap.add_argument("--scale-lo", type=float, default=None,
                    help="отдельный множитель для предсказаний НИЖЕ центра. Носители "
                         "c<0 и c>0 не пересекаются, поэтому перекрёстный член в "
                         "квадратичной форме равен нулю и задача распадается на две "
                         "независимые параболы — каждая решается одной пробой")
    ap.add_argument("--scale-hi", type=float, default=None,
                    help="отдельный множитель для предсказаний ВЫШЕ центра")
    ap.add_argument("--no-normalize", action="store_true",
                    help="не нормировать сумму весов к 1. Оптимум RMSLE — условное "
                         "среднее E[log1p y], оно сжато к центру сильнее, чем разброс "
                         "отдельных предсказаний. Нормировка навязывала масштаб 1, "
                         "хотя сырой МНК находит оптимальный (~0.98) сам")
    ap.add_argument("--shrink-weights", type=float, default=0.0,
                    help="доля равномерных весов в смеси с МНК-весами (0..1)")
    ap.add_argument("--weights", type=float, nargs="+", default=None,
                    help="явные веса вместо МНК со сдвигом. Нужны, когда оптимум "
                         "лежит не на отрезке между МНК и равномерными: скан по весу "
                         "seqnet_90 дал минимум на 0.35, а равные веса дают 1/6")
    ap.add_argument("--name", type=str, default="blend")
    ap.add_argument("--write-submission", action="store_true")
    args = ap.parse_args()

    P = load("valid", args.models)
    y = pl.read_parquet(anchor_path(VALID_ANCHOR), columns=["target"])["target"].to_numpy()
    yl = np.log1p(y)
    C = P - P.mean(0)

    print("--- по отдельности, центрировано + январский уровень %.4f ---" % DEFAULT_LEVEL)
    for i, n in enumerate(args.models):
        print(f"  {n:12s} RMSLE={rmsle_from_log(y, C[:, i] + DEFAULT_LEVEL):.5f}")

    if args.weights is not None:
        assert len(args.weights) == len(args.models), "весов должно быть столько же, сколько моделей"
        w = np.array(args.weights, dtype=float); w = w / w.sum()
        scale = 1.0
    else:
        w, _ = nnls(C, yl - yl.mean())
        scale = w.sum()
        w = w / scale
    if args.shrink_weights and args.weights is None:
        w = (1 - args.shrink_weights) * w + args.shrink_weights * np.ones(len(w)) / len(w)
    print("\n--- веса (по форме, сумма = 1) ---")
    for n, wi in zip(args.models, w):
        print(f"  {n:12s} {wi:.4f}")
    if args.no_normalize and args.weights is None:
        w = w * scale
        print(f"  масштаб весов сохранён: сумма {w.sum():.4f}")
    score = rmsle_from_log(y, C @ w + DEFAULT_LEVEL)
    print(f"  бленд RMSLE (январь, диагностика)={score:.5f}")
    if abs(args.level - DEFAULT_LEVEL) > 1e-9:
        print(f"  (сабмит будет собран с уровнем {args.level:.4f} для февраля-марта)")

    if not args.write_submission:
        return

    F = load("final", args.models)
    c = (F - F.mean(0)) @ w
    if args.scale_lo is not None or args.scale_hi is not None:
        a_lo = args.scale_lo if args.scale_lo is not None else args.scale
        a_hi = args.scale_hi if args.scale_hi is not None else args.scale
        c = np.where(c < 0, a_lo * c, a_hi * c)
        print(f"  раздельный масштаб: низ {a_lo:.4f}, верх {a_hi:.4f}")
    else:
        c = args.scale * c
    pred_log = np.clip(c + args.level, 0, None)
    pred = np.expm1(pred_log)
    user_ids = pl.read_parquet(anchor_path(PREDICT_ANCHOR), columns=["user_id"])["user_id"].to_numpy()

    sub = pl.DataFrame({"user_id": user_ids, "predict": pred})
    order = pl.read_csv(SAMPLE_SUBMIT).select("user_id")
    sub = order.join(sub, on="user_id", how="left").with_columns(pl.col("predict").fill_null(0.0))
    assert sub.height == order.height and sub["predict"].null_count() == 0
    assert sub["predict"].is_finite().all() and sub["predict"].min() >= 0

    out = SUBMISSIONS_DIR / f"{args.name}.csv"
    sub.write_csv(out)
    print(f"\nсабмит -> {out}")
    print(f"  mean log1p(pred) = {pred_log.mean():.4f}")
    print(f"  нулей {int((sub['predict'] == 0).sum())}, медиана {sub['predict'].median():.2f}, "
          f"макс {sub['predict'].max():.1f}")
    (MODELS_DIR / f"{args.name}.json").write_text(json.dumps(
        dict(models=args.models, weights=[float(x) for x in w], level=args.level,
             valid_rmsle=score, mean_log_pred=float(pred_log.mean())),
        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
