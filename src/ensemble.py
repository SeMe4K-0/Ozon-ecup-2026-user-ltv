"""Ансамбль разнородных моделей: обучение, сохранение предсказаний, блендинг.

Все модели работают в log1p-шкале, поэтому бленд — обычное взвешенное среднее
предсказаний, а не среднее в исходных рублях.

Базовый режим — обучение на де-меанированном по якорю таргете: это снимает
дрейф уровня площадки из обучающего сигнала (см. README) и даёт основной прирост.

Режимы:
  valid — учимся на якорях <= 2025-12-15, предсказываем 2026-01-14. Таргет известен,
          поэтому число итераций подбирается ранней остановкой и сохраняется;
  final — учимся на якорях <= 2026-01-14, предсказываем 2026-02-13, число итераций
          берётся из сохранённого валидационного прогона.
"""
from __future__ import annotations

import argparse
import gc
import json
from datetime import date

import lightgbm as lgb
import numpy as np
import polars as pl

from config import PRED_DIR, ARTIFACTS, MODELS_DIR, PREDICT_ANCHOR, SEED, VALID_ANCHOR, final_anchors, train_anchors
from features import anchor_path
from train import DROP, PARAMS, rmsle_from_log

ROUNDS_FILE = MODELS_DIR / "ensemble_rounds.json"

EARLY = 150
MAX_ROUNDS = 4000

SPECS = {
    "lgbm":      dict(kind="single", demean=True, params=dict(num_leaves=63)),
    "lgbm_deep": dict(kind="single", demean=True, params=dict(num_leaves=255)),
    "lgbm_slow": dict(kind="single", demean=True, params=dict(
        learning_rate=0.02, num_leaves=127, min_data_in_leaf=500,
        feature_fraction=0.5, lambda_l2=20.0)),
    # двухстадийная схема живёт на исходном таргете: разложение
    # E[log1p y] = P(y>0) * E[log1p y | y>0] точно только для него.
    # В ансамбле она нужна как источник разнообразия.
    "two_stage": dict(kind="two_stage", demean=False, params=dict(num_leaves=255)),
    "catboost":  dict(kind="catboost", demean=True, params=dict()),
}


def load_pool(anchors: list[date]):
    frames = [pl.read_parquet(anchor_path(a)) for a in anchors]
    feats = [c for c in frames[0].columns if c not in DROP]
    X = np.vstack([f.select(feats).to_numpy().astype(np.float32) for f in frames])
    y = np.concatenate([f["target"].to_numpy() for f in frames])
    aidx = np.concatenate([np.full(f.height, i, np.int16) for i, f in enumerate(frames)])
    del frames
    gc.collect()
    return X, y, aidx, feats


def load_test(anchor: date, feats: list[str]):
    f = pl.read_parquet(anchor_path(anchor))
    X = f.select(feats).to_numpy().astype(np.float32)
    y = f["target"].to_numpy() if "target" in f.columns else None
    return X, y, f["user_id"].to_numpy()


def demean_target(y: np.ndarray, aidx: np.ndarray) -> tuple[np.ndarray, float]:
    """log1p(y) минус средний уровень своего якоря.

    Убирает из обучающего сигнала дрейф площадки: без этого один и тот же
    паттерн признаков соответствует разным уровням таргета в разные месяцы,
    и модель тратит ёмкость на подгонку уровня вместо структуры.
    Возвращает также уровень, который надо вернуть обратно при предсказании.
    """
    yl = np.log1p(y)
    mus = np.array([yl[aidx == i].mean() for i in range(int(aidx.max()) + 1)])
    return yl - mus[aidx], float(mus.mean())


def _train_lgb(X, label, feats, params, seed, rounds, Xv=None, labelv=None):
    p = dict(PARAMS, seed=seed, bagging_seed=seed + 100, feature_fraction_seed=seed + 200)
    p.update(params)
    d = lgb.Dataset(X, label=label, feature_name=feats)
    if Xv is None:
        return lgb.train(p, d, num_boost_round=rounds), rounds
    dv = lgb.Dataset(Xv, label=labelv, reference=d)
    m = lgb.train(p, d, num_boost_round=MAX_ROUNDS, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(EARLY, verbose=False)])
    return m, m.best_iteration


def fit_predict(name: str, X, y, aidx, feats, Xt, *, yt=None, seeds=1,
                level=None, rounds=None):
    """Обучает модель и возвращает предсказание в log1p-шкале.

    Если передан yt, число итераций подбирается ранней остановкой по тесту
    (режим valid); иначе используется переданное rounds (режим final).
    """
    spec = SPECS[name]
    kind, demean, params = spec["kind"], spec["demean"], spec["params"]
    tune = yt is not None

    if demean:
        label, mu_add = demean_target(y, aidx)
        if level is not None:
            mu_add = level
        labelv = np.log1p(yt) - mu_add if tune else None
    else:
        label, mu_add = np.log1p(y), 0.0
        labelv = np.log1p(yt) if tune else None

    preds, used = [], {}
    for s in range(seeds):
        seed = SEED + 1000 * s
        if kind == "single":
            m, it = _train_lgb(X, label, feats, params, seed, rounds,
                               Xt if tune else None, labelv)
            preds.append(m.predict(Xt, num_iteration=it) + mu_add)
            used["rounds"] = it
        elif kind == "two_stage":
            pos = y > 0
            pc = dict(params, objective="binary", metric="binary_logloss")
            mc, itc = _train_lgb(X, pos.astype(np.float64), feats, pc, seed,
                                 (rounds or {}).get("cls") if isinstance(rounds, dict) else None,
                                 Xt if tune else None,
                                 (yt > 0).astype(np.float64) if tune else None)
            mr, itr = _train_lgb(X[pos], np.log1p(y[pos]), feats, params, seed,
                                 (rounds or {}).get("reg") if isinstance(rounds, dict) else None,
                                 Xt[yt > 0] if tune else None,
                                 np.log1p(yt[yt > 0]) if tune else None)
            preds.append(mc.predict(Xt, num_iteration=itc) * mr.predict(Xt, num_iteration=itr))
            used["rounds"] = dict(cls=itc, reg=itr)
        elif kind == "catboost":
            from catboost import CatBoostRegressor
            n = rounds or 3000
            m = CatBoostRegressor(iterations=n if not tune else 4000, learning_rate=0.05,
                                  depth=8, loss_function="RMSE", random_seed=seed,
                                  verbose=500, thread_count=12, l2_leaf_reg=5.0,
                                  early_stopping_rounds=EARLY if tune else None)
            m.fit(X, label, eval_set=(Xt, labelv) if tune else None)
            it = m.get_best_iteration() if tune else n
            preds.append(m.predict(Xt) + mu_add)
            used["rounds"] = int(it)
        else:
            raise ValueError(kind)
        print(f"    {name} сид {s} готов (rounds={used['rounds']})")
        gc.collect()
    return np.mean(preds, axis=0), used


def main() -> None:
    ap = argparse.ArgumentParser(description="Обучение моделей ансамбля")
    ap.add_argument("--mode", choices=["valid", "final"], required=True)
    ap.add_argument("--models", nargs="+", default=list(SPECS))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--n-anchors", type=int, default=None)
    ap.add_argument("--level", type=float, default=None,
                    help="уровень, возвращаемый де-меанированным моделям")
    args = ap.parse_args()

    anchors = (final_anchors() if args.mode == "final" else train_anchors())
    if args.n_anchors:
        anchors = anchors[-args.n_anchors:]
    test_anchor = VALID_ANCHOR if args.mode == "valid" else PREDICT_ANCHOR

    print(f"режим {args.mode}: якорей {len(anchors)} ({anchors[0]} .. {anchors[-1]}), "
          f"тест {test_anchor}")
    X, y, aidx, feats = load_pool(anchors)
    Xt, yt, user_ids = load_test(test_anchor, feats)
    print(f"train {X.shape}  test {Xt.shape}")

    saved = json.loads(ROUNDS_FILE.read_text()) if ROUNDS_FILE.exists() else {}
    for name in args.models:
        print(f"\n--- {name} ---")
        rounds = None if args.mode == "valid" else saved.get(name, {}).get("rounds")
        if args.mode == "final" and rounds is None:
            print(f"  пропуск: нет сохранённого числа итераций, сначала прогон --mode valid")
            continue
        pred, used = fit_predict(name, X, y, aidx, feats, Xt,
                                 yt=yt if args.mode == "valid" else None,
                                 seeds=args.seeds, level=args.level, rounds=rounds)
        np.save(PRED_DIR / f"{args.mode}_{name}.npy", pred)
        if yt is not None:
            print(f"  {name}: RMSLE={rmsle_from_log(yt, pred):.5f}")
            saved[name] = used
            ROUNDS_FILE.write_text(json.dumps(saved, indent=2))
        else:
            print(f"  {name}: mean log1p(pred)={np.clip(pred, 0, None).mean():.4f}")

    if yt is not None:
        np.save(PRED_DIR / "valid_y.npy", yt)
    np.save(PRED_DIR / f"{args.mode}_user_ids.npy", user_ids)


if __name__ == "__main__":
    main()
