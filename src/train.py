"""Обучение LightGBM на log1p(target).

RMSLE = RMSE в пространстве log1p, поэтому обучаем L2-регрессию прямо на
log1p(y): оптимум лосса и оптимум метрики совпадают, поправка на смещение
при обратном преобразовании не нужна.
"""
from __future__ import annotations

import argparse
import json
from datetime import date

import lightgbm as lgb
import numpy as np
import polars as pl

from config import (DROP, MODELS_DIR, PREDICT_ANCHOR, SEED, TRAIN_ANCHOR_STRIDE,
                    VALID_ANCHOR, train_anchors)
from features import anchor_path


PARAMS = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.04,
    num_leaves=63,        # подбор: 63 < 127 < 255 < 511, разница в пределах 0.001
    min_data_in_leaf=200,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=5.0,
    max_bin=255,
    num_threads=12,
    seed=SEED,
    verbosity=-1,
)


def load_anchor(a: date) -> pl.DataFrame:
    return pl.read_parquet(anchor_path(a))


def feature_names(frame: pl.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in DROP]


def stack(anchors: list[date], feats: list[str] | None = None
          ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    frames = [load_anchor(a) for a in anchors]
    feats = feats or feature_names(frames[0])
    X = np.vstack([f.select(feats).to_numpy().astype(np.float32) for f in frames])
    y = np.concatenate([f["target"].to_numpy() for f in frames])
    return X, y, feats


def rmsle_from_log(y_true: np.ndarray, log_pred: np.ndarray) -> float:
    """y_true в исходной шкале, log_pred — предсказание в log1p-шкале."""
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.clip(log_pred, 0, None)) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Обучение и валидация модели")
    ap.add_argument("--stride", type=int, default=TRAIN_ANCHOR_STRIDE)
    ap.add_argument("--n-anchors", type=int, default=None,
                    help="взять только N последних обучающих якорей")
    ap.add_argument("--rounds", type=int, default=5000)
    ap.add_argument("--early-stop", type=int, default=200)
    ap.add_argument("--tag", type=str, default="lgbm")
    args = ap.parse_args()

    anchors = train_anchors(stride=args.stride)
    if args.n_anchors:
        anchors = anchors[-args.n_anchors:]
    print(f"train anchors ({len(anchors)}): {anchors[0]} .. {anchors[-1]}")
    print(f"valid anchor : {VALID_ANCHOR}  -> предсказываем 30 дней после него")

    X, y, feats = stack(anchors)
    val = load_anchor(VALID_ANCHOR)
    Xv = val.select(feats).to_numpy().astype(np.float32)
    yv = val["target"].to_numpy()
    print(f"train {X.shape}  valid {Xv.shape}  features={len(feats)}")

    dtrain = lgb.Dataset(X, label=np.log1p(y), feature_name=feats, free_raw_data=True)
    dvalid = lgb.Dataset(Xv, label=np.log1p(yv), feature_name=feats, reference=dtrain)

    evals: dict = {}
    model = lgb.train(
        PARAMS, dtrain, num_boost_round=args.rounds,
        valid_sets=[dvalid], valid_names=["valid"],
        callbacks=[lgb.early_stopping(args.early_stop, verbose=False),
                   lgb.log_evaluation(100),
                   lgb.record_evaluation(evals)],
    )

    pred_log = model.predict(Xv, num_iteration=model.best_iteration)
    score = rmsle_from_log(yv, pred_log)

    # референсы для сравнения
    lag30 = val["gmv_s30"].to_numpy()
    naive = rmsle_from_log(yv, np.log1p(lag30))
    const = rmsle_from_log(yv, np.full_like(yv, np.mean(np.log1p(yv))))

    print("\n=== валидация (anchor %s) ===" % VALID_ANCHOR)
    print(f"  LightGBM        RMSLE = {score:.5f}   (best_iter={model.best_iteration})")
    print(f"  наивный лаг-30  RMSLE = {naive:.5f}")
    print(f"  лучшая константа RMSLE = {const:.5f}")
    print(f"  выигрыш к наивному: {naive - score:.5f}")

    imp = sorted(zip(feats, model.feature_importance("gain")),
                 key=lambda t: -t[1])
    print("\n--- топ-25 признаков по gain ---")
    for n, g in imp[:25]:
        print(f"  {n:32s} {g:14.0f}")

    model.save_model(str(MODELS_DIR / f"{args.tag}_valid.txt"),
                     num_iteration=model.best_iteration)
    meta = dict(tag=args.tag, rmsle=score, naive=naive, best_iteration=model.best_iteration,
                n_train_anchors=len(anchors), stride=args.stride,
                features=feats, params=PARAMS,
                importance=[(n, float(g)) for n, g in imp])
    (MODELS_DIR / f"{args.tag}_valid.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nмодель -> {MODELS_DIR / (args.tag + '_valid.txt')}")


if __name__ == "__main__":
    main()
