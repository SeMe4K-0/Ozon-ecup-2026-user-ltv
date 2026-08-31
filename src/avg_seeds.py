"""Усреднение предсказаний по сидам из сохранённых пофайловых прогонов.

seq_model.py кладёт предсказание каждого сида отдельно, поэтому новые сиды
можно доучить (`--seed-start`) и досыпать к уже посчитанным, не переобучая их.
Усреднение делается в log1p-шкале — той же, в которой работает бленд.
"""
from __future__ import annotations

import argparse

import numpy as np

from config import PRED_DIR


def main() -> None:
    ap = argparse.ArgumentParser(description="Собрать среднее по сидам")
    ap.add_argument("--mode", choices=["valid", "final"], required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    files = sorted(PRED_DIR.glob(f"{args.mode}_{args.tag}_seed*.npy"))
    if not files:
        raise SystemExit(f"нет файлов {args.mode}_{args.tag}_seed*.npy")
    preds = [np.load(f) for f in files]
    for f, p in zip(files, preds):
        print(f"  {f.name}  mean={np.clip(p, 0, None).mean():.4f}")
    avg = np.mean(preds, axis=0)
    out = PRED_DIR / f"{args.mode}_{args.tag}.npy"
    np.save(out, avg)
    print(f"усреднено {len(files)} сидов -> {out.name}  mean={np.clip(avg, 0, None).mean():.4f}")


if __name__ == "__main__":
    main()
