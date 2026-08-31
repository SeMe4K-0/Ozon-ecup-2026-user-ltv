"""Обучение на плотной панели: любые якоря с шагом 1 вместо 22 с шагом 7.

Запускать интерпретатором Python 3.11 (CUDA-сборка torch).

Плотное хранение (panel.py) сняло ограничение по памяти, которое всё это время
определяло конфигурацию: 22 якоря вместо доступных 200 и окно 180 дней вместо
доступных 409. Здесь используется и то, и другое.

Табличных признаков нет намеренно: считать их для 200 якорей нереально (72 ГБ),
но они и не нужны — их основная ценность была в агрегатах за всю историю, а
длинное окно даёт ту же информацию напрямую. Заодно модель получается максимально
непохожей на остальные, а по нашим замерам именно непохожесть определяет вклад
в ансамбль: hist_dist проигрывает в одиночку 0.0013 и получает наибольший вес 0.24.

Эпоха задана фиксированным числом шагов, а якорь выбирается случайно на каждом
шаге: так временное разнообразие растёт в девять раз без удорожания эпохи.
"""
from __future__ import annotations

import argparse
import os
import json
import time

import numpy as np
import torch
import torch.nn as nn

import panel
from config import (PRED_DIR, ARTIFACTS, HORIZON, MIN_HISTORY_DAYS, MODELS_DIR, PREDICT_ANCHOR,
                    SEED, SEQ_CHANNELS, SEQ_SCALE, VALID_ANCHOR)

from seq_model import SeqNet, build_bins, dist_labels, rmsle_from_log

VALID_LEVEL = 2.2421


def anchors_for(mode: str, seq_len: int, stride: int,
                min_history: int | None = None) -> np.ndarray:
    """Якоря обучения. В режиме valid таргет не должен заходить за валидационный."""
    days = panel.valid_anchor_days(seq_len if min_history is None else 1,
                                   min_history=min_history or MIN_HISTORY_DAYS)
    last = panel.day_index(VALID_ANCHOR)
    limit = last - HORIZON if mode == "valid" else last
    return days[days <= limit][::-1][::stride][::-1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Модель на плотной панели")
    ap.add_argument("--mode", choices=["valid", "final"], required=True)
    ap.add_argument("--seq-len", type=int, default=180)
    ap.add_argument("--anchor-stride", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--n-bins", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--stop-epoch", type=int, default=None)
    ap.add_argument("--weekly", type=int, default=0,
                    help="сколько недельных агрегатов ставить перед подневной частью. "
                         "0 — прежнее поведение. При 45 и --n-daily 90 охват 405 дней "
                         "за 135 шагов: приём hist_dist, но на всех якорях панели, "
                         "а не на 22 из parquet-сетки")
    ap.add_argument("--n-daily", type=int, default=90,
                    help="подневная часть при --weekly")
    ap.add_argument("--min-history", type=int, default=None,
                    help="минимум истории для якоря. При недельном режиме окно можно "
                         "уводить левее начала данных и добивать нулями, поэтому "
                         "глубина охвата больше не съедает якоря")
    ap.add_argument("--resume", action="store_true",
                    help="пропустить сиды, чей файл предсказания уже лежит на диске. "
                         "Гранулярность посидовая: прерывание теряет текущий сид "
                         "целиком, но не предыдущие")
    ap.add_argument("--init-from", type=str, default=None,
                    help="тег предобученного энкодера: начать не со "
                         "случайных весов, а с тех, что уже выучили ритм покупок "
                         "на всей панели, включая 60 свежих дней без таргета")
    ap.add_argument("--freeze-epochs", type=int, default=0,
                    help="сколько первых эпох держать энкодер замороженным, пока "
                         "случайная голова не перестанет портить его градиентами")
    ap.add_argument("--aux-horizons", type=str, default="",
                    help="вспомогательные горизонты через запятую, например 7,14,60. "
                         "Сумма за 30 дней — редкая и шумная величина; те же веса, "
                         "предсказывающие заодно 7 и 14 дней, обязаны выучить форму "
                         "кривой покупок, а не запомнить один срез. Горизонт 60 "
                         "заставляет отличать отток от паузы")
    ap.add_argument("--aux-weight", type=float, default=0.3,
                    help="вес вспомогательного лосса относительно основного")
    ap.add_argument("--tag", type=str, default="panel")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p, g, users = panel.load(mmap=False)
    raw = np.load(panel.RAW_PATH, mmap_mode="r") if args.weekly else None

    def getw(day, rows):
        """Вход сети: uint8-панель делится на масштаб, недельный режим уже в log1p."""
        if args.weekly:
            return panel.window_hist(p, raw, int(day), rows, args.n_daily, args.weekly)
        return panel.window(p, int(day), rows, args.seq_len).astype(np.float32) / SEQ_SCALE
    rows_all = np.arange(len(users), dtype=np.int64)

    tr_days = anchors_for(args.mode, args.seq_len, args.anchor_stride, args.min_history)
    test_day = panel.day_index(VALID_ANCHOR if args.mode == "valid" else PREDICT_ANCHOR)
    print(f"устройство {dev} | режим {args.mode} | якорей {len(tr_days)} "
          f"(шаг {args.anchor_stride}) | окно {args.seq_len} | тест день {test_day}")

    # таргеты всех обучающих якорей: 200 x 250k float32 = 0.2 ГБ
    T = np.stack([panel.target(g, int(d), rows_all) for d in tr_days])
    ys = [T[i] for i in range(len(tr_days))]
    edges, centers, mu_add, mus = build_bins(ys, args.n_bins)
    labels = np.stack([dist_labels(T[i], mus[i], edges, args.n_bins)
                       for i in range(len(tr_days))])
    print(f"корзин {args.n_bins}, уровень положительной части {mu_add:.4f}, "
          f"доля нулей {np.mean(T == 0):.3f}")
    del T

    aux_h = [int(x) for x in args.aux_horizons.split(",") if x.strip()]
    aux_y, aux_ok = None, None
    if aux_h:
        # Для каждого горизонта: де-меанированный log1p суммы + флаг наблюдаемости.
        # Срез за концом массива numpy молча укорачивает окно, поэтому якоря,
        # у которых горизонт не помещается, исключаются маской, а не усечением.
        n_days = g.shape[1]
        aux_y = np.zeros((len(aux_h), len(tr_days), len(users)), np.float16)
        aux_ok = np.zeros((len(aux_h), len(tr_days)), np.float32)
        for j, h in enumerate(aux_h):
            for i, d in enumerate(tr_days):
                if int(d) + h >= n_days:
                    continue
                v = np.log1p(panel.target(g, int(d), rows_all, horizon=h))
                aux_y[j, i] = (v - v.mean()).astype(np.float16)
                aux_ok[j, i] = 1.0
        print(f"вспомогательные горизонты {aux_h}, вес {args.aux_weight}, "
              f"наблюдаемых пар якорь-горизонт {int(aux_ok.sum())}/{aux_ok.size}")

    centers_t = torch.tensor(np.concatenate([[0.0], centers[1:] + mu_add]),
                             dtype=torch.float32, device=dev)
    yt = panel.target(g, test_day, rows_all) if args.mode == "valid" else None
    n_main = args.n_bins + 1
    zero_static = np.zeros((args.batch, 1), np.float32)

    @torch.no_grad()
    def predict(model):
        model.eval()
        out = []
        for i in range(0, len(users), args.batch):
            r = rows_all[i:i + args.batch]
            s = torch.from_numpy(getw(test_day, r)).to(dev)
            st = torch.zeros((len(r), 1), device=dev)
            q = torch.softmax(model(s, st)[:, :n_main], dim=1)
            out.append((q @ centers_t).cpu().numpy())
        return np.concatenate(out)

    rng = np.random.default_rng(SEED)
    preds, hist = [], []
    for sd in range(args.seed_start, args.seed_start + args.seeds):
        done = PRED_DIR / f"{args.mode}_{args.tag}_seed{sd}.npy"
        if args.resume and done.exists():
            print(f"  сид {sd}: уже посчитан, пропускаю")
            preds.append(np.load(done))
            hist.append(dict(seed=sd, best=None, best_epoch=None, resumed=True))
            continue
        torch.manual_seed(SEED + sd)
        model = SeqNet(len(SEQ_CHANNELS), 1, args.hidden,
                       n_out=n_main + len(aux_h)).to(dev)
        if args.init_from:
            ck = torch.load(MODELS_DIR / f"trunk_{args.init_from}.pt", map_location=dev)
            assert ck["hidden"] == args.hidden and ck["c_in"] == len(SEQ_CHANNELS),                 f"энкодер собран под hidden={ck['hidden']}, c_in={ck['c_in']}"
            missing = model.load_state_dict(ck["trunk"], strict=False)
            print(f"  энкодер загружен из trunk_{args.init_from}.pt, "
                  f"своих слоёв осталось {len(missing.missing_keys)}")
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=args.epochs * args.steps)
        lossf = nn.CrossEntropyLoss()

        best, best_ep, best_pred = np.inf, -1, None
        for ep in range(args.stop_epoch or args.epochs):
            t0 = time.time()
            if args.init_from and args.freeze_epochs:
                frozen = ep < args.freeze_epochs
                for n_, q_ in model.named_parameters():
                    if n_.startswith(("bn_in.", "conv.", "gru.", "proj.", "pos", "enc.")):
                        q_.requires_grad_(not frozen)
                if ep == args.freeze_epochs:
                    print("  энкодер разморожен")
            model.train()
            tot = 0.0
            for _ in range(args.steps):
                ai = rng.integers(len(tr_days))
                r = np.sort(rng.choice(len(users), args.batch, replace=False))
                s = torch.from_numpy(getw(tr_days[ai], r)).to(dev, non_blocking=True)
                st = torch.zeros((len(r), 1), device=dev)
                lab = torch.from_numpy(labels[ai][r]).to(dev, non_blocking=True)
                o = model(s, st)
                loss = lossf(o[:, :n_main], lab)
                for j in range(len(aux_h)):
                    if aux_ok[j, ai] == 0.0:
                        continue
                    ay = torch.from_numpy(aux_y[j, ai][r].astype(np.float32)
                                          ).to(dev, non_blocking=True)
                    loss = loss + args.aux_weight * ((o[:, n_main + j] - ay) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                sched.step()
                tot += loss.item()
            pv = predict(model)
            msg = f"  сид {sd} эпоха {ep+1}/{args.epochs} loss={tot/args.steps:.4f} {time.time()-t0:.0f}с"
            if yt is not None:
                sc = rmsle_from_log(yt, pv - pv.mean() + VALID_LEVEL)
                print(msg + f" valid RMSLE={sc:.5f}")
                if sc < best:
                    best, best_ep, best_pred = sc, ep + 1, pv
            else:
                print(msg)
                best_pred = pv
        np.save(PRED_DIR / f"{args.mode}_{args.tag}_seed{sd}.npy", best_pred)
        preds.append(best_pred)
        hist.append(dict(seed=sd, best=None if yt is None else best, best_epoch=best_ep))
        if yt is not None:
            print(f"  сид {sd}: лучший RMSLE={best:.5f} на эпохе {best_ep}")

    pred = np.mean(preds, axis=0)
    np.save(PRED_DIR / f"{args.mode}_{args.tag}.npy", pred - pred.mean() + VALID_LEVEL)
    if yt is not None:
        print(f"\nитог: RMSLE={rmsle_from_log(yt, pred - pred.mean() + VALID_LEVEL):.5f}")
        (MODELS_DIR / f"{args.tag}_valid.json").write_text(
            # см. seq_model.py: сохраняем полный набор аргументов, иначе тег
            # модели не описывает сам себя
            json.dumps(dict(hist=hist, args=vars(args),
                            env={k: os.environ.get(k) for k in
                                 ("FEATURE_VERSION", "LONG_WINDOWS", "SEQ_C13")}),
                       ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
