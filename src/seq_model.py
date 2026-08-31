"""Последовательная нейросеть по подневному поведению пользователя.

Запускать интерпретатором Python 3.11 (там стоит CUDA-сборка torch):
  py -3.11 seq_model.py --mode valid

Архитектура: свёрточный фронт сжимает 180 дней до 45 шагов, GRU читает их по
времени, а выход склеивается с табличными признаками (413 колонок в версии v5).
Табличная часть нужна,
чтобы сеть стартовала не хуже градиентного бустинга; ценность даёт свёрточно-
рекуррентная часть, которой у бустинга нет.

Таргет де-меанирован по якорю: уровень площадки дрейфует (доля активных дней
растёт с 0.268 до 0.351 по якорям), и без вычитания уровня сеть тратит ёмкость
на его подгонку. Уровень возвращается на этапе блендинга.

Батчи набираются целиком из одного якоря: это позволяет брать срез одним
fancy-index вместо поэлементного копирования и радикально ускоряет эпоху.
"""
from __future__ import annotations

import argparse
import os
import json
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from config import (PRED_DIR, ARTIFACTS, HORIZON, MODELS_DIR, PREDICT_ANCHOR, SEED, SEQ_CHANNELS,
                    SEQ_LEN, SEQ_SCALE, VALID_ANCHOR, DROP, final_anchors, train_anchors)
from features import anchor_path
from seq_data import seq_path
import seq_hist

PRED_DIR.mkdir(parents=True, exist_ok=True)
def norm_file(tag: str):
    """Нормировка своя у каждой модели.

    Она привязана к числу и составу колонок, а модели ансамбля живут на разных
    версиях признаков. Общий файл приводил к тому, что валидация одной модели
    молча ломала финальный прогон другой.
    """
    return MODELS_DIR / f"norm_{tag}.npz"

VALID_LEVEL = 2.2421          # истинный уровень январского окна (по умолчанию)


def window_level(y_true):
    """Истинный уровень окна = среднее log1p(таргета).

    Для второго валидационного окна январская константа неверна, а ошибка уровня
    входит в RMSLE квадратично и исказила бы сравнение моделей.
    """
    return float(np.log1p(y_true).mean())


# --------------------------------------------------------------------------- данные
def load_static(anchor, feats=None):
    f = pl.read_parquet(anchor_path(anchor))
    feats = feats or [c for c in f.columns if c not in DROP]
    X = f.select(feats).to_numpy().astype(np.float32)
    y = f["target"].to_numpy() if "target" in f.columns else None
    return X, y, feats, f["user_id"].to_numpy()


def signed_log(X):
    return np.sign(X) * np.log1p(np.abs(X))


def fit_norm(X):
    Z = signed_log(X)
    mu, sd = np.nanmean(Z, 0), np.nanstd(Z, 0)
    sd[~np.isfinite(sd) | (sd < 1e-6)] = 1.0
    mu[~np.isfinite(mu)] = 0.0
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_norm(X, mu, sd):
    Z = (signed_log(X) - mu) / sd
    return np.clip(np.nan_to_num(Z, nan=0.0, posinf=8.0, neginf=-8.0), -8, 8).astype(np.float32)


# --------------------------------------------------------------------------- модель
class SeqNet(nn.Module):
    """Свёрточно-рекуррентный энкодер последовательности + табличная ветка.

    При two_stage=True голова раздваивается на классификатор P(y>0) и условную
    регрессию E[log1p y | y>0], а предсказание — их произведение. Разложение
    E[log1p y] = P(y>0) * E[log1p y | y>0] точно, поскольку log1p(0) = 0.
    Именно оно делает `two_stage` лучшим вариантом среди бустингов, а 46% нулей
    в таргете дают этому разложению прямой смысл.
    """

    def __init__(self, c_in: int, n_static: int, hidden: int = 128, dropout: float = 0.1,
                 two_stage: bool = False, n_out: int = 1,
                 n_users: int = 0, user_dim: int = 0, arch: str = "gru"):
        super().__init__()
        self.two_stage = two_stage
        self.n_out = n_out
        self.arch = arch
        # Персональная предрасположенность, не выводимая из агрегатов и ряда.
        # Панель пользователей фиксирована, тот же user_id есть и в обучении, и в
        # тесте — это структура данных, а не утечка. Размерность держим малой:
        # 250k пользователей легко запомнить таргет, если дать простор.
        self.user_dim = user_dim if (n_users and user_dim) else 0
        if self.user_dim:
            self.user_emb = nn.Embedding(n_users, self.user_dim)
            nn.init.normal_(self.user_emb.weight, std=0.01)
            self.user_drop = nn.Dropout(0.2)
        self.bn_in = nn.BatchNorm1d(c_in)
        self.conv = nn.Sequential(
            nn.Conv1d(c_in, 64, 5, padding=2), nn.GELU(),
            nn.Conv1d(64, 96, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(96, 128, 5, stride=2, padding=2), nn.GELU(),
        )
        if arch == "transformer":
            # d_model = hidden, чтобы размерность головы совпадала с GRU-веткой
            self.proj = nn.Linear(128, hidden)
            self.pos = nn.Parameter(torch.zeros(1, 256, hidden))
            nn.init.normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=4, dim_feedforward=hidden * 2,
                dropout=dropout, batch_first=True, norm_first=True,
                activation="gelu")
            self.enc = nn.TransformerEncoder(layer, num_layers=3)
        else:
            self.gru = nn.GRU(128, hidden, batch_first=True)
        self.static = nn.Sequential(
            nn.Linear(n_static, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 3 + 128 + self.user_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 2 if two_stage else n_out),   # ziln => n_out=3
        )

    def encode_seq(self, seq):
        """Пошаговое представление ряда (B, T/4, hidden) без пулинга и головы.

        Нужно предобучению: оно восстанавливает замаскированные куски ряда и
        потому работает с позициями, а не со сжатым в вектор пользователем.
        Свёртки сжимают время вчетверо, поэтому одна позиция — это 4 дня.
        """
        h = self.conv(self.bn_in(seq.transpose(1, 2))).transpose(1, 2)
        if self.arch == "transformer":
            return self.enc(self.proj(h) + self.pos[:, :h.shape[1]])
        return self.gru(h)[0]

    def trunk_state(self):
        """Веса энкодера без головы — то, что переносится из предобучения."""
        keep = ("bn_in.", "conv.", "gru.", "proj.", "pos", "enc.")
        return {k: v for k, v in self.state_dict().items() if k.startswith(keep)}

    def forward(self, seq, static, uid=None):
        h = self.conv(self.bn_in(seq.transpose(1, 2)))       # (B, 128, T/4)
        h = h.transpose(1, 2)                                 # (B, T/4, 128)
        if self.arch == "transformer":
            o = self.enc(self.proj(h) + self.pos[:, :h.shape[1]])
            # у трансформера нет «последнего состояния»: берём последний шаг,
            # он ближе всего к якорю по времени
            pooled = [o[:, -1], o.mean(1), o.max(1).values]
        else:
            o, last = self.gru(h)
            pooled = [last[-1], o.mean(1), o.max(1).values]
        z = torch.cat(pooled + [self.static(static)], dim=1)
        if self.user_dim:
            z = torch.cat([z, self.user_drop(self.user_emb(uid))], dim=1)
        out = self.head(z)
        if self.n_out > 1:
            return out                      # логиты корзин, softmax берётся в лоссе
        if not self.two_stage:
            return out.squeeze(1)
        logit, raw = out[:, 0], out[:, 1]
        # softplus держит условное среднее неотрицательным: log1p(y) >= 0,
        # а произведение с вероятностью автоматически даёт нужную неотрицательность
        return torch.sigmoid(logit) * nn.functional.softplus(raw), logit


# узлы Гаусса-Эрмита для E[log1p(exp(Z))], Z ~ N(mu, sigma^2)
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(15)


def ziln_loss(out, y_pos, y_log, eps=1e-6):
    """Отрицательное лог-правдоподобие нуль-раздутой логнормали.

    Таргет — смесь точечной массы в нуле (46% пользователей) и логнормали на
    положительной части. В отличие от нашей корзинной головы, где хвост
    моделируется крайней корзиной, здесь он задан параметрически: три параметра
    вместо тридцати двух, и распределение продолжается за пределы обучающих
    значений.
    """
    logit, mu, log_sig = out[:, 0], out[:, 1], out[:, 2]
    sig = torch.nn.functional.softplus(log_sig) + eps
    # нулевая часть: обычная бинарная кросс-энтропия
    nll = torch.nn.functional.binary_cross_entropy_with_logits(
        logit, y_pos, reduction="none")
    # положительная часть: логнормаль по log(y)
    z = (y_log - mu) / sig
    ln = torch.log(sig) + 0.5 * z * z          # без констант, они не влияют на градиент
    return (nll + y_pos * ln).mean()


def ziln_expected_log1p(out, dev):
    """E[log1p Y] = P(Y>0) * E[log1p(exp(Z))], Z ~ N(mu, sigma^2).

    Метрика требует именно этого, а не E[Y]: log1p(E[Y]) != E[log1p Y].
    Интеграл берётся квадратурой Гаусса-Эрмита по 15 узлам.
    """
    logit, mu, log_sig = out[:, 0], out[:, 1], out[:, 2]
    p = torch.sigmoid(logit)
    sig = torch.nn.functional.softplus(log_sig) + 1e-6
    x = torch.tensor(_GH_X, dtype=torch.float32, device=dev)
    w = torch.tensor(_GH_W / np.sqrt(2 * np.pi), dtype=torch.float32, device=dev)
    z = mu[:, None] + sig[:, None] * x[None, :]
    return p * (torch.nn.functional.softplus(z) * w[None, :]).sum(1)


def build_bins(ys, n_bins: int):
    """Корзины по log1p(y) с отдельным классом для нуля.

    Класс 0 — ровно y == 0 (46% массы), классы 1..K — квантильные корзины по
    де-меанированному log1p среди покупавших. Де-меанирование внутри положительной
    части снимает дрейф уровня: без него одна и та же корзина означает разную
    покупательскую способность в июне и в декабре.

    Возвращает границы, центры корзин (с классом 0 в начале) и уровень, который
    надо вернуть при предсказании.
    """
    pos = [np.log1p(y[y > 0]) for y in ys]
    mus = np.array([p.mean() for p in pos])
    dm = np.concatenate([p - m for p, m in zip(pos, mus)])
    edges = np.quantile(dm, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.digitize(dm, edges[1:-1]), 0, n_bins - 1)
    centers = np.array([dm[idx == k].mean() if (idx == k).any() else 0.0
                        for k in range(n_bins)])
    return edges, np.concatenate([[0.0], centers]), float(mus.mean()), mus


def dist_labels(y, mu_a, edges, n_bins):
    """Метка класса: 0 для нуля, иначе 1 + номер корзины де-меанированного log1p."""
    lab = np.zeros(len(y), np.int64)
    m = y > 0
    lab[m] = 1 + np.clip(np.digitize(np.log1p(y[m]) - mu_a, edges[1:-1]), 0, n_bins - 1)
    return lab


def rmsle_from_log(y_true, log_pred):
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.clip(log_pred, 0, None)) ** 2)))


def to_seq(arr, dev):
    """uint8 -> log1p-шкала на GPU: делим на масштаб квантования."""
    t = torch.from_numpy(np.ascontiguousarray(arr)).to(dev, non_blocking=True).float()
    return t / SEQ_SCALE if arr.dtype == np.uint8 else t


@torch.no_grad()
def predict(model, seq, stat, batch, dev, centers=None, ziln_mode=False, clf_mode=False):
    model.eval()
    out = []
    for i in range(0, len(seq), batch):
        s = to_seq(seq[i:i + batch], dev)
        st = torch.from_numpy(stat[i:i + batch]).to(dev).float()
        # строки отсортированы по user_id одинаково во всех якорях,
        # поэтому индекс строки и есть индекс пользователя
        uid = (torch.arange(i, min(i + batch, len(seq)), device=dev)
               if model.user_dim else None)
        p = model(s, st, uid)
        if clf_mode:
            out.append(torch.sigmoid(p).cpu().numpy())
            continue
        if model.n_out == 3 and ziln_mode:
            out.append(ziln_expected_log1p(p, dev).cpu().numpy())
            continue
        if model.n_out > 1:
            # E[log1p y] = sum_k p_k * center_k; класс 0 (y == 0) даёт нулевой вклад
            q = torch.softmax(p, dim=1)
            out.append((q @ centers).cpu().numpy())
        else:
            out.append((p[0] if model.two_stage else p).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Нейросеть по подневным последовательностям")
    ap.add_argument("--mode", choices=["valid", "final"], required=True)
    ap.add_argument("--n-anchors", type=int, default=9)
    ap.add_argument("--stride", type=int, default=14)
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN,
                    help="брать последние N дней из готовых 180-дневных тензоров. "
                         "Другое рецептивное поле = другие ошибки = вклад в бленд, "
                         "и пересобирать данные не нужно")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1,
                    help="ключевой регуляризатор, до сих пор был захардкожен. Сеть "
                         "переобучается с 5-9 эпохи во ВСЕХ прогонах, а этот параметр "
                         "ни разу не подбирался")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--deterministic", action="store_true",
                    help="воспроизводимость ценой скорости. Без него задан только "
                         "torch.manual_seed, а недетерминированные ядра cuDNN дают "
                         "при повторе другие веса, и предсказания невозможно "
                         "воспроизвести побитово")
    ap.add_argument("--seed-start", type=int, default=0,
                    help="с какого сида начинать: позволяет доучить новые сиды, "
                         "не переобучая уже посчитанные")
    ap.add_argument("--no-static", action="store_true",
                    help="учить только по последовательности: слабее в одиночку, "
                         "но заметно менее коррелировано с бустингом")
    ap.add_argument("--tag", type=str, default="seqnet")
    ap.add_argument("--two-stage", action="store_true",
                    help="раздвоить голову на P(y>0) и E[log1p y|y>0]; таргет при "
                         "этом НЕ де-меанится — разложение точно только для исходного")
    ap.add_argument("--hist", action="store_true",
                    help="последовательность с полным охватом истории: 90 дней "
                         "подневно + 45 недельных агрегатов (405 дней в 135 шагах)")
    ap.add_argument("--resume", action="store_true",
                    help="пропустить сиды, чьё предсказание уже сохранено")
    ap.add_argument("--panel-seq", action="store_true",
                    help="резать последовательности из плотной панели, а не грузить "
                         "пофайлово. Файлы занимают 495 МБ на якорь и упирают нас в "
                         "22 якоря, панель занимает 1 ГБ целиком — на освободившуюся "
                         "память влезают табличные признаки для втрое большего числа "
                         "якорей. Соединяет силу dist32 с объёмом панельных моделей")
    ap.add_argument("--clf", action="store_true",
                    help="обучать ТОЛЬКО различение y>0 (BCE). Сейчас оно зашито "
                         "внутрь регрессии и даёт AUC 0.8543, при том что 46% "
                         "пользователей с нулевым таргетом дают 41.6% всей ошибки. "
                         "Отдельная модель отдаёт этой задаче всю ёмкость сети")
    ap.add_argument("--pos-only", action="store_true",
                    help="обучать ТОЛЬКО условное среднее среди купивших. Вместе с "
                         "--clf даёт точное разложение E[log1p y] = P(y>0) * E[log1p y|y>0], "
                         "точное потому, что log1p(0) = 0")
    ap.add_argument("--ziln", action="store_true",
                    help="нуль-раздутая логнормаль (Google, arXiv 1912.07753): три "
                         "параметра (p, mu, sigma) вместо 32 корзин, хвост задан "
                         "параметрически. Работает с СЫРЫМ таргетом, поэтому заодно "
                         "проверяет отказ от де-меанирования")
    ap.add_argument("--no-demean", action="store_true",
                    help="не вычитать уровень якоря из таргета. Де-меанирование "
                         "выбрано в первый день по январскому окну, которое врёт про "
                         "уровень и разброс; наши модели требуют растяжения 1.0275, "
                         "то есть недооценивают разброс — возможный след этого выбора")
    ap.add_argument("--dist", action="store_true",
                    help="распределительный режим: softmax по корзинам log1p(y) с "
                         "отдельным классом для нуля, предсказание = матожидание. "
                         "Оптимум RMSLE — это E[log1p y], и здесь оно считается "
                         "явно, а не подгоняется точечной регрессией")
    ap.add_argument("--n-bins", type=int, default=32)
    ap.add_argument("--aux-weight", type=float, default=0.2,
                    help="вес вспомогательного BCE-лосса на классификаторе")
    ap.add_argument("--arch", choices=["gru", "transformer"], default="gru",
                    help="энкодер последовательности. Трансформер — последний "
                         "непробованный класс архитектуры; свёрточный фронт остаётся, "
                         "он сжимает 180 дней до 45 шагов, иначе самовнимание на 180 "
                         "позициях вчетверо дороже без выигрыша")
    ap.add_argument("--user-emb", type=int, default=0,
                    help="размерность обучаемого эмбеддинга пользователя. Единственный "
                         "канал, добавляющий модели НОВУЮ информацию: персональную "
                         "предрасположенность, не выводимую из агрегатов и ряда. "
                         "ПРОВЕРЕНО И ПРОВАЛИЛОСЬ: размерность 16 дала итог 1.76378 "
                         "при базовых 1.666. Обучающая ошибка падала 2.2 -> 1.06, "
                         "валидационная росла до 2.09, лучшая эпоха первая — 250 тысяч "
                         "пользователей на 22 якорях запоминают таргет. Флаг оставлен "
                         "как задокументированный отрицательный результат")
    ap.add_argument("--emb-lr-scale", type=float, default=1.0,
                    help="множитель learning rate для эмбеддингов относительно сети")
    ap.add_argument("--valid-anchor", type=str, default=None,
                    help="альтернативный валидационный якорь (ГГГГ-ММ-ДД) для второго "
                         "окна. Все решения до сих пор принимались по одному январскому "
                         "окну, которое для моделей с утечкой завышено восьмикратно; "
                         "второе окно позволяет отличить реальный эффект от подгонки")
    ap.add_argument("--recency-halflife", type=float, default=None,
                    help="период полураспада веса якоря в днях. Якоря ближе к "
                         "предсказываемому окну релевантнее: добавление трёх свежих "
                         "якорей дало 0.0006 — крупнейший модельный эффект за работу. "
                         "Реализовано выборкой якорей пропорционально весу, а не "
                         "домножением лосса: Adam нормирует градиенты по RMS и "
                         "масштаб лосса частично сокращается, а частота выборки — нет")
    ap.add_argument("--stop-epoch", type=int, default=None,
                    help="оборвать обучение после N эпох, оставив расписание LR "
                         "рассчитанным на --epochs. Нужно в режиме final: там нет "
                         "валидации, а лучшая эпоха известна из прогона valid")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"устройство: {dev}"
          f"{' ' + torch.cuda.get_device_name(0) if dev.type == 'cuda' else ''}")

    from datetime import date as _date, timedelta as _timedelta
    valid_anchor = _date.fromisoformat(args.valid_anchor) if args.valid_anchor else VALID_ANCHOR
    if args.mode == "final":
        anchors = final_anchors(stride=args.stride, n=args.n_anchors)
    elif args.valid_anchor:
        # обучающий таргет не должен заходить за валидационный якорь
        # фильтруем ШТАТНУЮ сетку, а не строим новую от limit: сетка от другой
        # даты смещается на дни и указывает на несуществующие якоря
        limit = valid_anchor - _timedelta(days=HORIZON)
        pool = [a for a in train_anchors(stride=args.stride) if a <= limit]
        anchors = pool[-args.n_anchors:]
    else:
        anchors = train_anchors(stride=args.stride)[-args.n_anchors:]
    test_anchor = valid_anchor if args.mode == "valid" else PREDICT_ANCHOR
    print(f"режим {args.mode}: якорей {len(anchors)} ({anchors[0]} .. {anchors[-1]}), "
          f"тест {test_anchor}")

    # Признаки копились в float32 и ужимались лишь в конце: на 57 якорях это
    # пик в 23 ГБ и гарантированный отказ. Поэтому два прохода — сначала
    # статистики нормировки по разреженной подвыборке, потом загрузка каждого
    # якоря с немедленным сжатием в float16.
    Xs, ys, feats = [], [], None
    if args.mode == "valid":
        sub, feats = [], None
        for a in anchors:
            X, _, feats, _ = load_static(a, feats)
            sub.append(X[::7])
            del X
        mu, sd = fit_norm(np.vstack(sub))
        del sub
        np.savez(norm_file(args.tag), mu=mu, sd=sd, feats=np.array(feats))
    else:
        z = np.load(norm_file(args.tag), allow_pickle=True)
        mu, sd, feats = z["mu"], z["sd"], list(z["feats"])
    for a in anchors:
        X, y, feats, _ = load_static(a, feats)
        Xs.append(apply_norm(X, mu, sd).astype(np.float16))
        ys.append(y)
        del X
    Xt, yt, _, user_ids = load_static(test_anchor, feats)
    Xt = apply_norm(Xt, mu, sd).astype(np.float16)
    if args.no_static:
        # оставляем один нулевой столбец, чтобы не трогать сигнатуру модели
        Xs = [np.zeros((x.shape[0], 1), np.float16) for x in Xs]
        Xt = np.zeros((Xt.shape[0], 1), np.float16)
        feats = ["_none"]

    # Обычный режим: таргет де-меанится по своему якорю, уровень возвращается
    # при блендинге. Двухстадийный режим работает с исходным log1p — произведение
    # P(y>0) * E[log1p y|y>0] равно именно ему, а не отклонению от уровня.
    centers_t = None
    if args.clf:
        tg = [(y > 0).astype(np.float32) for y in ys]
        pos = None
        print(f"  классификатор: доля положительных "
              f"{np.mean([np.mean(y > 0) for y in ys]):.3f}")
    elif args.pos_only:
        # веса нулевых строк обнуляются, поэтому уровень считается по купившим
        tg = []
        for y in ys:
            v = np.log1p(y).astype(np.float32)
            m = v[y > 0].mean() if (y > 0).any() else 0.0
            tg.append(np.where(y > 0, v - m, 0.0).astype(np.float32))
        pos = [(y > 0).astype(np.float32) for y in ys]
    elif args.ziln:
        # ZILN учится на сыром log(y) для положительных и флаге положительности
        tg = [np.log(np.maximum(y, 1e-6)).astype(np.float32) for y in ys]
        pos = [(y > 0).astype(np.float32) for y in ys]
    elif args.dist:
        edges, centers, mu_add, mus = build_bins(ys, args.n_bins)
        if args.no_demean:
            # корзины по СЫРОМУ log1p: уровень якоря не вычитается
            mus = np.zeros(len(ys)); mu_add = 0.0
            edges, centers, _, _ = build_bins([np.concatenate(ys)], args.n_bins)
        tg = [dist_labels(y, m, edges, args.n_bins) for y, m in zip(ys, mus)]
        pos = None
        # вклад класса 0 (y == 0) в матожидание равен нулю по построению
        centers_t = torch.tensor(
            np.concatenate([[0.0], centers[1:] + mu_add]), dtype=torch.float32, device=dev)
        print(f"корзин {args.n_bins}, уровень положительной части {mu_add:.4f}, "
              f"доля нулей {np.mean([np.mean(y == 0) for y in ys]):.3f}")
    elif args.two_stage:
        tg = [np.log1p(y).astype(np.float32) for y in ys]
        pos = [(y > 0).astype(np.float32) for y in ys]
    else:
        tg = [(np.log1p(y) - np.log1p(y).mean()).astype(np.float32) for y in ys]
        pos = None

    t0 = time.time()
    if args.panel_seq:
        import panel as _pn
        assert not args.hist, "--panel-seq и --hist несовместимы"
        _P, _, _pu = _pn.load(mmap=False)
        # порядок пользователей в панели и в признаках совпадает (проверено)
        _days = [_pn.day_index(a) for a in anchors]
        _day_t = _pn.day_index(test_anchor)
        seqs = None
        get_seq = lambda i, u: _pn.window(_P, _days[i], u, args.seq_len)
        seq_t = _pn.window(_P, _day_t, np.arange(len(_pu)), args.seq_len)
        print(f"признаков {len(feats)}, последовательность {args.seq_len}x{len(SEQ_CHANNELS)} "
              f"из панели ({_P.nbytes / 2**30:.2f} ГБ на все якоря), "
              f"загрузка {time.time() - t0:.0f}с")
    else:
        path_fn = seq_hist.seq_path if args.hist else seq_path
        cut = (lambda a: a if args.hist or args.seq_len >= SEQ_LEN
               else np.ascontiguousarray(a[:, -args.seq_len:, :]))
        seqs = [cut(np.load(path_fn(a))) for a in anchors]
        get_seq = lambda i, u: seqs[i][u]
        seq_t = cut(np.load(path_fn(test_anchor)))
        print(f"признаков {len(feats)}, последовательность {seqs[0].shape[1]}x{len(SEQ_CHANNELS)}, "
              f"загрузка {time.time() - t0:.0f}с")

    anchor_w = None
    if args.recency_halflife:
        ages = np.array([(anchors[-1] - a).days for a in anchors], dtype=np.float64)
        anchor_w = 0.5 ** (ages / args.recency_halflife)
        anchor_w /= anchor_w.sum()
        print("веса якорей по свежести (полураспад %.0f дней): "
              "самый свежий %.4f, самый старый %.4f"
              % (args.recency_halflife, anchor_w[-1], anchor_w[0]))

    level = window_level(yt) if yt is not None else VALID_LEVEL
    if yt is not None and abs(level - VALID_LEVEL) > 1e-4:
        print(f"истинный уровень окна {test_anchor}: {level:.4f} "
              f"(январский был {VALID_LEVEL:.4f})")

    n_users = Xs[0].shape[0]
    starts = np.arange(0, n_users, args.batch)
    steps_per_epoch = len(anchors) * len(starts)
    rng = np.random.default_rng(SEED)

    preds, hist = [], []
    for s in range(args.seed_start, args.seed_start + args.seeds):
        done = PRED_DIR / f"{args.mode}_{args.tag}_seed{s}.npy"
        if args.resume and done.exists():
            print(f"  сид {s}: уже посчитан, пропускаю")
            preds.append(np.load(done))
            hist.append(dict(seed=s, best=None, best_epoch=None, resumed=True))
            continue
        torch.manual_seed(SEED + s)
        if args.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        model = SeqNet(len(SEQ_CHANNELS), len(feats), args.hidden, dropout=args.dropout,
                       two_stage=args.two_stage,
                       n_out=3 if args.ziln else (args.n_bins + 1 if args.dist else 1),
                       n_users=n_users if args.user_emb else 0,
                       user_dim=args.user_emb, arch=args.arch).to(dev)
        if args.user_emb:
            # эмбеддинги обновляются реже (каждый пользователь встречается раз за
            # проход по якорю), поэтому им отдельная группа параметров
            emb_p = list(model.user_emb.parameters())
            emb_ids = {id(q) for q in emb_p}
            rest = [q for q in model.parameters() if id(q) not in emb_ids]
            opt = torch.optim.AdamW([
                dict(params=rest, lr=args.lr, weight_decay=args.weight_decay),
                dict(params=emb_p, lr=args.lr * args.emb_lr_scale, weight_decay=1e-5),
            ])
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch)
        lossf = nn.CrossEntropyLoss() if args.dist else nn.MSELoss()
        bce = nn.BCEWithLogitsLoss()

        best, best_ep, best_pred = np.inf, -1, None
        for ep in range(args.stop_epoch or args.epochs):
            te = time.time()
            perms = [rng.permutation(n_users) for _ in anchors]
            if anchor_w is None:
                order = [(a, st) for a in range(len(anchors)) for st in starts]
                rng.shuffle(order)
            else:
                n_steps = len(anchors) * len(starts)
                picks = rng.choice(len(anchors), size=n_steps, p=anchor_w)
                order = [(int(a), int(rng.choice(starts))) for a in picks]

            model.train()
            tr_loss = 0.0
            for a, st in order:
                u = np.sort(perms[a][st:st + args.batch])
                sq = to_seq(get_seq(a, u), dev)
                stt = torch.from_numpy(Xs[a][u]).to(dev, non_blocking=True).float()
                tgt = torch.from_numpy(tg[a][u]).to(dev, non_blocking=True)
                uid = (torch.from_numpy(u).to(dev, non_blocking=True)
                       if args.user_emb else None)
                if args.clf:
                    loss = bce(model(sq, stt, uid), tgt)
                elif args.pos_only:
                    m_ = torch.from_numpy(pos[a][u]).to(dev, non_blocking=True)
                    if m_.sum() < 1:
                        continue
                    loss = (((model(sq, stt, uid) - tgt) ** 2) * m_).sum() / m_.sum()
                elif args.ziln:
                    loss = ziln_loss(model(sq, stt, uid), 
                                     torch.from_numpy(pos[a][u]).to(dev, non_blocking=True), tgt)
                elif args.dist:
                    loss = lossf(model(sq, stt, uid), tgt)
                elif args.two_stage:
                    p, logit = model(sq, stt, uid)
                    # вспомогательный BCE не даёт классификатору вырождаться:
                    # без него произведение может компенсировать плохое p ростом m
                    loss = lossf(p, tgt) + args.aux_weight * bce(
                        logit, torch.from_numpy(pos[a][u]).to(dev, non_blocking=True))
                else:
                    loss = lossf(model(sq, stt, uid), tgt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                sched.step()
                tr_loss += loss.item() * len(u)
            tr_loss /= len(anchors) * n_users

            pv = predict(model, seq_t, Xt, args.batch, dev, centers_t, args.ziln, args.clf)
            msg = (f"  сид {s} эпоха {ep+1:2d}/{args.epochs}  train MSE={tr_loss:.4f}  "
                   f"{time.time()-te:.0f}с")
            if yt is not None:
                if args.clf:
                    # у классификатора RMSLE по вероятностям бессмысленна:
                    # меряем AUC, а в `best` кладём 1-AUC, чтобы отбор шёл на минимум
                    from sklearn.metrics import roc_auc_score
                    auc = roc_auc_score(yt > 0, pv)
                    score = 1.0 - auc
                    print(msg + f"  valid AUC={auc:.5f}")
                else:
                    score = rmsle_from_log(yt, pv - pv.mean() + level)
                    print(msg + f"  valid RMSLE={score:.5f}")
                if score < best:
                    best, best_ep, best_pred = score, ep + 1, pv
            else:
                print(msg)
                best_pred = pv
        preds.append(best_pred)
        hist.append(dict(seed=s, best=None if yt is None else best, best_epoch=best_ep))
        # предсказание каждого сида сохраняется сразу: обрыв длинного прогона
        # тогда не обесценивает уже посчитанное
        np.save(PRED_DIR / f"{args.mode}_{args.tag}_seed{s}.npy", best_pred)
        if yt is not None:
            lbl = f"AUC={1-best:.5f}" if args.clf else f"RMSLE={best:.5f}"
            print(f"  сид {s}: лучший {lbl} на эпохе {best_ep}")

    pred = np.mean(preds, axis=0)
    # У классификатора сохраняем сырые вероятности: центрировать их и добавлять
    # уровень бессмысленно, они нужны сомножителем в разложении.
    np.save(PRED_DIR / f"{args.mode}_{args.tag}.npy",
            pred if args.clf else pred - pred.mean() + level)
    np.save(PRED_DIR / f"{args.mode}_user_ids.npy", user_ids)
    if yt is not None:
        print(f"\nитог: RMSLE={rmsle_from_log(yt, pred - pred.mean() + VALID_LEVEL):.5f}")
        (MODELS_DIR / f"{args.tag}_valid.json").write_text(json.dumps(
            # Полный набор аргументов и состояние среды: без этого тег модели
            # не описывает сам себя. Раньше сохранялись только epochs/batch/lr/
            # hidden/n_anchors, и по метаданным было не отличить dist32 от d32v5 —
            # их разделяют --dist, --n-bins и версия признаков.
            dict(hist=hist, args=vars(args),
                 env={k: os.environ.get(k) for k in
                      ("FEATURE_VERSION", "LONG_WINDOWS", "SEQ_C13")},
                 anchors=[str(a) for a in anchors], test_anchor=str(test_anchor)),
            ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
