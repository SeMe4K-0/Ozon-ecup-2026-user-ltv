"""Пути, даты и константы соревнования."""
from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TRAIN_PARQUET = ROOT / "train.parquet"
SAMPLE_SUBMIT = ROOT / "sample_submit.csv"

ARTIFACTS = ROOT / "artifacts"
FEATURES_DIR = ARTIFACTS / "features"
MODELS_DIR = ARTIFACTS / "models"
SUBMISSIONS_DIR = ROOT / "submissions"
# Предсказания моделей живут здесь. Раньше путь определялся в ensemble.py, а тот
# импортирует lightgbm на уровне модуля — из-за чего усреднение сидов и сборка
# бленда падали в окружении без lightgbm, хотя к нему отношения не имеют.
PRED_DIR = ARTIFACTS / "preds"

for _d in (FEATURES_DIR, MODELS_DIR, SUBMISSIONS_DIR, PRED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- календарь соревнования ---------------------------------------------
DATA_START = date(2025, 1, 1)
DATA_END = date(2026, 2, 13)          # последний день с данными
HORIZON = 30                          # горизонт прогноза, дней

# Якорь (anchor) = последний день, данные за который доступны для признаков.
# Таргет считается по окну [anchor + 1, anchor + HORIZON] включительно.
PREDICT_ANCHOR = DATA_END                       # 2026-02-13
PREDICT_START = PREDICT_ANCHOR + timedelta(days=1)      # 2026-02-14
PREDICT_END = PREDICT_ANCHOR + timedelta(days=HORIZON)  # 2026-03-15

# Валидационный якорь: единственное полностью наблюдаемое окно в конце данных.
VALID_ANCHOR = DATA_END - timedelta(days=HORIZON)       # 2026-01-14

# Обучающие якоря не должны заглядывать за VALID_ANCHOR своим таргетом,
# иначе валидация перестанет быть честной.
TRAIN_ANCHOR_LAST = VALID_ANCHOR - timedelta(days=HORIZON)   # 2025-12-15
TRAIN_ANCHOR_STRIDE = 14
MIN_HISTORY_DAYS = 180        # минимум истории под самое длинное окно признаков

N_USERS = 250_000

# Истинный уровень январского окна: среднее log1p(таргета) по всем пользователям.
# Раньше это число было продублировано в blend.py, seq_model.py и panel_model.py
# под тремя разными именами — при любой правке одно из мест отставало.
VALID_LEVEL = 2.2421

# --- признаки ------------------------------------------------------------
# Версия входит в имя файла кэша: разные версии признаков не перетирают друг
# друга, что позволяет сравнивать эксперименты без пересборки.
# Переопределяется переменной окружения: разные модели ансамбля могут
# жить на разных версиях признаков, лишь бы valid и final у одной
# модели совпадали (нормировка сети привязана к числу колонок).
FEATURE_VERSION = os.environ.get("FEATURE_VERSION", "v3")

# Данных 409 дней, а окна кончались на 180. panel240 показал, что глубина
# истории даёт панельной модели 0.0064 — самый крупный прирост базовой модели за
# всё время, но табличная часть этой глубины не видела вовсе.
# У якорей с короткой историей длинные окна выйдут усечёнными и несопоставимыми
# между якорями; ранговые версии (rk_*) считаются внутри якоря и потому корректны.
WINDOWS = [3, 7, 14, 30, 60, 90, 180]
if os.environ.get("LONG_WINDOWS"):
    WINDOWS = WINDOWS + [270, 365]

SUM_COLS = [
    "gmv", "gmv_search", "gmv_cat",
    "to_ord", "to_cart", "searches",
    "search_to_ord", "cat_to_ord", "search_to_cart", "cat_to_cart",
    "search", "cat",
]

# --- какие колонки не являются признаками ---------------------------------
# Признаки, однозначно выдающие дату якоря. Панель пользователей фиксирована с
# 2025-01-01, поэтому "дней с первой активности" растёт на единицу каждый день:
# классификатор train-против-test отличает строки по ним с AUC 1.0000. Модель
# по ним опознаёт якорь и запоминает его уровень, а на предсказании вынуждена
# экстраполировать (tenure ~409 против максимума ~348 в обучении).
# Ранговые версии (rk_tenure_days) сохранены — они сопоставимы между якорями.
ANCHOR_ID_COLS = {"tenure_days", "first_ord_db", "ord_lifespan"}

DROP = {"user_id", "anchor_date", "target"} | ANCHOR_ID_COLS

TARGET = "target"
SEED = 42

# --- последовательности для нейросети ------------------------------------
SEQ_DIR = ARTIFACTS / "seq"
SEQ_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 180          # дней истории на пользователя, последний день = якорь
# Канал "row" отмечает сам факт визита: 18% строк не содержат ни одного
# действия, и без него такие дни неотличимы от отсутствующих.
# search_to_ord / cat_to_ord — единственный кусок исходных данных, который сеть
# до сих пор не видела: разделение заказов по источнику. Деньги были разделены
# (gmv_search/gmv_cat), а сами заказы — нет, хотя именно их число доминирует
# в важности признаков у бустинга.
SEQ_CHANNEL_SET = "c11"
SEQ_CHANNELS = ["gmv", "to_ord", "to_cart", "searches", "gmv_search", "gmv_cat",
                "search_to_ord", "cat_to_ord", "search", "cat", "row"]
# search_to_cart и cat_to_cart — последний кусок сырых данных, которого сеть не
# видела ни разу: разделение добавлений в корзину по источнику. Заказы по
# источнику (search_to_ord/cat_to_ord) уже есть, корзины — нет. Табличная часть
# их использует, последовательности нет. Панель хранится под именем с числом
# каналов, поэтому 13ch не конфликтует с накопленным 11ch.
if os.environ.get("SEQ_C13"):
    SEQ_CHANNEL_SET = "c13"
    SEQ_CHANNELS = SEQ_CHANNELS[:-1] + ["search_to_cart", "cat_to_cart", "row"]
SEQ_LOG_CHANNELS = {"gmv", "to_ord", "to_cart", "searches", "gmv_search", "gmv_cat",
                    "search_to_ord", "cat_to_ord", "search_to_cart", "cat_to_cart"}

# Каналы хранятся в uint8: значения это log1p, максимум по данным ~11.2
# (log1p от максимального дневного gmv), при масштабе 20 это 224 < 255.
# Шаг квантования 0.05 в лог-шкале — на фоне сигнала пренебрежимо, зато
# 25 якорей занимают 10 ГБ вместо 19 и помещаются в память целиком.
SEQ_SCALE = 20.0


# Ограничение TRAIN_ANCHOR_LAST нужно только чтобы обучающий таргет не залез
# за валидационный якорь. Финальной модели валидация уже не нужна, поэтому ей
# доступны и более свежие якоря: их таргеты полностью наблюдаемы (последний
# кончается 2026-02-04). Это самые релевантные данные, и раньше они не
# использовались вовсе.
EXTRA_FINAL_ANCHORS = [date(2025, 12, 22), date(2025, 12, 29), date(2026, 1, 5)]


def final_anchors(stride: int = TRAIN_ANCHOR_STRIDE, n: int | None = None) -> list[date]:
    """Якоря финальной модели: обучающие + свежие + валидационный."""
    out = sorted(set(train_anchors(stride=stride)) | set(EXTRA_FINAL_ANCHORS)
                 | {VALID_ANCHOR})
    return out[-n:] if n else out


def train_anchors(last: date = TRAIN_ANCHOR_LAST,
                  stride: int = TRAIN_ANCHOR_STRIDE,
                  min_history: int = MIN_HISTORY_DAYS) -> list[date]:
    """Обучающие якоря с шагом `stride` назад от `last`, пока хватает истории."""
    earliest = DATA_START + timedelta(days=min_history - 1)
    out, a = [], last
    while a >= earliest:
        out.append(a)
        a -= timedelta(days=stride)
    return sorted(out)


def build_submission(user_ids, log_pred, name: str):
    """Собрать сабмит из предсказаний в log1p-шкале и проверить его.

    Логика была продублирована в blend.py и rebuild_final.py: порядок строк
    берётся из образца организаторов, пропуски заполняются нулями. Расхождение
    между копиями заметить трудно, а цена ошибки — недействительный сабмит.
    """
    import numpy as np
    import polars as pl

    log_pred = np.clip(log_pred, 0, None)
    sub = pl.DataFrame({"user_id": user_ids, "predict": np.expm1(log_pred)})
    order = pl.read_csv(SAMPLE_SUBMIT).select("user_id")
    sub = order.join(sub, on="user_id", how="left").with_columns(
        pl.col("predict").fill_null(0.0))

    assert sub.height == N_USERS, f"строк {sub.height}, ожидалось {N_USERS}"
    assert sub["user_id"].to_list() == order["user_id"].to_list(), "порядок пользователей нарушен"
    assert bool(sub["predict"].is_finite().all()), "есть бесконечные значения"
    assert sub["predict"].min() >= 0, "есть отрицательные предсказания"

    out = SUBMISSIONS_DIR / f"{name}.csv"
    sub.write_csv(out)
    return out, sub
