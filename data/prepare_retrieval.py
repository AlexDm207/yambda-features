"""Индекс для «поиска по истории» (target-aware retrieval, идея SIM — Pi et al., 2020).

Проблема: трансформер видит последние L событий (L=512 — это ~1 месяц у среднего пользователя,
~10% его истории), а лучшие признаки бустинга считаются по ВСЕЙ истории (окна inf).
Решение без ручных агрегатов: для каждого кандидата достаём из ВСЕЙ истории пользователя до дня показа
последние K событий с ЭТИМ ЖЕ треком и последние K событий с ЭТИМ ЖЕ артистом — сами события,
а не их статистики, — и отдаём модели вместе с числом найденных совпадений.

Индекс: события, отсортированные по (пользователь, трек, день) и (пользователь, артист, день);
поиск — двоичный (searchsorted), без перебора истории.
Выход: data/seq/retrieval.npz (+ сырые id кандидатов в cands_raw.npz).
"""
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SEQ = ROOT / "data/seq"
E = np.load(SEQ / "events.npz")
day, key_ud = E["day"], E["key"]
u = (key_ud // 1000).astype(np.int64)

raw = pl.read_parquet(ROOT / "data/processed/events.parquet", columns=["uid", "day", "item_id", "artist_id"])
assert raw.height == len(day) and np.array_equal(raw["day"].to_numpy(), day), "порядок событий не совпадает с events.npz"


def index(col):
    x = raw[col].fill_null(0).to_numpy().astype(np.int64)
    assert x.max() < 2 ** 24
    k = ((u << 24) | x) * 512 + day          # (пользователь, объект, день) в одном int64
    order = np.argsort(k, kind="stable").astype(np.int32)
    return k[order], order


ki, oi = index("item_id")
ka, oa = index("artist_id")
np.savez(SEQ / "retrieval.npz", key_item=ki, ord_item=oi, key_art=ka, ord_art=oa)
del raw

K = np.load(SEQ / "cands.npz")
imp = (pl.read_parquet(ROOT / "data/processed/impressions.parquet", columns=["imp_id", "group_id", "item_id", "artist_id"])
       .sort(["group_id", "imp_id"]))
assert np.array_equal(imp["imp_id"].to_numpy(), K["imp_id"]), "порядок кандидатов не совпадает с cands.npz"
np.savez(SEQ / "cands_raw.npz", item=imp["item_id"].fill_null(0).to_numpy().astype(np.int64),
         art=imp["artist_id"].fill_null(0).to_numpy().astype(np.int64))
print("индекс готов:", len(ki), "событий")
