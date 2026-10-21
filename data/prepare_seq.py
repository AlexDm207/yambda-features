"""Данные для трансформера над историей событий (день 4).

Та же информация, что у бустинга: для показа в день d модель видит события пользователя строго до дня d
(те же сплиты, те же показы-кандидаты). Разница только в том, что вместо ручных агрегатов модель
получает сами события и учится агрегировать сама.

Событие кодируется: трек, артист, тип (прослушивание с корзиной доли прослушивания / лайк / дизлайк /
снятие лайка / снятие дизлайка), органическое или из рекомендаций, день события.
Словари треков и артистов строятся ТОЛЬКО по дням train (<= последний день train) — без утечки.

Выход: data/seq/events.npz, data/seq/cands.npz, data/seq/meta.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "configs/base.yaml", encoding="utf-8"))
TRAIN_END = cfg["split"]["train_days"][1]
V_ITEM, V_ART = 200_000, 50_000  # 0 = паддинг, 1 = редкий/неизвестный
OUT = ROOT / "data/seq"
OUT.mkdir(parents=True, exist_ok=True)

EVP = ROOT / "data/processed/events.parquet"
COLS = ["uid", "timestamp", "day", "item_id", "artist_id", "etype", "is_organic", "ratio"]


def vocab(col, size):
    # словари считаем лениво по файлу: в память попадает только таблица частот
    return (pl.scan_parquet(EVP).filter((pl.col("day") <= TRAIN_END) & pl.col(col).is_not_null())
            .group_by(col).len().sort(["len", col], descending=[True, False]).head(size - 2).collect()
            .with_row_index("tok", offset=2).select(col, pl.col("tok").cast(pl.UInt32)))


vi, va = vocab("item_id", V_ITEM), vocab("artist_id", V_ART)
r = pl.col("ratio").cast(pl.Int32)
etype = (pl.when(pl.col("etype") == 0).then(1 + (r >= 10).cast(pl.Int32) + (r >= 30).cast(pl.Int32)
                                             + (r >= 50).cast(pl.Int32) + (r >= 80).cast(pl.Int32))
         .otherwise(5 + pl.col("etype").cast(pl.Int32)))  # 1..5 прослушивания по доле, 6 лайк, 7 дизлайк, 8, 9 снятия
users = pl.scan_parquet(EVP).select("uid").unique().collect()["uid"].sort()
uidx = pl.DataFrame({"uid": users, "u": np.arange(users.len(), dtype=np.int32)})
# Порциями по диапазонам пользователей (порядок сохраняется: пользователи по возрастанию, внутри — по времени).
# Так пиковая память — несколько сотен МБ вместо нескольких ГБ.
parts = {k: [] for k in ["item", "art", "ecode", "org", "day", "u"]}
n_chunks = 16
bounds = np.linspace(0, users.len(), n_chunks + 1).astype(int)
for c in range(n_chunks):
    lo, hi = users[int(bounds[c])], users[int(bounds[c + 1]) - 1]
    ch = (pl.scan_parquet(EVP).select(COLS).filter(pl.col("uid").is_between(lo, hi)).collect()
          .sort(["uid", "timestamp"], maintain_order=True)
          .join(vi.rename({"tok": "itok"}), on="item_id", how="left", maintain_order="left")
          .join(va.rename({"tok": "atok"}), on="artist_id", how="left", maintain_order="left")
          .join(uidx, on="uid", maintain_order="left")
          .with_columns(pl.col("itok").fill_null(1), pl.col("atok").fill_null(1), etype.alias("ecode")))
    parts["item"].append(ch["itok"].to_numpy().astype(np.int32))
    parts["art"].append(ch["atok"].to_numpy().astype(np.int32))
    parts["ecode"].append(ch["ecode"].to_numpy().astype(np.int8))
    parts["org"].append((ch["is_organic"].to_numpy() + 1).astype(np.int8))
    parts["day"].append(ch["day"].to_numpy().astype(np.int16))
    parts["u"].append(ch["u"].to_numpy().astype(np.int32))
    print(f"  порция {c + 1}/{n_chunks}: {ch.height:,} событий")
    del ch
E = {k: np.concatenate(v) for k, v in parts.items()}
del parts
u = E.pop("u")
assert np.all(np.diff(u) >= 0)
key = u.astype(np.int64) * 1000 + E["day"]
assert np.all(np.diff(key) >= 0), "события не отсортированы по (пользователь, день)"
np.savez(OUT / "events.npz", **E, key=key, ustart=np.searchsorted(u, np.arange(users.len())).astype(np.int64))
print(f"событий: {len(u):,}, пользователей: {users.len()}, "
      f"покрытие словаря треков: {(E['item'] > 1).mean():.3f}, артистов: {(E['art'] > 1).mean():.3f}")
del E, key, u

imp = (pl.read_parquet(ROOT / "data/processed/impressions.parquet",
                       columns=["imp_id", "uid", "day", "item_id", "artist_id", "length", "label", "split", "group_id"])
       .sort(["group_id", "imp_id"])
       .join(vi.rename({"tok": "itok"}), on="item_id", how="left", maintain_order="left")
       .join(va.rename({"tok": "atok"}), on="artist_id", how="left", maintain_order="left")
       .join(uidx, on="uid", maintain_order="left")
       .with_columns(pl.col("itok").fill_null(1), pl.col("atok").fill_null(1)))
ll = np.log1p(imp["length"].fill_null(0).to_numpy().astype(np.float64))
trm = imp["split"].to_numpy() == "train"
ll = ((ll - ll[trm].mean()) / ll[trm].std()).astype(np.float32)
split_code = {"train": 0, "val": 1, "test": 2}
np.savez(OUT / "cands.npz",
         imp_id=imp["imp_id"].to_numpy().astype(np.int64), item=imp["itok"].to_numpy().astype(np.int32),
         art=imp["atok"].to_numpy().astype(np.int32), loglen=ll, label=imp["label"].to_numpy().astype(np.float32),
         split=np.array([split_code[s] for s in imp["split"].to_list()], dtype=np.int8),
         group=imp["group_id"].to_numpy().astype(np.int64), u=imp["u"].to_numpy().astype(np.int32),
         day=imp["day"].to_numpy().astype(np.int16))
json.dump({"V_ITEM": V_ITEM, "V_ART": V_ART, "train_end": TRAIN_END, "n_users": users.len(),
           "cand_item_coverage": float((imp["itok"] > 1).mean())},
          open(OUT / "meta.json", "w"), indent=2)
print("кандидатов:", imp.height, " покрытие словаря у кандидатов:", float((imp["itok"] > 1).mean()))
