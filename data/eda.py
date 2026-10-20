"""Шаг 1б. EDA под постановку: какие сигналы вообще есть в показах и что из этого следует для признаков.
Все «исторические» величины считаются строго по дням < дня показа — так же, как будут считаться признаки."""
import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(200)

imp = pl.read_parquet(P / "impressions.parquet")
ev = pl.read_parquet(P / "events.parquet", columns=["uid", "day", "item_id", "artist_id", "etype", "is_organic", "ratio"])
lis = ev.filter(pl.col("etype") == 0)
res = {}

# 1. Типы событий и доля органики
print(ev.group_by("etype", "is_organic").agg(pl.len()).sort("etype", "is_organic"))

# 2. Динамика по дням: число показов и доля дослушиваний
daily = imp.group_by("day").agg(pl.len().alias("n"), pl.col("label").mean().alias("pos")).sort("day")
daily.write_csv(OUT / "eda_daily.csv")
print(daily.describe())


# 3. Прямые связи: слышал ли пользователь этот трек / артиста раньше (дни < d)
def first_day(keys):
    return lis.group_by(keys).agg(pl.col("day").min().alias("first_day"))


for name, keys in [("item", ["uid", "item_id"]), ("artist", ["uid", "artist_id"])]:
    x = imp.join(first_day(keys), on=keys, how="left").with_columns(
        (pl.col("first_day") < pl.col("day")).fill_null(False).alias(f"seen_{name}")
    )
    t = x.group_by("split", f"seen_{name}").agg(pl.len().alias("n"), pl.col("label").mean().alias("pos")).sort("split", f"seen_{name}")
    print(f"\n--- пользователь раньше слушал этот {name}? ---\n", t)
    res[f"seen_{name}"] = t.to_dicts()

# 4. Холодный старт треков: был ли трек хоть у кого-то до дня показа
x = imp.join(lis.group_by("item_id").agg(pl.col("day").min().alias("fd")), on="item_id", how="left")
t = x.group_by("split").agg((pl.col("fd") >= pl.col("day")).mean().alias("share_cold_item"))
print("\nдоля показов с треком, которого никто не слушал раньше:\n", t)
res["cold_item"] = t.to_dicts()

# 5. Насколько метка «личная»: разброс доли дослушиваний между пользователями и между треками (train)
tr = imp.filter(pl.col("split") == "train")
for k in ["uid", "item_id", "artist_id"]:
    s = tr.group_by(k).agg(pl.len().alias("n"), pl.col("label").mean().alias("p")).filter(pl.col("n") >= 20)
    sd = s["p"].std() if s.height > 1 else float("nan")
    print(f"{k}: объектов с >=20 показами {s.height}, std доли дослушиваний {sd:.3f}")
    res[f"std_{k}"] = sd

# 6. Размер групп пользователь-день (для NDCG@K)
gs = imp.group_by("split", "group_id").agg(pl.len().alias("n"))
print("\nразмер групп:\n", gs.group_by("split").agg([pl.col("n").quantile(q).alias(f"q{q}") for q in (0.1, 0.5, 0.9)]))

# 7. Позиция внутри дня: дослушивание в начале и в конце дневной сессии
pos = imp.with_columns(pl.int_range(pl.len()).over("group_id").alias("pos")).with_columns(pl.col("pos").clip(0, 50))
print("\nдоля дослушиваний по номеру показа за день (0..50):\n",
      pos.filter(pl.col("split") == "train").group_by("pos").agg(pl.col("label").mean()).sort("pos").head(12))

json.dump(res, open(OUT / "eda.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=str)
print("\nсохранено: reports/eda.json, reports/eda_daily.csv")
