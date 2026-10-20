"""Шаг 1. Единая таблица событий + таблица «показов» с метками и временным разбиением.

Показ   = прослушивание, которое запустила рекомендательная система (is_organic = 0).
Метка   = пользователь дослушал трек минимум до ratio_threshold % (аналог клика после показа).
Разбиение по дням: train / val / gap / test (см. configs/base.yaml).

Выход (data/processed):
  events.parquet       все события всех типов (для расчёта признаков и истории трансформера)
  impressions.parquet  показы из train/val/test с метками, imp_id и group_id (= пользователь-день)
  meta.json            параметры и сводная статистика
"""
import json
import sys
from pathlib import Path

import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "configs/base.yaml", encoding="utf-8"))
DAY = 86_400
S = cfg["split"]
OUT = ROOT / "data/processed"
OUT.mkdir(parents=True, exist_ok=True)

ETYPE = {"listen": 0, "like": 1, "dislike": 2, "unlike": 3, "undislike": 4}


def item_map(path: str) -> pl.DataFrame:
    """item_id -> один id (артиста или альбома). Если значений несколько, берём минимальный id."""
    df = pl.read_parquet(ROOT / path)
    col = [c for c in df.columns if c != "item_id"][0]
    out = col.replace("_id", "") + "_id"
    return df.group_by("item_id").agg(pl.col(col).min().cast(pl.UInt32).alias(out)).with_columns(
        pl.col("item_id").cast(pl.UInt32)
    )


# ---------- 1. Все события ----------
raw = Path(cfg["data"]["raw_path"])
lf = pl.scan_parquet(raw if raw.is_absolute() else ROOT / raw)
ev = lf.select(
    pl.col("uid").cast(pl.UInt32),
    pl.col("timestamp").cast(pl.Int64),
    (pl.col("timestamp") // DAY).cast(pl.Int16).alias("day"),
    pl.col("item_id").cast(pl.UInt32),
    pl.col("event_type").cast(pl.String).replace_strict(ETYPE, return_dtype=pl.UInt8).alias("etype"),
    pl.col("is_organic").cast(pl.UInt8),
    pl.col("played_ratio_pct").clip(0, 100).cast(pl.UInt8).alias("ratio"),  # null у лайков/дизлайков
    pl.col("track_length_seconds").cast(pl.UInt32).alias("length"),
).collect()
print("events:", ev.shape, " days:", ev["day"].min(), "...", ev["day"].max())

art, alb = item_map(cfg["data"]["artist_map"]), item_map(cfg["data"]["album_map"])
ev = ev.join(art, on="item_id", how="left").join(alb, on="item_id", how="left")
print("доля событий без артиста:", ev["artist_id"].is_null().mean(), " без альбома:", ev["album_id"].is_null().mean())
ev = ev.sort(["uid", "timestamp"], maintain_order=True)
ev.write_parquet(OUT / "events.parquet", compression="zstd")

# ---------- 2. Показы и метки ----------
imp = (
    ev.filter((pl.col("etype") == 0) & (pl.col("is_organic") == 0))
    .select("uid", "timestamp", "day", "item_id", "artist_id", "album_id", "ratio", "length")
    .with_columns((pl.col("ratio") >= cfg["label"]["ratio_threshold"]).cast(pl.UInt8).alias("label"))
    .sort("timestamp", maintain_order=True)
)
# вторичная метка: лайк того же трека в течение like_window_sec после начала прослушивания
likes = (
    ev.filter(pl.col("etype") == 1)
    .select("uid", "item_id", "timestamp", pl.col("timestamp").alias("like_ts"))
    .sort("timestamp", maintain_order=True)
)
imp = imp.join_asof(likes, on="timestamp", by=["uid", "item_id"], strategy="forward",
                    tolerance=cfg["label"]["like_window_sec"])
imp = imp.with_columns(pl.col("like_ts").is_not_null().cast(pl.UInt8).alias("like")).drop("like_ts")

d = pl.col("day")
split = (
    pl.when(d.is_between(*S["train_days"])).then(pl.lit("train"))
    .when(d.is_between(*S["val_days"])).then(pl.lit("val"))
    .when(d.is_between(*S["test_days"])).then(pl.lit("test"))
    .otherwise(pl.lit(None))
    .cast(pl.Enum(["train", "val", "test"]))
)
imp = (
    imp.with_columns(split.alias("split"))
    .filter(pl.col("split").is_not_null())
    .sort(["uid", "timestamp"], maintain_order=True)
    .with_row_index("imp_id")
    # группа для метрик ранжирования: пользователь + день
    .with_columns((pl.col("uid").cast(pl.UInt64) * 1000 + pl.col("day").cast(pl.UInt64)).alias("group_id"))
)
imp.write_parquet(OUT / "impressions.parquet", compression="zstd")

# ---------- 3. Статистика ----------
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(200)
g = imp.group_by("split", "group_id").agg(pl.len().alias("n"), pl.col("label").mean().alias("p"))
stats = (
    imp.group_by("split").agg(
        pl.len().alias("impressions"),
        pl.col("uid").n_unique().alias("users"),
        pl.col("item_id").n_unique().alias("items"),
        pl.col("label").mean().alias("pos_rate"),
        pl.col("like").mean().alias("like_rate"),
    )
    .join(
        g.group_by("split").agg(
            pl.len().alias("groups"),
            pl.col("n").median().alias("median_group_size"),
            ((pl.col("p") > 0) & (pl.col("p") < 1)).mean().alias("share_mixed_groups"),
        ),
        on="split",
    )
    .sort("split")
)
print(stats)

# ---------- 4. Проверки утечек и целостности ----------
rng = {s: (imp.filter(pl.col("split") == s)["day"].min(), imp.filter(pl.col("split") == s)["day"].max())
       for s in ["train", "val", "test"]}
assert rng["train"][1] < rng["val"][0] and rng["val"][1] < rng["test"][0], rng
assert imp["imp_id"].n_unique() == imp.height
# одна группа = один пользователь и один день, группы не пересекают сплиты
assert imp.group_by("group_id").agg(pl.col("split").n_unique()).filter(pl.col("split") > 1).height == 0
print("Проверки пройдены. Дни по сплитам:", rng)

json.dump(
    {"config": cfg, "day_ranges": {k: list(map(int, v)) for k, v in rng.items()},
     "stats": stats.to_dicts(), "n_events": ev.height},
    open(OUT / "meta.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=str,
)
