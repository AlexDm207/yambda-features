"""Движок признаков: любые оконные агрегаты по таблице событий через накопительные суммы по дням.

Идея. Для сущности E (пользователь, трек, артист, пара пользователь-трек, ...) один раз строим таблицу
    E, day -> накопительные суммы статистик за все дни <= day.
Тогда сумма статистики за окно из w дней, закончившееся ДО дня показа d (дни d-w ... d-1), равна
    S(d-1) - S(d-1-w),
где S(t) берётся asof-соединением (последняя строка с day <= t). Признак с любым окном стоит одного
соединения, поэтому перебор окон в автопоиске дешёвый. День показа d и всё, что после, в признаки
не попадают по построению.

Признак описывается строкой  kind:entity:stat:window
    sum:ua:full:30      сколько раз пользователь дослушал этого артиста за 30 дней
    rate:u:rec_full/rec:inf   доля дослушанных рекомендаций пользователя за всю историю
    share:ua:n:30       доля прослушиваний пользователя за 30 дней, пришедшихся на этого артиста
    recency:ui          сколько дней прошло с последнего события пары
    age:i               сколько дней прошло с первого прослушивания трека кем-либо
    static:length       логарифм длины трека (атрибут каталога)
"""
from __future__ import annotations

import gc
import os
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

# asof-соединение с by= не умеет проверять сортировку внутри групп; мы сортируем по дню явно
warnings.filterwarnings("ignore", message="Sortedness of columns cannot be checked")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed"
CACHE = ROOT / "data/features"
INF = 10_000  # «вся история»

ENTITIES = {
    "u": ["uid"],
    "i": ["item_id"],
    "a": ["artist_id"],
    "al": ["album_id"],
    "ui": ["uid", "item_id"],
    "ua": ["uid", "artist_id"],
    "ual": ["uid", "album_id"],
}

_listen = pl.col("etype") == 0
_r = pl.col("ratio").fill_null(0).cast(pl.Float32) / 100
STATS = {
    "n": _listen,                                                     # прослушивания
    "full": _listen & (pl.col("ratio") >= 50),                        # дослушанные
    "rec": _listen & (pl.col("is_organic") == 0),                     # показы рекомендаций
    "rec_full": _listen & (pl.col("is_organic") == 0) & (pl.col("ratio") >= 50),
    "org": _listen & (pl.col("is_organic") == 1),                     # сам выбрал
    "ratio": pl.when(_listen).then(_r).otherwise(0.0),                # сумма долей прослушивания
    "sec": pl.when(_listen).then(_r * pl.col("length").fill_null(0)).otherwise(0.0),  # секунды
    "like": pl.col("etype") == 1,
    "dislike": pl.col("etype") == 2,
    "unlike": pl.col("etype") == 3,
}
FLOAT_STATS = {"ratio", "sec"}


# ---------------------------------------------------------------- накопительные таблицы
# Пары (пользователь-трек и т.п.) — самые большие таблицы. Чтобы не упираться в память, всё, что
# связано с пользователями, считаем порциями: пользователи делятся на N_BUCKETS корзин по uid % N.
N_BUCKETS = int(os.environ.get("YF_BUCKETS", 16))
PAIR = {"ui", "ua", "ual"}


def bucket_of(uid: pl.Expr) -> pl.Expr:
    """Корзина пользователя. Простое uid % N не годится: в Yambda все uid кратны 4, и заполнялись бы
    только 4 корзины из 16. Перемешиваем uid мультипликативным хешем (Кнут), он детерминирован."""
    return ((uid.cast(pl.UInt64) * 2654435761) % 4294967296 // 65536) % N_BUCKETS


def _paths(ent: str, bucket: int):
    sfx = f"_h{bucket}" if ent in PAIR else ""  # _h: корзины по хешу uid
    return CACHE / f"cum_{ent}{sfx}.parquet", CACHE / f"first_{ent}{sfx}.parquet"


def _build(ent: str, bucket: int):
    path, fpath = _paths(ent, bucket)
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    keys = ENTITIES[ent]
    ev = pl.scan_parquet(PROC / "events.parquet").filter(pl.all_horizontal([pl.col(k).is_not_null() for k in keys]))
    if ent in PAIR:
        ev = ev.filter(bucket_of(pl.col("uid")) == bucket)
    aggs = [
        (STATS[s].cast(pl.Float64).sum() if s in FLOAT_STATS else STATS[s].cast(pl.UInt32).sum()).alias(s) for s in STATS
    ]
    daily = ev.group_by(keys + ["day"]).agg(aggs).collect().sort(keys + ["day"])
    daily.group_by(keys).agg(pl.col("day").min().alias("first_day")).write_parquet(fpath)
    daily = daily.with_columns([pl.col(s).cum_sum().over(keys).alias(s) for s in STATS])
    daily.sort("day").write_parquet(path)
    print(f"  [engine] cum_{ent}{'' if ent not in PAIR else f' корзина {bucket}'}: {daily.height:,} строк за {time.time() - t0:.0f} с")
    del daily
    gc.collect()


def cum_table(ent: str, stats: list[str], bucket: int, q: pl.DataFrame | None = None) -> pl.DataFrame:
    """E, day, накопительные суммы; отсортировано по day (нужно для asof). Если передан q, читаются
    только строки нужных ключей — так в память не попадает таблица целиком."""
    path, _ = _paths(ent, bucket)
    if not path.exists():
        _build(ent, bucket)
    keys = ENTITIES[ent]
    lf = pl.scan_parquet(path).select(keys + ["day"] + list(stats))
    if q is not None:
        lf = lf.join(q.select(keys).unique().lazy(), on=keys, how="semi")
    return lf.collect().sort("day")


def first_table(ent: str, bucket: int) -> pl.DataFrame:
    path, fpath = _paths(ent, bucket)
    if not fpath.exists():
        _build(ent, bucket)
    return pl.read_parquet(fpath)


def build_all(ents=None):
    """Построить все кэши заранее, по одной таблице за раз."""
    for ent in ents or ENTITIES:
        for b in (range(N_BUCKETS) if ent in PAIR else [0]):
            if not _paths(ent, b)[0].exists():
                _build(ent, b)


# ---------------------------------------------------------------- разбор спецификаций
def parse(spec: str) -> dict:
    p = spec.split(":")
    kind = p[0]
    if kind == "static":
        return {"kind": kind, "name": p[1]}
    if kind in ("recency", "age"):
        return {"kind": kind, "ent": p[1]}
    stats = p[2].split("/")
    if kind == "drift":  # rate за короткое окно минус rate за длинное: «меняется ли поведение»
        w1, w2 = [INF if x == "inf" else int(x) for x in p[3].split("-")]
        return {"kind": kind, "ent": p[1], "stats": stats, "w": (w1, w2)}
    w = INF if p[3] == "inf" else int(p[3])
    return {"kind": kind, "ent": p[1], "stats": stats, "w": w}


def _needs(specs: list[str]):
    """entity -> {window -> set(stats)}; плюс какие сущности нужны для recency/age."""
    need = defaultdict(lambda: defaultdict(set))
    rec = set()
    for s in specs:
        f = parse(s)
        if f["kind"] in ("sum", "rate", "share", "lift"):
            for st in f["stats"]:
                need[f["ent"]][f["w"]].add(st)
                if f["kind"] in ("share", "lift"):  # нужна та же статистика пользователя
                    need["u"][f["w"]].add(st)
        elif f["kind"] == "drift":
            for w in f["w"]:
                for st in f["stats"]:
                    need[f["ent"]][w].add(st)
        elif f["kind"] in ("recency", "age"):
            rec.add((f["kind"], f["ent"]))
    return need, rec


def _col(ent, st, w):
    return f"_{ent}_{st}_{w}"


# ---------------------------------------------------------------- расчёт
def compute(specs: list[str], rows: pl.DataFrame, verbose: bool = True) -> pl.DataFrame:
    """rows: imp_id, day + ключи сущностей (+ length). Возвращает imp_id + по колонке на spec.
    Считается порциями по корзинам пользователей, чтобы ограничить пиковую память."""
    t0 = time.time()
    build_all({parse(s).get("ent") for s in specs} - {None})
    parts = []
    for b in range(N_BUCKETS):
        rb = rows.filter(bucket_of(pl.col("uid")) == b)
        if rb.height:
            parts.append(_compute_chunk(specs, rb, b))
            gc.collect()
    res = pl.concat(parts).sort("imp_id")
    if verbose:
        print(f"  [engine] {len(specs)} признаков для {rows.height:,} строк за {time.time() - t0:.0f} с")
    return res


def _compute_chunk(specs: list[str], rows: pl.DataFrame, bucket: int) -> pl.DataFrame:
    need, rec = _needs(specs)
    rows = rows.select([c for c in ["imp_id", "day", "uid", "item_id", "artist_id", "album_id", "length"] if c in rows.columns])
    rows = rows.with_columns(pl.col("day").cast(pl.Int16))
    by_ent = defaultdict(list)
    for s in specs:
        f = parse(s)
        by_ent[f.get("ent", "static")].append(s)
    ents = sorted(set(need) | {e for _, e in rec}, key=lambda e: e != "u")  # пользователь первым: нужен для share
    base = rows.select("imp_id", *[c for c in ["length"] if c in rows.columns], *sorted({k for e in ents for k in ENTITIES[e]}), "day")
    uq = None
    for ent in ents:
        keys = ENTITIES[ent]
        q = rows.select(keys + ["day"]).unique().filter(pl.all_horizontal([pl.col(k).is_not_null() for k in keys]))
        stats = sorted(set().union(*need[ent].values())) if ent in need else []
        cum = cum_table(ent, stats, bucket, q).rename({"day": "_cd"})
        # S(d-1): общий для всех окон конец интервала; заодно даёт последний активный день
        q = q.with_columns((pl.col("day") - 1).alias("_t")).sort("_t")
        q = q.join_asof(cum.rename({s: f"_end_{s}" for s in stats}), left_on="_t", right_on="_cd", by=keys)
        q = q.with_columns((pl.col("day") - pl.col("_cd")).alias(f"_recency_{ent}")).drop("_cd")
        for w, sts in need.get(ent, {}).items():
            sts = sorted(sts)
            if w == INF:
                q = q.with_columns([pl.col(f"_end_{s}").fill_null(0).alias(_col(ent, s, w)) for s in sts])
                continue
            q = q.with_columns((pl.col("day") - 1 - w).alias("_t")).sort("_t")
            q = q.join_asof(cum.select(keys + ["_cd"] + sts).rename({s: f"_st_{s}" for s in sts}),
                            left_on="_t", right_on="_cd", by=keys).drop("_cd")
            q = q.with_columns([
                (pl.col(f"_end_{s}").fill_null(0) - pl.col(f"_st_{s}").fill_null(0)).alias(_col(ent, s, w)) for s in sts
            ]).drop([f"_st_{s}" for s in sts])
        del cum
        if ("age", ent) in rec:
            first = first_table(ent, bucket)
            q = q.join(first, on=keys, how="left").with_columns(
                pl.when(pl.col("first_day") < pl.col("day")).then(pl.col("day") - pl.col("first_day")).alias(f"_age_{ent}")
            ).drop("first_day")
        q = q.drop([c for c in q.columns if c.startswith("_end_")] + ["_t"])
        if ent == "u":
            uq = q
        if ent != "u" and any(parse(s)["kind"] in ("share", "lift") for s in by_ent[ent]):
            q = q.join(uq.select(["uid", "day"] + [c for c in uq.columns if c.startswith("_u_")]), on=["uid", "day"], how="left")
        # сразу сворачиваем промежуточные колонки в итоговые признаки: экономит память
        q = q.select(keys + ["day"] + [_final(s) for s in by_ent[ent]])
        base = base.join(q, on=keys + ["day"], how="left")
    return base.select(["imp_id"] + [_final(s) if parse(s)["kind"] == "static" else pl.col(s) for s in specs])


def _final(s: str) -> pl.Expr:
    """Итоговое выражение признака из промежуточных оконных сумм."""
    f = parse(s)
    k = f["kind"]
    if k == "static":
        if f["name"] == "length":
            e = pl.col("length").cast(pl.Float32).log1p()
        else:
            raise ValueError(s)
    elif k == "recency":
        e = pl.col(f"_recency_{f['ent']}").cast(pl.Float32)
    elif k == "age":
        e = pl.col(f"_age_{f['ent']}").cast(pl.Float32)
    elif k == "sum":
        e = pl.col(_col(f["ent"], f["stats"][0], f["w"])).cast(pl.Float32).fill_null(0)
    elif k == "rate":
        num = pl.col(_col(f["ent"], f["stats"][0], f["w"])).cast(pl.Float32).fill_null(0)
        den = pl.col(_col(f["ent"], f["stats"][1], f["w"])).cast(pl.Float32).fill_null(0)
        e = pl.when(den > 0).then(num / den)  # нет истории -> null (LightGBM понимает пропуски)
    elif k == "share":
        num = pl.col(_col(f["ent"], f["stats"][0], f["w"])).cast(pl.Float32).fill_null(0)
        den = pl.col(_col("u", f["stats"][0], f["w"])).cast(pl.Float32).fill_null(0)
        e = pl.when(den > 0).then(num / den)
    elif k == "drift":
        w1, w2 = f["w"]
        e = _rate(f["ent"], *f["stats"], w1) - _rate(f["ent"], *f["stats"], w2)
    elif k == "lift":  # насколько пользователь дослушивает этого артиста/трек лучше своего среднего
        e = _rate(f["ent"], *f["stats"], f["w"]) - _rate("u", *f["stats"], f["w"])
    else:
        raise ValueError(s)
    return e.alias(s)


def _rate(ent, num, den, w):
    n = pl.col(_col(ent, num, w)).cast(pl.Float32).fill_null(0)
    d = pl.col(_col(ent, den, w)).cast(pl.Float32).fill_null(0)
    return pl.when(d > 0).then(n / d)


def load_rows(splits=("train", "val", "test")) -> pl.DataFrame:
    return pl.read_parquet(PROC / "impressions.parquet").filter(pl.col("split").is_in(list(splits)))


def to_numpy(df: pl.DataFrame, specs: list[str]) -> np.ndarray:
    return df.select(specs).to_numpy().astype(np.float32)
