"""Проверка движка на утечки и ошибки: для случайных показов считаем признаки «в лоб» —
фильтруем сырые события условием day < дня показа и агрегируем — и сравниваем с движком.
Если движок где-то заглянул в день показа или в будущее, значения разойдутся."""
import os

os.environ.setdefault("POLARS_MAX_THREADS", "4")  # меньше потоков = ниже пиковая память и нагрев
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from features.engine import ENTITIES, INF, PROC, STATS, compute, load_rows, parse
from features.handcrafted import HANDCRAFTED

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--specs", default="", help="txt-файл с признаками (по умолчанию ручной набор)")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--grammar", type=int, default=0, help="проверить K случайных кандидатов из грамматики автопоиска")
a = ap.parse_args()
SPECS = [l.strip() for l in open(a.specs, encoding="utf-8") if l.strip()] if a.specs else HANDCRAFTED
if a.grammar:
    import random

    from features.grammar import all_candidates
    pool = [x for v in all_candidates().values() for x in v]
    SPECS = random.Random(0).sample(pool, a.grammar)
N = a.n
rows = load_rows().sample(N, seed=1)
eng = compute(SPECS, rows)
# читаем только события, которые относятся к выбранным показам (пользователи, треки, артисты, альбомы)
ev = pl.scan_parquet(PROC / "events.parquet").filter(
    pl.col("uid").is_in(rows["uid"].unique().implode())
    | pl.col("item_id").is_in(rows["item_id"].unique().implode())
    | pl.col("artist_id").is_in(rows["artist_id"].drop_nulls().unique().implode())
    | pl.col("album_id").is_in(rows["album_id"].drop_nulls().unique().implode())
).collect()
print(f"событий для проверки: {ev.height:,}")
ev = ev.with_columns([e.cast(pl.Float64).alias(f"_{s}") for s, e in STATS.items()])


def subsets(r):
    """события, относящиеся к показу r: по пользователю, треку, артисту, альбому (до фильтра по дню)"""
    d = r["day"]
    u = ev.filter((pl.col("uid") == r["uid"]) & (pl.col("day") < d))
    out = {"u": u, "ui": u.filter(pl.col("item_id") == r["item_id"]),
           "ua": u.filter(pl.col("artist_id") == r["artist_id"]), "ual": u.filter(pl.col("album_id") == r["album_id"])}
    for ent, k in [("i", "item_id"), ("a", "artist_id"), ("al", "album_id")]:
        out[ent] = ev.filter((pl.col(k) == r[k]) & (pl.col("day") < d)) if r[k] is not None else None
    return out


def brute(spec, r, sub_all):
    f = parse(spec)
    if f["kind"] == "static":
        return float(np.log1p(r["length"]))
    keys = ENTITIES[f["ent"]]
    if any(r[k] is None for k in keys):  # нет артиста/альбома у трека: признак = пропуск
        return None
    d = r["day"]
    sub = sub_all[f["ent"]]
    if f["kind"] == "recency":
        return None if sub.is_empty() else float(d - sub["day"].max())
    if f["kind"] == "age":
        return None if sub.is_empty() else float(d - sub["day"].min())
    def wsum(frame, w, st):
        win = frame if w == INF else frame.filter(pl.col("day") >= d - w)
        return float(win[f"_{st}"].sum())

    def rate(frame, w):
        num, den = wsum(frame, w, f["stats"][0]), wsum(frame, w, f["stats"][1])
        return num / den if den > 0 else None

    if f["kind"] in ("drift", "lift"):
        a_, b_ = (rate(sub, f["w"][0]), rate(sub, f["w"][1])) if f["kind"] == "drift" else (rate(sub, f["w"]), rate(sub_all["u"], f["w"]))
        return None if a_ is None or b_ is None else a_ - b_
    w = f["w"]
    win = sub if w == INF else sub.filter(pl.col("day") >= d - w)
    sm = [float(win[f"_{s}"].sum()) for s in f["stats"]]
    if f["kind"] == "sum":
        return sm[0]
    if f["kind"] == "rate":
        return sm[0] / sm[1] if sm[1] > 0 else None
    if f["kind"] == "share":
        uw = sub_all["u"]
        uw = uw if w == INF else uw.filter(pl.col("day") >= d - w)
        den = float(uw[f"_{f['stats'][0]}"].sum())
        return sm[0] / den if den > 0 else None


bad = 0
eng_rows = {r["imp_id"]: r for r in eng.iter_rows(named=True)}
for i, r in enumerate(rows.iter_rows(named=True)):
    e = eng_rows[r["imp_id"]]
    sub_all = subsets(r)
    for spec in SPECS:
        b, v = brute(spec, r, sub_all), e[spec]
        ok = (b is None and v is None) or (b is not None and v is not None and abs(b - v) <= 1e-3 * max(1, abs(b)))
        if not ok:
            bad += 1
            if bad <= 20:
                print(f"РАСХОЖДЕНИЕ imp_id={r['imp_id']} {spec}: движок={v} в лоб={b}")
    if i % 50 == 0:
        print(f"проверено показов: {i}")
print(f"\nпроверено {N} показов x {len(SPECS)} признаков, расхождений: {bad}")
assert bad == 0
print("Движок совпадает с расчётом в лоб: утечек нет.")
