"""Простые эвристики без обучения: нижняя планка, от которой отсчитываем пользу моделей.
Каждая эвристика — один исторический признак, используемый как score."""
import os

os.environ.setdefault("POLARS_MAX_THREADS", "4")  # меньше потоков = ниже пиковая память и нагрев
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from eval.metrics import evaluate, fmt
from features.engine import ROOT, compute, load_rows

ap = argparse.ArgumentParser()
ap.add_argument("--test", action="store_true")
args = ap.parse_args()

BASE = {
    "random": None,
    "item_popularity_30d": "sum:i:n:30",
    "item_rec_ctr": "rate:i:rec_full/rec:inf",
    "item_full_rate": "rate:i:full/n:inf",
    "user_rec_ctr": "rate:u:rec_full/rec:inf",   # внутри группы константа -> GAUC = 0.5
    "seen_track_count": "sum:ui:n:inf",
    "user_artist_share": "share:ua:n:inf",
    "user_artist_full_rate": "rate:ua:full/n:inf",
}
splits = ["val"] + (["test"] if args.test else [])
rows = load_rows(splits)
specs = [s for s in BASE.values() if s]
f = compute(specs, rows)
df = rows.select("imp_id", "split", "uid", "group_id", "label").join(f, on="imp_id")

res = {}
for split in splits:
    d = df.filter(pl.col("split") == split)
    for name, spec in BASE.items():
        if spec is None:
            sc = d.with_columns(pl.lit(0.5).alias("score"))
        else:
            # пропуски (нет истории) заполняем средним по сплиту — только для ранжирования
            sc = d.with_columns(pl.col(spec).fill_null(pl.col(spec).mean()).alias("score"))
        r = evaluate(sc, prob=False)  # эвристики не откалиброваны, logloss для них бессмыслен
        res[f"{split}/{name}"] = r
        print(f"{split:5s} {name:24s} {fmt(r)}")

out = ROOT / "reports/results"
out.mkdir(parents=True, exist_ok=True)
json.dump(res, open(out / "baselines.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
