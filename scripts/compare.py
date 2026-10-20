"""Парное сравнение моделей с доверительными интервалами: бутстреп по ПОЛЬЗОВАТЕЛЯМ.
Почему по пользователям: группы одного пользователя зависимы, бутстреп по группам занизил бы интервал.

  python scripts/compare.py hand hand_rank hand_no_user ...   # первая модель = эталон
  python scripts/compare.py hand auto --split test
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from eval.metrics import per_group

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("models", nargs="+")
ap.add_argument("--split", default="val")
ap.add_argument("--n_boot", type=int, default=1000)
args = ap.parse_args()

imp = pl.read_parquet(ROOT / "data/processed/impressions.parquet", columns=["imp_id", "uid", "group_id", "label", "split"])
imp = imp.filter(pl.col("split") == args.split)


def load(name):
    p = pl.read_parquet(ROOT / f"reports/preds/{name}_{args.split}.parquet")
    return per_group(imp.join(p, on="imp_id"))


ref_name, others = args.models[0], args.models[1:]
ref = load(ref_name)
users = ref["uid"].unique().sort()
rng = np.random.default_rng(0)
W = rng.poisson(1.0, size=(args.n_boot, users.len())).astype(np.float64)  # пуассоновский бутстреп


def user_sums(g, col):
    s = g.group_by("uid").agg(pl.col(col).sum().alias("s"), pl.len().alias("c"))
    s = pl.DataFrame({"uid": users}).join(s, on="uid", how="left").fill_null(0)
    return s["s"].to_numpy(), s["c"].to_numpy()


res = {}
print(f"эталон: {ref_name}, сплит {args.split}, групп {ref.height}, пользователей {users.len()}")
for name in others:
    g = load(name).join(ref.select("group_id", pl.col("auc").alias("auc0"), pl.col("ndcg").alias("ndcg0")), on="group_id")
    row = {}
    for m in ["auc", "ndcg"]:
        sd, c = user_sums(g.with_columns((pl.col(m) - pl.col(m + "0")).alias("d")), "d")
        point = sd.sum() / c.sum()
        boot = (W @ sd) / (W @ c)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        row[m] = {"diff": point, "lo": lo, "hi": hi}
    res[name] = row
    lab = {"auc": "GAUC", "ndcg": "NDCG@10"}
    print(f"{name:28s} " + "   ".join(
        f"{lab[m]} {v['diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]{' *' if v['lo'] > 0 or v['hi'] < 0 else ''}"
        for m, v in row.items()))
print("* — 95% интервал не содержит ноль")
out = ROOT / f"reports/results/compare_{ref_name}_{args.split}.json"
json.dump(res, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
