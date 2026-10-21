"""Смесь двух моделей: z = (1-w)·logit(p_a) + w·logit(p_b). Вес w подбирается по LogLoss на val,
затем (с --test) тем же весом считается test. Так сравнение со «сменными» моделями остаётся честным:
на test ничего не подбирается.

  python scripts/blend.py hand seqret --name blend_hand_seqret
  python scripts/blend.py hand seqret --name blend_hand_seqret --test
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl
from sklearn.metrics import log_loss

from eval.metrics import evaluate, fmt

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("--name", required=True)
ap.add_argument("--test", action="store_true")
args = ap.parse_args()

imp = pl.read_parquet(ROOT / "data/processed/impressions.parquet", columns=["imp_id", "uid", "group_id", "label", "split"])


def load(split):
    pa = pl.read_parquet(ROOT / f"reports/preds/{args.a}_{split}.parquet").rename({"score": "pa"})
    pb = pl.read_parquet(ROOT / f"reports/preds/{args.b}_{split}.parquet").rename({"score": "pb"})
    d = imp.filter(pl.col("split") == split).join(pa, on="imp_id").join(pb, on="imp_id")
    lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
    return d, lg(d["pa"].to_numpy()), lg(d["pb"].to_numpy())


def mix(za, zb, w):
    return 1 / (1 + np.exp(-((1 - w) * za + w * zb)))


dv, za, zb = load("val")
grid = np.round(np.arange(0, 1.0001, 0.05), 2)
ll = [log_loss(dv["label"].to_numpy(), mix(za, zb, w)) for w in grid]
w = float(grid[int(np.argmin(ll))])
print("LogLoss на val по весу w:", " ".join(f"{g:.2f}:{l:.4f}" for g, l in zip(grid, ll)))
out = {"name": args.name, "objective": f"blend({args.a},{args.b})", "n_features": "-", "w": w,
       "feature_seconds": 0, "train_seconds": 0}
for split in ["val"] + (["test"] if args.test else []):
    d, a, b = (dv, za, zb) if split == "val" else load(split)
    sc = d.select("imp_id", "uid", "group_id", "label").with_columns(pl.Series("score", mix(a, b, w)))
    out[split] = evaluate(sc)
    sc.select("imp_id", "score").write_parquet(ROOT / f"reports/preds/{args.name}_{split}.parquet")
    print(f"{split} (w={w}): {fmt(out[split])}")
json.dump(out, open(ROOT / f"reports/results/{args.name}.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
