"""LightGBM на признаках из движка. Обучение на train, ранняя остановка и все решения по val,
test считается только с флагом --test (в самом конце, один раз).

Примеры:
  python train/gbdt.py --name hand                         # ручной набор
  python train/gbdt.py --name hand_no_direct --drop pair_direct   # абляция группы
  python train/gbdt.py --name auto --specs reports/auto/selected.txt
"""
import os

os.environ.setdefault("POLARS_MAX_THREADS", "4")  # меньше потоков = ниже пиковая память и нагрев
import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np
import polars as pl

from eval.metrics import evaluate, fmt
from features.engine import CACHE, ROOT, compute, load_rows, to_numpy
from features.handcrafted import GROUPS, HANDCRAFTED

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--specs", default="", help="txt-файл со списком признаков; по умолчанию ручной набор")
ap.add_argument("--drop", nargs="*", default=[], help="группы ручного набора, которые выкинуть")
ap.add_argument("--train_last_days", type=int, default=0, help="обучаться только на последних N днях train")
ap.add_argument("--objective", default="binary", choices=["binary", "lambdarank"],
                help="binary: поточечная вероятность дослушивания; lambdarank: порядок внутри группы пользователь-день")
ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--leaves", type=int, default=127)
ap.add_argument("--min_leaf", type=int, default=200)
ap.add_argument("--rounds", type=int, default=5000)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--threads", type=int, default=4, help="потоки LightGBM")
ap.add_argument("--test", action="store_true", help="посчитать метрики на test")
args = ap.parse_args()

if args.specs:
    specs = [l.strip() for l in open(args.specs, encoding="utf-8") if l.strip() and not l.startswith("#")]
    set_name = Path(args.specs).stem
else:
    specs = [f for g, fs in GROUPS.items() if g not in args.drop for f in fs]
    set_name = "handcrafted"

# ---------- признаки: считаем один раз на все сплиты и кэшируем ----------
rows = load_rows()
fpath = CACHE / f"set_{set_name}.parquet"
t0 = time.time()
if fpath.exists() and set(specs) <= set(pl.read_parquet_schema(fpath)):
    feats = pl.read_parquet(fpath, columns=["imp_id"] + specs)
    feat_time = json.load(open(fpath.with_suffix(".json")))["seconds"]
    print(f"признаки из кэша {fpath.name}")
else:
    all_specs = HANDCRAFTED if set_name == "handcrafted" else specs
    feats = compute(all_specs, rows)
    feat_time = time.time() - t0
    feats.write_parquet(fpath)
    json.dump({"seconds": feat_time, "n_features": len(all_specs)}, open(fpath.with_suffix(".json"), "w"))
    feats = feats.select(["imp_id"] + specs)
print(f"признаков: {len(specs)}, расчёт: {feat_time:.0f} с")

data = rows.select("imp_id", "split", "day", "uid", "group_id", "label").join(feats, on="imp_id")
data = data.sort("group_id", "imp_id")  # строки одной группы подряд — нужно для lambdarank
tr = data.filter(pl.col("split") == "train")
if args.train_last_days:
    tr = tr.filter(pl.col("day") > tr["day"].max() - args.train_last_days)
va = data.filter(pl.col("split") == "val")
te = data.filter(pl.col("split") == "test")
del data, feats, rows
gc.collect()

def group_sizes(df):
    return df.group_by("group_id", maintain_order=True).len()["len"].to_numpy()


rank = args.objective == "lambdarank"
params = dict(
    objective=args.objective, learning_rate=args.lr, num_leaves=args.leaves, min_data_in_leaf=args.min_leaf,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    max_bin=255, verbose=-1, seed=args.seed, num_threads=args.threads,
)
if rank:
    params.update(metric="ndcg", eval_at=[10], lambdarank_truncation_level=30)
# сначала строим бинаризованный датасет LightGBM, потом освобождаем исходные массивы
dtr = lgb.Dataset(to_numpy(tr, specs), tr["label"].to_numpy(), feature_name=[f"f{i}" for i in range(len(specs))],
                  group=group_sizes(tr) if rank else None,
                  params={"max_bin": 255, "num_threads": args.threads}, free_raw_data=True).construct()
n_train = tr.height
del tr
gc.collect()
dva = lgb.Dataset(to_numpy(va, specs), va["label"].to_numpy(), reference=dtr,
                  group=group_sizes(va) if rank else None).construct()
t1 = time.time()
model = lgb.train(params, dtr, args.rounds, valid_sets=[dva], valid_names=["val"],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
train_time = time.time() - t1
print(f"обучение: {train_time:.0f} с, деревьев: {model.best_iteration}")

out = {"name": args.name, "objective": args.objective, "set": set_name, "n_features": len(specs), "drop": args.drop,
       "n_train": n_train, "feature_seconds": feat_time, "train_seconds": train_time,
       "best_iter": model.best_iteration, "params": params}
pred_dir = ROOT / "reports/preds"
pred_dir.mkdir(parents=True, exist_ok=True)
for split, df in [("val", va)] + ([("test", te)] if args.test else []):
    t2 = time.time()
    p = model.predict(to_numpy(df, specs), num_iteration=model.best_iteration)
    sc = df.select("imp_id", "uid", "group_id", "label").with_columns(pl.Series("score", p))
    out[split] = evaluate(sc, prob=not rank)  # у ранжировщика score не вероятность: без LogLoss
    out[split]["predict_seconds"] = time.time() - t2
    sc.select("imp_id", "score").write_parquet(pred_dir / f"{args.name}_{split}.parquet")
    print(f"{split}: {fmt(out[split])}")

imp = pl.DataFrame({"feature": specs, "gain": model.feature_importance("gain", iteration=model.best_iteration)})
imp = imp.with_columns((pl.col("gain") / pl.col("gain").sum()).alias("share")).sort("gain", descending=True)
res_dir = ROOT / "reports/results"
res_dir.mkdir(parents=True, exist_ok=True)
imp.write_csv(res_dir / f"{args.name}_importance.csv")
json.dump(out, open(res_dir / f"{args.name}.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(imp.head(20))
model.save_model(str(ROOT / "reports/results" / f"{args.name}.lgb"), num_iteration=model.best_iteration)
