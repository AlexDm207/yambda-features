"""Финальная оценка бустингов на test БЕЗ переобучения: берём сохранённые модели (.lgb) и кэш признаков.
Список моделей заморожен по результатам на val до первого обращения к test.

  python scripts/final_test.py hand hand_rank auto_scratch auto_hand auto_hand_g_rank
"""
import os

os.environ.setdefault("POLARS_MAX_THREADS", "2")
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import polars as pl

from eval.metrics import evaluate, fmt
from features.engine import CACHE, ROOT, load_rows, to_numpy
from features.handcrafted import GROUPS

rows = load_rows(["test"]).select("imp_id", "uid", "group_id", "label")
for name in sys.argv[1:]:
    rp = ROOT / f"reports/results/{name}.json"
    res = json.load(open(rp, encoding="utf-8"))
    if res["set"] == "handcrafted":
        specs = [f for g, fs in GROUPS.items() if g not in res.get("drop", []) for f in fs]
    else:
        specs = [l.strip() for l in open(ROOT / f"reports/auto/{res['set']}.txt", encoding="utf-8")
                 if l.strip() and not l.startswith("#")]
    assert len(specs) == res["n_features"], name
    feats = pl.read_parquet(CACHE / f"set_{res['set']}.parquet", columns=["imp_id"] + specs)
    df = rows.join(feats, on="imp_id")
    model = lgb.Booster(model_file=str(ROOT / f"reports/results/{name}.lgb"))
    sc = df.select("imp_id", "uid", "group_id", "label").with_columns(pl.Series("score", model.predict(to_numpy(df, specs))))
    res["test"] = evaluate(sc, prob=res.get("objective", "binary") == "binary")
    sc.select("imp_id", "score").write_parquet(ROOT / f"reports/preds/{name}_test.parquet")
    json.dump(res, open(rp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"{name:20s} test: {fmt(res['test'])}")
