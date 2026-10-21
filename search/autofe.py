"""Автопоиск признаков по грамматике (features/grammar.py). Этапы, каждый можно перезапустить:

  samples   выборки для поиска: train = последние 14 дней train у половины пользователей,
            sval = val тех же пользователей (по нему принимаются решения),
            hval = val ДРУГОЙ половины пользователей (только для контроля переобучения поиска)
  features  считаем всех ~1900 кандидатов на этих выборках (по сущностям, с кэшем)
  screen    одномерный отсев: маленький бустинг на одном признаке -> выигрыш в LogLoss и GAUC на sval
  pool      пул лучших кандидатов: топ по каждой сущности + удаление почти-дубликатов (|ρ| > 0.97)
  forward   жадный отбор: добавляем кандидатов по очереди, оставляем, если LogLoss на sval падает > delta
            --start scratch: с пустого набора (может ли поиск заменить человека)
            --start hand:    поверх ручного набора (может ли поиск помочь человеку)

  python search/autofe.py --stage all --start scratch
  python search/autofe.py --stage forward --start hand
"""
import os

os.environ.setdefault("POLARS_MAX_THREADS", "2")
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
from sklearn.metrics import log_loss

from eval.metrics import per_group
from features.engine import CACHE, N_BUCKETS, ROOT, bucket_of, compute, load_rows, parse
from features.grammar import all_candidates
from features.handcrafted import HANDCRAFTED

ap = argparse.ArgumentParser()
ap.add_argument("--stage", default="all", choices=["samples", "features", "screen", "pool", "forward", "all"])
ap.add_argument("--start", default="scratch", choices=["scratch", "hand"])
ap.add_argument("--search_days", type=int, default=14)
ap.add_argument("--screen_rows", type=int, default=300_000)
ap.add_argument("--pool_ll", type=int, default=15, help="лучших по LogLoss от каждой сущности")
ap.add_argument("--pool_gauc", type=int, default=8, help="лучших по GAUC от каждой сущности")
ap.add_argument("--corr", type=float, default=0.97)
ap.add_argument("--delta", type=float, default=1e-4, help="минимальное улучшение LogLoss на sval")
ap.add_argument("--threads", type=int, default=2, help="потоки LightGBM (меньше = тише и холоднее)")
ap.add_argument("--pool_mode", default="entity", choices=["entity", "family"],
                help="entity: топ от каждой сущности; family: топ от каждого семейства «вид:сущность:статистика» (разнообразнее)")
ap.add_argument("--per_family", type=int, default=2, help="family: сколько окон брать от семейства по LL и по GAUC")
ap.add_argument("--order_by", default="dll", choices=["dll", "gauc"], help="порядок очереди в пуле")
ap.add_argument("--criterion", default="ll", choices=["ll", "gauc"], help="критерий принятия признака в forward")
ap.add_argument("--pool_tag", default="", help="какой пул брать в forward (суффикс файла)")
ap.add_argument("--tag", default="", help="суффикс файлов результата (pool_<tag>.csv, forward_<start><tag>.jsonl)")
args = ap.parse_args()

OUT = ROOT / "reports/auto"
FS = CACHE / "search"
OUT.mkdir(parents=True, exist_ok=True)
FS.mkdir(parents=True, exist_ok=True)
COST = OUT / "cost.json"
cost = json.load(open(COST)) if COST.exists() else {}


def save_cost(key, sec):
    cost[key] = cost.get(key, 0) + sec
    json.dump(cost, open(COST, "w"), indent=2)


def run(stage):
    return args.stage in (stage, "all")


# ------------------------------------------------------------------ samples
if run("samples") and not (FS / "rows.parquet").exists():
    rows = load_rows(["train", "val"])
    half = bucket_of(pl.col("uid")) < N_BUCKETS // 2
    last = rows.filter(pl.col("split") == "train")["day"].max()
    part = (pl.when((pl.col("split") == "train") & (pl.col("day") > last - args.search_days) & half).then(pl.lit("train"))
            .when((pl.col("split") == "val") & half).then(pl.lit("sval"))
            .when(pl.col("split") == "val").then(pl.lit("hval")))
    rows = rows.with_columns(part.alias("part")).filter(pl.col("part").is_not_null())
    rows.write_parquet(FS / "rows.parquet")
    print(rows.group_by("part").agg(pl.len(), pl.col("uid").n_unique().alias("users")).sort("part"))

rows = pl.read_parquet(FS / "rows.parquet") if (FS / "rows.parquet").exists() else None
CANDS = all_candidates()

# ------------------------------------------------------------------ features
if run("features"):
    for ent, specs in list(CANDS.items()) + [("static", ["static:length"])]:
        p = FS / f"{ent}.parquet"
        if p.exists():
            continue
        t0 = time.time()
        print(f"сущность {ent}: {len(specs)} кандидатов")
        compute(specs, rows).write_parquet(p)
        save_cost(f"features_{ent}", time.time() - t0)
        gc.collect()

# общие данные для оценки
lab = rows.select("imp_id", "uid", "group_id", "label", "part")
y = {p: lab.filter(pl.col("part") == p)["label"].to_numpy() for p in ["train", "sval", "hval"]}


def gauc(part, score):
    d = lab.filter(pl.col("part") == part).with_columns(pl.Series("score", score))
    return float(per_group(d)["auc"].mean())


def col_values(ent_or_file, specs, part_mask=None):
    df = pl.read_parquet(FS / f"{ent_or_file}.parquet", columns=["imp_id"] + specs)
    return lab.select("imp_id", "part").join(df, on="imp_id", how="left", maintain_order="left")


# ------------------------------------------------------------------ screen
SCREEN = OUT / "screen.csv"
if run("screen"):
    done = set(pl.read_csv(SCREEN)["spec"]) if SCREEN.exists() else set()
    rng = np.random.default_rng(0)
    tr_idx = np.where(lab["part"].to_numpy() == "train")[0]
    tr_idx = np.sort(rng.choice(tr_idx, min(args.screen_rows, len(tr_idx)), replace=False))
    sv_idx = np.where(lab["part"].to_numpy() == "sval")[0]
    ytr, ysv = lab["label"].to_numpy()[tr_idx], y["sval"]
    ll0 = log_loss(ysv, np.full(len(ysv), ytr.mean()))
    params = dict(objective="binary", learning_rate=0.2, num_leaves=15, min_data_in_leaf=500, max_bin=63,
                  verbose=-1, num_threads=args.threads, seed=0, deterministic=True, force_row_wise=True)
    t0 = time.time()
    for ent, specs in list(CANDS.items()) + [("static", ["static:length"])]:
        todo = [s for s in specs if s not in done]
        if not todo:
            continue
        df = col_values(ent, todo)
        out = []
        for i, s in enumerate(todo):
            x = df[s].cast(pl.Float32).to_numpy()
            xtr, xsv = x[tr_idx], x[sv_idx]
            ok = ~np.isnan(xtr)
            if ok.sum() < 1000 or np.nanstd(xtr) == 0:
                out.append({"spec": s, "ent": ent, "kind": parse(s)["kind"], "dll": 0.0, "gauc": 0.5,
                            "null_share": float(np.isnan(xsv).mean())})
                continue
            m = lgb.train(params, lgb.Dataset(xtr.reshape(-1, 1), ytr), 30)
            p = m.predict(xsv.reshape(-1, 1))
            out.append({"spec": s, "ent": ent, "kind": parse(s)["kind"], "dll": ll0 - log_loss(ysv, p),
                        "gauc": gauc("sval", p), "null_share": float(np.isnan(xsv).mean())})
            if i % 50 == 0:
                print(f"  {ent}: {i}/{len(todo)}  {s}  dLL={out[-1]['dll']:+.5f}  GAUC={out[-1]['gauc']:.4f}")
        new = pl.DataFrame(out)
        (pl.concat([pl.read_csv(SCREEN), new], how="vertical_relaxed") if SCREEN.exists() else new).write_csv(SCREEN)
        done |= set(todo)
        save_cost("screen", time.time() - t0)
        t0 = time.time()
        del df
        gc.collect()

# ------------------------------------------------------------------ pool
POOL = OUT / f"pool{args.tag if args.stage == 'pool' else args.pool_tag}.csv"
if run("pool"):
    sc = pl.read_csv(SCREEN).filter(pl.col("dll") > 0)
    # семейство = признак без окна: rate:ui:full/n:30 -> rate:ui:full/n
    sc = sc.with_columns(pl.col("spec").map_elements(lambda x: ":".join(x.split(":")[:3]), return_dtype=pl.String).alias("family"))
    grp, k_ll, k_g = ("ent", args.pool_ll, args.pool_gauc) if args.pool_mode == "entity" else ("family", args.per_family, args.per_family)
    sel = []
    for g in sc[grp].unique().to_list():
        e = sc.filter(pl.col(grp) == g)
        sel += e.sort("dll", descending=True).head(k_ll)["spec"].to_list()
        sel += e.sort("gauc", descending=True).head(k_g)["spec"].to_list()
    cand = sc.filter(pl.col("spec").is_in(list(set(sel))))
    # очередь: по кругу между группами, внутри группы — по убыванию выбранного показателя
    key = args.order_by
    cand = cand.with_columns(pl.col(key).rank("ordinal", descending=True).over(grp).alias("r")).sort(["r", key], descending=[False, True])
    order = cand["spec"].to_list()
    # ранги на подвыборке для корреляции Спирмена; сравниваем и между собой, и с ручными признаками
    trp = lab.filter(pl.col("part") == "train")
    smp = trp.sample(min(100_000, trp.height), seed=0).select("imp_id")
    vals = smp
    for ent in cand["ent"].unique().to_list():
        sp = cand.filter(pl.col("ent") == ent)["spec"].to_list()
        vals = vals.join(pl.read_parquet(FS / f"{ent}.parquet", columns=["imp_id"] + sp), on="imp_id", how="left")
    hand = smp.join(pl.read_parquet(CACHE / "set_handcrafted.parquet"), on="imp_id", how="left")

    def ranks(df, specs):
        return np.column_stack([df[s].cast(pl.Float64).fill_null(-1e18).rank("average").to_numpy() for s in specs])

    R, H = ranks(vals, order), ranks(hand, HANDCRAFTED)
    R = (R - R.mean(0)) / (R.std(0) + 1e-12)
    H = (H - H.mean(0)) / (H.std(0) + 1e-12)
    corr_hand = np.abs(R.T @ H / len(R)).max(1)
    kept, rows_out = [], []
    for j, s in enumerate(order):
        c = max((abs(float(R[:, j] @ R[:, k]) / len(R)) for k in kept), default=0.0)
        if c < args.corr:
            kept.append(j)
            rows_out.append({"spec": s, "order": len(rows_out), "max_corr_pool": c,
                             "max_corr_hand": float(corr_hand[j]), "dup_hand": bool(corr_hand[j] >= args.corr or s in HANDCRAFTED)})
    pool = pl.DataFrame(rows_out).join(sc.select("spec", "ent", "kind", "dll", "gauc"), on="spec")
    pool.sort("order").write_csv(POOL)
    print(f"пул: {pool.height} кандидатов (из {len(order)} до удаления дубликатов), из них похожих на ручные: {pool['dup_hand'].sum()}")

# ------------------------------------------------------------------ forward
if run("forward"):
    pool = pl.read_csv(POOL).sort("order")
    if args.start == "hand":
        pool = pool.filter(~pl.col("dup_hand"))
    queue = pool["spec"].to_list()
    base = list(HANDCRAFTED) if args.start == "hand" else []
    # Память: держим в RAM только столбцы текущего набора, кандидата читаем с диска по одному.
    # (Раньше вся матрица пула 1.4 млн x 400 признаков собиралась сразу — это ~7 ГБ на пике.)
    parts = lab["part"].to_numpy()
    masks = {p: parts == p for p in ["train", "sval", "hval"]}
    lab_ids = lab["imp_id"].to_numpy()
    for f in set(["static"] + [parse(x)["ent"] for x in queue if not x.startswith("static")]):
        ids = pl.read_parquet(FS / f"{f}.parquet", columns=["imp_id"])["imp_id"].to_numpy()
        assert np.array_equal(ids, lab_ids), f"порядок строк в {f}.parquet не совпадает с выборкой"

    def split_parts(x):
        return {p: np.ascontiguousarray(x[m], dtype=np.float32) for p, m in masks.items()}

    def load_col(spec):
        f = "static" if spec.startswith("static") else parse(spec)["ent"]
        return split_parts(pl.read_parquet(FS / f"{f}.parquet", columns=[spec])[spec].cast(pl.Float32).to_numpy())

    cols = {}
    if base:
        hb = lab.select("imp_id").join(pl.read_parquet(CACHE / "set_handcrafted.parquet", columns=["imp_id"] + base),
                                       on="imp_id", how="left", maintain_order="left")
        for b in base:
            cols[b] = split_parts(hb[b].cast(pl.Float32).to_numpy())
        del hb
    gc.collect()
    params = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200, max_bin=255,
                  feature_fraction=1.0, bagging_fraction=1.0, lambda_l2=1.0, verbose=-1, num_threads=args.threads,
                  seed=0, deterministic=True, force_row_wise=True)

    def fit(names):
        t0 = time.time()
        if not names:
            p = {k: np.full(len(y[k]), y["train"].mean()) for k in ("sval", "hval")}
            it = 0
        else:
            X = {k: np.column_stack([cols[s][k] for s in names]) for k in ("train", "sval", "hval")}
            dtr = lgb.Dataset(X["train"], y["train"], free_raw_data=True)
            dsv = lgb.Dataset(X["sval"], y["sval"], reference=dtr)
            m = lgb.train(params, dtr, 500, valid_sets=[dsv], callbacks=[lgb.early_stopping(30, verbose=False)])
            p = {k: m.predict(X[k], num_iteration=m.best_iteration) for k in ("sval", "hval")}
            del X, dtr, dsv
            it = m.best_iteration
        return {"n": len(names), "iters": it, "sec": time.time() - t0,
                **{f"{k}_ll": float(log_loss(y[k], p[k])) for k in ("sval", "hval")},
                **{f"{k}_gauc": gauc(k, p[k]) for k in ("sval", "hval")}}

    LOG = OUT / f"forward_{args.start}{args.tag}.jsonl"
    hist = [json.loads(l) for l in open(LOG)] if LOG.exists() else []
    cur = list(base)
    for h in hist:
        if h.get("accepted") and h["spec"] != "<start>":
            cur.append(h["spec"])
            cols[h["spec"]] = load_col(h["spec"])
    tried = {h["spec"] for h in hist}
    if not hist:
        r0 = fit(cur)
        r0.update(spec="<start>", accepted=True)
        hist.append(r0)
        open(LOG, "a").write(json.dumps(r0) + "\n")
    best = [h for h in hist if h["accepted"]][-1]
    print(f"старт ({args.start}): признаков {len(cur)}, sval LL={best['sval_ll']:.5f} GAUC={best['sval_gauc']:.4f}")
    t_all = time.time()
    for s in queue:
        if s in tried or s in cur:
            continue
        cols[s] = load_col(s)
        r = fit(cur + [s])
        r["spec"] = s
        if args.criterion == "ll":
            r["accepted"] = best["sval_ll"] - r["sval_ll"] > args.delta
        else:  # GAUC: порядок треков внутри дня пользователя
            r["accepted"] = r["sval_gauc"] - best["sval_gauc"] > args.delta
        if r["accepted"]:
            cur.append(s)
            best = r
        else:
            del cols[s]
        gc.collect()
        open(LOG, "a").write(json.dumps(r) + "\n")
        print(f"{'+' if r['accepted'] else ' '} {s:32s} n={len(cur):3d}  sval LL={r['sval_ll']:.5f} GAUC={r['sval_gauc']:.4f}"
              f"  | hval GAUC={r['hval_gauc']:.4f}  ({r['sec']:.0f} с)")
    save_cost(f"forward_{args.start}{args.tag}", time.time() - t_all)
    sel = [s for s in cur]
    open(OUT / f"selected_{args.start}{args.tag}.txt", "w", encoding="utf-8").write("\n".join(sel) + "\n")
    print(f"отобрано признаков: {len(sel)} -> reports/auto/selected_{args.start}{args.tag}.txt")
