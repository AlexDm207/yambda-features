"""Обучение трансформера над историей (models/seq_ranker.py). Протокол как у бустинга:
обучение на train, выбор эпохи по LogLoss на val, test — только с --test.

  python train/seq.py --name seq                 # L=256
  python train/seq.py --name seq_L64 --L 64      # короче история: как качество зависит от её длины
  python train/seq.py --name seq --test --eval_only   # посчитать test лучшим чекпойнтом
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl
import torch

from eval.metrics import evaluate, fmt
from models.seq_ranker import SeqRanker

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--name", default="seq")
ap.add_argument("--L", type=int, default=256, help="длина истории (последние L событий)")
ap.add_argument("--C", type=int, default=64, help="кандидатов в одном примере (большие дни режутся)")
ap.add_argument("--d", type=int, default=128)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--train_last_days", type=int, default=0)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--test", action="store_true")
ap.add_argument("--eval_only", action="store_true")
ap.add_argument("--throttle", type=float, default=0.0, help="пауза после каждого шага, с (снижает нагрев)")
ap.add_argument("--threads", type=int, default=2, help="потоки CPU для torch")
ap.add_argument("--shuffle_train_labels", action="store_true",
                help="проверка на утечку: метки train перемешаны -> на val должно быть GAUC ~ 0.5")
args = ap.parse_args()

torch.manual_seed(args.seed)
torch.set_num_threads(args.threads)
rng = np.random.default_rng(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
t_prep = time.time()
E = dict(np.load(ROOT / "data/seq/events.npz"))
K = dict(np.load(ROOT / "data/seq/cands.npz"))
meta = json.load(open(ROOT / "data/seq/meta.json"))
if args.shuffle_train_labels:
    tm = K["split"] == 0
    K["label"][tm] = rng.permutation(K["label"][tm])
AGO_EDGES = np.array([2, 3, 4, 8, 15, 31, 61, 121, 181, 241])  # дней назад -> корзина 1..11


def make_samples(split_code, last_days=0):
    """Пример = (пользователь-день, до C кандидатов). Возвращает конец истории и диапазон кандидатов."""
    m = np.where(K["split"] == split_code)[0]
    if last_days:
        m = m[K["day"][m] > K["day"][m].max() - last_days]
    g = K["group"][m]
    # границы групп считаем в позициях внутри m: между двумя группами одного сплита в K могут лежать
    # строки других сплитов (у пользователя после последнего дня train идут его дни val/test).
    # Ошибка здесь = обучение на метках val/test; её поймала проверка на случайных метках.
    pa = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    pb = np.r_[pa[1:], len(m)]
    s, c = [], []
    for a, b in zip(m[pa], pb - pa):
        assert np.all(K["split"][a:a + b] == split_code) and len(np.unique(K["group"][a:a + b])) == 1
        for x in range(a, a + b, args.C):
            s.append(x)
            c.append(min(args.C, a + b - x))
    s, c = np.array(s), np.array(c)
    key = K["u"][s].astype(np.int64) * 1000 + K["day"][s]
    hend = np.searchsorted(E["key"], key, side="left")  # первое событие дня показа: история строго до него
    return s, c, hend, E["ustart"][K["u"][s]]


def batch(idx, S):
    s, c, hend, ust = (x[idx] for x in S)
    L, C = args.L, args.C
    hi = hend[:, None] - L + np.arange(L)[None, :]
    hm = hi >= ust[:, None]  # не выходить за начало истории пользователя
    hi = np.where(hm, hi, 0)
    days = K["day"][s]
    ago = np.digitize(days[:, None] - E["day"][hi], AGO_EDGES) + 1
    ci = s[:, None] + np.arange(C)[None, :]
    cm = np.arange(C)[None, :] < c[:, None]
    ci = np.where(cm, ci, s[:, None])
    t = lambda a, dt=torch.long: torch.from_numpy(np.ascontiguousarray(a)).to(dev, dt, non_blocking=True)
    h = [t(np.where(hm, E[k][hi], 0)) for k in ("item", "art", "ecode", "org")] + [t(np.where(hm, ago, 0))]
    cand = [t(np.where(cm, K["item"][ci], 0)), t(np.where(cm, K["art"][ci], 0)), t(K["loglen"][ci], torch.float32)]
    return h, cand, t(K["label"][ci], torch.float32), t(cm, torch.bool), ci, cm


S = {"train": make_samples(0, args.train_last_days), "val": make_samples(1), "test": make_samples(2)}
t_prep = time.time() - t_prep
print(f"примеров: train {len(S['train'][0]):,}, val {len(S['val'][0]):,}; подготовка {t_prep:.0f} с; устройство {dev}")

model = SeqRanker(meta["V_ITEM"], meta["V_ART"], d=args.d, L=args.L, layers=args.layers).to(dev)
print(f"параметров: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} млн")
ckpt = ROOT / f"checkpoints/{args.name}.pt"
ckpt.parent.mkdir(exist_ok=True)


@torch.no_grad()
def predict(split):
    model.eval()
    Sp = S[split]
    out = np.zeros(len(K["imp_id"]), dtype=np.float32)
    for i in range(0, len(Sp[0]), 256):
        idx = np.arange(i, min(i + 256, len(Sp[0])))
        h, cand, _, _, ci, cm = batch(idx, Sp)
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            p = torch.sigmoid(model(*h, *cand).float()).cpu().numpy()
        out[ci[cm]] = p[cm]
    m = K["split"] == {"train": 0, "val": 1, "test": 2}[split]
    return pl.DataFrame({"imp_id": K["imp_id"][m], "score": out[m]})


def score(split, pred):
    imp = pl.read_parquet(ROOT / "data/processed/impressions.parquet", columns=["imp_id", "uid", "group_id", "label"])
    return evaluate(imp.join(pred, on="imp_id"))


res = {"name": args.name, "objective": "transformer", "n_features": 0, "args": vars(args), "feature_seconds": t_prep}
if not args.eval_only:
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    n_steps = args.epochs * (len(S["train"][0]) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=n_steps, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    lossf = torch.nn.BCEWithLogitsLoss(reduction="none")
    best, t0, step = 1e9, time.time(), 0
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(len(S["train"][0]))
        run_loss = 0.0
        for b in range(len(perm) // args.bs):
            h, cand, y, cm, _, _ = batch(perm[b * args.bs:(b + 1) * args.bs], S["train"])
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                logit = model(*h, *cand)
            loss = (lossf(logit.float(), y) * cm).sum() / cm.sum()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run_loss += loss.item()
            if args.throttle:
                time.sleep(args.throttle)
            if b % 500 == 0:
                print(f"  эпоха {ep} шаг {b}/{len(perm) // args.bs} loss {run_loss / (b + 1):.4f}  {time.time() - t0:.0f} с")
        r = score("val", pv := predict("val"))
        print(f"эпоха {ep}: val {fmt(r)}")
        if r["logloss"] < best:
            best = r["logloss"]
            torch.save(model.state_dict(), ckpt)
            res["best_epoch"], res["val"] = ep, r
            pv.write_parquet(ROOT / f"reports/preds/{args.name}_val.parquet")
    res["train_seconds"] = time.time() - t0
    res["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0
else:
    old = ROOT / f"reports/results/{args.name}.json"
    res = json.load(open(old, encoding="utf-8")) if old.exists() else res
model.load_state_dict(torch.load(ckpt, map_location=dev))
if args.test:
    pt = predict("test")
    pt.write_parquet(ROOT / f"reports/preds/{args.name}_test.parquet")
    res["test"] = score("test", pt)
    print(f"test: {fmt(res['test'])}")
res.setdefault("train_seconds", 0)
json.dump(res, open(ROOT / f"reports/results/{args.name}.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"лучшая эпоха {res.get('best_epoch')}: val {fmt(res['val'])}")
