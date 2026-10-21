"""Графики для отчёта -> reports/figures/*.png (по сохранённым результатам, ничего не пересчитывает)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
R, A, OUT = ROOT / "reports/results", ROOT / "reports/auto", ROOT / "reports/figures"
OUT.mkdir(parents=True, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"]  # категориальные слоты 1-4 (проверенная палитра)
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 11, "axes.titlecolor": INK, "lines.linewidth": 2,
})


def save(fig, name, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)
    print("  ", name)


def res(name):
    p = R / f"{name}.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


# 1. Качество признака от окна (одномерный отсев автопоиска)
sc = {r["spec"]: r for r in pl.read_csv(A / "screen.csv").iter_rows(named=True)}
W = ["1", "2", "3", "5", "7", "10", "14", "21", "30", "45", "60", "90", "inf"]
fams = [("rate:u:rec_full/rec", "CTR рекомендаций пользователя"), ("rate:ui:full/n", "доля дослушиваний трека пользователем"),
        ("rate:i:rec_full/rec", "CTR трека в рекомендациях"), ("lift:ua:rec_full/rec", "lift артиста у пользователя")]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k, (f, lab) in enumerate(fams):
    xs = [i for i, w in enumerate(W) if f"{f}:{w}" in sc]
    ax[0].plot(xs, [sc[f"{f}:{W[i]}"]["dll"] * 1e3 for i in xs], "-o", ms=4, color=C[k], label=lab)
    ax[1].plot(xs, [sc[f"{f}:{W[i]}"]["gauc"] for i in xs], "-o", ms=4, color=C[k], label=lab)
for a, t in zip(ax, ["Выигрыш в LogLoss ×10³ (один признак)", "GAUC (один признак)"]):
    a.set_xticks(range(len(W)), [w if w != "inf" else "вся" for w in W])
    a.set_xlabel("окно, дней")
    a.set_title(t, loc="left")
    for i in (4, 8):  # окна 7 и 30 из ручного набора
        a.axvline(i, color=INK2, lw=0.8, ls=":")
ax[1].axhline(0.5, color=INK2, lw=0.8)
h_, l_ = ax[0].get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, fontsize=9, loc="lower center", ncol=4)
fig.tight_layout(rect=(0, 0.08, 1, 1))
save(fig, "window_curves.png", tight=False)

# 2. Жадный отбор: GAUC на выборке поиска (sval) и на скрытых пользователях (hval)
runs = [("forward_scratch.jsonl", "с нуля, критерий LogLoss"), ("forward_scratch_fam_gauc.jsonl", "с нуля, критерий GAUC"),
        ("forward_hand.jsonl", "поверх ручного, критерий LogLoss"), ("forward_hand_fam_gauc.jsonl", "поверх ручного, критерий GAUC")]
fig, ax = plt.subplots(1, 4, figsize=(14, 3.6), sharey=False)
for a, (f, t) in zip(ax, runs):
    if not (A / f).exists():
        continue
    H = [h for h in (json.loads(l) for l in open(A / f)) if h["accepted"]]
    x = range(len(H))
    a.plot(x, [h["sval_gauc"] for h in H], "-o", ms=4, color=C[0], label="sval (решения поиска)")
    a.plot(x, [h["hval_gauc"] for h in H], "-o", ms=4, color=C[1], label="hval (скрытые пользователи)")
    a.set_title(t, loc="left", fontsize=10)
    a.set_xlabel("принято признаков")
ax[0].set_ylabel("GAUC")
ax[0].legend(frameon=False, fontsize=8.5)
save(fig, "forward_curves.png")

# 3. Сравнение с ручным бустингом на test: разница GAUC и 95% интервал (бутстреп по пользователям)
cmp = json.load(open(R / "compare_hand_test.json", encoding="utf-8"))
cr = R / "compare_hand_rank_test.json"
names = {"auto_scratch": "автопоиск с нуля (18 пр.)", "auto_hand": "ручной + автопоиск", "seq_L512": "трансформер, 512 событий",
         "seqret": "трансформер + поиск по истории", "blend_hand_seqret": "смесь: бустинг + трансформер с поиском",
         "hand_rank": "ручной, lambdarank"}
items = sorted(cmp.items(), key=lambda kv: kv[1]["auc"]["diff"])
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for a, m, t in [(ax[0], "auc", "Δ GAUC к ручному бустингу (test)"), (ax[1], "ndcg", "Δ NDCG@10 к ручному бустингу (test)")]:
    for i, (n, v) in enumerate(items):
        d = v[m]
        a.plot([d["lo"], d["hi"]], [i, i], color=C[0], lw=2, solid_capstyle="round")
        a.plot(d["diff"], i, "o", ms=8, color=C[0], mec=BG, mew=2)
    a.axvline(0, color=INK2, lw=1)
    a.set_title(t, loc="left")
ax[0].set_yticks(range(len(items)), [names.get(n, n) for n, _ in items])
save(fig, "test_compare.png")

# 4. Качество против вычислительных затрат (test GAUC, минуты на признаки + обучение)
models = [("hand", "ручной", 0), ("hand_rank", "ручной, lambdarank", 0), ("auto_scratch", "авто с нуля", 0),
          ("auto_hand", "ручной+авто", 0), ("auto_hand_g_rank", "ручной+авто, lambdarank", 0),
          ("seq_L512", "трансформер L=512", 1), ("seqret", "трансформер + поиск", 1), ("blend_hand_seqret", "смесь", 2)]
fam_lab = ["бустинг", "трансформер", "смесь"]
fig, ax = plt.subplots(figsize=(8, 4.5))
seen = set()
for n, lab, fam in models:
    r = res(n)
    if not r or "test" not in r:
        continue
    t = (r.get("feature_seconds", 0) + r.get("train_seconds", 0)) / 60
    if n == "blend_hand_seqret":
        t = sum((res(x)["feature_seconds"] + res(x)["train_seconds"]) / 60 for x in ("hand", "seqret"))
    ax.plot(t, r["test"]["gauc"], "o", ms=9, color=C[fam], mec=BG, mew=2, label=None if fam in seen else fam_lab[fam])
    seen.add(fam)
    off = {"hand": (6, -12, "left"), "auto_hand_g_rank": (6, -12, "left"), "auto_hand": (-8, 4, "right")}.get(n, (6, 4, "left"))
    ax.annotate(lab, (t, r["test"]["gauc"]), xytext=off[:2], textcoords="offset points", fontsize=8.5, color=INK, ha=off[2])
ax.set_xscale("log")
ax.set_xticks([1, 2, 3, 5, 10, 20, 30], ["1", "2", "3", "5", "10", "20", "30"])
ax.minorticks_off()
ax.set_xlabel("время: признаки + обучение, мин (лог. шкала)")
ax.set_ylabel("GAUC на test")
ax.set_title("Качество против затрат (ноутбук, RTX 3050 Ti)", loc="left")
ax.legend(frameon=False)
save(fig, "cost_quality.png")

# 5. Трансформер: длина истории и поиск по всей истории (val)
seqs = [("seq_L64", "64 последних"), ("seq", "256 последних"), ("seq_L512", "512 последних"), ("seqret", "256 + поиск по всей истории")]
fig, ax = plt.subplots(figsize=(8, 3.4))
vals = [(lab, res(n)["val"]["gauc"]) for n, lab in seqs if res(n)]
for i, (lab, g) in enumerate(vals):
    ax.plot([0.5, g], [i, i], color=GRID, lw=2)
    ax.plot(g, i, "o", ms=9, color=C[0], mec=BG, mew=2)
    ax.annotate(f"{g:.4f}", (g, i), xytext=(8, -3), textcoords="offset points", fontsize=9, color=INK)
h = res("hand")["val"]["gauc"]
ax.axvline(h, color=C[1], lw=1.5, ls="--")
ax.annotate(f"ручной бустинг {h:.4f}", (h, 1.5), xytext=(-6, 0), textcoords="offset points",
            ha="right", fontsize=9, color=INK)
ax.set_yticks(range(len(vals)), [v[0] for v in vals])
ax.set_xlim(0.53, max(h, max(v[1] for v in vals)) + 0.01)
ax.set_xlabel("GAUC на val")
ax.set_title("Трансформер: сколько истории он видит", loc="left")
save(fig, "history_length.png")
