"""Сводная таблица всех запусков из reports/results/*.json -> reports/summary.md (и печать)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
split = sys.argv[1] if len(sys.argv) > 1 else "val"
rows = []
for p in sorted((ROOT / "reports/results").glob("*.json")):
    r = json.load(open(p, encoding="utf-8"))
    if p.stem == "baselines":
        for k, m in r.items():
            if k.startswith(split + "/"):
                rows.append(("эвристика: " + k.split("/", 1)[1], "-", "-", m))
        continue
    if split in r:
        rows.append((r["name"], r.get("objective", "binary"), r["n_features"], r[split] | {
            "t": r["feature_seconds"] + r["train_seconds"]}))
cols = ["logloss", "auc", "gauc", "ndcg@5", "ndcg@10", "skip@10", "t"]
lines = [f"| модель | цель | признаков | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 3)]
for name, obj, nf, m in rows:
    vals = [f"{m[c]:.4f}" if isinstance(m.get(c), float) and c != "t" else (f"{m[c]:.0f} с" if c in m and c == "t" else "-") for c in cols]
    lines.append(f"| {name} | {obj} | {nf} | " + " | ".join(vals) + " |")
text = "\n".join(lines)
print(text)
(ROOT / f"reports/summary_{split}.md").write_text(text + "\n", encoding="utf-8")
