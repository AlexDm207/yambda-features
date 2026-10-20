"""Шаг 2а. Построить накопительные таблицы движка признаков по одной (можно прервать и запустить снова:
готовые таблицы пропускаются)."""
import os

os.environ.setdefault("POLARS_MAX_THREADS", "4")
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features.engine import N_BUCKETS, build_all

print(f"корзин пользователей: {N_BUCKETS}")
build_all()
print("Все накопительные таблицы готовы.")
