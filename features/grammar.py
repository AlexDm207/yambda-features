"""Грамматика признаков для автопоиска: все комбинации «сущность × статистика × окно × вид».
Ручной набор (handcrafted.py) — небольшое подмножество этого пространства."""
from features.engine import ENTITIES

WINDOWS = [1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, "inf"]
SUM_STATS = ["n", "full", "rec", "rec_full", "org", "sec", "like", "dislike", "unlike"]
RATES = ["full/n", "rec_full/rec", "ratio/n", "org/n", "like/n", "dislike/n", "sec/n", "rec/n"]
SHARE_STATS = ["n", "full", "rec", "like"]
DRIFT_RATES = ["full/n", "rec_full/rec"]
DRIFT_WINDOWS = ["1-30", "3-14", "7-30", "7-inf", "30-inf"]
LIFT_RATES = ["full/n", "rec_full/rec", "ratio/n"]
PAIR = {"ui", "ua", "ual"}


def candidates(ent: str) -> list[str]:
    out = []
    for w in WINDOWS:
        out += [f"sum:{ent}:{s}:{w}" for s in SUM_STATS]
        out += [f"rate:{ent}:{r}:{w}" for r in RATES]
        if ent in PAIR:
            out += [f"share:{ent}:{s}:{w}" for s in SHARE_STATS]
            out += [f"lift:{ent}:{r}:{w}" for r in LIFT_RATES]
    out += [f"drift:{ent}:{r}:{w}" for r in DRIFT_RATES for w in DRIFT_WINDOWS]
    out += [f"recency:{ent}", f"age:{ent}"]
    return out


def all_candidates() -> dict[str, list[str]]:
    return {e: candidates(e) for e in ENTITIES}


if __name__ == "__main__":
    c = all_candidates()
    for e, l in c.items():
        print(e, len(l))
    print("всего:", sum(map(len, c.values())) + 1, "(+ static:length)")
