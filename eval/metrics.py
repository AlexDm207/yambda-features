"""Метрики: поточечные (LogLoss, ROC-AUC) и ранжирования внутри групп «пользователь-день».

Почему два вида. Разброс доли дослушиваний между пользователями (std 0.27) больше, чем между треками
(0.16), поэтому общий ROC-AUC во многом измеряет «угадала ли модель пользователя-скипера». Для
рекомендаций важнее другое: правильно ли упорядочены треки ОДНОГО пользователя. Это меряют GAUC и
NDCG@K внутри группы. Группы, где все метки одинаковые, ранжированию ничего не дают и исключаются.
"""
import numpy as np
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score


def evaluate(df: pl.DataFrame, score_col: str = "score", ks=(5, 10), seed: int = 0, prob: bool = True) -> dict:
    """df: group_id, uid, label, score. Ничьи в score разбиваются случайно (иначе константная модель
    «ранжировала» бы по порядку показов, а он информативен)."""
    y = df["label"].to_numpy()
    s = df[score_col].to_numpy().astype(np.float64)
    res = {"n": int(len(y)), "pos_rate": float(y.mean())}
    if prob:
        res["logloss"] = float(log_loss(y, np.clip(s, 1e-6, 1 - 1e-6)))
    res["auc"] = float(roc_auc_score(y, s))

    rng = np.random.default_rng(seed)
    d = df.select("group_id", "label", pl.col(score_col).alias("s")).with_columns(
        pl.Series("tb", rng.random(df.height))
    )
    d = d.with_columns(
        pl.col("label").sum().over("group_id").alias("npos"),
        pl.len().over("group_id").alias("ng"),
    ).filter((pl.col("npos") > 0) & (pl.col("npos") < pl.col("ng")))
    res["groups_mixed"] = int(d["group_id"].n_unique())

    # GAUC: AUC внутри группы через ранги, среднее по группам (без весов и с весом = размер группы)
    d = d.with_columns((pl.col("s") + pl.col("tb") * 1e-9).rank("average").over("group_id").alias("r"))
    g = d.group_by("group_id").agg(
        pl.col("r").filter(pl.col("label") == 1).sum().alias("rs"),
        pl.col("npos").first(), pl.col("ng").first(),
    ).with_columns(
        ((pl.col("rs") - pl.col("npos") * (pl.col("npos") + 1) / 2)
         / (pl.col("npos") * (pl.col("ng") - pl.col("npos")))).alias("auc")
    )
    res["gauc"] = float(g["auc"].mean())
    res["gauc_w"] = float((g["auc"] * g["ng"]).sum() / g["ng"].sum())

    # NDCG@K: позиции по убыванию score внутри группы
    d = d.sort(["group_id", "s", "tb"], descending=[False, True, True]).with_columns(
        (pl.int_range(pl.len()).over("group_id") + 1).alias("pos")
    )
    for k in ks:
        dcg = d.filter(pl.col("pos") <= k).group_by("group_id").agg(
            (pl.col("label") / (pl.col("pos") + 1).log(2)).sum().alias("dcg"),
            pl.col("npos").first(),
        )
        ideal = np.cumsum(1 / np.log2(np.arange(2, k + 2)))
        idcg = ideal[np.minimum(dcg["npos"].to_numpy(), k) - 1]
        res[f"ndcg@{k}"] = float(np.mean(dcg["dcg"].to_numpy() / idcg))
        # доля скипов среди первых K: «сколько плохих треков пользователь увидит первыми»
        res[f"skip@{k}"] = float(1 - d.filter(pl.col("pos") <= k)["label"].mean())
    return res


def fmt(r: dict) -> str:
    keys = ["logloss", "auc", "gauc", "ndcg@5", "ndcg@10", "skip@10"]
    return "  ".join(f"{k}={r[k]:.4f}" for k in keys if k in r)


def per_group(df: pl.DataFrame, score_col: str = "score", k: int = 10, seed: int = 0) -> pl.DataFrame:
    """Метрики по каждой смешанной группе: group_id, uid, auc, ndcg (для бутстрепа по пользователям)."""
    rng = np.random.default_rng(seed)
    d = df.select("group_id", "uid", "label", pl.col(score_col).alias("s")).with_columns(
        pl.Series("tb", rng.random(df.height))
    ).with_columns(
        pl.col("label").sum().over("group_id").alias("npos"), pl.len().over("group_id").alias("ng"),
    ).filter((pl.col("npos") > 0) & (pl.col("npos") < pl.col("ng")))
    d = d.with_columns((pl.col("s") + pl.col("tb") * 1e-9).rank("average").over("group_id").alias("r"))
    d = d.sort(["group_id", "s", "tb"], descending=[False, True, True]).with_columns(
        (pl.int_range(pl.len()).over("group_id") + 1).alias("pos"))
    g = d.group_by("group_id").agg(
        pl.col("uid").first(), pl.col("npos").first(), pl.col("ng").first(),
        pl.col("r").filter(pl.col("label") == 1).sum().alias("rs"),
        (pl.col("label") / (pl.col("pos") + 1).log(2)).filter(pl.col("pos") <= k).sum().alias("dcg"),
    )
    ideal = np.cumsum(1 / np.log2(np.arange(2, k + 2)))
    g = g.with_columns(
        ((pl.col("rs") - pl.col("npos") * (pl.col("npos") + 1) / 2) / (pl.col("npos") * (pl.col("ng") - pl.col("npos")))).alias("auc"),
    )
    g = g.with_columns(pl.Series("ndcg", g["dcg"].to_numpy() / ideal[np.minimum(g["npos"].to_numpy(), k) - 1]))
    return g.select("group_id", "uid", "auc", "ndcg").sort("group_id")
