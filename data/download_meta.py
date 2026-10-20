"""Скачивает маленькие справочники Yambda: трек -> артист, трек -> альбом.
Сами события (multi_event.parquet) уже лежат в D:/yambda-rl, путь задаётся в configs/base.yaml."""
from huggingface_hub import HfApi, hf_hub_download
import polars as pl

REPO = "yandex/yambda"

info = HfApi().dataset_info(REPO, files_metadata=True)
files = [s for s in info.siblings if s.rfilename.startswith(("artist", "album"))]
for s in files:
    print(f"{(s.size or 0) / 1e6:8.1f} MB  {s.rfilename}")

for s in files:
    if not s.rfilename.endswith(".parquet"):
        continue
    p = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=s.rfilename, local_dir="data/raw")
    df = pl.read_parquet(p)
    print("\n", p, df.shape)
    print(df.schema)
    print(df.head(3))
    # Сколько треков имеют больше одного артиста/альбома
    key = [c for c in df.columns if c != "item_id"][0]
    multi = df.group_by("item_id").agg(pl.col(key).n_unique().alias("k")).filter(pl.col("k") > 1).height
    print(f"треков с >1 значением {key}: {multi} из {df['item_id'].n_unique()}")
