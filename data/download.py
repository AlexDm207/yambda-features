"""Скачать события Yambda-50M (multi_event.parquet, ~370 МБ) в data/raw/flat/50m/.
После скачивания укажите в configs/base.yaml: raw_path: data/raw/flat/50m/multi_event.parquet"""
from huggingface_hub import hf_hub_download

p = hf_hub_download(repo_id="yandex/yambda", repo_type="dataset", filename="flat/50m/multi_event.parquet", local_dir="data/raw")
print("saved to:", p)
