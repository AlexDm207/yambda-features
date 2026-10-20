# Ручные признаки vs автоматические признаки vs трансформер над историей (Yambda-50M)

Черновик, заполняется по ходу работы.

## Задача
Показ = трек, запущенный рекомендательной системой (`is_organic = 0`). Метка = дослушал ≥ 50%.
Признаки для показа в день *d* считаются только по событиям дней < *d*.

## Воспроизведение
```
python data/download_meta.py   # справочники трек→артист, трек→альбом
python data/prepare.py         # события, показы, метки, разбиение, проверки утечек
python data/eda.py             # EDA под постановку
python scripts/build_cache.py             # накопительные таблицы движка признаков (по одной, с докачкой)
python train/gbdt.py --name hand          # ручные признаки + LightGBM
python scripts/check_leaks.py             # сверка движка признаков с расчётом «в лоб»
python scripts/baselines.py               # эвристики без обучения
.\scripts\run_day2.ps1                     # lambdarank + абляции групп
.\scripts\run_day2b.ps1                    # шум от seed + интервалы (scripts/compare.py)
.\scripts\run_day3.ps1                     # автопоиск признаков (search/autofe.py)
```
