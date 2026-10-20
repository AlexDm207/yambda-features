# День 3: автопоиск признаков. Долгий (~4-5 ч), лучше на ночь и от сети.
# Все этапы продолжаются с места остановки: при сбое просто запусти скрипт снова.
$py = "D:\yambda-rl\.venv\Scripts\python.exe"
& $py scripts/check_leaks.py --grammar 300 --n 100        # проверка движка на новых видах признаков
& $py search/autofe.py --stage samples
& $py search/autofe.py --stage features                   # ~30 мин
& $py search/autofe.py --stage screen                     # ~30-40 мин
& $py search/autofe.py --stage pool
& $py search/autofe.py --stage forward --start scratch    # ~1-1.5 ч
& $py search/autofe.py --stage forward --start hand       # ~1.5-2 ч
& $py train/gbdt.py --name auto_scratch --specs reports/auto/selected_scratch.txt
& $py train/gbdt.py --name auto_hand --specs reports/auto/selected_hand.txt
& $py scripts/compare.py hand auto_scratch auto_hand
& $py scripts/summary.py val
