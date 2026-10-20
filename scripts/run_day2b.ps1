# День 2б: шум от случайного зерна + интервалы для абляций. Запуск: .\scripts\run_day2b.ps1
$py = "D:\yambda-rl\.venv\Scripts\python.exe"
& $py train/gbdt.py --name hand_s7 --seed 7
& $py train/gbdt.py --name hand_s13 --seed 13
& $py scripts/compare.py hand hand_s7 hand_s13 hand_rank hand_no_user hand_no_item hand_no_pair_direct hand_no_pair_indirect hand_no_artist_album
