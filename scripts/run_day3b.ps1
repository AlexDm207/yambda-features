# День 3б: автопоиск v2 — разнообразный пул (по семействам) и критерий GAUC; ранжирующие версии.
# Щадящий режим: 2 потока, пониженный приоритет. Можно перезапускать — продолжит с места остановки.
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'   # дочерние процессы наследуют приоритет
$env:POLARS_MAX_THREADS = "2"
$py = "D:\yambda-rl\.venv\Scripts\python.exe"
if (-not (Test-Path reports/auto/pool_fam.csv)) { & $py search/autofe.py --stage pool --pool_mode family --order_by gauc --tag _fam }
& $py search/autofe.py --stage forward --start scratch --pool_tag _fam --criterion gauc --delta 0.0005 --tag _fam_gauc --threads 2
& $py search/autofe.py --stage forward --start hand --pool_tag _fam --criterion gauc --delta 0.0005 --tag _fam_gauc --threads 2
& $py train/gbdt.py --name auto_scratch_g --specs reports/auto/selected_scratch_fam_gauc.txt --threads 2
& $py train/gbdt.py --name auto_hand_g --specs reports/auto/selected_hand_fam_gauc.txt --threads 2
& $py train/gbdt.py --name auto_hand_rank --specs reports/auto/selected_hand.txt --objective lambdarank --threads 2
& $py train/gbdt.py --name auto_hand_g_rank --specs reports/auto/selected_hand_fam_gauc.txt --objective lambdarank --threads 2
& $py scripts/compare.py hand auto_scratch auto_hand auto_scratch_g auto_hand_g
& $py scripts/compare.py hand_rank auto_hand_rank auto_hand_g_rank
& $py scripts/summary.py val
