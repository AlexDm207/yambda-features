# День 4: трансформер над историей событий. Щадящий режим: пониженный приоритет, 2 потока CPU,
# пауза 0.05 с после каждого шага обучения (видеокарта не работает на 100% непрерывно).
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
$env:POLARS_MAX_THREADS = "2"
$py = "D:\yambda-rl\.venv\Scripts\python.exe"
if (-not (Test-Path data/seq/cands.npz)) { & $py data/prepare_seq.py }      # события и кандидаты в numpy, порциями
& $py train/seq.py --name seq_shuf --epochs 1 --shuffle_train_labels --throttle 0.05   # проверка утечки: ждём GAUC ~ 0.5
& $py train/seq.py --name seq --L 256 --throttle 0.05                      # основная модель
& $py train/seq.py --name seq_L64 --L 64 --throttle 0.05                   # короче история
& $py train/seq.py --name seq_L512 --L 512 --bs 64 --throttle 0.05         # длиннее история
& $py scripts/compare.py hand seq_shuf seq seq_L64 seq_L512 hand_rank
& $py scripts/summary.py val
