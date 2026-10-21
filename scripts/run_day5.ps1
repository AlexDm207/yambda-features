# День 5 (только val): трансформер с поиском по всей истории, регуляризация, смеси с бустингом.
# Щадящий режим. ~2,5-3 ч. Test здесь НЕ трогаем.
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
$env:POLARS_MAX_THREADS = "2"
$py = "D:\yambda-rl\.venv\Scripts\python.exe"
if (-not (Test-Path data/seq/retrieval.npz)) { & $py data/prepare_retrieval.py }   # индекс поиска по истории
$common = "--epochs 2 --bs 64 --dropout 0.2 --wd 0.05 --throttle 0.05".Split(" ")
& $py train/seq.py --name seqret_shuf --epochs 1 --bs 64 --retrieve 32 --shuffle_train_labels --throttle 0.05   # ждём GAUC ~ 0.5
& $py train/seq.py --name seq512_reg --L 512 @common                    # без поиска, но с регуляризацией (честное сравнение)
& $py train/seq.py --name seqret --L 256 --retrieve 32 @common          # + поиск по всей истории
& $py scripts/blend.py hand seqret --name blend_hand_seqret
& $py scripts/blend.py hand seq512_reg --name blend_hand_seq512
& $py scripts/compare.py hand seqret_shuf seq_L512 seq512_reg seqret blend_hand_seq512 blend_hand_seqret
& $py scripts/summary.py val
