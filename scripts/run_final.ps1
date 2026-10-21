# Финал: ОДНОКРАТНАЯ оценка замороженного списка моделей на test (без переобучения). ~15-20 мин.
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
$env:POLARS_MAX_THREADS = "2"
$py = "D:\yambda-rl\.venv\Scripts\python.exe"
& $py scripts/baselines.py --test
& $py scripts/final_test.py hand hand_rank auto_scratch auto_hand auto_hand_g_rank
& $py train/seq.py --name seq_L512 --L 512 --eval_only --test
& $py train/seq.py --name seqret --L 256 --retrieve 32 --eval_only --test
& $py scripts/blend.py hand seqret --name blend_hand_seqret --test
& $py scripts/compare.py hand auto_scratch auto_hand seq_L512 seqret blend_hand_seqret hand_rank --split test
& $py scripts/compare.py hand_rank auto_hand_g_rank --split test
& $py scripts/summary.py test
