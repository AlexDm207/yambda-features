# День 2: ранжирующая цель и абляции групп ручных признаков (val). Запуск: .\scripts\run_day2.ps1
$py = "D:\yambda-rl\.venv\Scripts\python.exe"
& $py scripts/build_cache.py
& $py train/gbdt.py --name hand_rank --objective lambdarank
foreach ($g in "user", "item", "artist_album", "pair_direct", "pair_indirect") {
    & $py train/gbdt.py --name "hand_no_$g" --drop $g
}
& $py scripts/summary.py val
