#!/usr/bin/env bash
# Полный прогон от сырых parquet до answer.csv. Тяжёлые шаги идут по одному, суммарно ~4 часа на M1.
# Если уже скачаны артефакты (data/models/e5-small-ft, data/rerank/model_rr_all_ens5_v5.joblib),
# достаточно последних двух команд.
set -euo pipefail

PY=${PY:-python}

$PY -m src.build_dataset

# валидация: сначала проверочный сплит, потом обучающий — без пересечения по тексту запроса
$PY -m src.validation --n 5000 --seed 42
$PY -m src.validation --n 40000 --seed 7 --name rr_train_n40000_s7 --exclude-split val_n5000_s42

# дообучение энкодера (~70 мин), сиды зафиксированы внутри
$PY -m src.finetune.e5_contrastive

# признаки и пять моделей ансамбля (~2 часа, кэш признаков пишется в data/rerank)
$PY -m src.rerank.train

$PY -m src.rerank.predict --model rr_all_ens5 --out answer.csv
$PY -m src.submission answer.csv
