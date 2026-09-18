"""Запись answer.csv и проверка формата перед отправкой."""

from __future__ import annotations

import csv
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

MAX_ITEMS = 50
QUERY_ID_RE = re.compile(r"^[0-9A-Za-z]{16}$")
ITEM_ID_RE = re.compile(r"^[0-9a-f]{16}$")

RAW = Path("dataset")


def write_answer(predictions: Mapping[str, Sequence[str]], path: str | Path) -> None:
    """Пишет answer.csv. Дубли убираются с сохранением порядка, ответ обрезается до 50."""
    rows = [(qid, " ".join(list(dict.fromkeys(items))[:MAX_ITEMS])) for qid, items in predictions.items()]
    # Всё строками: id нельзя превращать в числа (потеря ведущих нулей, экспоненциальная запись).
    pd.DataFrame(rows, columns=["query_id", "answer"], dtype=str).to_csv(path, index=False)


def validate_answer(path: str | Path, query_ids: set[str], corpus_ids: set[str]) -> list[str]:
    """Возвращает список ошибок; пустой список — файл корректен."""
    errors: list[str] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != ["query_id", "answer"]:
            return [f"заголовок должен быть ['query_id', 'answer'], получено {header}"]
        seen: set[str] = set()
        n_short = 0
        for line_no, row in enumerate(reader, start=2):
            if len(row) != 2:
                errors.append(f"строка {line_no}: ожидалось 2 колонки, получено {len(row)}")
                continue
            qid, answer = row
            if not QUERY_ID_RE.match(qid):
                errors.append(f"строка {line_no}: query_id {qid!r} не 16 символов [0-9A-Za-z]")
            if qid in seen:
                errors.append(f"строка {line_no}: повтор query_id {qid}")
            seen.add(qid)
            if qid not in query_ids:
                errors.append(f"строка {line_no}: query_id {qid} нет в benchmark_queries")
            items = answer.split(" ") if answer else []
            if answer != answer.strip() or "" in items:
                errors.append(f"строка {line_no}: лишние пробелы в answer")
            if len(items) > MAX_ITEMS:
                errors.append(f"строка {line_no}: {len(items)} item_id > {MAX_ITEMS}")
            if len(set(items)) != len(items):
                errors.append(f"строка {line_no}: повторы item_id внутри строки")
            bad_format = [i for i in items if i and not ITEM_ID_RE.match(i)]
            if bad_format:
                errors.append(f"строка {line_no}: item_id не 16 hex в нижнем регистре: {bad_format[:3]}")
            not_in_corpus = [i for i in items if ITEM_ID_RE.match(i) and i not in corpus_ids]
            if not_in_corpus:
                errors.append(f"строка {line_no}: item_id нет в корпусе: {not_in_corpus[:3]}")
            n_short += len(items) < MAX_ITEMS
    missing = query_ids - seen
    if missing:
        errors.append(f"нет строк для {len(missing)} query_id, например {sorted(missing)[:3]}")
    if n_short and not errors:
        # Не ошибка формата, но пустые слоты — упущенный recall: порядок не важен, заполнять стоит все 50.
        print(f"предупреждение: {n_short} строк содержат меньше {MAX_ITEMS} item_id", file=sys.stderr)
    return errors


def main(path: str) -> None:
    query_ids = set(pd.read_parquet(RAW / "benchmark_queries.parquet", columns=["query_id"]).query_id)
    corpus_ids = set(pd.read_parquet(RAW / "benchmark_items.parquet", columns=["item_id"]).item_id)
    errors = validate_answer(path, query_ids, corpus_ids)
    if errors:
        print(f"НАЙДЕНО ОШИБОК: {len(errors)}")
        print("\n".join(errors[:50]))
        sys.exit(1)
    print("answer.csv корректен")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "answer.csv")
