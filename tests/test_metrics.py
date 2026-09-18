"""Тесты метрики: пример из условия задачи и крайние случаи, влияющие на итоговую цифру."""

import pytest

from src.metrics import recall_at_k, recall_per_query


def test_example_from_task_statement():
    # Условие: A — 1 из 1, Б — 1 из 2, В — 0 из 1 → (1 + 0.5 + 0) / 3 = 0.5.
    truth = {"A": {"a1"}, "B": {"b1", "b2"}, "C": {"c1"}}
    predictions = {"A": ["a1", "x"], "B": ["b1", "y"], "C": ["z"]}
    assert recall_at_k(predictions, truth) == pytest.approx(0.5)


def test_missing_prediction_counts_as_zero():
    truth = {"A": {"a1"}, "B": {"b1"}}
    assert recall_at_k({"A": ["a1"]}, truth) == pytest.approx(0.5)


def test_only_first_k_unique_ids_count():
    truth = {"A": {"hit"}}
    # Дубли не должны «сдвигать» правильный id в топ-k.
    assert recall_per_query({"A": ["x", "x", "hit"]}, truth, k=2)["A"] == 1.0
    assert recall_per_query({"A": ["x", "y", "hit"]}, truth, k=2)["A"] == 0.0


def test_top_from_tiers_exclude_fills_all_slots():
    # Регрессия: исключённые индексы в маленьком ярусе не должны съедать места.
    import numpy as np

    from src.retrieval.pools import top_from_tiers

    key = np.arange(10, dtype=float)
    out = top_from_tiers(key, [np.array([0, 1, 2]), np.arange(10)], k=5, exclude=np.array([1, 2]))
    assert len(out) == 5 and not set(out) & {1, 2}
    assert list(out[:1]) == [0]
