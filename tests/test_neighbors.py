"""Тест запросозависимой истории (H12): max / mean сходства по текстам объявления через reduceat."""

import numpy as np

from src.rerank.neighbors import QueryNeighbors


def test_item_text_sims_max_mean_count():
    nb = object.__new__(QueryNeighbors)  # без загрузки данных: проверяем только арифметику
    nb.text_emb = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype=np.float32)
    nb.item_texts = {"a": np.array([0, 1]), "b": np.array([2])}
    q = np.array([1.0, 0.0], dtype=np.float32)
    mx, mean, cnt = nb.item_text_sims(q, np.array(["a", "x", "b"]))
    assert np.allclose(mx[[0, 2]], [1.0, 0.6])
    assert np.allclose(mean[[0, 2]], [0.5, 0.6])
    assert np.isnan(mx[1]) and cnt[1] == 0
    assert list(cnt[[0, 2]]) == [2, 1]
