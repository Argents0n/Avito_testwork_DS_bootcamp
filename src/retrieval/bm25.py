"""BM25 на разреженных матрицах scipy.

Своя реализация, потому что нужны скоры по всему корпусу — их потом маскируем гео-пулами.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vocab: dict[str, int] = {}
        self.weights: sp.csc_matrix | None = None

    def fit(self, docs: Sequence[str]) -> BM25:
        """docs — строки стеммов через пробел (формат items_text.parquet)."""
        rows, cols, vals = [], [], []
        doc_len = np.zeros(len(docs), dtype=np.float32)
        vocab = self.vocab
        for i, doc in enumerate(docs):
            toks = doc.split()
            doc_len[i] = len(toks)
            for tok, tf in Counter(toks).items():
                j = vocab.setdefault(tok, len(vocab))
                rows.append(i)
                cols.append(j)
                vals.append(tf)
        tf = sp.csr_matrix((np.asarray(vals, dtype=np.float32), (rows, cols)), shape=(len(docs), len(vocab)))
        df = np.bincount(tf.indices, minlength=len(vocab)).astype(np.float32)
        # Вариант idf без отрицательных значений (как в Lucene): частые слова весят мало, но не штрафуют.
        idf = np.log1p((len(docs) - df + 0.5) / (df + 0.5))
        norm = self.k1 * (1 - self.b + self.b * doc_len / max(doc_len.mean(), 1.0))
        tf = tf.tocoo()
        w = idf[tf.col] * tf.data * (self.k1 + 1) / (tf.data + norm[tf.row])
        # CSC: быстрый срез столбцов (термов запроса).
        self.weights = sp.csc_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)
        return self

    def scores(self, query: str) -> np.ndarray:
        """Скоры запроса по всем документам. Повторы слов в запросе считаются один раз:"""
        idx = sorted({self.vocab[t] for t in query.split() if t in self.vocab})
        if not idx:
            return np.zeros(self.weights.shape[0], dtype=np.float32)
        return np.asarray(self.weights[:, idx].sum(axis=1)).ravel()

    def term_hits(self, query: str, rows: np.ndarray) -> np.ndarray:
        """Сколько разных термов запроса встречается в каждом из документов rows."""
        idx = sorted({self.vocab[t] for t in query.split() if t in self.vocab})
        if not idx:
            return np.zeros(len(rows), dtype=np.float32)
        sub = self.weights[:, idx].tocsr()[rows]
        return np.asarray((sub > 0).sum(axis=1), dtype=np.float32).ravel()
