"""Обёртка над HuggingFace-энкодерами.

Не sentence-transformers: у USER-base конфиг не читается актуальной версией.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    pooling: str  # "mean" или "cls" — как модель обучалась
    query_prefix: str = ""  # e5 и USER обучены с префиксами «query: » / «passage: », без них качество падает
    doc_prefix: str = ""


# Характеристики взяты из карточек моделей на HuggingFace.
ENCODERS = {
    "rubert-tiny2": EncoderSpec("cointegrated/rubert-tiny2", "cls"),
    "e5-small": EncoderSpec("intfloat/multilingual-e5-small", "mean", "query: ", "passage: "),
    "e5-base": EncoderSpec("intfloat/multilingual-e5-base", "mean", "query: ", "passage: "),
    "user-base": EncoderSpec("deepvk/USER-base", "mean", "query: ", "passage: "),
    # e5-small, дообученный на парах трейна (src/finetune/e5_contrastive.py, H8).
    "e5-small-ft": EncoderSpec("data/models/e5-small-ft", "mean", "query: ", "passage: "),
    # Промежуточная копия после 1000 шагов из 2286: сравниваем с финальной, чтобы не взять переобученную модель.
    "e5-small-ft-step1000": EncoderSpec("data/models/e5-small-ft-step1000", "mean", "query: ", "passage: "),
    # Второй круг дообучения: старт с e5-small-ft, 400 тыс. пар (H15).
    "e5-small-ft2": EncoderSpec("data/models/e5-small-ft2", "mean", "query: ", "passage: "),
}


def pick_device() -> str:
    """CUDA на арендованной GPU-машине, MPS на Mac, иначе CPU — код не меняется при переезде."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Encoder:
    def __init__(self, spec: EncoderSpec, max_length: int = 128, device: str | None = None):
        self.spec = spec
        self.max_length = max_length
        self.device = device or pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(spec.name)
        self.model = AutoModel.from_pretrained(spec.name).to(self.device).eval()

    @torch.inference_mode()
    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        out = []
        # Сортировка по длине уменьшает паддинг в батчах — на MPS это ускоряет кодирование в разы.
        order = np.argsort([len(t) for t in texts])
        for start in range(0, len(texts), batch_size):
            batch = [texts[i] for i in order[start : start + batch_size]]
            enc = self.tokenizer(
                batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            ).to(self.device)
            hidden = self.model(**enc).last_hidden_state
            if self.spec.pooling == "cls":
                emb = hidden[:, 0]
            else:
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(torch.nn.functional.normalize(emb, dim=-1).float().cpu().numpy())
        emb = np.concatenate(out)
        result = np.empty_like(emb)
        result[order] = emb
        return result

    def encode_queries(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        return self._encode([self.spec.query_prefix + t for t in texts], batch_size)

    def encode_docs(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self._encode([self.spec.doc_prefix + t for t in texts], batch_size)
