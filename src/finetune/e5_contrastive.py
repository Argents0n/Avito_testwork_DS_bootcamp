"""Дообучение e5-small на парах «запрос → выбранное объявление».

InfoNCE с негативами батча плюс жёсткий негатив: объявление того же города с мест 5–30 по исходному e5.
Сессии валидации и обучения переранжирования исключены.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from ..embed_corpus import doc_texts
from ..retrieval.encoder import ENCODERS, pick_device

PROCESSED = Path("data/processed")
VALIDATION = Path("data/validation")
OUT_DIR = Path("data/models/e5-small-ft")
# Второй круг (H15): стартуем от уже дообученной модели и проходим больший набор пар с меньшим шагом обучения.
HARD_NEG_CACHE_TEMPLATE = "data/finetune/hard_negatives_{n}.parquet"
EXCLUDE_SPLITS = ("val_n5000_s42", "rr_train_n40000_s7")
HARD_NEG_RANKS = (5, 30)
TEMPERATURE = 0.05
QUERY_MAX_LEN = 32
DOC_MAX_LEN = 128
SEED = 17


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def training_pairs(max_pairs: int, rng: np.random.Generator) -> pd.DataFrame:
    """Уникальные пары (текст запроса, объявление) без сессий валидации и обучения переранжирования."""
    queries = pd.read_parquet(PROCESSED / "queries.parquet", columns=["query_id", "source", "search_query"])
    pairs = pd.read_parquet(PROCESSED / "train_pairs.parquet")
    excluded = set()
    for name in EXCLUDE_SPLITS:
        excluded |= set(pd.read_parquet(VALIDATION / name / "val_queries.parquet").query_id)
    pairs = pairs[~pairs.query_id.isin(excluded)].merge(queries[["query_id", "search_query"]], on="query_id")
    # Один и тот же текст + объявление из разных сессий — одна пара; вес не нужен, частые пары и так частые.
    pairs = pairs.drop_duplicates(["search_query", "item_id"])
    items = pd.read_parquet(PROCESSED / "items.parquet", columns=["item_id", "item_microcat_id", "item_location_id"])
    pairs = pairs.merge(items, on="item_id")
    if len(pairs) > max_pairs:
        pairs = pairs.sample(max_pairs, random_state=int(rng.integers(1 << 31)))
    log(f"пар для обучения: {len(pairs)} (исключено сессий: {len(excluded)})")
    return pairs.reset_index(drop=True)


def mine_hard_negatives(pairs: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Жёсткий негатив для каждой пары: объявление того же города с места HARD_NEG_RANKS по исходному e5-small."""
    cache_path = Path(HARD_NEG_CACHE_TEMPLATE.format(n=len(pairs)))
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if len(cached) == len(pairs) and (cached.item_id.to_numpy() == pairs.item_id.to_numpy()).all():
            return cached.hard_negative
    from ..embed_corpus import ensure_embeddings, ensure_query_embeddings

    all_items = pd.read_parquet(PROCESSED / "items.parquet", columns=["item_id", "item_location_id"])
    item_emb = ensure_embeddings("e5-small", pd.Index(all_items.item_id)).astype(np.float32)
    q_emb = ensure_query_embeddings("e5-small", pairs.search_query.tolist()).astype(np.float32)
    queries = pd.read_parquet(PROCESSED / "queries.parquet", columns=["query_id", "search_query"])
    chosen = pd.read_parquet(PROCESSED / "train_pairs.parquet").merge(queries, on="query_id")
    chosen_by_text = chosen.groupby("search_query").item_id.agg(set).to_dict()

    item_ids = all_items.item_id.to_numpy()
    by_loc = all_items.groupby("item_location_id").indices
    lo, hi = HARD_NEG_RANKS
    result = np.empty(len(pairs), dtype=object)
    for loc, pair_idx in pairs.groupby("item_location_id").indices.items():
        cand = np.asarray(by_loc[loc])
        top_k = hi + 10  # запас на исключённые «выбранные по этому тексту» объявления
        if len(cand) <= top_k:
            continue  # в маленьком городе «трудных» соседей нет — пара останется без жёсткого негатива
        for start in range(0, len(pair_idx), 512):
            chunk = pair_idx[start : start + 512]
            sims = q_emb[chunk] @ item_emb[cand].T
            top = np.argpartition(-sims, top_k, axis=1)[:, :top_k]
            for row, pi in enumerate(chunk):
                order = top[row][np.argsort(-sims[row, top[row]])]
                banned = chosen_by_text.get(pairs.search_query.iat[pi], set())
                ranked = [item_ids[cand[j]] for j in order if item_ids[cand[j]] not in banned]
                pool = ranked[lo:hi]
                if pool:
                    result[pi] = pool[int(rng.integers(len(pool)))]
    out = pd.DataFrame({"item_id": pairs.item_id.to_numpy(), "hard_negative": result})
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_path, index=False)
    log(f"жёсткие негативы: найдены для {out.hard_negative.notna().mean():.3f} пар")
    return out.hard_negative


def make_batches(pairs: pd.DataFrame, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Половина батчей — из одной подкатегории (трудные негативы), половина — случайные."""
    idx = rng.permutation(len(pairs))
    half = len(idx) // 2
    topical, random_part = idx[:half], idx[half:]
    batches = []
    sub = pairs.iloc[topical].assign(_i=topical).sort_values("item_microcat_id", kind="stable")
    order = sub._i.to_numpy()
    for start in range(0, len(order), batch_size):
        batches.append(order[start : start + batch_size])
    for start in range(0, len(random_part), batch_size):
        batches.append(random_part[start : start + batch_size])
    rng.shuffle(batches)
    item_ids, texts = pairs.item_id.to_numpy(), pairs.search_query.to_numpy()
    clean = []
    for batch in batches:
        keep = (~pd.Series(item_ids[batch]).duplicated().to_numpy()) & (
            ~pd.Series(texts[batch]).duplicated().to_numpy()
        )
        batch = batch[keep]
        if len(batch) >= 8:  # в слишком маленьком батче почти нет негативов
            clean.append(batch)
    return clean


def embed(model, tokenizer, texts: list[str], max_len: int, device: str) -> torch.Tensor:
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
    hidden = model(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    return F.normalize((hidden * mask).sum(1) / mask.sum(1).clamp(min=1), dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pairs", type=int, default=150_000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-steps", type=int, default=0, help="ограничение шагов для замера скорости (0 — без)")
    parser.add_argument("--init-model", default="e5-small", help="с какой модели стартовать (второй круг: e5-small-ft)")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    spec = ENCODERS["e5-small"]  # префиксы и формат текстов одинаковы у исходной и дообученной модели
    init = ENCODERS[args.init_model].name
    out_dir = Path(args.out_dir)
    device = pick_device()
    log(f"старт с модели {init}, результат в {out_dir}")
    tokenizer = AutoTokenizer.from_pretrained(init)
    model = AutoModel.from_pretrained(init).to(device)
    model.train()

    pairs = training_pairs(args.max_pairs, rng)
    log("жёсткие негативы")
    pairs["hard_negative"] = mine_hard_negatives(pairs, rng).to_numpy()
    pairs = pairs.dropna(subset=["hard_negative"]).reset_index(drop=True)
    log(f"пар с жёстким негативом: {len(pairs)}")
    log("тексты документов")
    item_ids = pd.Index(pd.unique(pd.concat([pairs.item_id, pairs.hard_negative])))
    doc_text = pd.Series(doc_texts(item_ids), index=item_ids)
    q_texts = (spec.query_prefix + pairs.search_query).to_numpy()
    d_texts = (spec.doc_prefix + doc_text.reindex(pairs.item_id)).to_numpy()
    n_texts = (spec.doc_prefix + doc_text.reindex(pairs.hard_negative)).to_numpy()

    batches_per_epoch = len(make_batches(pairs, args.batch_size, np.random.default_rng(0)))
    total_steps = batches_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup = max(1, int(0.05 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, 0.5 * (1 + math.cos(math.pi * s / total_steps)))
    )

    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        batches = make_batches(pairs, args.batch_size, rng)
        running = 0.0
        for b in batches:
            q = embed(model, tokenizer, q_texts[b].tolist(), QUERY_MAX_LEN, device)
            d = embed(model, tokenizer, d_texts[b].tolist() + n_texts[b].tolist(), DOC_MAX_LEN, device)
            # Документы батча: B позитивов + B жёстких негативов; правильный для i-го запроса — i-й документ.
            logits = q @ d.T / TEMPERATURE
            labels = torch.arange(len(b), device=device)
            # Обратное направление (документ → запрос) только для позитивов: у негативов нет «своего» запроса.
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits[:, : len(b)].T, labels)) / 2
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            running += loss.item()
            if step % 100 == 0:
                speed = step / (time.time() - t0)
                log(
                    f"эпоха {epoch + 1}, шаг {step}/{total_steps}, loss {running / 100:.4f}, "
                    f"{speed:.2f} шаг/с, осталось ~{(total_steps - step) / speed / 60:.0f} мин"
                )
                running = 0.0
            if args.max_steps and step >= args.max_steps:
                log(f"стоп по --max-steps: {step / (time.time() - t0):.2f} шаг/с")
                return
            if step % 1000 == 0:
                out_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(out_dir)
                tokenizer.save_pretrained(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    log(f"модель сохранена: {out_dir}")


if __name__ == "__main__":
    main()
