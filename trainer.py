#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import random
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import (
    AddedToken,
    AutoModelForMaskedLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class Cfg:
    input_format: str = "json_tweets"
    input_json: Optional[str] = None
    txt_dir: Optional[str] = None
    annotations_csv: Optional[str] = None
    out_dir: str = "outputs/run"
    base_model: str = "bert-base-uncased"
    usr_prefix: str = "usr"
    max_len: int = 128
    epochs: int = 4
    batch_size: int = 128
    users_per_batch: int = 128
    lr: float = 5e-5
    mlm_prob: float = 0.15
    p_user_mask: float = 0.30
    seed: int = 42
    num_workers: int = max(2, (os.cpu_count() or 8) // 2)
    tokenize_chunk: int = 4096
    min_posts_per_user: int = 2
    export_kv: bool = False
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    warmup_ratio: float = 0.03
    per_user_cap: int = 200
    freeze_epochs: int = 1
    max_tweets_per_user: Optional[int] = None
    align_use_hidden: bool = False
    align_lambda: float = 0.2
    con_weight: float = 0.0
    con_temperature: float = 0.07
    soft_prompt_len: int = 0

    @classmethod
    def from_json(cls, path: str) -> "Cfg":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


def set_seeds(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def l2n_rows(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[(n == 0) | (~np.isfinite(n))] = 1.0
    return M / n


def uid_to_token(uid: str, tok, prefix: str = "usr") -> str:
    h = hashlib.sha1(str(uid).encode("utf-8")).hexdigest()[:10]
    token = f"{prefix}{h}"
    return token.lower() if getattr(tok, "do_lower_case", False) else token


def _file_iter(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        head = ""
        while True:
            ch = f.read(1)
            if not ch:
                break
            if ch.isspace():
                continue
            head = ch
            break
        f.seek(0)

        if head == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise RuntimeError("Top-level JSON must be a list.")
            for obj in data:
                if isinstance(obj, dict):
                    yield obj
        else:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _tweet_text(tweet: dict) -> Optional[str]:
    if not isinstance(tweet, dict):
        return None

    text = tweet.get("full_text")
    if isinstance(text, str) and text.strip():
        return " ".join(text.split()).strip()

    ext = tweet.get("extended_tweet")
    if isinstance(ext, dict):
        ext_text = ext.get("full_text")
        if isinstance(ext_text, str) and ext_text.strip():
            return " ".join(ext_text.split()).strip()

    text = tweet.get("text")
    if isinstance(text, str) and text.strip():
        return " ".join(text.split()).strip()
    return None


def _derive_uid_from_tweets(tweets: List[dict]) -> Optional[str]:
    for tweet in tweets or []:
        user = tweet.get("user") or {}
        cand = user.get("id_str") or user.get("id") or user.get("screen_name")
        if cand is not None:
            return str(cand)
    return None


def _source_to_label(
    source: Optional[str],
    account_type: Optional[str],
    label_field: Optional[str],
) -> Optional[str]:
    if isinstance(label_field, str):
        label = label_field.strip().lower()
        if label in {"bot", "human"}:
            return label

    if isinstance(account_type, str):
        account = account_type.strip().lower()
        if account in {"bot", "human"}:
            return account

    source = (source or "").strip().lower()
    if not source:
        return None
    if "fox8" in source:
        return "bot"
    for key in ("botometer-feedback", "gilani-17", "midterm-2018", "varol-icwsm", "human"):
        if key in source:
            return "human"
    return None


def load_user_json(
    json_path: Path,
    min_posts_per_user: int,
    max_tweets_per_user: Optional[int],
) -> Tuple[List[Tuple[str, str]], Dict[str, Dict[str, Any]]]:
    if not json_path.is_file():
        raise SystemExit(f"Input JSON not found: {json_path}")

    samples: List[Tuple[str, str]] = []
    users_meta: Dict[str, Dict[str, Any]] = {}

    for row in _file_iter(json_path):
        if not isinstance(row, dict):
            continue

        tweets = row.get("tweets") or row.get("user_tweets") or []
        if not isinstance(tweets, list) or not tweets:
            continue

        uid = row.get("user_id") or row.get("id_str") or row.get("id") or row.get("screen_name")
        if uid is None:
            uid = _derive_uid_from_tweets(tweets)
        if uid is None:
            continue
        uid = str(uid)

        label = _source_to_label(row.get("source"), row.get("account_type"), row.get("label"))
        texts: List[str] = []
        for tweet in tweets:
            text = _tweet_text(tweet)
            if text:
                texts.append(text)
            if max_tweets_per_user is not None and len(texts) >= int(max_tweets_per_user):
                break

        if len(texts) < max(1, int(min_posts_per_user)):
            continue

        users_meta.setdefault(uid, {"texts": [], "label": label})
        users_meta[uid]["texts"].extend(texts)

    if not users_meta:
        raise SystemExit("No users parsed from JSON input.")

    users: Dict[str, Dict[str, Any]] = {}
    for uid, meta in users_meta.items():
        texts = meta["texts"]
        for text in texts:
            samples.append((uid, text))
        users[uid] = {
            "n_posts": len(texts),
            "label_majority": meta.get("label"),
            "targets": [],
        }

    if not samples:
        raise SystemExit("No posts after filtering.")
    return samples, users


def load_txt_csv(
    txt_dir: Path,
    csv_path: Path,
    min_posts_per_user: int,
) -> Tuple[List[Tuple[str, str]], Dict[str, Dict[str, Any]]]:
    if not txt_dir.is_dir():
        raise SystemExit(f"TXT dir not found: {txt_dir}")
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not {"file_id", "user_id"}.issubset(reader.fieldnames or []):
            raise SystemExit("CSV must contain file_id and user_id columns.")
        for row in reader:
            rows.append({k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()})

    existing = {p.stem: p for p in txt_dir.glob("*.txt")}
    post_counts = Counter()
    label_counts = defaultdict(Counter)
    samples: List[Tuple[str, str]] = []

    for row in rows:
        file_id = row.get("file_id", "")
        uid = str(row.get("user_id", "")).strip()
        file_path = existing.get(file_id)
        if not file_id or not uid or not file_path:
            continue

        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            with file_path.open("rb") as fh:
                text = fh.read().decode("utf-8", "ignore")

        text = " ".join(text.split()).strip()
        if not text:
            continue

        samples.append((uid, text))
        post_counts[uid] += 1

        label = (row.get("label") or "").strip()
        if label:
            label_counts[uid][label] += 1

    allowed = {uid for uid, count in post_counts.items() if count >= max(1, int(min_posts_per_user))}
    samples = [(uid, text) for uid, text in samples if uid in allowed]
    if not samples:
        raise SystemExit("No posts after filtering.")

    users = {
        uid: {
            "n_posts": post_counts[uid],
            "label_majority": (label_counts[uid].most_common(1)[0][0] if label_counts[uid] else None),
            "targets": [],
        }
        for uid in allowed
    }
    return samples, users


class EncodedDataset(Dataset):
    def __init__(self, enc: Dict[str, List[List[int]]]):
        self.enc = enc

    def __len__(self) -> int:
        return len(self.enc["input_ids"])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            key: torch.tensor(self.enc[key][idx], dtype=torch.long)
            for key in ("input_ids", "attention_mask", "special_tokens_mask")
        }


def batch_encode(tok, texts: List[str], max_len: int, chunk: int) -> Dict[str, List[List[int]]]:
    ids, attn, special = [], [], []
    for i in range(0, len(texts), chunk):
        enc = tok(
            texts[i : i + chunk],
            truncation=True,
            max_length=max_len,
            padding=False,
            return_special_tokens_mask=True,
        )
        ids.extend(enc["input_ids"])
        attn.extend(enc["attention_mask"])
        special.extend(enc["special_tokens_mask"])
    return {"input_ids": ids, "attention_mask": attn, "special_tokens_mask": special}


class BalancedUserBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        sample_uids: List[str],
        batch_size: int,
        users_per_batch: int,
        per_user_cap: int = 0,
        drop_last: bool = False,
    ):
        self.bs = int(batch_size)
        self.upb = int(max(1, users_per_batch))
        self.drop_last = drop_last
        self.cap = int(per_user_cap)
        self.s_per_user = max(1, self.bs // self.upb)

        bins = defaultdict(list)
        for idx, uid in enumerate(map(str, sample_uids)):
            bins[uid].append(idx)
        self.bins = {uid: np.array(indexes, dtype=int) for uid, indexes in bins.items()}
        self.n_samples = len(sample_uids)

    def __iter__(self):
        rng = np.random.default_rng()
        local = {
            uid: deque(
                rng.permutation(indexes)[: self.cap].tolist()
                if self.cap > 0 and len(indexes) > self.cap
                else rng.permutation(indexes).tolist()
            )
            for uid, indexes in self.bins.items()
        }
        pool = [uid for uid, items in local.items() if items]

        while pool:
            batch, emptied = [], []
            take = min(self.upb, len(pool))
            for uid in rng.choice(pool, size=take, replace=False):
                for _ in range(self.s_per_user):
                    if local[uid]:
                        batch.append(local[uid].popleft())
                    else:
                        break
                if not local[uid]:
                    emptied.append(uid)
                if len(batch) >= self.bs:
                    break

            pool = [uid for uid in pool if uid not in emptied]
            if not batch:
                break
            if len(batch) < self.bs and self.drop_last:
                break
            yield batch[: self.bs]

    def __len__(self) -> int:
        return max(1, math.ceil(self.n_samples / max(1, self.bs)))


class UserAwareMLMCollator:
    def __init__(
        self,
        tok,
        mlm_prob: float = 0.15,
        p_user_mask: float = 0.30,
        user_token_ids: Optional[torch.Tensor] = None,
    ):
        self.tok = tok
        self.mlm = float(mlm_prob)
        self.pusr = float(p_user_mask)
        self.user_ids = (
            user_token_ids if isinstance(user_token_ids, torch.Tensor) else torch.tensor([], dtype=torch.long)
        )
        self.never_ids = [
            tok.cls_token_id,
            tok.sep_token_id,
            tok.pad_token_id,
            tok.unk_token_id,
            tok.mask_token_id,
        ]
        self.never_ids = [idx for idx in self.never_ids if idx is not None]

    def _mask(self, ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        device = ids.device
        never = (
            torch.isin(ids, torch.tensor(self.never_ids, device=device))
            if self.never_ids
            else torch.zeros_like(ids, dtype=torch.bool)
        )
        is_user = (
            torch.isin(ids, self.user_ids.to(device))
            if len(self.user_ids) > 0
            else torch.zeros_like(ids, dtype=torch.bool)
        )

        allowed_nonuser = attn & (~never) & (~is_user)
        allowed_user = attn & (~never) & is_user
        mask = (torch.rand_like(ids, dtype=torch.float) < self.mlm) & allowed_nonuser
        if len(self.user_ids) > 0:
            mask |= (torch.rand_like(ids, dtype=torch.float) < self.pusr) & allowed_user

        for b in range(ids.size(0)):
            if not mask[b].any():
                allowed = (allowed_nonuser[b] | allowed_user[b]).nonzero(as_tuple=False).flatten()
                if allowed.numel() > 0:
                    j = torch.randint(0, allowed.numel(), (1,), device=device)
                    mask[b, allowed[j]] = True
        return mask

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = self.tok.pad(features, padding=True, return_tensors="pt")
        ids = batch["input_ids"]
        attn = batch["attention_mask"].bool()
        user_first_ids = ids[:, 0].clone()

        labels = ids.clone()
        mask = self._mask(ids, attn)
        labels[~mask] = -100

        rand = torch.rand_like(ids, dtype=torch.float)
        mask80 = mask & (rand < 0.8)
        mask10 = mask & (rand >= 0.8) & (rand < 0.9)
        ids[mask80] = self.tok.mask_token_id

        user_set = set(self.user_ids.tolist())
        never_set = set(self.never_ids)
        safe = torch.tensor(
            [idx for idx in range(len(self.tok)) if idx not in user_set and idx not in never_set],
            device=ids.device,
            dtype=torch.long,
        )
        if mask10.any():
            rand_ids = safe[torch.randint(0, safe.numel(), ids.shape, device=ids.device)]
            ids[mask10] = rand_ids[mask10]

        return {
            "input_ids": ids,
            "attention_mask": attn.long(),
            "labels": labels,
            "user_first_ids": user_first_ids,
        }


class SoftPromptTable(nn.Module):
    def __init__(self, num_users: int, dim: int, prompt_len: int):
        super().__init__()
        self.emb = nn.Embedding(num_users, prompt_len * dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        self.dim = dim
        self.prompt_len = prompt_len

    def forward(self, row_idx: torch.Tensor) -> torch.Tensor:
        x = self.emb(row_idx)
        return x.view(x.size(0), self.prompt_len, self.dim)


def export_user_kv(model, tokenizer, usr_tokens_map: Dict[str, str], out_path: Path) -> None:
    try:
        from gensim.models import KeyedVectors
    except Exception:
        print("gensim not found; skipping KV export")
        return

    W = model.get_input_embeddings().weight.detach().cpu().numpy().astype("float32")
    W = l2n_rows(W)
    vocab = tokenizer.get_vocab()
    rows = []
    for uid, tok_str in usr_tokens_map.items():
        idx = vocab.get(tok_str)
        if idx is not None:
            rows.append((f"USR:{uid}", W[idx]))

    if not rows:
        print("No user tokens available for KV export.")
        return

    keys, vecs = zip(*rows)
    kv = KeyedVectors(vector_size=W.shape[1])
    kv.add_vectors(list(keys), np.vstack(vecs))
    kv.fill_norms()
    kv.save(str(out_path))


def load_samples(cfg: Cfg) -> Tuple[List[Tuple[str, str]], Dict[str, Dict[str, Any]]]:
    if cfg.input_format == "json_tweets":
        if not cfg.input_json:
            raise SystemExit("input_json is required when input_format=json_tweets")
        return load_user_json(Path(cfg.input_json), cfg.min_posts_per_user, cfg.max_tweets_per_user)

    if cfg.input_format == "txt_csv":
        if not cfg.txt_dir or not cfg.annotations_csv:
            raise SystemExit("txt_dir and annotations_csv are required when input_format=txt_csv")
        return load_txt_csv(Path(cfg.txt_dir), Path(cfg.annotations_csv), cfg.min_posts_per_user)

    raise SystemExit(f"Unsupported input_format: {cfg.input_format}")


def train(cfg: Cfg) -> None:
    set_seeds(cfg.seed)
    out_dir = Path(cfg.out_dir)
    (out_dir / "model").mkdir(parents=True, exist_ok=True)

    samples, users = load_samples(cfg)
    print(f"samples={len(samples)} users={len(users)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(cfg.base_model)

    user_tokens_map = {uid: uid_to_token(uid, tok, prefix=cfg.usr_prefix) for uid in users}
    tok.add_tokens([AddedToken(t, single_word=True, normalized=True) for t in user_tokens_map.values()])

    texts = [f"{user_tokens_map[uid]} {text}" for uid, text in samples]
    sample_uids = [uid for uid, _ in samples]

    vocab = tok.get_vocab()
    user_ids = {t: vocab[t] for t in user_tokens_map.values() if t in vocab}
    assert len(user_ids) == len(user_tokens_map), "Some user tokens were not added to the vocabulary."

    def _first_is_user_token(text: str) -> bool:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        return len(ids) > 0 and ids[0] in user_ids.values()

    check_n = min(1000, len(texts))
    ok = sum(_first_is_user_token(texts[i]) for i in range(check_n))
    assert ok == check_n, "User token is not the first token."

    enc = batch_encode(tok, texts, cfg.max_len, cfg.tokenize_chunk)
    ds = EncodedDataset(enc)
    user_tensor = torch.tensor([vocab[t] for t in user_tokens_map.values()], dtype=torch.long)
    collator = UserAwareMLMCollator(tok, cfg.mlm_prob, cfg.p_user_mask, user_tensor)

    users_per_batch = min(cfg.users_per_batch, max(1, len(users)))
    sampler = BalancedUserBatchSampler(
        sample_uids,
        cfg.batch_size,
        users_per_batch,
        per_user_cap=cfg.per_user_cap,
        drop_last=False,
    )
    dl = DataLoader(
        ds,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )

    model = AutoModelForMaskedLM.from_pretrained(cfg.base_model)
    model.resize_token_embeddings(len(tok))
    model.to(device)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    emb_layer = model.get_input_embeddings()

    token_id_list = [vocab[user_tokens_map[uid]] for uid in users]
    tokid2row = {tid: idx for idx, tid in enumerate(token_id_list)}
    soft_prompt = None
    if cfg.soft_prompt_len > 0:
        soft_prompt = SoftPromptTable(len(token_id_list), emb_layer.embedding_dim, cfg.soft_prompt_len).to(device)

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if any(key in name for key in ["bias", "LayerNorm.weight"]) else decay).append(param)
    if soft_prompt is not None:
        decay.append(soft_prompt.emb.weight)

    fused_ok = (device.type == "cuda") and ("fused" in inspect.signature(AdamW).parameters)
    opt = AdamW(
        [
            {"params": decay, "weight_decay": 0.01},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        eps=1e-6,
        betas=(0.9, 0.98),
        **({"fused": True} if fused_ok else {}),
    )

    use_bf16 = (device.type == "cuda") and torch.cuda.is_bf16_supported()
    use_fp16 = (device.type == "cuda") and not use_bf16
    scaler = GradScaler(enabled=use_fp16)

    total_steps = cfg.epochs * max(1, len(dl)) // max(1, cfg.grad_accum_steps)
    warmup = max(100, int(cfg.warmup_ratio * total_steps))
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=total_steps)

    embed_param_ids = {id(p) for p in emb_layer.parameters()}

    def set_backbone_trainable(trainable: bool) -> None:
        for param in model.parameters():
            param.requires_grad = id(param) in embed_param_ids or trainable

    set_backbone_trainable(False)
    model.train()

    user_token_tensor = torch.tensor([vocab[t] for t in user_tokens_map.values()], device=device, dtype=torch.long)
    gstep, running, t0 = 0, 0.0, time.time()

    for epoch in range(cfg.epochs):
        if epoch >= cfg.freeze_epochs:
            set_backbone_trainable(True)

        for batch in dl:
            user_first_ids = batch.pop("user_first_ids").to(device)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            need_hid = bool(cfg.align_use_hidden or cfg.con_weight > 0.0)

            inputs = {"attention_mask": batch["attention_mask"], "labels": batch["labels"]}
            if soft_prompt is not None and cfg.soft_prompt_len > 0:
                row_idx = torch.tensor(
                    [tokid2row[int(x)] for x in user_first_ids.tolist()],
                    device=device,
                    dtype=torch.long,
                )
                prefix = soft_prompt(row_idx)
                word_emb = emb_layer(batch["input_ids"])
                inputs_embeds = torch.cat([prefix, word_emb], dim=1)
                attn_prefix = torch.ones(
                    (inputs_embeds.size(0), cfg.soft_prompt_len),
                    dtype=batch["attention_mask"].dtype,
                    device=device,
                )
                labels_prefix = torch.full(
                    (inputs_embeds.size(0), cfg.soft_prompt_len),
                    -100,
                    dtype=batch["labels"].dtype,
                    device=device,
                )
                inputs["inputs_embeds"] = inputs_embeds
                inputs["attention_mask"] = torch.cat([attn_prefix, batch["attention_mask"]], dim=1)
                inputs["labels"] = torch.cat([labels_prefix, batch["labels"]], dim=1)
            else:
                inputs = {**batch}

            if use_bf16:
                with autocast("cuda", dtype=torch.bfloat16):
                    out = model(**inputs, output_hidden_states=need_hid)
                    loss = out.loss
            elif use_fp16:
                with autocast("cuda", dtype=torch.float16):
                    out = model(**inputs, output_hidden_states=need_hid)
                    loss = out.loss
            else:
                out = model(**inputs, output_hidden_states=need_hid)
                loss = out.loss

            ids = batch["input_ids"]
            attn = batch["attention_mask"].bool()
            never_ids = torch.tensor(
                [tok.cls_token_id, tok.sep_token_id, tok.pad_token_id, tok.unk_token_id, tok.mask_token_id],
                device=ids.device,
            )
            never_ids = never_ids[never_ids >= 0]
            is_never = torch.isin(ids, never_ids)
            is_user = torch.isin(ids, user_token_tensor)
            ctx_mask = attn & (~is_never) & (~is_user)

            e_usr = emb_layer(user_first_ids)
            if cfg.align_use_hidden and need_hid and out.hidden_states is not None:
                last_hid = out.hidden_states[-1]
                if soft_prompt is not None and cfg.soft_prompt_len > 0:
                    pad = torch.zeros((ids.size(0), cfg.soft_prompt_len), dtype=torch.bool, device=ids.device)
                    ctx_mask_ext = torch.cat([pad, ctx_mask], dim=1)
                    ctx = torch.where(ctx_mask_ext.unsqueeze(-1), last_hid, torch.zeros_like(last_hid))
                else:
                    ctx = torch.where(ctx_mask.unsqueeze(-1), last_hid, torch.zeros_like(last_hid))
            else:
                input_emb = emb_layer(ids)
                ctx = torch.where(ctx_mask.unsqueeze(-1), input_emb, torch.zeros_like(input_emb))

            ctx_sum = ctx.sum(dim=1)
            ctx_cnt = (ctx_mask.sum(dim=1).clamp_min(1)).unsqueeze(-1).to(ctx_sum.dtype)
            e_ctx_mean = ctx_sum / ctx_cnt
            cos = nn.functional.cosine_similarity(e_usr, e_ctx_mean.detach(), dim=-1, eps=1e-8)
            L_align = (1.0 - cos).mean()
            loss_total = loss + cfg.align_lambda * L_align

            if cfg.con_weight > 0.0:
                u = nn.functional.normalize(e_usr, dim=-1)
                c = nn.functional.normalize(e_ctx_mean, dim=-1)
                logits_uc = (u @ c.t()) / cfg.con_temperature
                logits_cu = (c @ u.t()) / cfg.con_temperature
                target = torch.arange(u.shape[0], device=ids.device)
                ce = nn.CrossEntropyLoss()
                L_con = 0.5 * (ce(logits_uc, target) + ce(logits_cu, target))
                loss_total = loss_total + cfg.con_weight * L_con
            else:
                L_con = torch.tensor(0.0, device=ids.device)

            if not torch.isfinite(loss_total):
                print("[warn] non-finite loss; skipping step")
                opt.zero_grad(set_to_none=True)
                continue

            if use_fp16:
                scaler.scale(loss_total / cfg.grad_accum_steps).backward()
                if (gstep + 1) % cfg.grad_accum_steps == 0:
                    if cfg.grad_clip > 0:
                        scaler.unscale_(opt)
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
            else:
                (loss_total / cfg.grad_accum_steps).backward()
                if (gstep + 1) % cfg.grad_accum_steps == 0:
                    if cfg.grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    sched.step()

            gstep += 1
            running += float(loss.item())
            if gstep % 100 == 0:
                sps = gstep / max(1e-9, time.time() - t0)
                print(
                    f"epoch={epoch + 1} step={gstep} mlm={running / 100:.4f} "
                    f"align={float(L_align):.4f} con={float(L_con):.4f} {sps:.2f}sps"
                )
                running = 0.0

    model.save_pretrained(str(out_dir / "model"))
    tok.save_pretrained(str(out_dir / "model"))

    with (out_dir / "users.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["user_id", "token", "n_posts", "label_majority", "targets"],
        )
        writer.writeheader()
        for uid, meta in users.items():
            writer.writerow(
                {
                    "user_id": uid,
                    "token": user_tokens_map[uid],
                    "n_posts": meta.get("n_posts", 1),
                    "label_majority": meta.get("label_majority"),
                    "targets": ",".join(meta.get("targets") or []),
                }
            )

    meta = {
        **asdict(cfg),
        "n_users": len(users),
        "n_texts": len(texts),
        "inputs": {
            "input_json": cfg.input_json,
            "txt_dir": cfg.txt_dir,
            "annotations_csv": cfg.annotations_csv,
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if cfg.export_kv:
        export_user_kv(model, tok, user_tokens_map, out_dir / "user_embeddings.kv")
    print(f"Saved to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a POLAR embedding space model.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    args = parser.parse_args()
    train(Cfg.from_json(args.config))


if __name__ == "__main__":
    main()
