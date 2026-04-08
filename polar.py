#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class Cfg:
    out_dir: str = "outputs/run"
    model_dir: Optional[str] = None
    users_csv: Optional[str] = None
    meta_json: Optional[str] = None
    output_csv: Optional[str] = None
    attributes_path: str = "configs/attributes.example.json"
    usr_prefix_fallback: str = "usr"
    min_posts_default: int = 1
    alpha_bh: float = 0.05
    mc_samples: int = 2000
    seed: int = 123
    print_every: int = 500

    @classmethod
    def from_json(cls, path: str) -> "Cfg":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


def _l2n_rows(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[(n == 0) | (~np.isfinite(n))] = 1.0
    return M / n


def _l2n(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 or not np.isfinite(n) else v / n


def _word_vec(
    word: str,
    tok,
    W: np.ndarray,
    cache: Dict[str, Optional[np.ndarray]],
) -> Optional[np.ndarray]:
    if word in cache:
        return cache[word]
    w = word.lower() if getattr(tok, "do_lower_case", False) else word
    ids = tok.encode(w, add_special_tokens=False)
    cache[word] = None if not ids else _l2n(W[ids].mean(axis=0).astype("float32"))
    return cache[word]


def _bh_fdr(p: np.ndarray, alpha: float) -> np.ndarray:
    p = np.asarray(p, float)
    p[np.isnan(p)] = np.inf
    n = len(p)
    if n == 0:
        return np.zeros(0, bool)
    order = np.argsort(p)
    ranks = np.empty(n, int)
    ranks[order] = np.arange(1, n + 1)
    thresh = (ranks / n) * alpha
    passed = p <= thresh
    mx = (ranks * passed).max() if passed.any() else 0
    return (p <= (mx / n * alpha)) if mx > 0 else np.zeros(n, bool)


def _perm_p(
    d_all: np.ndarray,
    m: int,
    rng: np.random.RandomState,
    s_obs: float,
    sd_all: float,
    mc: int,
) -> float:
    N = d_all.shape[0]
    n = N - m
    if m <= 0 or n <= 0 or not np.isfinite(s_obs) or sd_all == 0 or not np.isfinite(sd_all):
        return np.nan
    inv_m, inv_n, inv_sd = 1.0 / m, 1.0 / n, 1.0 / sd_all
    total = d_all.sum()
    extreme = denom = 0
    for _ in range(mc):
        idx = rng.choice(N, size=m, replace=False)
        s = (((d_all[idx].sum() * inv_m) - ((total - d_all[idx].sum()) * inv_n)) * inv_sd)
        if np.isfinite(s):
            denom += 1
            if abs(s) >= abs(s_obs):
                extreme += 1
    return (extreme + 1) / (denom + 1) if denom > 0 else np.nan


def _load_users_and_tokens(
    users_csv: Path,
    min_posts: int,
    meta_json: Optional[Path],
    fallback_prefix: str,
) -> Dict[str, Dict[str, Any]]:
    prefix = fallback_prefix
    min_posts = int(min_posts)
    try:
        if meta_json and meta_json.exists():
            meta = json.loads(meta_json.read_text(encoding="utf-8"))
            if isinstance(meta.get("usr_prefix"), str) and meta["usr_prefix"].strip():
                prefix = meta["usr_prefix"].strip()
            if "min_posts_per_user" in meta:
                min_posts = max(int(meta["min_posts_per_user"]), 1)
    except Exception:
        pass

    out: Dict[str, Dict[str, Any]] = {}
    with users_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_token_col = "token" in (reader.fieldnames or [])
        for row in reader:
            uid = (row.get("user_id") or "").strip()
            if not uid:
                continue
            try:
                n_posts = int(row.get("n_posts", "0"))
            except Exception:
                n_posts = 0
            if n_posts < min_posts:
                continue

            token = (row.get("token") or "").strip() if has_token_col else ""
            if not token:
                token = f"{prefix}{uid}"
            out[uid] = {
                "n_posts": n_posts,
                "token": token,
                "label_majority": (row.get("label_majority") or "").strip() or None,
                "targets": (row.get("targets") or "").strip() or None,
            }
    return out


def _resolve_paths(cfg: Cfg) -> Tuple[Path, Path, Optional[Path], Path]:
    out_dir = Path(cfg.out_dir)
    model_dir = Path(cfg.model_dir) if cfg.model_dir else out_dir / "model"
    users_csv = Path(cfg.users_csv) if cfg.users_csv else out_dir / "users.csv"
    meta_json = Path(cfg.meta_json) if cfg.meta_json else out_dir / "meta.json"
    output_csv = Path(cfg.output_csv) if cfg.output_csv else out_dir / "per_user_scores.csv"
    return model_dir, users_csv, meta_json, output_csv


def run(cfg: Cfg) -> None:
    np.random.seed(cfg.seed)
    rng = np.random.RandomState(cfg.seed)

    model_dir, users_csv, meta_json, output_csv = _resolve_paths(cfg)
    users = _load_users_and_tokens(users_csv, cfg.min_posts_default, meta_json, cfg.usr_prefix_fallback)
    if not users:
        raise SystemExit("No users after filtering.")

    attrs = json.loads(Path(cfg.attributes_path).read_text(encoding="utf-8"))
    print(f"users={len(users)} (from {users_csv.name})")

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForMaskedLM.from_pretrained(str(model_dir))
    with torch.no_grad():
        W = model.get_input_embeddings().weight.detach().cpu().numpy().astype("float32")
    W = _l2n_rows(W)
    vocab = tok.get_vocab()

    cache: Dict[str, Optional[np.ndarray]] = {}
    pair_mats: Dict[str, Tuple[np.ndarray, np.ndarray, int, int]] = {}
    for name, pair in attrs.items():
        pos = [w for w in (pair.get("pos", []) or []) if isinstance(w, str) and w.strip()]
        neg = [w for w in (pair.get("neg", []) or []) if isinstance(w, str) and w.strip()]
        A = [v for w in pos if (v := _word_vec(w, tok, W, cache)) is not None]
        B = [v for w in neg if (v := _word_vec(w, tok, W, cache)) is not None]
        A = np.vstack(A) if A else np.empty((0, W.shape[1]), "float32")
        B = np.vstack(B) if B else np.empty((0, W.shape[1]), "float32")
        pair_mats[name] = (A, B, len(A), len(B))
        if len(A) > 0 and len(B) > 0:
            sep = float(np.dot(_l2n(A.mean(0)), _l2n(B.mean(0))))
            print(f"[{name}] pos={len(A)} neg={len(B)} cos(centroids)={sep:+.3f}")
        else:
            print(f"[{name}] pos={len(A)} neg={len(B)}")

    user_vecs: Dict[str, np.ndarray] = {}
    missing = []
    for uid, meta in users.items():
        tok_id = vocab.get(meta["token"])
        if tok_id is None:
            missing.append((uid, meta["token"]))
            continue
        user_vecs[uid] = _l2n(W[tok_id])

    if not user_vecs:
        raise SystemExit("No user tokens found in the saved model vocabulary.")
    if missing:
        print(f"Missing {len(missing)} user tokens (showing up to 5): {missing[:5]}")

    rows: List[Dict[str, Any]] = []
    uids = list(user_vecs.keys())
    t0 = time.time()
    for pidx, (pair_name, (A, B, m, n)) in enumerate(pair_mats.items(), 1):
        if m == 0 or n == 0:
            for uid in uids:
                rows.append(
                    {
                        "user_id": uid,
                        "pair": pair_name,
                        "s": np.nan,
                        "p_perm": np.nan,
                        "n_posts": users[uid]["n_posts"],
                        "n_pos_attr": m,
                        "n_neg_attr": n,
                        "label_majority": users[uid].get("label_majority"),
                        "targets": users[uid].get("targets"),
                    }
                )
            continue

        AB = np.vstack([A, B])
        for k, uid in enumerate(uids, 1):
            u = user_vecs[uid]
            d_all = AB @ u
            sd = float(np.std(d_all))
            s_obs = (d_all[:m].mean() - d_all[m:].mean()) / sd if sd > 0 and np.isfinite(sd) else np.nan
            p_val = _perm_p(d_all, m, rng, s_obs, sd, cfg.mc_samples)
            rows.append(
                {
                    "user_id": uid,
                    "pair": pair_name,
                    "s": float(s_obs),
                    "p_perm": float(p_val),
                    "n_posts": users[uid]["n_posts"],
                    "n_pos_attr": m,
                    "n_neg_attr": n,
                    "label_majority": users[uid].get("label_majority"),
                    "targets": users[uid].get("targets"),
                }
            )
            if k % cfg.print_every == 0:
                print(f"pair {pidx}/{len(pair_mats)} {k}/{len(uids)} users elapsed={time.time() - t0:.1f}s")

    df = pd.DataFrame(rows)
    df["signif_bh_fdr_0.05"] = False
    for pair_name, sub in df.groupby("pair", sort=False):
        df.loc[sub.index, "signif_bh_fdr_0.05"] = _bh_fdr(sub["p_perm"].to_numpy(), cfg.alpha_bh)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"saved -> {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run POLAR per-user association scoring.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    args = parser.parse_args()
    run(Cfg.from_json(args.config))


if __name__ == "__main__":
    main()
