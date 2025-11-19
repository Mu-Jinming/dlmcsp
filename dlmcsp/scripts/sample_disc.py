#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, os
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dlmcsp.models.llada import LLaDA
from dlmcsp.tokenization.tokenizer import MaterialTokenizer

# -----------------------
# 与 train 中一致的类型划分
# -----------------------

TOKEN_TYPE_P = {
    "LATT_BIN":   0.60,
    "LATT_DR":    0.00,
    "PARAM_BASE": 0.70,
    "PARAM_FINE": 0.50,
    "PARAM_DR":   0.00,
    "SG":       0.00,
    "WY":       0.00,
    "EL":       0.00,
    "FORMULA":  0.00,
    "PUNCT":    0.00,
    "OTHER":    0.00,
}

PUNCTS = {
    "FORMULA=", "NATOMS=", "SG=", "LATT=", "SITES=",
    "EL:", "WY:", "PARAM:", "u:", "v:", "w:",
    "(", ")", "->", ";", ","
}

PREDICT_TYPES = {"LATT_BIN", "PARAM_BASE", "PARAM_FINE"}  # 我们要生成的 token 类型


def classify_token(tok_str: str) -> str:
    s = tok_str
    up = s.upper()

    if "_BIN_" in up:
        if any(x in up for x in ("A_BIN_", "B_BIN_", "C_BIN_", "ALPHA_BIN_", "BETA_BIN_", "GAMMA_BIN_")):
            return "LATT_BIN"
    if "_DR_" in up:
        if any(x in up for x in ("A_DR_", "B_DR_", "C_DR_", "ALPHA_DR_", "BETA_DR_", "GAMMA_DR_")):
            return "LATT_DR"
        return "PARAM_DR"

    if up.startswith("BASE_"):
        return "PARAM_BASE"
    if up.startswith("FINE_"):
        return "PARAM_FINE"

    if up.startswith("SG_") or s == "SG=":
        return "SG"
    if s.startswith("WY:") or s == "WY:":
        return "WY"
    if s in PUNCTS:
        return "PUNCT"

    return "OTHER"


# -----------------------
# Dataset & collate
# -----------------------

class MsJsonlDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.recs: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                obj = json.loads(ln)
                assert "tokens" in obj
                self.recs.append(obj)
        if not self.recs:
            raise ValueError(f"empty dataset: {path}")

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        # 返回 index 方便回写
        return i, self.recs[i]


def collate(batch, tok: MaterialTokenizer):
    PAD_id = tok.vocab.token_id("<PAD>")
    B = len(batch)
    maxL = max(len(rec["tokens"]) for _, rec in batch)
    ids = torch.full((B, maxL), PAD_id, dtype=torch.long)
    idxs: List[int] = []

    for b, (idx, rec) in enumerate(batch):
        idxs.append(idx)
        seq = rec["tokens"]
        ids[b, :len(seq)] = torch.tensor(seq, dtype=torch.long)

    return torch.tensor(idxs, dtype=torch.long), ids


# -----------------------
# 构建类型表，方便快速判断哪些 token 需要预测
# -----------------------

def build_token_types(vocab) -> List[str]:
    id2tok = vocab._id2tok
    types = []
    type_count = {}
    for s in id2tok:
        t = classify_token(s)
        types.append(t)
        type_count[t] = type_count.get(t, 0) + 1
    print("[INFO] token type counts:", type_count)
    return types


# -----------------------
# 主逻辑：单步 infilling
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="输入 jsonl (ms 数据)")
    ap.add_argument("--vocab", required=True, help="vocab.yaml")
    ap.add_argument("--ckpt", required=True, help="训练好的离散 LLaDA ckpt")
    ap.add_argument("--out", required=True, help="输出 jsonl，带 tokens_pred")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--t_fixed", type=float, default=-1.0,
                    help="若 >=0 则使用固定 t；否则每个 batch 随机采样 t~U(0,1)")
    args = ap.parse_args()

    device = torch.device(args.device)

    # 数据 & tokenizer
    ds = MsJsonlDataset(args.data)
    tok = MaterialTokenizer.from_yaml(args.vocab)
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,  # VERY IMPORTANT: 保持与原始顺序一致，方便回写
        collate_fn=lambda b: collate(b, tok),
    )

    vocab = tok.vocab
    V = len(vocab._id2tok)
    PAD_id = vocab.token_id("<PAD>")
    MASK_id = vocab.token_id("<MASK>")

    # 构建 token 类型表
    token_types = build_token_types(vocab)
    token_types = list(token_types)  # index -> type

    # 加载 ckpt
    ck = torch.load(args.ckpt, map_location=device)
    if "cfg" in ck:
        cfg = ck["cfg"]
        hidden = cfg.get("hidden", 512)
        layers = cfg.get("layers", 12)
        heads = cfg.get("heads", 8)
        vocab_size = cfg.get("vocab_size", V)
        assert vocab_size == V, f"ckpt vocab_size={vocab_size}, but current V={V}"
    else:
        # 没 cfg 就用默认 / 当前 args
        hidden, layers, heads = 512, 12, 8

    model = LLaDA(
        vocab_size=V,
        hidden=hidden,
        n_layers=layers,
        n_heads=heads,
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    print(f"[INFO] Loaded ckpt from {args.ckpt}")

    # 逐 batch 预测
    with torch.no_grad():
        for idxs, ids in dl:
            idxs = idxs.tolist()
            ids = ids.to(device)          # (B, L)
            B, L = ids.shape

            # 构建 mask：只 mask PREDICT_TYPES 对应的 token
            mask_bool = torch.zeros_like(ids, dtype=torch.bool)  # (B, L)
            for b in range(B):
                for pos in range(L):
                    tid = ids[b, pos].item()
                    if tid == PAD_id:
                        continue
                    t = token_types[tid]
                    if t in PREDICT_TYPES:
                        mask_bool[b, pos] = True

            inputs = ids.clone()
            inputs[mask_bool] = MASK_id

            # 时间 t：可固定，也可随机采样
            if args.t_fixed >= 0.0:
                t = torch.full((B,), float(args.t_fixed), device=device)
            else:
                t = torch.rand(B, device=device)

            logits = model(inputs, t)  # (B, L, V)
            # softmax 可选，这里直接 argmax on logits 即可
            # logits_masked: (N_mask, V)
            logits_masked = logits[mask_bool]
            if logits_masked.numel() == 0:
                # 没有可预测位置，直接跳过
                for idx in idxs:
                    # 原样复制 tokens 作为 tokens_pred
                    ds.recs[idx]["tokens_pred"] = list(ds.recs[idx]["tokens"])
                continue

            pred_ids_flat = torch.argmax(logits_masked, dim=-1)  # (N_mask,)
            ids_pred = ids.clone()
            ids_pred[mask_bool] = pred_ids_flat

            # 回写到 dataset.recs
            for b, idx in enumerate(idxs):
                # 恢复到原始长度（去掉 PAD）
                orig_len = len(ds.recs[idx]["tokens"])
                ds.recs[idx]["tokens_pred"] = ids_pred[b, :orig_len].tolist()

    # 输出 jsonl
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in ds.recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[完成] 写出预测结果到 {args.out}")


if __name__ == "__main__":
    main()
