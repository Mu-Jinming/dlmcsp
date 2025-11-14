#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GT 条件训练脚本（LLaDA）：
- 数据：*.ms.jsonl（每行含 "tokens": [int,...]）
- 词表：configs/vocab.yaml（必要时扩展 runtime 以覆盖 max token id）
- 掩码：按 token 类型选择性掩码（只训练 lattice/参数）
"""
from __future__ import annotations
import argparse, json, os, math, time
from pathlib import Path
from typing import List, Dict

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from dlmcsp.models.llada import LLaDA
from dlmcsp.tokenization.tokenizer import MaterialTokenizer

# =========================
# 掩码类型与概率（GT 模式）
# =========================
TOKEN_TYPE_P = {
    "LATT_BIN": 0.60,
    "LATT_DR":  0.00,   # 初期置 0，稳定后可开到 0.15
    "PARAM_BASE": 0.70,
    "PARAM_FINE": 0.50,
    "PARAM_DR":  0.00,
    "SG": 0.00,
    "WY": 0.00,
    "EL": 0.00,
    "FORMULA": 0.00,
    "PUNCT": 0.00,
    "OTHER": 0.05,
}

PUNCTS = {"FORMULA=","NATOMS=","LATT=","SITES=","EL:","PARAM:","(",")","->",";",",",":"}

def classify_token(tok_str: str) -> str:
    s = tok_str
    up = s.upper()
    if "_BIN_" in up:
        if any(x in up for x in ("A_BIN_","B_BIN_","C_BIN_","ALPHA_BIN_","BETA_BIN_","GAMMA_BIN_")):
            return "LATT_BIN"
    if "_DR_" in up:
        if any(x in up for x in ("A_DR_","B_DR_","C_DR_","ALPHA_DR_","BETA_DR_","GAMMA_DR_")):
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
    # 元素判断可以更严谨；保守起见不掩即可
    return "OTHER"

# ==============
# 数据集
# ==============
class MsJsonlDataset(Dataset):
    def __init__(self, path: str):
        self.items: List[List[int]] = []
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                obj = json.loads(ln)
                toks = obj.get("tokens", None)
                if not toks:
                    raise ValueError("数据项缺少 'tokens' 字段；请确保使用 preprocess 生成的 *.ms.jsonl")
                self.items.append([int(x) for x in toks])
        self.max_len = max(len(x) for x in self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return torch.tensor(self.items[i], dtype=torch.long)

def collate_pad(batch: List[torch.Tensor], pad_id: int):
    maxL = max(x.size(0) for x in batch)
    out = torch.full((len(batch), maxL), pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        out[i, :x.size(0)] = x
    return out

# =========================
# 训练主函数
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="*.ms.jsonl")
    ap.add_argument("--vocab", required=True, help="configs/vocab.yaml")
    ap.add_argument("--save", required=True, help="checkpoint path")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=1000)
    args = ap.parse_args()

    device = args.device
    Path(os.path.dirname(args.save)).mkdir(parents=True, exist_ok=True)

    # 1) 读数据
    ds = MsJsonlDataset(args.data)
    print(f"[INFO] dataset loaded. size={len(ds)} max_len={ds.max_len}")

    # 2) 构建 tokenizer，必要时扩展 runtime vocab 以覆盖数据的 token id
    tok = MaterialTokenizer.from_yaml(args.vocab)
    vocab = tok.vocab
    PAD_ID = vocab.token_id("<PAD>")
    MASK_ID = vocab.token_id("<MASK>")

    # 找到数据中的最大 token id，若超出当前词表，顺序补齐占位符
    max_token_id_in_data = 0
    for ids in ds.items:
        if ids and max(ids) > max_token_id_in_data:
            max_token_id_in_data = max(ids)
    if max_token_id_in_data >= len(vocab._id2tok):
        need = max_token_id_in_data + 1 - len(vocab._id2tok)
        for k in range(need):
            vocab._add(f"__RUNTIME_{len(vocab._id2tok)}__")
        print(f"[INFO] expanded runtime vocab to size={len(vocab._id2tok)} to cover max token id={max_token_id_in_data}")

    # 保存 runtime 词表顺序，便于采样复现
    runtime_vocab_path = Path(args.vocab).with_suffix(".runtime.yaml")
    with open(runtime_vocab_path, "w", encoding="utf-8") as fo:
        fo.write("runtime_vocab:\n")
        for t in vocab._id2tok:
            fo.write(f"  - {t}\n")
    print(f"[INFO] runtime vocab saved to {runtime_vocab_path} (tokens={len(vocab._id2tok)})")

    # 3) 构建类型掩码概率表（按 id）
    V = len(vocab._id2tok)
    p_by_id = torch.zeros(V)
    type_count = {k:0 for k in TOKEN_TYPE_P.keys()}
    for i, s in enumerate(vocab._id2tok):
        t = classify_token(s)
        p_by_id[i] = float(TOKEN_TYPE_P.get(t, TOKEN_TYPE_P["OTHER"]))
        if t in type_count: type_count[t] += 1
    print("[INFO] token types (counts):", {k:v for k,v in type_count.items() if v>0})

    # 4) 模型
    model = LLaDA(
        vocab_size=V,
        hidden=args.hidden,
        n_layers=args.layers,
        n_heads=args.heads
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 5) dataloader
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=lambda x: collate_pad(x, PAD_ID), drop_last=True)

    # 6) 训练循环
    model.train()
    step = 0
    t0 = time.time()
    while step < args.steps:
        for batch in dl:
            step += 1
            batch = batch.to(device)  # [B, L]
            B, L = batch.size()

            # 按 id 概率掩码（GT 条件）
            probs = p_by_id.to(device)[batch]   # [B, L]
            randu = torch.rand_like(probs)
            mask = randu < probs                # bool
            inputs = batch.clone()
            inputs[mask] = MASK_ID
            targets = batch.clone()
            targets[~mask] = -100

            # 随机扩散时间 t ~ U(0,1)
            t = torch.rand(B, device=device)

            logits = model(inputs, t)  # [B, L, V]
            loss = criterion(logits.view(-1, V), targets.view(-1))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % args.log_every == 0:
                # 统计各类型的实际掩码率（仅日志显示，不耗时）
                with torch.no_grad():
                    # 抽取当前 batch 中各类型 token 的掩码比例
                    masked_ratio = {}
                    ids_cpu = batch.detach().cpu()
                    mask_cpu = mask.detach().cpu()
                    for name in ["LATT_BIN","LATT_DR","PARAM_BASE","PARAM_FINE","PARAM_DR","SG","WY","EL","PUNCT","OTHER"]:
                        # 找到该类型对应的 id 集
                        ids_of_type = [i for i,s in enumerate(vocab._id2tok) if classify_token(s)==name]
                        if not ids_of_type:
                            continue
                        m = (ids_cpu.unsqueeze(-1) == torch.tensor(ids_of_type)[None, None, :]).any(-1)
                        total = int(m.sum().item())
                        if total == 0:
                            continue
                        masked = int((m & mask_cpu).sum().item())
                        masked_ratio[name] = round(masked / max(total,1), 3)
                print(f"step {step} | loss {loss.item():.4f} | B {B} L {L} | vocab {V} | masked {masked_ratio}")

            if step % args.save_every == 0 or step == args.steps:
                ck = {
                    "cfg": {"hidden": args.hidden, "layers": args.layers, "heads": args.heads},
                    "model": model.state_dict(),
                    "runtime_vocab": vocab._id2tok,  # 保存顺序供采样复现
                }
                torch.save(ck, args.save)
                print(f"[CKPT] saved to {args.save}")

            if step >= args.steps:
                break

    print(f"[DONE] train steps={step} time={(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
