#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, math, sys
from typing import Dict, List

import torch
import torch.nn.functional as F

# 确保能找到包（如果在项目根目录运行，通常不需要这两行，但在某些环境下需要）
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from dlmcsp.models.llada import LLaDA
from dlmcsp.tokenization.tokenizer import MaterialTokenizer

# -----------------------------
# 基础定义
# -----------------------------
PUNCTS = {
    "FORMULA=", "NATOMS=", "SG=", "LATT=", "SITES=",
    "EL:", "WY:", "PARAM:", "u:", "v:", "w:",
    "(", ")", "->", ";", ","
}

PREDICT_TYPES = {"LATT_BIN", "PARAM_BASE", "PARAM_FINE"}

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
    if up.startswith("BASE_"): return "PARAM_BASE"
    if up.startswith("FINE_"): return "PARAM_FINE"
    if up.startswith("SG_") or s == "SG=": return "SG"
    if s.startswith("WY:") or s == "WY:": return "WY"
    if s in PUNCTS: return "PUNCT"
    return "OTHER"

def build_token_types(vocab):
    id2tok = vocab._id2tok
    types: List[str] = []
    type_count: Dict[str, int] = {}
    for s in id2tok:
        t = classify_token(s)
        types.append(t)
        type_count[t] = type_count.get(t, 0) + 1
    print(f"[INFO] Token types built. V={len(types)}")
    return types, type_count

def build_id_is_predict(token_types: List[str]) -> torch.Tensor:
    V = len(token_types)
    arr = torch.zeros(V, dtype=torch.bool)
    for tid in range(V):
        if token_types[tid] in PREDICT_TYPES:
            arr[tid] = True
    return arr

def build_id2type_id(token_types: List[str], type_id_map: Dict[str, int]) -> torch.Tensor:
    V = len(token_types)
    id2type = torch.zeros(V, dtype=torch.long)
    for tid in range(V):
        tname = token_types[tid]
        id2type[tid] = type_id_map.get(tname, 0)
    return id2type

# -----------------------------
# 调度器函数
# -----------------------------
def get_mask_ratio(step: int, total_steps: int, method="cosine", r_init=1.0, r_final=0.0, gamma=1.0) -> float:
    """
    计算当前步的 mask ratio。
    step: 0 -> total_steps (不包含 total_steps)
    """
    # 归一化进度 p: 0.0 -> 1.0
    if total_steps <= 1:
        return r_final
    p = step / total_steps 
    
    if method == "linear":
        # 从 r_init 线性降到 r_final
        return r_init - p * (r_init - r_final)
    
    elif method == "cosine":
        # maskgit 风格 cosine schedule
        return math.cos(p * math.pi / 2) * (r_init - r_final) + r_final
        
    else:
        # 默认线性
        return r_init - p * (r_init - r_final)

def main():
    print("[INFO] Script started...") # 调试输出，确保进入了 main
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    
    # 新增参数以匹配你的 Shell 脚本
    ap.add_argument("--r_init", type=float, default=1.0, help="初始 Mask 比例")
    ap.add_argument("--r_final", type=float, default=0.0, help="最终 Mask 比例")
    ap.add_argument("--gamma", type=float, default=1.0, help="调度参数(保留备用)")
    
    args = ap.parse_args()
    print(f"[INFO] Arguments parsed. Output will be: {args.out}")

    device = torch.device(args.device)

    # 1. Load
    tok = MaterialTokenizer.from_yaml(args.vocab)
    vocab = tok.vocab
    V = len(vocab._id2tok)
    PAD_id = vocab.token_id("<PAD>")
    MASK_id = vocab.token_id("<MASK>")

    token_types, _ = build_token_types(vocab)

    ck = torch.load(args.ckpt, map_location=device)
    print(f"[INFO] Checkpoint loaded from {args.ckpt}")

    cfg = ck.get("cfg", {})
    type_id_map = ck.get("type_id_map", None)
    if type_id_map is None:
        uniq_types = sorted(set(token_types))
        type_id_map = {tname: i for i, tname in enumerate(uniq_types)}

    id2type_id = build_id2type_id(token_types, type_id_map).to(device)
    id_is_predict = build_id_is_predict(token_types).to(device)

    n_token_types = cfg.get("n_token_types", len(type_id_map))
    model = LLaDA(
        vocab_size=V,
        hidden=cfg.get("hidden", 512),
        n_layers=cfg.get("layers", 12),
        n_heads=cfg.get("heads", 8),
        dropout=cfg.get("dropout", 0.0),
        max_len=cfg.get("max_len", 4096),
        n_token_types=n_token_types,
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    total = 0
    with open(args.data, "r", encoding="utf-8") as fin, \
         open(args.out, "w", encoding="utf-8") as fout:

        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            tokens = rec["tokens"]
            ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
            B, L = ids.shape

            if L > cfg.get("max_len", 4096):
                continue

            is_predict = id_is_predict[ids] & (ids != PAD_id)

            if not is_predict.any():
                rec["tokens_pred"] = tokens
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1
                continue

            type_ids = id2type_id[ids]
            ids_cur = ids.clone()
            ids_cur[is_predict] = MASK_id
            keep_mask = torch.zeros_like(is_predict, dtype=torch.bool)

            T = args.steps
            
            for step in range(T):
                # 1. 计算当前 step 的时间 t (用于输入模型) 和 目标 mask 比例 r
                # MaskGIT 逻辑: t 随着 step 增加从 1 -> 0 (mask程度减小)
                # 你的脚本给了 r_init=1.0, r_final=0.1
                
                # 当前这一步预期的 mask 比例
                r_t = get_mask_ratio(step + 1, T, "linear", args.r_init, args.r_final)
                
                # 模型输入的时间 embedding t: 可以直接用当前的 mask 比例，或者 (1 - progress)
                # 这里简单使用 (1 - step/T)
                t_input = 1.0 - (step / T)
                t_vec = torch.full((B,), t_input, dtype=torch.float32, device=device)

                with torch.no_grad():
                    logits = model(ids_cur, t_vec, token_type_ids=type_ids)
                    logp = F.log_softmax(logits, dim=-1)

                pred_ids = logp.argmax(dim=-1)
                probs = logp.exp()
                conf = probs.gather(-1, pred_ids.unsqueeze(-1)).squeeze(-1)

                mask_can_refine = is_predict & (~keep_mask)
                if not mask_can_refine.any():
                    break

                # 2. 计算要保留多少个 Token (Mask 剩下的数量)
                # n_total_predict = is_predict.sum().item()
                # n_keep = int(n_total_predict * r_t) 
                # 这里的逻辑是：MaskGIT 策略，保留置信度低的为 Mask，置信度高的变为 Token
                # 因此，我们要把 mask_can_refine 中置信度最高的 top-k 变成 token
                
                # 当前还剩下多少个是 mask
                n_remaining = mask_can_refine.sum().item()
                
                # 目标：让总的 mask 数量降低到 n_total_predict * r_t
                n_total_target = is_predict.sum().item()
                n_target_mask = int(n_total_target * r_t)
                
                # 本轮需要解开(Unmask)的数量
                # 如果当前剩下的 mask 数量已经是目标值或更少，就不动了
                # 实际上因为是迭代，通常 n_remaining > n_target_mask
                k_to_unmask = n_remaining - n_target_mask
                if k_to_unmask < 1:
                    k_to_unmask = 1 # 至少解开一个，防止死锁
                
                k_to_unmask = min(k_to_unmask, n_remaining)

                # 屏蔽掉非候选区域
                conf_step = conf.clone()
                conf_step[~mask_can_refine] = -1e9

                # 选出置信度最高的 k 个进行 Unmask
                topk_conf, topk_idx = torch.topk(conf_step.view(-1), k_to_unmask)
                
                sel = torch.zeros_like(conf_step, dtype=torch.bool)
                sel.view(-1)[topk_idx] = True

                ids_cur[sel] = pred_ids[sel]
                keep_mask |= sel
                ids_cur[is_predict & (~keep_mask)] = MASK_id

            rec["tokens_pred"] = ids_cur[0].tolist()
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += 1
            if total % 100 == 0:
                print(f"[Progress] {total} samples processed", end="\r")

    print(f"\n[Done] Total {total} samples -> {args.out}")

if __name__ == "__main__":
    main()