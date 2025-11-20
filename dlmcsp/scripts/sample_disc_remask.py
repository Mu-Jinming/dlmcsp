#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, math, sys
from typing import Dict, List

import torch
import torch.nn.functional as F
from pathlib import Path

# 保证 python -m 能找到包
sys.path.append(str(Path(__file__).resolve().parents[2]))

from dlmcsp.models.llada import LLaDA
from dlmcsp.tokenization.tokenizer import MaterialTokenizer

# -----------------------------
# 基础定义 & token 类型
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
        if any(x in up for x in ("A_BIN_", "B_BIN_", "C_BIN_",
                                 "ALPHA_BIN_", "BETA_BIN_", "GAMMA_BIN_")):
            return "LATT_BIN"
    if "_DR_" in up:
        if any(x in up for x in ("A_DR_", "B_DR_", "C_DR_",
                                 "ALPHA_DR_", "BETA_DR_", "GAMMA_DR_")):
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


def build_token_types(vocab):
    id2tok = vocab._id2tok
    types: List[str] = []
    type_count: Dict[str, int] = {}
    for s in id2tok:
        t = classify_token(s)
        types.append(t)
        type_count[t] = type_count.get(t, 0) + 1
    print(f"[INFO] Token types built. V={len(types)}")
    print("[INFO] token type counts:",
          {k: v for k, v in type_count.items() if v > 0})
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


def build_type_allowed_mask(id2type_id: torch.Tensor,
                            n_token_types: int,
                            ban_ids: List[int] | None = None) -> torch.Tensor:
    """
    type_allowed[tid, v] = True 表示 type_id=tid 的位置允许 vocab_id=v。
    ban_ids: 强制禁止生成的 vocab id（如 PAD / MASK）
    """
    V = id2type_id.numel()
    type_allowed = torch.zeros(n_token_types, V, dtype=torch.bool)
    for vid in range(V):
        tid = int(id2type_id[vid].item())
        type_allowed[tid, vid] = True

    if ban_ids is not None:
        for bid in ban_ids:
            if 0 <= bid < V:
                type_allowed[:, bid] = False

    return type_allowed


# -----------------------------
# mask ratio 调度器
# -----------------------------
def get_mask_ratio(step: int,
                   total_steps: int,
                   method: str = "cosine",
                   r_init: float = 1.0,
                   r_final: float = 0.0,
                   gamma: float = 1.0) -> float:
    """
    step: 1..total_steps
    返回“希望还保持为 MASK 的比例”
    """
    if total_steps <= 1:
        return r_final

    p = float(step) / float(total_steps)  # 0..1

    if method == "linear":
        r = r_init - p * (r_init - r_final)
    else:
        base = math.cos(p * math.pi / 2.0)  # 1 -> 0
        base = max(0.0, min(1.0, base))
        base = base ** gamma
        r = base * (r_init - r_final) + r_final

    r = max(min(r, max(r_init, r_final)), min(r_init, r_final))
    return r


# -----------------------------
# 主流程（支持 batch）
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")

    # mask schedule
    ap.add_argument("--r_init", type=float, default=1.0)
    ap.add_argument("--r_final", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=1.5)
    ap.add_argument("--schedule", type=str, default="cosine",
                    choices=["linear", "cosine"])

    # sampling
    ap.add_argument("--temp_init", type=float, default=1.5)
    ap.add_argument("--temp_final", type=float, default=0.6)
    ap.add_argument("--gumbel_scale", type=float, default=0.1)

    args = ap.parse_args()
    device = torch.device(args.device)

    # 1. 加载 vocab & 模型
    tok = MaterialTokenizer.from_yaml(args.vocab)
    vocab = tok.vocab
    V = len(vocab._id2tok)
    PAD_id = vocab.token_id("<PAD>")
    MASK_id = vocab.token_id("<MASK>")

    token_types, _ = build_token_types(vocab)

    print(f"[INFO] Loading checkpoint: {args.ckpt}")
    ck = torch.load(args.ckpt, map_location=device)
    cfg = ck.get("cfg", {})

    type_id_map = ck.get("type_id_map", None)
    if type_id_map is None:
        uniq_types = sorted(set(token_types))
        type_id_map = {tname: i for i, tname in enumerate(uniq_types)}

    id2type_id_cpu = build_id2type_id(token_types, type_id_map)
    n_token_types = cfg.get("n_token_types", len(type_id_map))

    # 类型约束 & ban 掉 PAD / MASK
    type_allowed = build_type_allowed_mask(
        id2type_id_cpu,
        n_token_types,
        ban_ids=[PAD_id, MASK_id],
    ).to(device)
    id2type_id = id2type_id_cpu.to(device)
    id_is_predict = build_id_is_predict(token_types).to(device)

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

    max_len_cfg = cfg.get("max_len", 4096)

    # 2. 读入全部样本
    all_records = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_records.append(json.loads(line))
    print(f"[INFO] Total samples: {len(all_records)}")

    def get_batch(start_idx: int, batch_size: int):
        end_idx = min(start_idx + batch_size, len(all_records))
        batch_recs = all_records[start_idx:end_idx]
        max_l = max(len(r["tokens"]) for r in batch_recs)
        max_l = min(max_l, max_len_cfg)

        batch_ids = torch.full((len(batch_recs), max_l),
                               PAD_id, dtype=torch.long)
        for i, r in enumerate(batch_recs):
            seq = r["tokens"][:max_l]
            batch_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        return batch_ids, batch_recs

    fout = open(args.out, "w", encoding="utf-8")
    total_processed = 0
    T = args.steps

    # 3. Batch 采样
    for start_idx in range(0, len(all_records), args.batch):
        ids, batch_recs = get_batch(start_idx, args.batch)
        ids = ids.to(device)
        B, L = ids.shape

        # 哪些位置需要预测
        is_predict = id_is_predict[ids] & (ids != PAD_id)  # [B,L]
        type_ids = id2type_id[ids]                         # [B,L]

        # 初始化：预测位置全 MASK
        ids_cur = ids.clone()
        ids_cur[is_predict] = MASK_id

        # keep_mask: True = 已解开（固定）；非预测 token 从一开始就 True
        keep_mask = ~is_predict.clone()

        # [B,L,V] 合法 vocab mask
        batch_allowed_mask = type_allowed[type_ids]

        for step in range(T):
            # 当前仍可 refine 的位置
            mask_can_refine = is_predict & (~keep_mask)     # [B,L]
            if not mask_can_refine.any():
                break

            n_pred_total = is_predict.sum(dim=1)            # [B]
            n_mask_remain = mask_can_refine.sum(dim=1)      # [B]
            n_keep_now = (is_predict & keep_mask).sum(dim=1)  # [B]

            # 对于没有预测位的样本，后面直接跳过
            has_pred = n_pred_total > 0

            # 目标还保留的 mask 比例（全 batch 共用同一个 r_t）
            r_t = get_mask_ratio(
                step + 1,
                T,
                method=args.schedule,
                r_init=args.r_init,
                r_final=args.r_final,
                gamma=args.gamma,
            )

            # 目标的 mask 数 / keep 数
            n_target_mask = (n_pred_total.float() * r_t).long()
            n_target_mask = torch.clamp(n_target_mask, min=0)

            desired_keep = n_pred_total - n_target_mask
            # 单调性：不能比已经 keep 的还少
            desired_keep = torch.max(desired_keep, n_keep_now)
            desired_keep = torch.clamp(desired_keep, max=n_pred_total)

            n_new = desired_keep - n_keep_now               # 每个样本本轮要解开的新位置数

            # t 输入：用真实 mask 比例（和训练一致）
            t_vec = torch.zeros(B, device=device, dtype=torch.float32)
            valid = (n_pred_total > 0)
            t_vec[valid] = (
                n_mask_remain[valid].float() /
                n_pred_total[valid].float()
            )

            # 温度退火
            prog = step / max(1, T - 1)
            temp = args.temp_init + prog * (args.temp_final - args.temp_init)
            temp = max(0.1, float(temp))

            with torch.no_grad():
                logits = model(ids_cur, t_vec, token_type_ids=type_ids)  # [B,L,V]
                logits = logits.masked_fill(~batch_allowed_mask, float("-inf"))
                probs = (logits / temp).softmax(dim=-1)                  # [B,L,V]

                pred_token = probs.argmax(dim=-1)                        # [B,L]
                conf = probs.gather(-1, pred_token.unsqueeze(-1)).squeeze(-1)  # [B,L]

                score = conf
                if args.gumbel_scale > 0.0:
                    g = -torch.empty_like(score).exponential_().log()
                    score = score + args.gumbel_scale * g

            # 按 batch 逐样本选 top-k，逻辑清晰，不玩阈值/kthvalue 那套骚操作
            for b in range(B):
                if not has_pred[b]:
                    continue

                k_new = int(n_new[b].item())
                if k_new <= 0:
                    continue

                # 候选位置：当前仍是 MASK 的预测位
                cand_mask = mask_can_refine[b]              # [L]
                cand_idx = torch.nonzero(cand_mask, as_tuple=False).squeeze(-1)
                if cand_idx.numel() == 0:
                    continue

                k_new = min(k_new, cand_idx.numel())
                scores_b = score[b, cand_idx]              # [k_remain]
                topk_score, topk_rel = torch.topk(scores_b, k_new)
                chosen_idx = cand_idx[topk_rel]            # 这些位置本轮解开

                keep_mask[b, chosen_idx] = True
                ids_cur[b, chosen_idx] = pred_token[b, chosen_idx]

            # 未解开的预测位仍然保持 MASK
            ids_cur[is_predict & (~keep_mask)] = MASK_id

        # 最后兜底：如果还有 MASK（极少数情况），用 t=0 再解一次
        final_mask = (ids_cur == MASK_id) & is_predict
        if final_mask.any():
            with torch.no_grad():
                t_zero = torch.zeros(B, device=device)
                logits = model(ids_cur, t_zero, token_type_ids=type_ids)
                logits = logits.masked_fill(~batch_allowed_mask, float("-inf"))
                preds = logits.argmax(dim=-1)
            ids_cur[final_mask] = preds[final_mask]

        # 写回 JSON
        for b in range(B):
            rec = batch_recs[b]
            orig_len = len(rec["tokens"])
            pred_seq = ids_cur[b, :orig_len].tolist()
            rec["tokens_pred"] = pred_seq
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        total_processed += B
        print(f"[Progress] {total_processed}/{len(all_records)}", end="\r")

    fout.close()
    print(f"\n[Done] Results saved to {args.out}")


if __name__ == "__main__":
    main()
