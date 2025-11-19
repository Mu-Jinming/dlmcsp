#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json
from typing import Any, Dict, List

from dlmcsp.tokenization.tokenizer import MaterialTokenizer

# 与其他脚本保持一致
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


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            recs.append(json.loads(ln))
    if not recs:
        raise ValueError(f"empty jsonl: {path}")
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="GT jsonl（训练/验证 ms 数据）")
    ap.add_argument("--pred", required=True, help="sample_disc.py 输出 jsonl（含 tokens_pred）")
    ap.add_argument("--vocab", required=True, help="vocab.yaml，用于解析 token 类型")
    args = ap.parse_args()

    gt_recs = load_jsonl(args.gt)
    pred_recs = load_jsonl(args.pred)

    if len(gt_recs) != len(pred_recs):
        raise RuntimeError(f"len(gt)={len(gt_recs)} != len(pred)={len(pred_recs)}")

    tok = MaterialTokenizer.from_yaml(args.vocab)
    vocab = tok.vocab

    total_masked_tokens = 0
    correct_masked_tokens = 0

    total_seqs = 0
    exact_match_masked = 0  # 所有需要预测位置都对的序列

    # 可选：全 token 统计
    total_all_tokens = 0
    correct_all_tokens = 0
    exact_match_all = 0

    for i, (gt, pr) in enumerate(zip(gt_recs, pred_recs)):
        gt_ids = gt["tokens"]
        pred_ids = pr.get("tokens_pred", None)
        if pred_ids is None:
            raise RuntimeError(f"[错误] pred 第 {i} 条缺少 tokens_pred 字段")

        if len(gt_ids) != len(pred_ids):
            raise RuntimeError(
                f"[错误] 第 {i} 条：gt len={len(gt_ids)} vs pred len={len(pred_ids)}"
            )

        L = len(gt_ids)
        total_seqs += 1

        # 全 token 统计
        all_equal = True
        for g, p in zip(gt_ids, pred_ids):
            if g == p:
                correct_all_tokens += 1
            else:
                all_equal = False
            total_all_tokens += 1
        if all_equal:
            exact_match_all += 1

        # 只在 PREDICT_TYPES 上统计
        masked_equal = True
        for pos in range(L):
            tid = gt_ids[pos]
            tok_str = vocab.id_token(tid)
            t_type = classify_token(tok_str)
            if t_type in PREDICT_TYPES:
                total_masked_tokens += 1
                if pred_ids[pos] == tid:
                    correct_masked_tokens += 1
                else:
                    masked_equal = False

        if masked_equal:
            exact_match_masked += 1

    print("===== EVAL DISC (token 级 & 序列级) =====")
    print(f"#samples                     : {total_seqs}")
    print(f"#all_tokens                  : {total_all_tokens}")
    print(f"#masked(PREDICT_TYPES) tokens: {total_masked_tokens}")

    if total_all_tokens > 0:
        acc_all = correct_all_tokens / total_all_tokens
        em_all = exact_match_all / total_seqs
        print(f"[ALL TOKENS] token acc       : {acc_all:.4f}")
        print(f"[ALL TOKENS] seq exact-match : {em_all:.4f}")

    if total_masked_tokens > 0:
        acc_masked = correct_masked_tokens / total_masked_tokens
        em_masked = exact_match_masked / total_seqs
        print(f"[MASKED TOKENS] token acc    : {acc_masked:.4f}")
        print(f"[MASKED TOKENS] seq EM (on masked positions): {em_masked:.4f}")
    else:
        print("[WARN] 没有任何 PREDICT_TYPES token 被统计（检查 PREDICT_TYPES 设置和数据）")


if __name__ == "__main__":
    main()
