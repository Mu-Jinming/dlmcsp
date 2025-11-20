#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json
from typing import Any, Dict, List

from dlmcsp.tokenization.tokenizer import MaterialTokenizer

PUNCTS = {
    "FORMULA=", "NATOMS=", "SG=", "LATT=", "SITES=",
    "EL:", "WY:", "PARAM:", "u:", "v:", "w:",
    "(", ")", "->", ";", ","
}

# 核心 DOF：跟训练里的 PREDICT_TYPES 保持一致
DOF_TYPES = {"LATT_BIN", "PARAM_BASE", "PARAM_FINE"}

# 结构相关：SG / WY
SYM_TYPES = {"SG", "WY"}


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
    ap.add_argument("--gt", required=True, help="GT jsonl（train/val ms 数据）")
    ap.add_argument("--pred", required=True, help="sample_disc_remask 输出 jsonl（含 tokens_pred）")
    ap.add_argument("--vocab", required=True, help="vocab.yaml，用于解析 token 类型")
    args = ap.parse_args()

    gt_recs = load_jsonl(args.gt)
    pred_recs = load_jsonl(args.pred)

    if len(gt_recs) != len(pred_recs):
        raise RuntimeError(f"len(gt)={len(gt_recs)} != len(pred)={len(pred_recs)}")

    tok = MaterialTokenizer.from_yaml(args.vocab)
    vocab = tok.vocab

    # ---- 全 token 统计 ----
    total_seqs = 0
    total_all_tokens = 0
    correct_all_tokens = 0
    exact_match_all = 0

    # ---- DOF (LATT_BIN / PARAM_BASE / PARAM_FINE) 统计 ----
    total_dof_tokens = 0
    correct_dof_tokens = 0
    exact_match_dof = 0

    # ---- SYM (SG / WY) 统计 ----
    total_sym_tokens = 0
    correct_sym_tokens = 0
    exact_match_sym = 0

    # ---- 分类型统计：看清哪一类拖后腿 ----
    type_total: Dict[str, int] = {
        "LATT_BIN": 0,
        "PARAM_BASE": 0,
        "PARAM_FINE": 0,
        "SG": 0,
        "WY": 0,
    }
    type_correct: Dict[str, int] = {k: 0 for k in type_total.keys()}

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

        # ---------- 全 token acc / EM ----------
        all_equal = True
        for g, p in zip(gt_ids, pred_ids):
            if g == p:
                correct_all_tokens += 1
            else:
                all_equal = False
            total_all_tokens += 1
        if all_equal:
            exact_match_all += 1

        # ---------- DOF / SYM / 分类型 ----------
        dof_equal = True   # 该序列在所有 DOF 位置是否全对
        sym_equal = True   # 该序列在所有 SYM 位置是否全对

        for pos in range(L):
            tid = gt_ids[pos]
            tok_str = vocab.id_token(tid)
            t_type = classify_token(tok_str)

            # 分类型统计
            if t_type in type_total:
                type_total[t_type] += 1
                if pred_ids[pos] == tid:
                    type_correct[t_type] += 1

            # DOF 部分
            if t_type in DOF_TYPES:
                total_dof_tokens += 1
                if pred_ids[pos] == tid:
                    correct_dof_tokens += 1
                else:
                    dof_equal = False

            # SYM 部分
            if t_type in SYM_TYPES:
                total_sym_tokens += 1
                if pred_ids[pos] == tid:
                    correct_sym_tokens += 1
                else:
                    sym_equal = False

        if dof_equal:
            exact_match_dof += 1
        if sym_equal:
            exact_match_sym += 1

    # ================= 输出 =================
    print("===== EVAL DISC (token 级 & 序列级) =====")
    print(f"#samples                     : {total_seqs}")
    print(f"#all_tokens                  : {total_all_tokens}")
    print(f"#DOF tokens (LATT/PARAM)     : {total_dof_tokens}")
    print(f"#SYM tokens (SG/WY)          : {total_sym_tokens}")

    # 全 token
    if total_all_tokens > 0:
        acc_all = correct_all_tokens / total_all_tokens
        em_all = exact_match_all / total_seqs
        print(f"[ALL TOKENS] token acc       : {acc_all:.4f}")
        print(f"[ALL TOKENS] seq exact-match : {em_all:.4f}")

    # DOF
    if total_dof_tokens > 0:
        acc_dof = correct_dof_tokens / total_dof_tokens
        em_dof = exact_match_dof / total_seqs
        print(f"[DOF (LATT_BIN+PARAM)] token acc    : {acc_dof:.4f}")
        print(f"[DOF (LATT_BIN+PARAM)] seq EM       : {em_dof:.4f}")
    else:
        print("[WARN] 没有任何 DOF token 被统计（检查 DOF_TYPES 设置和数据）")

    # SYM
    if total_sym_tokens > 0:
        acc_sym = correct_sym_tokens / total_sym_tokens
        em_sym = exact_match_sym / total_seqs
        print(f"[SYM (SG+WY)] token acc      : {acc_sym:.4f}")
        print(f"[SYM (SG+WY)] seq EM         : {em_sym:.4f}")
    else:
        print("[WARN] 没有任何 SYM token 被统计（检查 SYM_TYPES 设置和数据）")

    # 分类型细节
    print("----- Per-type token accuracy -----")
    for tname in ["LATT_BIN", "PARAM_BASE", "PARAM_FINE", "SG", "WY"]:
        tot = type_total[tname]
        cor = type_correct[tname]
        if tot > 0:
            acc = cor / tot
            print(f"[{tname}] token acc          : {acc:.4f}  ({cor}/{tot})")
        else:
            print(f"[{tname}] token acc          : N/A       (0/0)")


if __name__ == "__main__":
    main()
