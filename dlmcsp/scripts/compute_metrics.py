#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLaDA-cont: ms_pred(jsonl) vs test.clean.ms.jsonl 的 matching rate 计算（带 debug log）.
(多进程版本)
"""

from __future__ import annotations
import argparse, json
from typing import Dict, Any, List

# NEW: 导入 multiprocessing 和 os
import multiprocessing
import os

from tqdm import tqdm
from pymatgen.analysis.structure_matcher import StructureMatcher

from dlmcsp.representation.ms_to_cif import ms_to_structure
from dlmcsp.eval.validators import quick_validate_structure


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            recs.append(json.loads(ln))
    return recs


def _build_gt_dict(ms_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for ms in ms_list:
        mid = ms.get("material_id")
        if mid is None:
            continue
        out[str(mid)] = ms
    return out


# --- NEW: 多进程辅助函数 ---

# NEW: 为子进程准备的全局变量
G_ARGS = None
G_GT_BY_ID = None
G_MATCHER = None

def init_worker(args: argparse.Namespace, gt_by_id: Dict[str, Dict[str, Any]]):
    """
    初始化子进程。
    将只读数据和进程专属的 matcher 存入子进程的全局变量。
    """
    global G_ARGS, G_GT_BY_ID, G_MATCHER
    G_ARGS = args
    G_GT_BY_ID = gt_by_id
    G_MATCHER = StructureMatcher(
        stol=G_ARGS.stol,
        ltol=G_ARGS.ltol,
        angle_tol=G_ARGS.angle_tol,
    )

def process_record(rec: Dict[str, Any]) -> (str, float | None):
    """
    处理单个 pred 记录。这是原 for 循环的主体。
    返回一个元组 (status_code, rms_value)，用于主进程统计。
    """
    # 从子进程的全局变量中读取
    global G_ARGS, G_GT_BY_ID, G_MATCHER

    # 1) 取预测的 ms_pred 和对应的 material_id
    if "ms_pred" in rec:
        ms_pred = rec["ms_pred"]
        mid = rec.get("orig_material_id") or rec.get("material_id")
    else:
        ms_pred = rec
        mid = rec.get("material_id")

    if mid is None:
        return "mid_none", None
    mid = str(mid)

    if G_ARGS.only_geom_ok and ("geom_ok" in rec) and (not rec.get("geom_ok", False)):
        return "geom_flag_skip", None

    ms_gt = G_GT_BY_ID.get(mid)
    if ms_gt is None:
        return "no_gt", None

    if G_ARGS.require_same_formula:
        f_pred = ms_pred.get("formula")
        f_gt = ms_gt.get("formula")
        if f_pred is not None and f_gt is not None and str(f_pred) != str(f_gt):
            return "formula_mismatch", None  # 返回一个新状态

    # 2) 转 Structure
    try:
        struct_pred = ms_to_structure(ms_pred, G_ARGS.vocab)
        struct_gt = ms_to_structure(ms_gt, G_ARGS.vocab)
    except Exception as e:
        return "build_fail", None

    # ok_pred, _ = quick_validate_structure(struct_pred)
    # ok_gt, _ = quick_validate_structure(struct_gt)
    # if not (ok_pred and ok_gt):
    #     return "geom_invalid", None

    # sg 不一致：我们把它当作一对“considered 但不匹配”
    sg_pred = int(ms_pred.get("sg", -1))
    sg_gt = int(ms_gt.get("sg", -1))
    if sg_pred != -1 and sg_gt != -1 and sg_pred != sg_gt:
        return "sg_mismatch", None  # 算 n_total=1, match=0

    # 3) StructureMatcher
    try:
        rms = G_MATCHER.get_rms_dist(struct_pred, struct_gt)
    except Exception as e:
        return "rms_exception", None  # 算 n_total=1, match=0
    
    if rms is not None:
        return "match", float(rms[0])
    else:
        return "rms_none", None  # 算 n_total=1, match=0

# --- END: 多进程辅助函数 ---


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_ms", required=True, help="llada_cont.test_samples.jsonl")
    ap.add_argument("--gt_ms", required=True, help="test.clean.ms.jsonl")
    ap.add_argument("--vocab", required=True, help="vocab.yaml 路径")
    ap.add_argument("--stol", type=float, default=0.5)
    ap.add_argument("--ltol", type=float, default=0.3)
    ap.add_argument("--angle_tol", type=float, default=10.0)
    ap.add_argument("--require_same_formula", action="store_true",
                    help="若开启，则 formula 不同的 pair 直接跳过")
    ap.add_argument("--only_geom_ok", action="store_true",
                    help="若预测记录中有 geom_ok 字段，则只用 geom_ok=True 的样本")
    ap.add_argument("-j", "--jobs", type=int, default=12,
                    help="Number of processes to use (default: 12)")
    args = ap.parse_args()

    preds_raw = _load_jsonl(args.pred_ms)
    gts_raw = _load_jsonl(args.gt_ms)
    gt_by_id = _build_gt_dict(gts_raw)
    
    total_preds = len(preds_raw)

    print("========== DEBUG: basic stats ==========")
    print(f"[DEBUG] #pred records     = {total_preds}")
    print(f"[DEBUG] #gt records       = {len(gts_raw)}")
    print(f"[DEBUG] #gt with mid      = {len(gt_by_id)}")
    
    # 看看前几个 GT 记录长啥样
    for i in range(min(3, len(gts_raw))):
        rec = gts_raw[i]
        print(f"[DEBUG] gt[{i}] keys          = {list(rec.keys())}")
        print(f"[DEBUG] gt[{i}].material_id   = {rec.get('material_id')}")

    # 统计 pred 里各种 id 情况
    pred_ids = []
    pred_orig_ids = []
    pred_has_ms_pred = 0
    for rec in preds_raw:
        if "ms_pred" in rec:
            pred_has_ms_pred += 1
        mid1 = rec.get("orig_material_id")
        mid2 = rec.get("material_id")
        if mid1 is not None:
            pred_orig_ids.append(str(mid1))
        if mid2 is not None:
            pred_ids.append(str(mid2))

    set_gt_ids = set(gt_by_id.keys())
    set_pred_all_ids = set(pred_ids) | set(pred_orig_ids)
    inter_ids = set_gt_ids & set_pred_all_ids

    print("---------- DEBUG: id overview ----------")
    print(f"[DEBUG] #pred with ms_pred      = {pred_has_ms_pred}")
    print(f"[DEBUG] #pred with orig_material = {len(pred_orig_ids)}")
    print(f"[DEBUG] #pred with material_id  = {len(pred_ids)}")
    print(f"[DEBUG] #distinct pred ids total = {len(set_pred_all_ids)}")
    print(f"[DEBUG] #id intersection(pred,gt)= {len(inter_ids)}")

    # 打印 intersection 里前几个 id
    sample_ids = list(inter_ids)[:10]
    print(f"[DEBUG] example intersect ids   = {sample_ids}")

    # 再看一下前几个 pred 记录的结构
    for i in range(min(3, len(preds_raw))):
        rec = preds_raw[i]
        print(f"[DEBUG] pred[{i}] keys            = {list(rec.keys())}")
        print(f"[DEBUG] pred[{i}].orig_material_id= {rec.get('orig_material_id')}")
        print(f"[DEBUG] pred[{i}].material_id     = {rec.get('material_id')}")
        if "ms_pred" in rec:
            ms_pred = rec["ms_pred"]
            print(f"[DEBUG] pred[{i}].ms_pred keys    = {list(ms_pred.keys())}")

    # matcher = StructureMatcher(...)

    # 各种 continue 路径的计数
    cnt_mid_none = 0
    cnt_no_gt = 0
    cnt_geom_flag_skip = 0
    cnt_formula_mismatch = 0
    cnt_build_fail = 0
    # cnt_geom_invalid = 0
    cnt_sg_mismatch = 0
    cnt_rms_exception = 0
    cnt_rms_none = 0

    n_total = 0
    n_match = 0
    rms_list: List[float] = []
    
    n_jobs = args.jobs
    if n_jobs <= 0:
        n_jobs = os.cpu_count() or 1
    print(f"---------- Starting matching with {n_jobs} processes ----------")

    with multiprocessing.Pool(processes=n_jobs,
                            initializer=init_worker,
                            initargs=(args, gt_by_id)) as pool:
        
        # 使用 imap_unordered 获取最佳性能，结果在完成时立即返回
        results_iter = pool.imap_unordered(process_record, preds_raw)
        
        # 在主进程中收集结果并显示进度条
        for status, rms in tqdm(results_iter, total=total_preds, desc="matching"):
            # 1. 处理那些不计入 n_total 的 continue 情况
            if status == "mid_none":
                cnt_mid_none += 1
                continue
            if status == "no_gt":
                cnt_no_gt += 1
                continue
            if status == "geom_flag_skip":
                cnt_geom_flag_skip += 1
                continue
            if status == "formula_mismatch":
                cnt_formula_mismatch += 1
                continue
            if status == "build_fail":
                cnt_build_fail += 1
                continue
            
            # 移除 geom_invalid 逻辑
            # if status == "geom_invalid":
            #     cnt_geom_invalid += 1
            #     continue

            # 2. 处理计入 n_total 的情况
            # (能运行到这里，说明是一个 "considered" 的 pair)
            n_total += 1
            
            if status == "sg_mismatch":
                cnt_sg_mismatch += 1
            elif status == "rms_exception":
                cnt_rms_exception += 1
            elif status == "rms_none":
                cnt_rms_none += 1
            elif status == "match":
                if rms is not None:
                    n_match += 1
                    rms_list.append(rms)
                else:
                    # 理论上 status=="match" 时 rms 不该是 None，但做个保护
                    cnt_rms_none += 1
            # else:
            #   (不应该有其他 status)

    # --- 循环结束 ---

    # 计算两种matching rate
    match_rate_vs_total = n_match / max(1, total_preds)
    match_rate_vs_considered = n_match / max(1, n_total)
    mean_rms = sum(rms_list) / len(rms_list) if rms_list else None

    print("========== LLaDA-cont matching ==========")
    print(f"#total pred records: {total_preds}")
    print(f"#pairs considered : {n_total}")
    print(f"#pairs matched    : {n_match}")
    print(f"matching rate (vs total)     : {match_rate_vs_total:.4f}")
    print(f"matching rate (vs considered): {match_rate_vs_considered:.4f}")
    if mean_rms is not None:
        print(f"mean RMS distance : {mean_rms:.6f}")

    print("========== DEBUG: path counters ==========")
    print(f"[DEBUG] mid is None             : {cnt_mid_none}")
    print(f"[DEBUG] no gt for this mid      : {cnt_no_gt}")
    print(f"[DEBUG] geom_ok flag skip       : {cnt_geom_flag_skip}")
    print(f"[DEBUG] formula mismatch (skip) : {cnt_formula_mismatch}")
    print(f"[DEBUG] ms_to_structure fail    : {cnt_build_fail}")
    # print(f"[DEBUG] geom invalid (validator): {cnt_geom_invalid}")
    print(f"[DEBUG] sg mismatch (counted)   : {cnt_sg_mismatch}")
    print(f"[DEBUG] rms exception           : {cnt_rms_exception}")
    print(f"[DEBUG] rms is None             : {cnt_rms_none}")


if __name__ == "__main__":
    main()