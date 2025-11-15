#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLaDA-cont: ms_pred(jsonl) vs test.clean.ms.jsonl 的 matching rate 计算（带 debug log）.
"""

from __future__ import annotations
import argparse, json
from typing import Dict, Any, List

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
    args = ap.parse_args()

    preds_raw = _load_jsonl(args.pred_ms)
    gts_raw = _load_jsonl(args.gt_ms)
    gt_by_id = _build_gt_dict(gts_raw)

    print("========== DEBUG: basic stats ==========")
    print(f"[DEBUG] #pred records     = {len(preds_raw)}")
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
    print(f"[DEBUG] #pred with ms_pred       = {pred_has_ms_pred}")
    print(f"[DEBUG] #pred with orig_material = {len(pred_orig_ids)}")
    print(f"[DEBUG] #pred with material_id   = {len(pred_ids)}")
    print(f"[DEBUG] #distinct pred ids total = {len(set_pred_all_ids)}")
    print(f"[DEBUG] #id intersection(pred,gt)= {len(inter_ids)}")

    # 打印 intersection 里前几个 id
    sample_ids = list(inter_ids)[:10]
    print(f"[DEBUG] example intersect ids    = {sample_ids}")

    # 再看一下前几个 pred 记录的结构
    for i in range(min(3, len(preds_raw))):
        rec = preds_raw[i]
        print(f"[DEBUG] pred[{i}] keys            = {list(rec.keys())}")
        print(f"[DEBUG] pred[{i}].orig_material_id= {rec.get('orig_material_id')}")
        print(f"[DEBUG] pred[{i}].material_id     = {rec.get('material_id')}")
        if "ms_pred" in rec:
            ms_pred = rec["ms_pred"]
            print(f"[DEBUG] pred[{i}].ms_pred keys   = {list(ms_pred.keys())}")

    matcher = StructureMatcher(
        stol=args.stol,
        ltol=args.ltol,
        angle_tol=args.angle_tol,
    )

    # 各种 continue 路径的计数
    cnt_mid_none = 0
    cnt_no_gt = 0
    cnt_geom_flag_skip = 0
    cnt_build_fail = 0
    cnt_geom_invalid = 0
    cnt_sg_mismatch = 0
    cnt_rms_exception = 0
    cnt_rms_none = 0

    n_total = 0
    n_match = 0
    rms_list: List[float] = []

    for rec in tqdm(preds_raw, desc="matching"):
        # 1) 取预测的 ms_pred 和对应的 material_id
        if "ms_pred" in rec:
            ms_pred = rec["ms_pred"]
            mid = rec.get("orig_material_id") or rec.get("material_id")
        else:
            ms_pred = rec
            mid = rec.get("material_id")

        if mid is None:
            cnt_mid_none += 1
            continue
        mid = str(mid)

        if args.only_geom_ok and ("geom_ok" in rec) and (not rec.get("geom_ok", False)):
            cnt_geom_flag_skip += 1
            continue

        ms_gt = gt_by_id.get(mid)
        if ms_gt is None:
            cnt_no_gt += 1
            continue

        if args.require_same_formula:
            f_pred = ms_pred.get("formula")
            f_gt = ms_gt.get("formula")
            if f_pred is not None and f_gt is not None and str(f_pred) != str(f_gt):
                # 这里也可以单独计数
                continue

        # 2) 转 Structure
        try:
            struct_pred = ms_to_structure(ms_pred, args.vocab)
            struct_gt = ms_to_structure(ms_gt, args.vocab)
        except Exception as e:
            cnt_build_fail += 1
            # 你如果想看具体错误，可以打开下面注释
            # print(f"[DEBUG] build_fail mid={mid} err={type(e).__name__}:{e}")
            continue

        ok_pred, _ = quick_validate_structure(struct_pred)
        ok_gt, _ = quick_validate_structure(struct_gt)
        if not (ok_pred and ok_gt):
            cnt_geom_invalid += 1
            continue

        # sg 不一致：我们把它当作一对“considered 但不匹配”
        sg_pred = int(ms_pred.get("sg", -1))
        sg_gt = int(ms_gt.get("sg", -1))
        if sg_pred != -1 and sg_gt != -1 and sg_pred != sg_gt:
            n_total += 1
            cnt_sg_mismatch += 1
            continue

        # 3) StructureMatcher
        try:
            rms = matcher.get_rms_dist(struct_pred, struct_gt)
        except Exception as e:
            cnt_rms_exception += 1
            rms = None

        n_total += 1
        if rms is not None:
            n_match += 1
            rms_list.append(float(rms[0]))
        else:
            cnt_rms_none += 1

    match_rate = n_match / max(1, n_total)
    mean_rms = sum(rms_list) / len(rms_list) if rms_list else None

    print("========== LLaDA-cont matching ==========")
    print(f"#pairs considered : {n_total}")
    print(f"#pairs matched    : {n_match}")
    print(f"matching rate     : {match_rate:.4f}")
    if mean_rms is not None:
        print(f"mean RMS distance : {mean_rms:.6f}")

    print("========== DEBUG: path counters ==========")
    print(f"[DEBUG] mid is None           : {cnt_mid_none}")
    print(f"[DEBUG] no gt for this mid    : {cnt_no_gt}")
    print(f"[DEBUG] geom_ok flag skip     : {cnt_geom_flag_skip}")
    print(f"[DEBUG] ms_to_structure fail  : {cnt_build_fail}")
    print(f"[DEBUG] geom invalid (validator): {cnt_geom_invalid}")
    print(f"[DEBUG] sg mismatch (counted) : {cnt_sg_mismatch}")
    print(f"[DEBUG] rms exception         : {cnt_rms_exception}")
    print(f"[DEBUG] rms is None           : {cnt_rms_none}")


if __name__ == "__main__":
    main()
