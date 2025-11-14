# -*- coding: utf-8 -*-
"""
审计 material-string jsonl （多进程版）：

- ms -> Structure（不经 CIF）；
- 用 spglib 重建 SG，检查 sg 一致性；
- 校验晶系角度与规范一致（含 R 群，在 preprocess 中已 HEX/Rhombo 归一）；
- 检查 u/v/w ∈ [0,1)；
- 统计失败类型与比例；
- 统计 R 群（146–167）的 gamma 分布。
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from typing import Dict, Any, Tuple, List

import numpy as np
from tqdm import tqdm
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from dlmcsp.representation.ms_to_cif import ms_to_structure


def _crystal_system(sg: int) -> str:
    if 1 <= sg <= 2:
        return "triclinic"
    if 3 <= sg <= 15:
        return "monoclinic"
    if 16 <= sg <= 74:
        return "orthorhombic"
    if 75 <= sg <= 142:
        return "tetragonal"
    if 143 <= sg <= 167:
        return "trigonal"
    if 168 <= sg <= 194:
        return "hexagonal"
    if 195 <= sg <= 230:
        return "cubic"
    return "unknown"


def _angle_checks(
    cs: str,
    a: float,
    b: float,
    c: float,
    al: float,
    be: float,
    ga: float,
    sg: int,
) -> List[str]:
    """按晶系做几何角度规范检查，返回错误标签列表。"""

    def near(x, y, tol=0.5):
        return abs(x - y) <= tol

    errs: List[str] = []
    if cs == "cubic":
        if not (near(al, 90) and near(be, 90) and near(ga, 90)):
            errs.append("angle_cubic")
    elif cs == "tetragonal":
        if not (near(al, 90) and near(be, 90) and near(ga, 90)):
            errs.append("angle_tetra")
    elif cs == "orthorhombic":
        if not (near(al, 90) and near(be, 90) and near(ga, 90)):
            errs.append("angle_ortho")
    elif cs == "hexagonal":
        if not (near(al, 90) and near(be, 90) and near(ga, 120)):
            errs.append("angle_hex")
    elif cs == "trigonal":
        # 统一按 hex setting 检查：α≈β≈90, γ≈120
        if not (near(al, 90) and near(be, 90) and near(ga, 120)):
            errs.append("angle_trig_hex")
    # monoclinic / triclinic 不强制
    return errs


def _audit_one(rec: Dict[str, Any], vocab_path: str) -> Tuple[Dict[str, int], float | None]:
    """
    对单条 record 做审计，返回：
      - local_errs: {error_type: count}
      - r_gamma: 若为 R 群则返回 gamma，否则 None
    """
    local = Counter()
    r_gamma = None

    sg_gt = int(rec["sg"])

    # 1) ms -> Structure
    try:
        struct = ms_to_structure(rec, vocab_path)
    except Exception:
        local["ms_to_struct_fail"] += 1
        return dict(local), None

    # 2) spglib SG 重构
    try:
        sga = SpacegroupAnalyzer(struct, symprec=1e-2, angle_tolerance=5)
        sg2 = int(sga.get_space_group_number())
    except Exception:
        local["spglib_fail"] += 1
        return dict(local), None

    if sg2 != sg_gt:
        local["sg_mismatch"] += 1

    # 3) 晶系角度检查
    latt = struct.lattice
    a, b, c = latt.a, latt.b, latt.c
    al, be, ga = latt.alpha, latt.beta, latt.gamma

    cs = _crystal_system(sg_gt)
    ang_errs = _angle_checks(cs, a, b, c, al, be, ga, sg_gt)
    for e in ang_errs:
        local[e] += 1

    # 4) R 群 gamma 统计
    if 146 <= sg_gt <= 167:
        r_gamma = ga

    # 5) u/v/w 数值范围检查（value 字段）
    for site in rec.get("sites", []):
        params = site.get("params", {})
        for k in ("u", "v", "w"):
            node = params.get(k)
            if isinstance(node, dict) and "value" in node:
                v = float(node["value"])
                if not (0.0 - 1e-6 <= v < 1.0 + 1e-6):
                    local["uvw_out_of_range"] += 1
                    break

    return dict(local), r_gamma


# ---- 多进程支持 ----

_AUDIT_VOCAB_PATH: str | None = None


def _init_audit_worker(vocab_path: str):
    global _AUDIT_VOCAB_PATH
    _AUDIT_VOCAB_PATH = vocab_path


def _audit_worker(rec: Dict[str, Any]) -> Tuple[Dict[str, int], float | None]:
    global _AUDIT_VOCAB_PATH
    assert _AUDIT_VOCAB_PATH is not None
    return _audit_one(rec, _AUDIT_VOCAB_PATH)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="material-string jsonl")
    ap.add_argument("--vocab", required=True, help="vocab.yaml")
    ap.add_argument("--limit", type=int, default=0, help="只审计前 N 条（0=全部）")
    ap.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="并行 worker 数量；0 或 1 表示单进程",
    )
    args = ap.parse_args()

    # 预读 jsonl，避免在 worker 里重复 I/O
    records: List[Dict[str, Any]] = []
    with open(args.data, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            rec = json.loads(ln)
            records.append(rec)
            if args.limit and len(records) >= args.limit:
                break

    bad = Counter()
    r_gamma: List[float] = []

    # 单进程
    if args.num_workers <= 1:
        for rec in tqdm(records, desc="audit(single)"):
            local_dict, gamma = _audit_one(rec, args.vocab)
            bad.update(local_dict)
            if gamma is not None:
                r_gamma.append(gamma)
    # 多进程
    else:
        from multiprocessing import Pool

        with Pool(
            processes=args.num_workers,
            initializer=_init_audit_worker,
            initargs=(args.vocab,),
        ) as pool:
            for local_dict, gamma in tqdm(
                pool.imap(_audit_worker, records),
                total=len(records),
                desc=f"audit(mp, {args.num_workers} workers)",
            ):
                bad.update(local_dict)
                if gamma is not None:
                    r_gamma.append(gamma)

    total = len(records)
    print("=== AUDIT SUMMARY ===")
    print(f"total: {total}, bad_total: {sum(bad.values())}")
    for k, v in bad.most_common():
        print(f"{k}: {v}")

    if r_gamma:
        arr = np.array(r_gamma)
        print(
            f"R-group gamma stats: mean={arr.mean():.3f}, "
            f"std={arr.std():.3f}, min={arr.min():.3f}, max={arr.max():.3f}"
        )
    else:
        print("No R-group samples found or parsed.")


if __name__ == "__main__":
    main()
