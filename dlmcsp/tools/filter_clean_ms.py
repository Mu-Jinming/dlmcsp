# -*- coding: utf-8 -*-
"""
从 material-string jsonl 里筛出“干净样本”：

保留样本需满足：
  - ms -> Structure 成功；
  - quick_validate_structure 成功（几何合法、体积合理等）；
  - spglib 重新识别的 SG 与 ms["sg"] 一致；
  - 晶系角度满足我们的规范（cubic/tetra/ortho/hex/trig）；
  - 所有 u/v/w 的 value ∈ [0,1)（带一点浮点容差）。

输出：
  - <out>.jsonl：干净样本（建议命名为 train.clean.ms.jsonl）
  - <out>.rejects.jsonl：被过滤掉的样本 + 原因计数（便于排查）
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Any, Tuple, List

from tqdm import tqdm
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from dlmcsp.representation.ms_to_cif import ms_to_structure
from dlmcsp.eval.validators import quick_validate_structure


# ---------- 晶系 & 角度检查 ----------

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
) -> bool:
    """
    返回 True 表示角度不一致（应当 reject）。
    """
    def near(x, y, tol=0.5):
        return abs(x - y) <= tol

    if cs == "cubic":
        return not (near(al, 90) and near(be, 90) and near(ga, 90))
    if cs == "tetragonal":
        return not (near(al, 90) and near(be, 90) and near(ga, 90))
    if cs == "orthorhombic":
        return not (near(al, 90) and near(be, 90) and near(ga, 90))
    if cs == "hexagonal":
        return not (near(al, 90) and near(be, 90) and near(ga, 120))
    if cs == "trigonal":
        # 统一按 hex setting 检查（R 群 + 非 R 群）
        return not (near(al, 90) and near(be, 90) and near(ga, 120))
    # monoclinic / triclinic 不强制
    return False


# ---------- 单样本判定逻辑（可被多进程复用） ----------

def _filter_one(rec: Dict[str, Any], vocab_path: str) -> Tuple[bool, str | None, str | None]:
    """
    对单条样本做判定：

    返回：
      keep: 是否保留
      reason_code: 若不保留，对应的简短错误类型（如 'geom_fail', 'sg_mismatch' 等），保留则为 None
      reason_detail: 更详细的文本（用于写 rejects），保留则为 None
    """
    mid = rec.get("material_id")
    sg_gt = int(rec["sg"])

    # 1) ms -> Structure
    try:
        struct = ms_to_structure(rec, vocab_path)
    except Exception as e:
        return False, "ms_to_struct_fail", f"{type(e).__name__}:{str(e)}"

    # 2) quick_validate_structure 几何检查
    ok_geom, why = quick_validate_structure(struct)
    if not ok_geom:
        return False, "geom_fail", why

    # 3) SG 一致性
    try:
        sga = SpacegroupAnalyzer(struct, symprec=1e-2, angle_tolerance=5)
        sg2 = int(sga.get_space_group_number())
    except Exception as e:
        return False, "spglib_fail", f"{type(e).__name__}:{str(e)}"

    if sg2 != sg_gt:
        return False, "sg_mismatch", f"{sg_gt}->{sg2}"

    # 4) 晶系角度检查
    latt = struct.lattice
    a, b, c = latt.a, latt.b, latt.c
    al, be, ga = latt.alpha, latt.beta, latt.gamma
    cs = _crystal_system(sg_gt)
    if _angle_checks(cs, a, b, c, al, be, ga, sg_gt):
        detail = f"cs={cs},angles=({al:.2f},{be:.2f},{ga:.2f})"
        return False, "angle_inconsistent", detail

    # 5) u/v/w ∈ [0,1)
    for site in rec.get("sites", []):
        params = site.get("params", {})
        for k in ("u", "v", "w"):
            node = params.get(k)
            if isinstance(node, dict) and "value" in node:
                v = float(node["value"])
                if not (0.0 - 1e-6 <= v < 1.0 + 1e-6):
                    return False, "uvw_out_of_range", f"{k}={v}"

    # 全部通过
    return True, None, None


# ---------- 多进程封装 ----------

_FILTER_VOCAB_PATH: str | None = None


def _init_worker(vocab_path: str):
    global _FILTER_VOCAB_PATH
    _FILTER_VOCAB_PATH = vocab_path


def _worker_task(rec: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, str | None, str | None]:
    """
    worker 里处理一条记录，返回：
      rec: 原始记录（主进程需要写 clean/reject）
      keep: 是否保留
      reason_code: 错误类型
      reason_detail: 错误详情
    """
    global _FILTER_VOCAB_PATH
    assert _FILTER_VOCAB_PATH is not None
    keep, code, detail = _filter_one(rec, _FILTER_VOCAB_PATH)
    return rec, keep, code, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="原始 ms jsonl（e.g. train.ms.jsonl）")
    ap.add_argument("--vocab", required=True, help="vocab.yaml")
    ap.add_argument("--out", required=True, help="输出 clean jsonl 路径")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条（0=全部）")
    ap.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="并行 worker 数量；0 或 1 表示单进程",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rejects_path = str(out_path) + ".rejects.jsonl"

    # 预读 jsonl
    records: List[Dict[str, Any]] = []
    with open(args.data, "r", encoding="utf-8") as fi:
        for ln in fi:
            if not ln.strip():
                continue
            rec = json.loads(ln)
            records.append(rec)
            if args.limit and len(records) >= args.limit:
                break

    bad_reason = Counter()
    ok_cnt, bad_cnt = 0, 0
    total = len(records)

    # 单进程
    if args.num_workers <= 1:
        with open(args.out, "w", encoding="utf-8") as fo, \
             open(rejects_path, "w", encoding="utf-8") as fr:

            for rec in tqdm(records, desc="filter_clean_ms(single)"):
                keep, code, detail = _filter_one(rec, args.vocab)
                mid = rec.get("material_id")

                if keep:
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    ok_cnt += 1
                else:
                    bad_cnt += 1
                    if code is not None:
                        bad_reason[code] += 1
                        reason_str = code if detail is None else f"{code}:{detail}"
                    else:
                        reason_str = "unknown"
                    fr.write(json.dumps(
                        {"material_id": mid, "reason": reason_str},
                        ensure_ascii=False
                    ) + "\n")

    # 多进程
    else:
        from multiprocessing import Pool

        with open(args.out, "w", encoding="utf-8") as fo, \
             open(rejects_path, "w", encoding="utf-8") as fr, \
             Pool(
                 processes=args.num_workers,
                 initializer=_init_worker,
                 initargs=(args.vocab,),
             ) as pool:

            for rec, keep, code, detail in tqdm(
                pool.imap(_worker_task, records),
                total=len(records),
                desc=f"filter_clean_ms(mp, {args.num_workers} workers)",
            ):
                mid = rec.get("material_id")
                if keep:
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    ok_cnt += 1
                else:
                    bad_cnt += 1
                    if code is not None:
                        bad_reason[code] += 1
                        reason_str = code if detail is None else f"{code}:{detail}"
                    else:
                        reason_str = "unknown"
                    fr.write(json.dumps(
                        {"material_id": mid, "reason": reason_str},
                        ensure_ascii=False
                    ) + "\n")

    print("=== FILTER CLEAN SUMMARY ===")
    print(f"total: {total}, ok={ok_cnt}, bad={bad_cnt}")
    for k, v in bad_reason.most_common():
        print(f"{k}: {v}")
    print(f"[OUT] clean={args.out} rejects={rejects_path}")


if __name__ == "__main__":
    main()
