# -*- coding: utf-8 -*-
"""
读取 MP CSV → 标准化（conventional 优先）→ material-string v2 → tokens → jsonl

规范（已钉死，除非你显式改 CLI）：
  - 统一使用 conventional cell（非原胞）
  - R 群(146–167) 强制 HEX setting（α=β≈90°, γ≈120°）
  - 单斜默认为 b-unique（沿用 pymatgen/spglib 常规）
  - 角度单位：度；长度单位：Å
  - material-string v2 中带 value 真值（lattice 与 u/v/w）

输出：
  - <out>.jsonl：每行一个样本
  - <out>.rejects.jsonl：失败样本与原因
  - manifest.json：记录本次预处理的关键决策/容差，保证可复现

新增：
  - 支持多进程预处理：--num_workers N
    * N=1: 单进程（行为与旧版一致）
    * N<=0: 自动取 min(8, cpu_count())
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple, List
from multiprocessing import Pool, cpu_count

import pandas as pd
from tqdm import tqdm
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from dlmcsp.representation.material_string import to_material_string_v2
from dlmcsp.tokenization.tokenizer import MaterialTokenizer
from dlmcsp.eval.validators import quick_validate_structure

# 统一的 spglib 容差（写入 manifest，训练/采样都应打印出来，避免漂移）
SYMPREC: float = 1e-2
ANGLE_TOL: float = 5.0

# === worker 全局状态（通过 initializer 填充） ===
_G_TOKENIZER: MaterialTokenizer | None = None
_G_VOCAB_PATH: str | None = None
_G_R_SETTING: str = "hex"


def _read_mp_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, engine="python")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def _pick_cif(row: Dict[str, Any]) -> str:
    """优先选 'cif'，其次 'cif.conv'，否则空字符串。"""
    c = row.get("cif")
    if isinstance(c, str) and len(c) > 10:
        return c
    cc = row.get("cif.conv")
    if isinstance(cc, str) and len(cc) > 10:
        return cc
    return ""


def _is_rhombohedral(sgnum: int) -> bool:
    """R（三方菱方）晶系号段：146–167"""
    return 146 <= int(sgnum) <= 167


def _standardize_conventional_hexR(cif_text: str, r_setting: str) -> Tuple[Structure, int, Dict[str, Any]]:
    """
    统一 conventional；R 群根据 r_setting='hex'/'rhombo' 选择 HEX 常规胞或 rhombo 原胞。
    其他群一律 conventional。
    返回：std 结构、标准化后的 sgnum、以及记录到 meta 的信息。
    """
    meta: Dict[str, Any] = {}
    s = Structure.from_str(cif_text, fmt="cif")
    sga0 = SpacegroupAnalyzer(s, symprec=SYMPREC, angle_tolerance=ANGLE_TOL)
    sg_in = int(sga0.get_space_group_number())

    if _is_rhombohedral(sg_in):
        if r_setting == "hex":
            std = sga0.get_conventional_standard_structure()
            meta["r_setting"] = "hex"
        elif r_setting == "rhombo":
            std = sga0.get_primitive_standard_structure()
            meta["r_setting"] = "rhombo"
        else:
            # 默认走 HEX，更贴合大多数 Wy 模板/约束与我们的词表
            std = sga0.get_conventional_standard_structure()
            meta["r_setting"] = "hex(default)"
    else:
        # 非 R 群：强制 conventional
        std = sga0.get_conventional_standard_structure()
        meta["r_setting"] = None

    sg_out = int(SpacegroupAnalyzer(std, symprec=SYMPREC, angle_tolerance=ANGLE_TOL).get_space_group_number())

    meta["sgnum_in"] = sg_in
    meta["sgnum_out"] = sg_out
    meta["symprec"] = SYMPREC
    meta["angle_tol"] = ANGLE_TOL
    meta["cell_type"] = "conventional"  # 明确记录
    meta["monoclinic_unique_axis"] = "b"  # pymatgen 默认

    return std, sg_out, meta


def process_row(
    row: Dict[str, Any],
    tokenizer: MaterialTokenizer,
    vocab_path: str,
    r_setting: str,
) -> Tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
    """
    单样本处理：选 CIF → 标准化 → 校验 → material-string v2（含 value）→ token → 组装字典
    失败则返回 None 与错误原因（用于写入 rejects）
    """
    cif_text = _pick_cif(row)
    if not cif_text:
        return None, {"reason": "empty_cif"}

    try:
        std, sgnum, meta = _standardize_conventional_hexR(cif_text, r_setting=r_setting)
    except Exception as e:
        return None, {"reason": f"std_fail:{type(e).__name__}:{str(e)}"}

    ok, why = quick_validate_structure(std)
    if not ok:
        return None, {"reason": f"geom_fail:{why}"}

    try:
        # 注意：to_material_string_v2 应已写入 latt[*]["value"] 与 sites[*].params[*]["value"]
        ms = to_material_string_v2(std, sgnum, vocab_path)
    except Exception as e:
        return None, {"reason": f"ms_fail:{type(e).__name__}:{str(e)}"}

    try:
        tokens = tokenizer.encode(ms)
    except Exception as e:
        return None, {"reason": f"tok_fail:{type(e).__name__}:{str(e)}"}

    item = {
        "material_id": row.get("material_id"),
        "formula": ms["formula"],
        "n_atoms": ms["n_atoms"],
        "sg": ms["sg"],
        "latt": ms["latt"],   # value: a/b/c (Å), alpha/beta/gamma (deg)
        "sites": ms["sites"], # value: u/v/w ∈ [0,1)
        "tokens": tokens,
        "meta": meta,
    }
    return item, None


# === 多进程相关 ===

def _init_worker(vocab_path: str, r_setting: str):
    """
    在每个 worker 进程里初始化 tokenizer / 配置，避免每条样本重复构建。
    """
    global _G_TOKENIZER, _G_VOCAB_PATH, _G_R_SETTING
    _G_VOCAB_PATH = vocab_path
    _G_R_SETTING = r_setting
    _G_TOKENIZER = MaterialTokenizer.from_yaml(vocab_path)


def _process_row_worker(row: Dict[str, Any]) -> Tuple[Dict[str, Any] | None, Dict[str, Any] | None, Any]:
    """
    worker 入口：从全局拿 tokenizer / vocab_path / r_setting。
    返回 (item, err, material_id)，方便主进程写 rejects。
    """
    global _G_TOKENIZER, _G_VOCAB_PATH, _G_R_SETTING
    assert _G_TOKENIZER is not None and _G_VOCAB_PATH is not None
    item, err = process_row(
        row,
        tokenizer=_G_TOKENIZER,
        vocab_path=_G_VOCAB_PATH,
        r_setting=_G_R_SETTING,
    )
    mid = row.get("material_id")
    return item, err, mid


def _write_manifest(out_path: str, vocab_path: str, csv_path: str, r_setting: str, ok_cnt: int, bad_cnt: int):
    manifest = {
        "source_csv": os.path.abspath(csv_path),
        "output_jsonl": os.path.abspath(out_path),
        "vocab_yaml": os.path.abspath(vocab_path),
        "policy": {
            "cell": "conventional",
            "rhombohedral_setting": r_setting,
            "monoclinic_unique_axis": "b",
        },
        "spglib": {
            "symprec": SYMPREC,
            "angle_tolerance": ANGLE_TOL,
        },
        "counts": {"ok": ok_cnt, "bad": bad_cnt},
    }
    man_path = os.path.join(os.path.dirname(out_path), "manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[MANIFEST] saved to {man_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="MP-20 等数据的 CSV 路径")
    ap.add_argument("--out", required=True, help="输出 jsonl 路径")
    ap.add_argument("--vocab", required=True, help="vocab.yaml 路径")
    ap.add_argument(
        "--setting",
        type=str,
        default="hex",
        choices=["hex", "rhombo"],
        help="R 群 cell setting：hex=六方常规胞，rhombo=菱方原胞（不建议）",
    )
    ap.add_argument("--limit", type=int, default=0, help="可选，仅处理前 N 条")
    ap.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="worker 进程数；1=单进程；<=0 自动取 min(8, cpu_count())",
    )
    args = ap.parse_args()

    df = _read_mp_csv(args.csv)
    rows: List[Dict[str, Any]] = df.to_dict("records")
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rejects_path = args.out + ".rejects.jsonl"

    # 计算实际使用的 worker 数量
    if args.num_workers is None or args.num_workers <= 0:
        n_workers = max(1, min(8, cpu_count() or 1))
    else:
        n_workers = max(1, args.num_workers)

    print(f"[INFO] preprocess_mp20 using {n_workers} worker(s)")

    ok_cnt, bad_cnt = 0, 0

    if n_workers == 1:
        # === 单进程模式（与旧版行为一致） ===
        tokenizer = MaterialTokenizer.from_yaml(args.vocab)
        with open(args.out, "w", encoding="utf-8") as fo, open(rejects_path, "w", encoding="utf-8") as fr:
            for r in tqdm(rows, desc="mp→material-string v2 (conventional + R=HEX, single)"):
                item, err = process_row(
                    r,
                    tokenizer=tokenizer,
                    vocab_path=args.vocab,
                    r_setting=args.setting,
                )
                if item is not None:
                    fo.write(json.dumps(item, ensure_ascii=False) + "\n")
                    ok_cnt += 1
                else:
                    rec = {"material_id": r.get("material_id")}
                    if err:
                        rec.update(err)
                    fr.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    bad_cnt += 1
    else:
        # === 多进程模式 ===
        with open(args.out, "w", encoding="utf-8") as fo, open(rejects_path, "w", encoding="utf-8") as fr:
            with Pool(
                processes=n_workers,
                initializer=_init_worker,
                initargs=(args.vocab, args.setting),
            ) as pool:
                for item, err, mid in tqdm(
                    pool.imap_unordered(_process_row_worker, rows),
                    total=len(rows),
                    desc=f"mp→material-string v2 (conventional + R=HEX, workers={n_workers})",
                ):
                    if item is not None:
                        fo.write(json.dumps(item, ensure_ascii=False) + "\n")
                        ok_cnt += 1
                    else:
                        rec = {"material_id": mid}
                        if err:
                            rec.update(err)
                        fr.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        bad_cnt += 1

    total = ok_cnt + bad_cnt
    rate = ok_cnt / total if total > 0 else 0.0
    print(f"[DONE] ok={ok_cnt} bad={bad_cnt} rate={rate:.3f} out={args.out} rejects={rejects_path}")

    _write_manifest(
        out_path=args.out,
        vocab_path=args.vocab,
        csv_path=args.csv,
        r_setting=args.setting,
        ok_cnt=ok_cnt,
        bad_cnt=bad_cnt,
    )


if __name__ == "__main__":
    main()
