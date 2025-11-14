# -*- coding: utf-8 -*-
"""
简单检查 material-string v2 -> CIF -> Structure 的闭环
"""
from __future__ import annotations
import json
from pathlib import Path

from pymatgen.core import Structure
from dlmcsp.representation.ms_to_cif import ms_to_cif
from dlmcsp.eval.validators import quick_validate_structure

MS_PATH = Path("dlmcsp/data/mp_20/train.ms.jsonl")
VOCAB = "configs/vocab.yaml"


def main():
    # 读第一条记录
    with MS_PATH.open("r", encoding="utf-8") as f:
        first = json.loads(next(f))

    mid = first.get("material_id")
    print(f"=== first record material_id: {mid}")

    # 1) ms -> CIF（字符串）
    cif = ms_to_cif(first, VOCAB)
    print("CIF built:")
    print(cif)

    # 2) CIF -> Structure
    struct = Structure.from_str(cif, fmt="cif")

    # 3) 快速几何合法性检查
    ok, why = quick_validate_structure(struct)
    print("quick_validate_structure:", ok, why)


if __name__ == "__main__":
    main()
