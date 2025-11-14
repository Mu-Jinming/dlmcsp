# -*- coding: utf-8 -*-
# dlmcsp/constraints/wyckoff_gt.py
from __future__ import annotations
from typing import Dict, List, Tuple
from math import gcd
from functools import reduce

from dlmcsp.representation.wyckoff_table import WyckoffDB

def _gcd_list(xs: List[int]) -> int:
    return reduce(gcd, xs) if xs else 1

def normalize_wy_letters(sg: int, wy_letters: List[str], vocab_yaml: str) -> List[str]:
    """
    规范化 Wy 字符串：支持 "adg" 或 ["2a","2d","4g"] 两种写法。
    返回统一形式：["<mult><letter>", ...]，比如 ["2a","2d","4g"]。
    """
    db = WyckoffDB(sg, vocab_yaml=vocab_yaml, try_pyxtal=True)
    valid = {w.letter: w.mult for w in db.all_sites_sorted()}  # 'a'->mult, 'b'->mult, ...
    out: List[str] = []
    if len(wy_letters) == 1 and wy_letters[0].isalpha():
        # 形如 "adg"
        for ch in wy_letters[0]:
            if ch not in valid:
                raise ValueError(f"Invalid Wy letter '{ch}' for SG {sg}")
            out.append(f"{valid[ch]}{ch}")
        return out
    # 形如 ["2a","2d","4g"] 或混合 ["a","4g"]
    for w in wy_letters:
        w = w.strip()
        if not w:
            continue
        if w[0].isdigit():
            # "4f" 形式
            mult = ""
            i = 0
            while i < len(w) and w[i].isdigit():
                mult += w[i]; i += 1
            letter = w[i:]
            if letter not in valid:
                raise ValueError(f"Invalid Wy letter '{letter}' for SG {sg}")
            if int(mult) != valid[letter]:
                # 容忍用户写错倍数？这里严苛一些，防止计数错位
                raise ValueError(f"Wy multiplicity mismatch: '{w}' vs SG {sg} expects {valid[letter]}{letter}")
            out.append(w)
        else:
            # "f" 形式
            letter = w
            if letter not in valid:
                raise ValueError(f"Invalid Wy letter '{letter}' for SG {sg}")
            out.append(f"{valid[letter]}{letter}")
    return out

def infer_Z_and_natoms(
    sg: int, wy_norm: List[str], elements: List[str], formula_counts: Dict[str, int], vocab_yaml: str
) -> Tuple[int, int, Dict[str, int]]:
    """
    根据已知 Wy 模板与元素分配，计算 Z 与 natoms，并校验与组分一致。
    elements 的长度需要与 wy_norm 长度一致（一个 Wy 条目对应一个元素种类）。
    """
    if len(wy_norm) != len(elements):
        raise ValueError(f"length mismatch: wyckoff_letters({len(wy_norm)}) vs elements({len(elements)})")
    db = WyckoffDB(sg, vocab_yaml=vocab_yaml, try_pyxtal=True)
    mult_map = {f"{s.mult}{s.letter}": s.mult for s in db.all_sites_sorted()}
    # 统计每个元素总原子数（由该元素所占的 Wy 多重性相加）
    totals: Dict[str, int] = {}
    for wy, el in zip(wy_norm, elements):
        mult = mult_map.get(wy)
        if mult is None:
            raise ValueError(f"Unknown Wy '{wy}' for SG {sg}")
        totals[el] = totals.get(el, 0) + mult
    # 由 totals 与 formula_counts 推出 Z（要求 totals[el] = Z * formula_counts[el]）
    Z_vals = []
    for el, n_base in formula_counts.items():
        if n_base <= 0:
            continue
        if el not in totals:
            raise ValueError(f"Element '{el}' absent in wy assignment")
        t = totals[el]
        if t % n_base != 0:
            raise ValueError(f"Z not integer: totals[{el}]={t}, base={n_base}")
        Z_vals.append(t // n_base)
    if not Z_vals:
        raise ValueError("empty composition?")
    Z = Z_vals[0]
    for z in Z_vals[1:]:
        if z != Z:
            raise ValueError(f"inconsistent Z across elements: {Z_vals}")
    natoms = sum(totals.values())
    return Z, natoms, totals

class WyAssign:
    __slots__ = ("wy", "element")
    def __init__(self, wy: str, element: str):
        self.wy = wy
        self.element = element

def build_plan_from_gt(
    sg: int, wy_letters: List[str], elements: List[str],
    formula_counts: Dict[str, int], vocab_yaml: str
) -> Tuple[int, int, List[WyAssign]]:
    """
    输出 (Z, natoms, plan)，plan 用于 build_ms_skeleton：[(wy, element), ...]
    """
    wy_norm = normalize_wy_letters(sg, wy_letters, vocab_yaml)
    Z, natoms, _ = infer_Z_and_natoms(sg, wy_norm, elements, formula_counts, vocab_yaml)
    plan = [WyAssign(wy=w, element=e) for w, e in zip(wy_norm, elements)]
    return Z, natoms, plan
