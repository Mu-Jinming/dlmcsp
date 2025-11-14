# -*- coding: utf-8 -*-
"""
material-string v2 -> Structure / CIF

核心原则：
- 优先使用 ms 中的 value 真值；没有 value 时才用 token 反量化；
- Wyckoff 展开全部走 WyckoffDB.expand_positions(letter, params)；
- 不在这里做任何 CIF round-trip 审计逻辑；
- R 群在 preprocess 阶段已经统一成我们想要的 cell（hex/rhombo），这里只负责按 ms 恢复。

提供两个主入口：
- ms_to_structure(ms, vocab_yaml) -> pymatgen.Structure
- ms_to_cif(ms, vocab_yaml) -> CIF 字符串（内部先调用 ms_to_structure）
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple

import numpy as np
from pymatgen.core import Lattice, Structure, Element
from pymatgen.io.cif import CifWriter

from dlmcsp.representation.wyckoff_table import WyckoffDB
from dlmcsp.tokenization.vocab_utils import (
    load_vocab_yaml,
    get_lattice_conf,
    get_param_conf,
    inv_bin_lattice_scalar,
    inv_quantize_param,
)


def _get_lattice(ms: Dict[str, Any], vocab_yaml: str) -> Lattice:
    """
    从 ms["latt"] 构造 Lattice：
    - 如果该维度有 "value"，优先用 value；
    - 否则用 (bin, dr) + vocab 配置做反量化。
    """
    vocab = load_vocab_yaml(vocab_yaml)
    latt_conf = get_lattice_conf(vocab)
    L = ms["latt"]

    def val_or_bin(name: str) -> float:
        node = L[name]
        if isinstance(node, dict) and "value" in node:
            return float(node["value"])
        # 兼容旧格式：只存 bin/dr
        return float(
            inv_bin_lattice_scalar(
                int(node["bin"]),
                int(node.get("dr", 0)),
                latt_conf[name],
            )
        )

    a = val_or_bin("a")
    b = val_or_bin("b")
    c = val_or_bin("c")
    alpha = val_or_bin("alpha")
    beta = val_or_bin("beta")
    gamma = val_or_bin("gamma")

    return Lattice.from_parameters(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)


def _params_value_or_token(
    p: Dict[str, Any] | str,
    param_conf: Dict[str, Any],
) -> Dict[str, float]:
    """
    从 site["params"] 中拿 u/v/w：
    - 如果节点里有 "value"，直接用；
    - 否则用 inv_quantize_param(node, conf=param_conf) 反量化。
    返回 { "u": float, "v": float, "w": float }，都已经 mod 1.
    """
    if p == "-" or p is None:
        return {}

    out: Dict[str, float] = {}
    for k in ("u", "v", "w"):
        node = p.get(k)
        if node is None:
            continue

        if isinstance(node, dict) and "value" in node:
            out[k] = float(node["value"]) % 1.0
        else:
            # 新接口：inv_quantize_param(tok, conf=param_conf)
            out[k] = float(inv_quantize_param(node, conf=param_conf)) % 1.0
    return out


def ms_to_structure(ms: Dict[str, Any], vocab_yaml: str) -> Structure:
    """
    核心接口：material-string v2 -> pymatgen.Structure

    不经过 CIF，直接构建：
      1) lattice: _get_lattice
      2) WyckoffDB(sg).expand_positions(letter, params)
      3) 把展开后的分数坐标去重（避免数值噪声导致重复点）

    注意：
      - 不使用 ms["sites"][*]["occ"] 来倍增原子数，occupancy 一律视作 1；
      - 这是训练/审计/采样内部用的“干净结构”。
    """
    sg = int(ms["sg"])
    vocab = load_vocab_yaml(vocab_yaml)
    param_conf = get_param_conf(vocab)
    wydb = WyckoffDB(sg, vocab_yaml=vocab_yaml, try_pyxtal=True)

    lattice = _get_lattice(ms, vocab_yaml)
    species: List[str] = []
    coords: List[Tuple[float, float, float]] = []

    for site in ms["sites"]:
        el = str(site["el"])
        wy = str(site["wy"])  # e.g. "4f"
        # 解析 multiplicity / letter（主要用 letter 去查 WyckoffDB）
        mult_str = "".join(ch for ch in wy if ch.isdigit())
        letter = "".join(ch for ch in wy if ch.isalpha()).lower()
        _ = int(mult_str) if mult_str else None  # 当前不强依赖 mult

        params = _params_value_or_token(site.get("params", {}), param_conf)

        # Wyckoff 展开：letter + params -> 等价的分数坐标
        ex = wydb.expand_positions(letter, params)  # List[Tuple[x,y,z]]
        if not isinstance(ex, list) or len(ex) == 0:
            raise RuntimeError(f"Wyckoff expand failed for wy={wy} params={params}")

        # 去重：有些群/setting 差异会导致等价点重复（在数值误差范围内）
        uniq: List[Tuple[float, float, float]] = []
        seen = set()
        for (x, y, z) in ex:
            u = (round(x % 1.0, 8), round(y % 1.0, 8), round(z % 1.0, 8))
            if u not in seen:
                seen.add(u)
                uniq.append((u[0], u[1], u[2]))

        for (x, y, z) in uniq:
            species.append(el)
            coords.append((x % 1.0, y % 1.0, z % 1.0))

    struct = Structure(lattice, [Element(s) for s in species], coords, to_unit_cell=True)
    return struct


def ms_to_cif(ms: Dict[str, Any], vocab_yaml: str) -> str:
    """
    material-string v2 -> CIF 字符串。

    内部先构造 Structure，再交给 pymatgen.CifWriter：
    - 不在这里做任何对称性检查；
    - 只作为对外导出 / 评测用的中间格式。
    """
    struct = ms_to_structure(ms, vocab_yaml)
    cif = CifWriter(struct).__str__()
    return cif
