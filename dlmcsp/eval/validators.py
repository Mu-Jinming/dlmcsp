# -*- coding: utf-8 -*-
"""
轻量校验器：
- quick_validate_structure: 几何快速过滤（最小原子间距、体积/密度阈）
- sg_number: 统一用 spglib 获取 SG 国际号
"""
from __future__ import annotations
from typing import Tuple
import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def sg_number(struct: Structure) -> int:
    sga = SpacegroupAnalyzer(struct, symprec=1e-2, angle_tolerance=5)
    return int(sga.get_space_group_number())


def quick_validate_structure(struct: Structure,
                             min_distance: float = 0.9,
                             min_volume: float = 5.0) -> Tuple[bool, str]:
    """
    极简快速检查，避免明显畸形结构进入训练/采样环节：
    - 最小原子间距 < min_distance → 拒绝
    - 晶胞体积 < min_volume → 拒绝
    """
    vol = float(struct.lattice.volume)
    if not np.isfinite(vol) or vol < min_volume:
        return False, "volume_too_small"

    # 最小距离检查：O(N^2) 对 MP-20 级别可接受
    coords = np.array([s.frac_coords for s in struct.sites], dtype=float)
    cart = np.dot(coords, struct.lattice.matrix)  # frac→cart
    n = cart.shape[0]
    mind = 1e9
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(cart[i] - cart[j])
            if d < mind:
                mind = d
                if mind < min_distance:
                    return False, "min_distance_violation"
    return True, "ok"
