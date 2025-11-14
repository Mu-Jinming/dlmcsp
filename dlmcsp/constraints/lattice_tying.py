# -*- coding: utf-8 -*-
# dlmcsp/constraints/lattice_tying.py
from __future__ import annotations
from typing import Dict, List, Tuple
import math

from dlmcsp.tokenization.vocab_utils import load_vocab_yaml, inv_bin_lattice_scalar

# --- SG -> 晶系 ---
def crystal_system_from_sg(sg: int) -> str:
    # 1 triclinic, 2 monoclinic, 3-15 monoclinic?（标准：1-2 triclinic, 3-15 monoclinic）
    if sg in (1, 2):
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
    return "triclinic"

def _nearest_bin_for_value(name: str, value: float, cfg: Dict) -> int:
    nb = int(cfg[name]["num_bins"])
    best, be = 0, 1e9
    for i in range(nb):
        v = inv_bin_lattice_scalar(i, 0, cfg[name])
        e = abs(v - value)
        if e < be:
            be, best = e, i
    return best

def lattice_tie_plan(sg: int, vocab_yaml: str) -> Dict[str, Dict]:
    """
    返回 tie 方案：
    - 'masters': 需要模型预测的量（顺序靠前）
    - 'slaves': 由 master 复制或固定到常数（给出固定目标的 bin）
    """
    sys = crystal_system_from_sg(sg)
    cfg = load_vocab_yaml(vocab_yaml)["lattice_bins"]

    # 把变量表述成 ("a","b","c","alpha","beta","gamma") 的 bin，DR 我们固定 0（先求稳）
    plan = {"masters": [], "slaves": []}

    if sys == "cubic":
        plan["masters"] = ["a"]
        # b,c 绑定 a；角全 90
        plan["slaves"] = [("b", "a"), ("c", "a"),
                          ("alpha", ("fix", _nearest_bin_for_value("alpha", 90.0, cfg))),
                          ("beta",  ("fix", _nearest_bin_for_value("beta",  90.0, cfg))),
                          ("gamma", ("fix", _nearest_bin_for_value("gamma", 90.0, cfg)))]
    elif sys == "tetragonal":
        plan["masters"] = ["a", "c"]
        plan["slaves"] = [("b", "a"),
                          ("alpha", ("fix", _nearest_bin_for_value("alpha", 90.0, cfg))),
                          ("beta",  ("fix", _nearest_bin_for_value("beta",  90.0, cfg))),
                          ("gamma", ("fix", _nearest_bin_for_value("gamma", 90.0, cfg)))]
    elif sys == "hexagonal":
        plan["masters"] = ["a", "c"]
        plan["slaves"] = [("b", "a"),
                          ("alpha", ("fix", _nearest_bin_for_value("alpha", 90.0, cfg))),
                          ("beta",  ("fix", _nearest_bin_for_value("beta",  90.0, cfg))),
                          ("gamma", ("fix", _nearest_bin_for_value("gamma", 120.0, cfg)))]
    elif sys == "trigonal":
        # 用六角设置更稳（a=b, gamma=120），角(alpha,beta)=90 只是六角胞写法；真正菱方设置 alpha=beta=gamma!=90
        plan["masters"] = ["a", "c"]
        plan["slaves"] = [("b", "a"),
                          ("alpha", ("fix", _nearest_bin_for_value("alpha", 90.0, cfg))),
                          ("beta",  ("fix", _nearest_bin_for_value("beta",  90.0, cfg))),
                          ("gamma", ("fix", _nearest_bin_for_value("gamma", 120.0, cfg)))]
    elif sys == "orthorhombic":
        plan["masters"] = ["a", "b", "c"]
        plan["slaves"] = [("alpha", ("fix", _nearest_bin_for_value("alpha", 90.0, cfg))),
                          ("beta",  ("fix", _nearest_bin_for_value("beta",  90.0, cfg))),
                          ("gamma", ("fix", _nearest_bin_for_value("gamma", 90.0, cfg)))]
    elif sys == "monoclinic":
        plan["masters"] = ["a", "b", "c", "beta"]  # 约定 b 为唯一轴
        plan["slaves"] = [("alpha", ("fix", _nearest_bin_for_value("alpha", 90.0, cfg))),
                          ("gamma", ("fix", _nearest_bin_for_value("gamma", 90.0, cfg)))]
    else:  # triclinic
        plan["masters"] = ["a", "b", "c", "alpha", "beta", "gamma"]
        plan["slaves"] = []
    return plan
