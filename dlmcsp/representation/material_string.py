# -*- coding: utf-8 -*-
"""
material-string v2 的构造（可逆、鲁棒）：
- 只信 spglib 标准化得到的 SG；
- Wyckoff 自由度优先由 pyxtal 推断；失败则回退到 vocab 的 wy_tokens（保守设 dof=3, u/v/w=True）；
- 对每个等价类生成 (EL, WY, PARAM)；
- PARAM:
    * 使用 vocab.yaml 中的 param_tokens 量化（BASE/FINE）
    * 同时写入 "value" 真值，方便后续连续训练 / 反量化
- 晶格：
    * 使用 vocab.yaml 中的 lattice_bins 量化，同时写入 "value" 真值
"""

from __future__ import annotations
from typing import Dict, Any
from collections import defaultdict

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from dlmcsp.representation.wyckoff_table import WyckoffDB
from dlmcsp.tokenization.vocab_utils import (
    load_vocab_yaml,
    bin_lattice_scalar,
    get_lattice_conf,
    get_param_conf,
    quantize_param_scalar,
)


def _lattice_tokens(std_struct, vocab_yaml: str) -> Dict[str, Dict[str, float]]:
    """
    从结构中抽晶格，按 vocab.yaml 的 lattice_bins 量化，同时带上 "value" 真值。
    """
    vocab = load_vocab_yaml(vocab_yaml)
    latt_conf = get_lattice_conf(vocab)

    a, b, c = std_struct.lattice.abc
    alpha, beta, gamma = std_struct.lattice.angles

    out: Dict[str, Dict[str, float]] = {}
    for name, val in [
        ("a", a),
        ("b", b),
        ("c", c),
        ("alpha", alpha),
        ("beta", beta),
        ("gamma", gamma),
    ]:
        q = bin_lattice_scalar(name, float(val), latt_conf[name])
        q["value"] = float(val)
        out[name] = q
    return out


def _representative_sites(std_struct) -> Dict[int, Dict[str, Any]]:
    """
    使用 spglib 的 symmetry_dataset，将 equivalent_atoms 分组，
    每个等价类选第一个原子作为代表，并记录 element, wy letter, frac coords, multiplicity。
    """
    sga = SpacegroupAnalyzer(std_struct, symprec=1e-2, angle_tolerance=5)
    ds = sga.get_symmetry_dataset()
    eq = ds["equivalent_atoms"]      # shape=(n_atoms,)
    wy = ds["wyckoffs"]              # shape=(n_atoms,) like 'a','b',...
    site_groups = defaultdict(list)
    for i, site in enumerate(std_struct.sites):
        site_groups[int(eq[i])].append(i)

    rep: Dict[int, Dict[str, Any]] = {}
    for cid, indices in site_groups.items():
        j = indices[0]
        site = std_struct.sites[j]
        rep[cid] = {
            "el": site.specie.symbol,
            "wy_letter": str(wy[j]).lower(),        # e.g., 'c'
            "frac": site.frac_coords,               # np.array([u,v,w])
            "multiplicity": len(indices),           # 当前 conventional cell 中等价原子数
        }
    return rep


def to_material_string_v2(std_struct, sgnum: int, vocab_yaml: str) -> Dict[str, Any]:
    """
    输出 ms_dict：
    {
      "formula": ...,
      "n_atoms": ...,
      "sg": ...,
      "latt": {
        a:     {bin, dr, value},
        b:     {bin, dr, value},
        c:     {bin, dr, value},
        alpha: {bin, dr, value},
        beta:  {bin, dr, value},
        gamma: {bin, dr, value},
      },
      "sites": [
        {
          "el": "Ga",
          "wy": "4f",
          "params": {
             "u": {"mode":"BASE"/"FINE", ... , "value": float},
             "v": {...},
             "w": {...}
          },
          "mult_schema": 4,
          "mult_cell": 4,
          "occ": 1.0,
        },
        ...
      ]
    }
    """
    sgnum = int(sgnum)
    vocab = load_vocab_yaml(vocab_yaml)
    param_conf = get_param_conf(vocab)
    wydb = WyckoffDB(sgnum, vocab_yaml=vocab_yaml, try_pyxtal=True)

    ms: Dict[str, Any] = {
        "formula": std_struct.composition.reduced_formula,
        "n_atoms": len(std_struct.sites),
        "sg": sgnum,
        "latt": _lattice_tokens(std_struct, vocab_yaml),
        "sites": [],
    }

    rep = _representative_sites(std_struct)
    # 规范排序：按 multiplicity 降序，再按元素名、wy letter
    items = sorted(
        rep.items(),
        key=lambda kv: (-kv[1]["multiplicity"], kv[1]["el"], kv[1]["wy_letter"]),
    )

    for _, info in items:
        letter = info["wy_letter"]
        schema = wydb.param_schema(letter)  # {"mult","dof","mask"}
        mult_schema = int(schema["mult"])
        mask = dict(schema["mask"])
        wy_str = f"{mult_schema}{letter}"   # 词表里也是 "4f" 这种形式

        u, v, w = [float(x) % 1.0 for x in info["frac"]]
        params: Any = "-"

        if int(schema.get("dof", 0)) > 0:
            params = {}
            if mask.get("u", False):
                q = quantize_param_scalar(u, conf=param_conf)
                q["value"] = u
                params["u"] = q
            if mask.get("v", False):
                q = quantize_param_scalar(v, conf=param_conf)
                q["value"] = v
                params["v"] = q
            if mask.get("w", False):
                q = quantize_param_scalar(w, conf=param_conf)
                q["value"] = w
                params["w"] = q

            # 如果 schema.dof>0 但 mask 全 False，就退化为全开（极少数兜底）
            if not params:
                params = {}
                for axis, val in (("u", u), ("v", v), ("w", w)):
                    q = quantize_param_scalar(val, conf=param_conf)
                    q["value"] = val
                    params[axis] = q

        ms["sites"].append({
            "el": info["el"],
            "wy": wy_str,               # "3c"
            "params": params,           # "-" or dict
            "mult_schema": mult_schema, # schema 的理论多重性
            "mult_cell": info["multiplicity"],  # 当前 cell 里的实际等价数
            "occ": 1.0,                 # 我们目前只处理非部分占位
        })

    return ms
