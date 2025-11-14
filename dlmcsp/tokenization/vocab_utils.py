# -*- coding: utf-8 -*-
"""
数值离散化与反量化工具（纯新接口版）：
- 晶格常数 / 角度：支持 μ-law / log / linear 分桶 + 残差
- Wyckoff 参数 u/v/w：基分数集合 + 均匀细网格 (denom 从 vocab.yaml 读)
- 统一从 vocab.yaml 读取 lattice_bins / param_tokens 配置

接口约定（只保留新接口）：
- 晶格：
    bin_lattice_scalar(name, x, conf) -> {"bin": int, "dr": int}
    inv_bin_lattice_scalar(bin_idx, dr, conf) -> float
- Wyckoff 参数：
    pc = get_param_conf(vocab_dict)
    quantize_param_scalar(x, conf=pc) -> {"mode":"BASE"/"FINE", ...}
    inv_quantize_param(tok, conf=pc) -> float ∈ [0,1)
"""

from __future__ import annotations
import math
from typing import Dict, Tuple, Any, List

import yaml


# -----------------------------------------------------------------------------
# YAML 读写
# -----------------------------------------------------------------------------

def load_vocab_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_vocab_yaml(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


# -----------------------------------------------------------------------------
# 晶格量化
# -----------------------------------------------------------------------------

def _mu_law_encode(x: float, x_min: float, x_max: float, mu: float = 255.0) -> float:
    x = max(x_min, min(x, x_max))
    t = (2.0 * (x - x_min) / (x_max - x_min)) - 1.0
    t = max(-1.0, min(1.0, t))
    y = math.copysign(math.log1p(mu * abs(t)) / math.log1p(mu), t)
    return 0.5 * (y + 1.0)


def _mu_law_decode(ratio: float, x_min: float, x_max: float, mu: float = 255.0) -> float:
    y = 2.0 * ratio - 1.0
    t = math.copysign((math.expm1(abs(y) * math.log1p(mu)) / mu), y)
    x = (t + 1.0) * 0.5 * (x_max - x_min) + x_min
    return x


def _linear_encode(x: float, x_min: float, x_max: float) -> float:
    x = max(x_min, min(x, x_max))
    return (x - x_min) / (x_max - x_min)


def _linear_decode(ratio: float, x_min: float, x_max: float) -> float:
    return ratio * (x_max - x_min) + x_min


def _log_encode(x: float, x_min: float, x_max: float) -> float:
    x = max(x_min, min(x, x_max))
    if x_min <= 0:
        raise ValueError("log 编码要求 x_min>0")
    lx = math.log(x)
    lmin, lmax = math.log(x_min), math.log(x_max)
    return (lx - lmin) / (lmax - lmin)


def _log_decode(ratio: float, x_min: float, x_max: float) -> float:
    if x_min <= 0:
        raise ValueError("log 解码要求 x_min>0")
    lmin, lmax = math.log(x_min), math.log(x_max)
    lx = ratio * (lmax - lmin) + lmin
    return math.exp(lx)


def _encode_ratio(x: float, conf: Dict[str, Any]) -> float:
    scale = conf.get("scale", "mu")
    x_min, x_max = float(conf["range"][0]), float(conf["range"][1])
    if scale == "mu":
        return _mu_law_encode(x, x_min, x_max)
    elif scale == "log":
        return _log_encode(x, x_min, x_max)
    elif scale == "linear":
        return _linear_encode(x, x_min, x_max)
    else:
        raise ValueError(f"未知 scale: {scale}")


def _decode_ratio(idx: int, num_bins: int, conf: Dict[str, Any]) -> float:
    # 取 bin 中心
    ratio = (idx + 0.5) / float(num_bins)
    scale = conf.get("scale", "mu")
    x_min, x_max = float(conf["range"][0]), float(conf["range"][1])
    if scale == "mu":
        return _mu_law_decode(ratio, x_min, x_max)
    elif scale == "log":
        return _log_decode(ratio, x_min, x_max)
    elif scale == "linear":
        return _linear_decode(ratio, x_min, x_max)
    else:
        raise ValueError(f"未知 scale: {scale}")


def bin_lattice_scalar(name: str, x: float, conf: Dict[str, Any]) -> Dict[str, int]:
    """
    按 vocab.yaml 中对应维度的 lattice_bins 量化一个标量。
    返回 {"bin": idx, "dr": dr}，当前 dr 统一为 0 或中间值。
    """
    num_bins = int(conf["num_bins"])
    residuals = list(conf.get("residuals", [-2, -1, 0, 1, 2]))
    ratio = _encode_ratio(x, conf)
    idx = int(round(ratio * (num_bins - 1)))
    idx = max(0, min(num_bins - 1, idx))

    # 残差目前不真用，统一写 0 或中间值（但接口保留）
    if 0 in residuals:
        dr = 0
    else:
        dr = residuals[len(residuals) // 2]

    return {"bin": idx, "dr": dr}


def inv_bin_lattice_scalar(bin_idx: int, dr: int, conf: Dict[str, Any]) -> float:
    """
    反量化晶格标量：根据 bin index + conf 解码为实数。
    当前忽略 dr，仅返回 bin 中心。
    """
    num_bins = int(conf["num_bins"])
    x_center = _decode_ratio(int(bin_idx), num_bins, conf)
    return x_center


def get_lattice_conf(vocab: Dict[str, Any]) -> Dict[str, Any]:
    """从 vocab.yaml 的解析 dict 中取出 lattice_bins 配置。"""
    return vocab.get("lattice_bins", {})


# -----------------------------------------------------------------------------
# Wyckoff 参数量化（只保留 conf 接口）
# -----------------------------------------------------------------------------

_BASE_FRACTIONS: List[str] = [
    "0", "1/2", "1/3", "2/3",
    "1/4", "3/4",
    "1/6", "5/6",
    "1/8", "3/8", "5/8", "7/8",
]


def _frac_to_float(s: str) -> float:
    if s == "0":
        return 0.0
    a, b = s.split("/")
    return (float(a) / float(b)) % 1.0


def _nearest_base_fraction(x: float, base_list: List[str]) -> Tuple[str, float]:
    x = x % 1.0
    best = base_list[0]
    best_err = 10.0
    for s in base_list:
        v = _frac_to_float(s)
        # 在周期 [0,1) 上的距离
        dx = abs(((x - v + 0.5) % 1.0) - 0.5)
        if dx < best_err:
            best_err = dx
            best = s
    return best, best_err


def get_param_conf(vocab: Dict[str, Any]) -> Dict[str, Any]:
    """
    从 vocab 里解析 param_tokens：
      param_tokens:
        base: ["0","1/2",...]
        fine: {denom: 96}
        dr: [-1,0,1]

    返回标准化后的配置：
      {
        "base": [...],
        "fine": {"denom": int},
        "dr": [...]
      }
    """
    pc_raw = vocab.get("param_tokens", {})
    base = list(pc_raw.get("base", _BASE_FRACTIONS))
    fine = dict(pc_raw.get("fine", {}))
    denom = int(fine.get("denom", 96))
    fine["denom"] = denom
    dr = list(pc_raw.get("dr", [-1, 0, 1]))
    return {"base": base, "fine": fine, "dr": dr}


def quantize_param_scalar(x: float, conf: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wyckoff 自由度离散化（只支持新接口）：

      pc = get_param_conf(vocab)
      quantize_param_scalar(x, conf=pc)

    逻辑：
      - 先尝试贴最近的 base 分数，若误差 < 0.5 / denom，则用 BASE
      - 否则用 FINE idx，均匀分成 denom 份，idx ∈ [0, denom-1]
    """
    if conf is None:
        raise ValueError("quantize_param_scalar 需要 conf 参数（来自 get_param_conf）。")

    x = x % 1.0
    base_list = list(conf["base"])
    denom = int(conf["fine"]["denom"])

    base, err = _nearest_base_fraction(x, base_list)
    if err < 1.0 / (2 * denom):
        return {"mode": "BASE", "base": base}

    idx = int(math.floor(x * denom)) % denom
    return {"mode": "FINE", "idx": idx, "dr": 0}


def frac_str_to_float(s: str) -> float:
    """公开版本，供其它模块直接使用。"""
    return _frac_to_float(s)


def inv_quantize_param(tok: Dict[str, Any], conf: Dict[str, Any]) -> float:
    """
    反量化 Wyckoff 参数：

      pc = get_param_conf(vocab)
      inv_quantize_param(tok, conf=pc)

    BASE: 返回基分数对应的小数
    FINE: 返回 (idx+0.5)/denom 的 bin 中心
    """
    if conf is None:
        raise ValueError("inv_quantize_param 需要 conf 参数（来自 get_param_conf）。")

    if tok["mode"] == "BASE":
        return _frac_to_float(tok["base"])

    denom = int(conf["fine"]["denom"])
    idx = int(tok["idx"])
    return ((idx + 0.5) / float(denom)) % 1.0
