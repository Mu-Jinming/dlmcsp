# -*- coding: utf-8 -*-
"""
动态硬掩码：
- 给定当前位置类型（SG/WY/EL/LATT_x/参数U/V/W），屏蔽非法 token
- 依赖 Vocab 与 WyckoffDB
注意：这里不直接生成 token 序列，只提供“这一位置允许哪些 token”的 0/1 掩码。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List

import numpy as np

from dlmcsp.representation.wyckoff_table import WyckoffDB
from dlmcsp.tokenization.tokenizer import Vocab


@dataclass
class ConstraintState:
    sg: int
    natoms: int
    elements_left: Dict[str, int]
    phase: int = 1
    # 已选模板（Phase-1 的结果）
    template: List[Dict[str, Any]] = field(default_factory=list)  # [{"wy":"3c","el":"O"}...]
    # 晶格与参数的部分结果
    lattice_partial: Dict[str, Dict[str, int]] = field(default_factory=dict)  # name -> {"bin":i,"dr":j}
    params_partial: Dict[int, Dict[str, Any]] = field(default_factory=dict)   # site_id -> {"u":..., "v":..., "w":...}


def build_token_mask(position: str, state: ConstraintState, vocab: Vocab, vocab_yaml: str) -> np.ndarray:
    """
    position: 当前要预测的字段（'SG','WY','EL','LATT_a'...'ALPHA'...,'U_BASE','U_FINE','U_DR',...）
    返回：长度=|vocab| 的 0/1 掩码向量
    """
    size = len(vocab._id2tok)
    mask = np.zeros(size, dtype=np.float32)

    if position == "SG":
        # 全部 SG token 开放（若需晶系过滤，可在此屏蔽）
        for tid in vocab.sg_tokens:
            mask[tid] = 1.0
        return mask

    wydb = WyckoffDB(state.sg, vocab_yaml=vocab_yaml, try_pyxtal=True)

    if position == "WY":
        rest = sum(state.elements_left.values())
        # 合法 Wy：不超过剩余总数
        for s in wydb.all_sites_sorted():
            if s.mult <= rest:
                tok = f"WY:{s.wy}"  # 我们编码阶段使用通用 "WY:{wy}" token
                tid = vocab._tok2id.get(tok, vocab._add(tok))
                mask[tid] = 1.0
        return mask

    if position == "EL":
        # 仅允许剩余>0 的元素
        for el, cnt in state.elements_left.items():
            if cnt > 0:
                tid = vocab._tok2id.get(el, vocab._add(el))
                mask[tid] = 1.0
        return mask

    # 晶格维度：name in {"a","b","c","alpha","beta","gamma"}
    if position.startswith("LATT_"):
        name = position[len("LATT_"):].lower()
        # 暂时：开放此维度所有 bin 与 dr（组合一致性在更高层约束）
        for tid in vocab.lattice_bins(name):
            mask[tid] = 1.0
        for tid in vocab.lattice_drs(name):
            mask[tid] = 1.0
        return mask

    # 参数维度
    if position in ("U_BASE", "V_BASE", "W_BASE"):
        for base, tid in vocab.param_base_ids().items():
            mask[tid] = 1.0
        return mask
    if position in ("U_FINE", "V_FINE", "W_FINE"):
        for tid in vocab.param_fine_ids():
            mask[tid] = 1.0
        return mask
    if position in ("U_DR", "V_DR", "W_DR"):
        for _, tid in vocab.param_dr_ids().items():
            mask[tid] = 1.0
        return mask

    # 未知字段
    raise ValueError(f"未知 position: {position}")
