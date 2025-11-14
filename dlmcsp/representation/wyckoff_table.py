# -*- coding: utf-8 -*-
"""
WyckoffDB: 查询某空间群下的 Wyckoff 位点元数据（多重性、自由度、参数掩码）。
策略：优先用 pyxtal 获取精确信息；若 pyxtal 缺失或接口差异导致失败，则回退到 vocab.yaml 的 wy_tokens。
回退模式下：
- 解析 wy 字符串（如 "3c"）得到 multiplicity=3, letter="c"
- 保守设定 dof=3，mask={"u":True,"v":True,"w":True}（编码端足够；生成端再用硬约束收紧）
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import re
import yaml


@dataclass(frozen=True)
class WySiteDef:
    wy: str                 # 形如 "3c"
    mult: int               # 3
    letter: str             # "c"
    dof: int                # 0..3
    mask: Dict[str, bool]   # {"u": True/False, "v":..., "w":...}


def _parse_wy_str(wy: str) -> Tuple[int, str]:
    """从 '3c' 解析 (3, 'c')；从 '1a' 解析 (1, 'a')。"""
    m = re.match(r"^\s*(\d+)\s*([A-Za-z])\s*$", str(wy))
    if not m:
        raise ValueError(f"非法 Wyckoff 字符串: {wy}")
    return int(m.group(1)), m.group(2).lower()


class WyckoffDB:
    """
    API:
      - all_sites_sorted(): List[WySiteDef]  按 mult↓, dof↑, letter↑ 排序
      - find_by_wy("3c"): WySiteDef
      - param_schema("c"): {"mult":..., "dof":..., "mask":{...}}  （letter 版）
    """
    def __init__(self, sgnum: int, vocab_yaml: Optional[str] = None, try_pyxtal: bool = True):
        self.sgnum = int(sgnum)
        if not (1 <= self.sgnum <= 230):
            raise ValueError(f"非法空间群号: {self.sgnum}")
        self._sites: List[WySiteDef] = []
        self._built_from = "none"

        # 优先 pyxtal
        if try_pyxtal:
            try:
                self._sites = self._build_from_pyxtal(self.sgnum)
                self._built_from = "pyxtal"
            except Exception:
                # 回退到 vocab
                pass

        if not self._sites:
            if not vocab_yaml:
                raise RuntimeError("WyckoffDB 回退需要 vocab_yaml，但未提供。")
            self._sites = self._build_from_vocab(self.sgnum, vocab_yaml)
            self._built_from = "vocab"

        # 排序：mult 降序、dof 升序、letter 升序
        self._sites.sort(key=lambda s: (-s.mult, s.dof, s.letter))

    # ---------- 构建实现 ----------

    def _build_from_pyxtal(self, sgnum: int) -> List[WySiteDef]:
        try:
            from pyxtal.symmetry import Group  # type: ignore
        except Exception as e:
            raise RuntimeError("pyxtal 未安装或版本不兼容") from e

        g = Group(int(sgnum))
        results: List[WySiteDef] = []
        seen = set()

        # 不同 pyxtal 版本中，group.wyckoffs 的内部结构略有差异；我们尽可能鲁棒处理。
        wy_list = getattr(g, "wyckoffs", None)
        if wy_list is None:
            raise AttributeError("pyxtal.Group 无 wyckoffs 属性")

        for entries in wy_list:
            if not entries:
                continue
            # 取代表（通常 entries[0]）
            w = entries[0]

            # multiplicity
            mult = None
            for attr in ("multiplicity", "mult"):
                if hasattr(w, attr):
                    try:
                        mult = int(getattr(w, attr))
                        break
                    except Exception:
                        pass
            if mult is None:
                # 尝试 group 的 general multiplicity 逻辑（退化）
                continue

            # letter
            letter = None
            for attr in ("letter", "symbol", "wyckoff_letter"):
                if hasattr(w, attr):
                    try:
                        letter = str(getattr(w, attr)).lower()
                        break
                    except Exception:
                        pass
            if letter is None:
                continue

            key = (mult, letter)
            if key in seen:
                continue
            seen.add(key)

            # dof & mask
            dof = None
            for attr in ("dof", "degrees_of_freedom", "n_variable", "n_variables"):
                if hasattr(w, attr):
                    try:
                        dof = int(getattr(w, attr))
                        break
                    except Exception:
                        pass
            if dof is None:
                # 粗略猜：general 位点往往 dof=3，否则 0；这里不做激进假设，设 0，留给上游根据坐标编码
                dof = 0

            mask = {"u": False, "v": False, "w": False}
            # 变量名：x,y,z → u,v,w
            var_letters = None
            for name in ("variables", "variable", "var_list"):
                if hasattr(w, name):
                    try:
                        var_letters = list(getattr(w, name))
                        break
                    except Exception:
                        pass
            if var_letters:
                tr = {"x": "u", "y": "v", "z": "w"}
                for v in var_letters:
                    v = str(v).lower()
                    if v in tr:
                        mask[tr[v]] = True
            else:
                # 若未知，按照 dof 从 u 起置 True
                if dof >= 1: mask["u"] = True
                if dof >= 2: mask["v"] = True
                if dof >= 3: mask["w"] = True

            results.append(WySiteDef(wy=f"{mult}{letter}", mult=mult, letter=letter, dof=int(dof), mask=mask))
        if not results:
            raise RuntimeError("pyxtal 构建 wyckoff 失败或为空")
        return results

    def _build_from_vocab(self, sgnum: int, vocab_yaml: str) -> List[WySiteDef]:
        with open(vocab_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        wy_tokens: Dict[str, List[str]] = cfg.get("wy_tokens", {})
        key = f"SG_{int(sgnum)}"
        if key not in wy_tokens:
            raise KeyError(f"vocab.yaml 未包含 {key} 的 wy_tokens")

        results: List[WySiteDef] = []
        letters_seen = set()
        for wy in wy_tokens[key]:
            mult, letter = _parse_wy_str(wy)
            # 保守：编码期全参数可写；生成期再用硬约束收紧
            dof = 3
            mask = {"u": True, "v": True, "w": True}
            # 若同 letter 多次出现（极少），保留 multiplicity 最大的那条
            if letter in letters_seen:
                # 若已存在，替换成 mult 最大的
                prev_idx = None
                for i, s in enumerate(results):
                    if s.letter == letter:
                        prev_idx = i
                        break
                if prev_idx is not None and mult > results[prev_idx].mult:
                    results[prev_idx] = WySiteDef(wy=f"{mult}{letter}", mult=mult, letter=letter, dof=dof, mask=mask)
            else:
                results.append(WySiteDef(wy=f"{mult}{letter}", mult=mult, letter=letter, dof=dof, mask=mask))
                letters_seen.add(letter)

        if not results:
            raise RuntimeError(f"vocab 构建 wyckoff 失败: {key}")
        return results


    # ---------- 展开 Wyckoff 等价位置（供 ms_to_cif 使用） ----------

    def expand_positions(self, wy: str, params: Dict[str, float]) -> List[Tuple[float, float, float]]:
        """
        给定 Wyckoff 位点和 (u,v,w) 参数，使用 pyxtal 展开所有等价原子坐标。

        参数:
            wy: 可以是 "4f" 也可以是 "f"（只给字母）
            params: {"u": float, "v": float, "w": float}，缺失的维度当 0.0 处理

        返回:
            List[(x,y,z)]，都已经 mod 1 归一化到 [0,1)
        """
        try:
            from pyxtal.symmetry import Group, Wyckoff_position  # type: ignore
        except Exception as e:
            raise RuntimeError("expand_positions 需要 pyxtal，但当前环境不可用") from e

        # 统一成完整的 "mult+letter" 字符串，例如 "4f"
        wy = str(wy).strip()
        if any(ch.isdigit() for ch in wy):
            site = wy.lower()
        else:
            # 只给了字母，补上 multiplicity
            schema = self.param_schema(wy)
            site = f"{int(schema['mult'])}{str(wy).lower()}"

        # 构造 Group，用它的 hall_number 来保证 R/特殊 setting 一致
        G = Group(self.sgnum)
        hn = getattr(G, "hall_number", None)

        # 从 group+letter 得到 Wyckoff_position 对象
        wp = Wyckoff_position.from_group_and_letter(G.number, site, dim=3, hn=hn)

        # 组装 (x,y,z) = (u,v,w)，缺失就填 0
        u = float(params.get("u", 0.0))
        v = float(params.get("v", 0.0))
        w = float(params.get("w", 0.0))
        pt = [u, v, w]

        # apply_ops 返回 N×3 的坐标数组
        coords = wp.apply_ops(pt)

        out: List[Tuple[float, float, float]] = []
        for p in coords:
            x = float(p[0]) % 1.0
            y = float(p[1]) % 1.0
            z = float(p[2]) % 1.0
            out.append((x, y, z))
        return out

    # ---------- 查询 API ----------

    def all_sites_sorted(self) -> List[WySiteDef]:
        return list(self._sites)

    def find_by_wy(self, wy: str) -> WySiteDef:
        mult, letter = _parse_wy_str(wy)
        for s in self._sites:
            if s.mult == mult and s.letter == letter:
                return s
        # 没找到就回退到 letter 匹配
        for s in self._sites:
            if s.letter == letter:
                return s
        raise KeyError(f"未找到 Wyckoff 位点: {wy} (SG={self.sgnum}, from={self._built_from})")

    def param_schema(self, wy_letter: str) -> Dict[str, Any]:
        wy_letter = str(wy_letter).lower()
        for s in self._sites:
            if s.letter == wy_letter or s.wy.endswith(wy_letter):
                return {"mult": s.mult, "dof": s.dof, "mask": dict(s.mask)}
        # 兜底：可编码但不严格
        return {"mult": 1, "dof": 3, "mask": {"u": True, "v": True, "w": True}}
