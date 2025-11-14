# -*- coding: utf-8 -*-
"""
Wyckoff 模板采样（无 ILP/DP 版本）
- 输入：SG、NATOMS、目标元素配额（从 formula 解析）
- 输出：一条可行的 (wy, element) 序列，满足总原子数与合法 Wy 限制
- 剪枝：不超配；剩余原子总数可由 multiplicity 集合的子集和凑出
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

from collections import Counter

from dlmcsp.representation.wyckoff_table import WyckoffDB, WySiteDef


@dataclass
class WyAssign:
    wy: str            # "3c"
    mult: int          # 3
    element: str       # "O"


class WyTemplateSampler:
    """
    使用 WyckoffDB 的位点表（按 mult↓, dof↑ 排序）做模板采样。
    - 不用 ILP/DP；使用 bitset 子集和可达性 + 有界回溯。
    """
    def __init__(self, sgnum: int, natoms: int, element_counts: Dict[str, int],
                 vocab_yaml: str, max_backtracks: int = 2000):
        self.sgnum = int(sgnum)
        self.natoms = int(natoms)
        self.left = Counter(element_counts)  # 剩余每种元素数量
        self.db = WyckoffDB(self.sgnum, vocab_yaml=vocab_yaml, try_pyxtal=True)
        self.max_backtracks = max_backtracks
        self.backtracks = 0

        # multiplicity 集合与可达性 bitset（高效判断“剩余能否被凑出”）
        self.mults = sorted({s.mult for s in self.db.all_sites_sorted()}, reverse=True)
        self.reach = self._precompute_reachable(self.mults, limit=max(512, self.natoms * 2))

        # 候选位点（固定顺序：mult↓, dof↑, letter↑）
        self.candidates: List[WySiteDef] = self.db.all_sites_sorted()

    @staticmethod
    def _precompute_reachable(mults: List[int], limit: int) -> List[bool]:
        """
        bitset 方式：reach[k]=True 表示 k 可由 mults 的非负线性组合表示（允许重复）
        """
        limit = int(limit)
        reach = [False] * (limit + 1)
        reach[0] = True
        for m in mults:
            for k in range(m, limit + 1):
                if reach[k - m]:
                    reach[k] = True
        return reach

    def _total_left(self) -> int:
        return sum(self.left.values())

    def _sum_reachable(self, k: int) -> bool:
        k = int(k)
        return 0 <= k < len(self.reach) and self.reach[k]

    def _feasible(self, wy: WySiteDef, el: str) -> bool:
        # 1) 不超配
        if self.left[el] - wy.mult < 0:
            return False
        # 2) 剩余总数可达
        rest = self._total_left() - wy.mult
        if rest < 0:
            return False
        if rest > 0 and not self._sum_reachable(rest):
            return False
        return True

    def sample(self) -> List[WyAssign]:
        """
        产生一条完整模板：(wy, element) 序列，使 sum(mult)=natoms 且各元素配额满足。
        """
        plan: List[WyAssign] = []
        # 每层候选指针（回溯需要）
        idx_stack: List[int] = [0]

        while self._total_left() > 0:
            if self.backtracks > self.max_backtracks:
                raise RuntimeError("wyckoff backtrack overflow")

            rest = self._total_left()
            avail = [s for s in self.candidates if s.mult <= rest]
            i = idx_stack[-1]
            progressed = False

            while i < len(avail):
                wy = avail[i]
                # 选择一个元素（贪心：剩余多的优先）
                elements = [e for e, c in self.left.items() if c > 0]
                elements.sort(key=lambda x: -self.left[x])
                placed = False
                for el in elements:
                    if self._feasible(wy, el):
                        # 接受此选择
                        self.left[el] -= wy.mult
                        plan.append(WyAssign(wy=wy.wy, mult=wy.mult, element=el))
                        # 进入下一层
                        idx_stack[-1] = i
                        idx_stack.append(0)
                        progressed = True
                        placed = True
                        break
                if progressed:
                    break
                i += 1

            if not progressed:
                # 回溯
                self.backtracks += 1
                if not plan:
                    raise RuntimeError("no feasible wyckoff template")
                last = plan.pop()
                self.left[last.element] += last.mult
                idx_stack.pop()           # 弹出一层
                idx_stack[-1] += 1        # 父层指针前进

        return plan
