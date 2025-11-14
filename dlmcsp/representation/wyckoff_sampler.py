# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List, Dict, Tuple
from dlmcsp.representation.wyckoff_table import WyckoffDB

@dataclass
class WySite:
    wy: str      # e.g., "3c"
    mult: int    # 3
    dof: int     # 0..3
    mask: Dict   # {"u":True,"v":False,"w":False}

class WyTemplateSampler:
    """
    DiffCSP++ 风格：在给定 SG 与 NATOMS、元素配额下，按 Wyckoff 序列逐步采样。
    - 不做ILP/DP；用有界回溯 + 强剪枝（不超配、不留死余数）。
    - 返回一条可行模板序列：[ (wy="3c", el="O"), (wy="1a", el="Ca"), ... ]
    """
    def __init__(self, sgnum: int, natoms: int, element_counts: Dict[str, int], max_backtracks: int = 2000):
        self.db = WyckoffDB(sgnum)
        self.natoms = natoms
        self.left = dict(element_counts)
        self.backtracks = 0
        self.max_backtracks = max_backtracks
        self.candidates: List[WySite] = self.db.all_sites_sorted()  # 按 mult降序、dof升序、字母序

    def feasible(self, choose: WySite, el: str) -> bool:
        # 1) 不超配
        if self.left[el] - choose.mult < 0: return False
        # 2) 余数可凑（强剪枝）：剩余原子总数是否能由任何mult组合表示
        rest_total = sum(v for v in self.left.values()) - choose.mult
        if rest_total < 0: return False
        if rest_total > 0 and not self.db.can_sum_to(rest_total):  # 仅用mult集合做快速可达判定
            return False
        return True

    def sample(self) -> List[Tuple[WySite, str]]:
        plan: List[Tuple[WySite, str]] = []
        site_idx = 0
        cand_idx = [0]  # 每层当前候选指针
        while sum(self.left.values()) > 0:
            if self.backtracks > self.max_backtracks:
                raise RuntimeError("wyckoff backtrack overflow")
            # 候选列表：不超过余数的Wy
            avail = [s for s in self.candidates if s.mult <= sum(self.left.values())]
            i = cand_idx[-1]
            progressed = False
            while i < len(avail):
                wy = avail[i]
                # 尝试将此位点分配给某个元素（优先剩余大的元素）
                els = sorted([e for e in self.left if self.left[e] > 0], key=lambda k: -self.left[k])
                placed = False
                for el in els:
                    if self.feasible(wy, el):
                        # 接受
                        self.left[el] -= wy.mult
                        plan.append((wy, el))
                        cand_idx[-1] = i  # 记录本层选择
                        # 下一层
                        cand_idx.append(0)
                        progressed = True
                        placed = True
                        break
                if progressed: break
                i += 1
            if not progressed:
                # 回溯
                self.backtracks += 1
                if not plan: raise RuntimeError("no wyckoff template feasible")
                wy, el = plan.pop()
                self.left[el] += wy.mult
                cand_idx.pop()
                cand_idx[-1] += 1
        return plan
