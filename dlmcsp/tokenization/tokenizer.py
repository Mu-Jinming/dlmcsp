# -*- coding: utf-8 -*-
"""
MaterialTokenizer:
- 从 vocab.yaml 构建 token→id / id→token 映射；
- 提供 encode(ms_dict) → List[int]；
- 数值 token 不走 BPE，全部来自 vocab.yaml 规则；
- 文本关键字用固定表。
"""
from __future__ import annotations
from typing import Dict, Any, List

from .vocab_utils import load_vocab_yaml, get_param_conf


# 和 vocab_utils 里的 _BASE_FRACTIONS 保持一致，作为兜底
_DEFAULT_BASE_FRAC = ["0", "1/2", "1/3", "2/3",
                      "1/4", "3/4", "1/6", "5/6",
                      "1/8", "3/8", "5/8", "7/8"]


class Vocab:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self._tok2id: Dict[str, int] = {}
        self._id2tok: List[str] = []

        # ---------- special ----------
        for t in cfg.get("special_tokens", ["<BOS>", "<EOS>", "<PAD>", "<MASK>"]):
            self._add(t)

        # ---------- keywords ----------
        for t in [
            "FORMULA=", "NATOMS=", "SG=", "LATT=", "SITES=",
            "EL:", "WY:", "PARAM:", "->", ";", ",", "(", ")",
            "u:", "v:", "w:",
        ]:
            self._add(t)

        # ---------- SG tokens ----------
        self.sg_tokens: List[int] = []
        for t in cfg.get("sg_tokens", []):
            self.sg_tokens.append(self._add(t))

        # ---------- element tokens ----------
        self.element_tokens: List[int] = []
        for t in cfg.get("element_tokens", []):
            self.element_tokens.append(self._add(t))

        # ---------- Wyckoff tokens (flatten) ----------
        # key = (sg_str, wy_str) -> token_id
        self._wy_map: Dict[tuple[str, str], int] = {}
        for sg, wy_list in cfg.get("wy_tokens", {}).items():
            for wy in wy_list:
                self._wy_map[(sg, wy)] = self._add(f"WY:{sg}:{wy}")

        # ---------- lattice bins ----------
        self._latt_conf = cfg.get("lattice_bins", {})
        self._latt_bins_tok: Dict[str, List[int]] = {}  # name -> bin token ids
        self._latt_dr_tok: Dict[str, List[int]] = {}    # name -> dr token ids
        for name, conf in self._latt_conf.items():
            nb = int(conf["num_bins"])
            # BIN tokens
            self._latt_bins_tok[name] = [
                self._add(f"{name.upper()}_BIN_{i}") for i in range(nb)
            ]
            # DR tokens
            drs = list(conf.get("residuals", [-2, -1, 0, 1, 2]))
            self._latt_dr_tok[name] = [
                self._add(f"{name.upper()}_DR_{d}") for d in drs
            ]

        # ---------- param tokens ----------
        pc = get_param_conf(cfg)  # 期望结构: {base:[..], fine:{denom:..}, dr:[..]}
        self._param_conf = pc

        base_list = pc.get("base", _DEFAULT_BASE_FRAC)
        fine_conf = pc.get("fine", {})
        denom = int(fine_conf.get("denom", 96))
        dr_list = pc.get("dr", [-1, 0, 1])

        # BASE_xx
        self._param_base_tok: Dict[str, int] = {
            s: self._add(f"BASE_{s}") for s in base_list
        }
        # FINE_i
        self._param_fine_tok: List[int] = [
            self._add(f"FINE_{i}") for i in range(denom)
        ]
        # DR_d
        self._param_dr_tok: Dict[int, int] = {
            int(d): self._add(f"DR_{int(d)}") for d in dr_list
        }

    # ========== basic mapping ==========

    def _add(self, t: str) -> int:
        if t in self._tok2id:
            return self._tok2id[t]
        idx = len(self._id2tok)
        self._tok2id[t] = idx
        self._id2tok.append(t)
        return idx

    def token_id(self, t: str) -> int:
        return self._tok2id[t]

    def id_token(self, i: int) -> str:
        return self._id2tok[i]

    # ========== helpers for SG/WY/lattice/param ==========

    def wy_token_id(self, sg_token: str, wy: str) -> int:
        key = (sg_token, wy)
        if key not in self._wy_map:
            # 兜底：动态加入（不推荐频繁触发）
            return self._add(f"WY:{sg_token}:{wy}")
        return self._wy_map[key]

    def lattice_bins(self, name: str) -> List[int]:
        return self._latt_bins_tok[name]

    def lattice_drs(self, name: str) -> List[int]:
        return self._latt_dr_tok[name]

    def param_base_ids(self) -> Dict[str, int]:
        return self._param_base_tok

    def param_fine_ids(self) -> List[int]:
        return self._param_fine_tok

    def param_dr_ids(self) -> Dict[int, int]:
        return self._param_dr_tok


class MaterialTokenizer:
    """
    encode(ms_dict)：
    输入 ms_dict 形如：
    {
      "formula": "CaTiO3",
      "n_atoms": 5,
      "sg": 221,
      "latt": { "a": {"bin":173,"dr":+1}, ... },
      "sites": [
          {"el":"Ca","wy":"1a","params":"-"},
          {"el":"Ti","wy":"1b","params":"-"},
          {"el":"O","wy":"3c","params":{"u":{...},"v":{...},"w":{...}}}
      ]
    }
    """
    def __init__(self, vocab: Vocab):
        self.vocab = vocab

    @classmethod
    def from_yaml(cls, path: str) -> "MaterialTokenizer":
        cfg = load_vocab_yaml(path)
        return cls(Vocab(cfg))

    def encode(self, ms: Dict[str, Any]) -> List[int]:
        v = self.vocab
        toks: List[int] = []

        # <BOS>
        toks.append(v.token_id("<BOS>"))

        # ---------- FORMULA ----------
        toks.append(v.token_id("FORMULA="))
        for ch in ms["formula"]:
            toks.append(v._add(ch))  # 字面字符纳入词表
        toks.append(v.token_id(";"))

        # ---------- NATOMS ----------
        toks.append(v.token_id("NATOMS="))
        for ch in str(int(ms["n_atoms"])):
            toks.append(v._add(ch))
        toks.append(v.token_id(";"))

        # ---------- SG ----------
        toks.append(v.token_id("SG="))
        sg_tok = f"SG_{int(ms['sg'])}"
        toks.append(v._add(sg_tok))
        toks.append(v.token_id(";"))

        # ---------- LATT ----------
        toks.append(v.token_id("LATT="))
        # 先写 6 个 bin
        for name in ["a", "b", "c", "alpha", "beta", "gamma"]:
            info = ms["latt"][name]
            bin_tok = v.lattice_bins(name)[int(info["bin"])]
            toks.append(bin_tok)
            if name != "gamma":
                toks.append(v.token_id(","))
        # 再写 6 个 dr
        for name in ["a", "b", "c", "alpha", "beta", "gamma"]:
            info = ms["latt"][name]
            dr_tok_list = v.lattice_drs(name)
            dr_idx = self._dr_index(name, int(info.get("dr", 0)), v)
            toks.append(dr_tok_list[dr_idx])
            if name != "gamma":
                toks.append(v.token_id(","))
        toks.append(v.token_id(";"))

        # ---------- SITES ----------
        toks.append(v.token_id("SITES="))
        for i, site in enumerate(ms["sites"]):
            toks.append(v.token_id("("))

            # EL
            toks.append(v.token_id("EL:"))
            el = str(site["el"])
            toks.append(v.token_id(el) if el in v._tok2id else v._add(el))
            toks.append(v.token_id(","))

            # WY（不在这里区分 SG，只存 "WY:3c" 这类）
            toks.append(v.token_id("WY:"))
            wy = str(site["wy"])   # e.g. "3c"
            toks.append(v._add(f"WY:{wy}"))
            toks.append(v.token_id(","))

            # PARAM
            toks.append(v.token_id("PARAM:"))
            params = site.get("params", "-")
            if params == "-" or params is None:
                toks.append(v._add("-"))
            else:
                first = True
                for axis in ("u", "v", "w"):
                    if axis not in params:
                        continue
                    if not first:
                        toks.append(v.token_id(","))
                    first = False
                    toks.append(v._add(f"{axis}:"))
                    p = params[axis]
                    if p["mode"] == "BASE":
                        base_id = v.param_base_ids()[p["base"]]
                        toks.append(base_id)
                    else:
                        # FINE + 可选 DR
                        idx = int(p["idx"])
                        toks.append(v.param_fine_ids()[idx])
                        dr = int(p.get("dr", 0))
                        toks.append(v.param_dr_ids()[dr])

            toks.append(v.token_id(")"))
            if i != len(ms["sites"]) - 1:
                toks.append(v.token_id("->"))
        toks.append(v.token_id(";"))

        # <EOS>
        toks.append(v.token_id("<EOS>"))
        return toks

    @staticmethod
    def _dr_index(name: str, dr: int, v: Vocab) -> int:
        """
        在该维度的 DR token 列表里找到对应 dr 的索引；
        若找不到，则退到中间值。
        """
        tok_ids = v.lattice_drs(name)
        dr_values = []
        for tid in tok_ids:
            t = v.id_token(tid)          # e.g. "ALPHA_DR_-1"
            try:
                d = int(t.split("_")[-1])
            except Exception:
                d = 0
            dr_values.append(d)

        if dr in dr_values:
            return dr_values.index(dr)
        return len(dr_values) // 2
