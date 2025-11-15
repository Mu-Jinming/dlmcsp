# -*- coding: utf-8 -*-
"""
MaterialTokenizer:
- vocab 只来自 vocab.yaml，训练/推理都不扩表；
- formula 用元素级别编码（按 formula unit 展开元素）；
- NATOMS 用离散 token NATOMS_k (k<=512)；
- Wy token 用 WY:SG_x:wy，完全由 wy_tokens 决定；
"""

from __future__ import annotations
from typing import Dict, Any, List

from pymatgen.core import Composition

from .vocab_utils import load_vocab_yaml, get_param_conf

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
            "-",   # 无参数占位
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

        # ---------- Wyckoff tokens (SG-aware) ----------
        # key = (sg_str, wy_str) -> token_id, e.g. ("SG_194","4f") -> "WY:SG_194:4f"
        self._wy_map: Dict[tuple[str, str], int] = {}
        for sg, wy_list in cfg.get("wy_tokens", {}).items():
            for wy in wy_list:
                tok_str = f"WY:{sg}:{wy}"
                self._wy_map[(sg, wy)] = self._add(tok_str)

        # ---------- lattice bins ----------
        self._latt_conf = cfg.get("lattice_bins", {})
        self._latt_bins_tok: Dict[str, List[int]] = {}
        self._latt_dr_tok: Dict[str, List[int]] = {}
        for name, conf in self._latt_conf.items():
            nb = int(conf["num_bins"])
            self._latt_bins_tok[name] = [
                self._add(f"{name.upper()}_BIN_{i}") for i in range(nb)
            ]
            drs = list(conf.get("residuals", [-2, -1, 0, 1, 2]))
            self._latt_dr_tok[name] = [
                self._add(f"{name.upper()}_DR_{d}") for d in drs
            ]

        # ---------- param tokens ----------
        pc = get_param_conf(cfg)
        self._param_conf = pc

        base_list = pc.get("base", _DEFAULT_BASE_FRAC)
        fine_conf = pc.get("fine", {})
        denom = int(fine_conf.get("denom", 96))
        dr_list = pc.get("dr", [-1, 0, 1])

        self._param_base_tok: Dict[str, int] = {
            s: self._add(f"BASE_{s}") for s in base_list
        }
        self._param_fine_tok: List[int] = [
            self._add(f"FINE_{i}") for i in range(denom)
        ]
        self._param_dr_tok: Dict[int, int] = {
            int(d): self._add(f"DR_{int(d)}") for d in dr_list
        }

        # ---------- NATOMS tokens ----------
        self._natoms_tok: Dict[int, int] = {}
        for t in cfg.get("natoms_tokens", []):
            idx = self._add(t)
            if t.startswith("NATOMS_"):
                try:
                    n = int(t.split("_")[1])
                    self._natoms_tok[n] = idx
                except Exception:
                    # 忽略坏格式，audit_vocab 时可检查
                    pass

    # ===== 基础映射 =====

    def _add(self, t: str) -> int:
        if t in self._tok2id:
            return self._tok2id[t]
        idx = len(self._id2tok)
        self._tok2id[t] = idx
        self._id2tok.append(t)
        return idx

    def token_id(self, t: str) -> int:
        if t not in self._tok2id:
            raise KeyError(f"[Vocab] token '{t}' not found in vocab.yaml")
        return self._tok2id[t]

    def id_token(self, i: int) -> str:
        return self._id2tok[i]

    # ===== helpers =====

    def wy_token_id(self, sg_token: str, wy: str) -> int:
        key = (sg_token, wy)
        if key not in self._wy_map:
            raise KeyError(
                f"[Vocab] wy-token for ({sg_token},{wy}) not found in wy_tokens; "
                f"请检查 vocab.yaml 的 wy_tokens 是否覆盖该 SG 的该 Wy letter"
            )
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

    def natoms_token_id(self, n: int) -> int:
        if n not in self._natoms_tok:
            raise KeyError(
                f"[Vocab] NATOMS_{n} 不在 natoms_tokens 里；"
                f"请在 vocab.yaml.natoms_tokens 中补齐（当前范围={sorted(self._natoms_tok.keys())[:5]}...）"
            )
        return self._natoms_tok[n]


class MaterialTokenizer:
    """
    encode(ms_dict)：
    - formula：按 formula unit 展开元素，用 element_tokens；
    - NATOMS：用单个 NATOMS_k token；
    - 其他布局与原来一致。
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
        comp = Composition(ms["formula"])
        # 这里使用 reduced formula 的计数（formula unit）
        elems: List[str] = []
        for el, amt in comp.items():
            cnt = int(round(float(amt)))
            elems.extend([el.symbol] * cnt)

        for i, el in enumerate(elems):
            toks.append(v.token_id(el))  # 必须在 element_tokens 里
            if i != len(elems) - 1:
                toks.append(v.token_id(","))
        toks.append(v.token_id(";"))

        # ---------- NATOMS ----------
        toks.append(v.token_id("NATOMS="))
        natoms = int(ms["n_atoms"])
        toks.append(v.natoms_token_id(natoms))  # NATOMS_k
        toks.append(v.token_id(";"))

        # ---------- SG ----------
        toks.append(v.token_id("SG="))
        sg_tok = f"SG_{int(ms['sg'])}"
        toks.append(v.token_id(sg_tok))
        toks.append(v.token_id(";"))

        # ---------- LATT ----------
        toks.append(v.token_id("LATT="))
        # 6 bins
        for name in ["a", "b", "c", "alpha", "beta", "gamma"]:
            info = ms["latt"][name]
            bin_tok = v.lattice_bins(name)[int(info["bin"])]
            toks.append(bin_tok)
            if name != "gamma":
                toks.append(v.token_id(","))
        # 6 dr
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
        sg_tok = f"SG_{int(ms['sg'])}"
        for i, site in enumerate(ms["sites"]):
            toks.append(v.token_id("("))

            # EL
            toks.append(v.token_id("EL:"))
            el = str(site["el"])
            toks.append(v.token_id(el))
            toks.append(v.token_id(","))

            # WY (SG-aware)
            toks.append(v.token_id("WY:"))
            wy = str(site["wy"])      # e.g. "4f"
            wy_id = v.wy_token_id(sg_tok, wy)
            toks.append(wy_id)
            toks.append(v.token_id(","))

            # PARAM
            toks.append(v.token_id("PARAM:"))
            params = site.get("params", "-")
            if params == "-" or params is None:
                toks.append(v.token_id("-"))
            else:
                first = True
                for axis in ("u", "v", "w"):
                    if axis not in params:
                        continue
                    if not first:
                        toks.append(v.token_id(","))
                    first = False
                    toks.append(v.token_id(f"{axis}:"))
                    p = params[axis]
                    if p["mode"] == "BASE":
                        base_id = v.param_base_ids()[p["base"]]
                        toks.append(base_id)
                    else:
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
        tok_ids = v.lattice_drs(name)
        dr_values = []
        for tid in tok_ids:
            t = v.id_token(tid)
            try:
                d = int(t.split("_")[-1])
            except Exception:
                d = 0
            dr_values.append(d)
        if dr in dr_values:
            return dr_values.index(dr)
        return len(dr_values) // 2
