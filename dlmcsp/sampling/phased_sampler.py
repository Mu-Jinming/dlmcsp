# -*- coding: utf-8 -*-
"""
三阶段采样（只给组分也行）：
P0 选空间群（可多次尝试，保证 natoms 可由该 SG 的 Wy 多重性凑出）
P1 硬约束模板：WyTemplateSampler（DiffCSP++风格回溯）
P2 用 LLaDA 填 LATT 与参数（参数仅填 BASE 分数，先求稳，后续再开 FINE+DR）
"""
from __future__ import annotations
import random
from typing import Dict, Any, List, Tuple
from collections import Counter

import torch
import numpy as np
import re
from pymatgen.core import Composition

from dlmcsp.constraints.lattice_tying import lattice_tie_plan, crystal_system_from_sg
from dlmcsp.representation.wyckoff_table import WyckoffDB
from dlmcsp.constraints.wyckoff_sampler import WyTemplateSampler
from dlmcsp.tokenization.tokenizer import MaterialTokenizer, Vocab
from dlmcsp.tokenization.vocab_utils import load_vocab_yaml, inv_bin_lattice_scalar, inv_quantize_param


def parse_formula(formula: str) -> Tuple[Dict[str, int], int, str]:
    comp = Composition(formula)
    el_counts = {el.symbol: int(round(float(amount))) for el, amount in comp.items()}
    natoms = int(sum(el_counts.values()))
    pretty = comp.get_reduced_formula_and_factor()[0]
    return el_counts, natoms, pretty



def build_ms_skeleton(formula_pretty: str, natoms: int, sg: int,
                      template: List[Tuple[str, str]],  # [(wy, el), ...]
                      vocab_yaml: str) -> Dict[str, Any]:
    """
    仅放占位：LATT 全 <MASK>；参数轴按 Wy 自由度决定（仅占位，不含值）
    """
    wydb = WyckoffDB(sg, vocab_yaml=vocab_yaml, try_pyxtal=True)
    sites = []
    for wy, el in template:
        # 查 mask 决定是否给 u/v/w
        mult, letter = int(wy[:-1]), wy[-1]
        schema = wydb.param_schema(letter)
        params = {}
        for ax in ("u", "v", "w"):
            if schema["mask"].get(ax, False):
                params[ax] = {"mode": "BASE", "base": "0"}  # 先放个可解析的默认值，稍后被模型覆盖
        if not params:
            params = "-"  # 固定点
        sites.append({"el": el, "wy": f"{schema['mult']}{letter}", "params": params})

    # LATT 占位（bin/dr 用 -1 表示需要填）
    latt = {k: {"bin": -1, "dr": 0} for k in ["a", "b", "c", "alpha", "beta", "gamma"]}

    ms = {
        "formula": formula_pretty,
        "n_atoms": natoms,
        "sg": int(sg),
        "latt": latt,
        "sites": sites,
    }
    return ms

# --------- 用 LLaDA 填充 ---------

def _split_formula_elements(formula: str) -> List[str]:
    """
    把化学式按元素切分：
      "GaTe"      -> ["Ga", "Te"]
      "CaTiO3"    -> ["Ca", "Ti", "O"]    # 目前忽略数字，只编码元素
      "Li2MnO3"   -> ["Li", "Mn", "O"]
    """
    if not formula:
        return []
    # 一个元素符号：大写 + 可选小写，如 Ca, Ti, O, Ga
    return re.findall(r"[A-Z][a-z]?", str(formula))

def _token_id_or_add(vocab: Vocab, t: str) -> int:
    """
    采样阶段禁止扩表：token 必须来自 vocab.yaml。
    """
    if t not in vocab._tok2id:
        raise KeyError(
            f"[LLaDA-cont sample] token '{t}' 不在 vocab 里；"
            f"请在 vocab.yaml 中补齐（element_tokens / sg_tokens / wy_tokens / natoms_tokens 等），"
            f"并重新 preprocess + 训练。"
        )
    return vocab._tok2id[t]



def _position_slots_for_lattice(vocab: Vocab) -> List[Tuple[str, List[int]]]:
    """
    返回 [(position_name, allowed_token_ids), ...] for 6维 bin + 6维 dr
    """
    slots = []
    for name in ["a", "b", "c", "alpha", "beta", "gamma"]:
        bins = vocab.lattice_bins(name)
        drs  = vocab.lattice_drs(name)
        slots.append((f"LATT_{name}_BIN", bins))
        slots.append((f"LATT_{name}_DR",  drs))
    return slots

def _position_slots_for_params(vocab: Vocab, ms: Dict[str, Any]) -> List[Tuple[str, List[int], Tuple[int,int,str]]]:
    """
    返回参数槽位：[(pos_name, allowed_token_ids, (site_idx, order_idx, axis)), ...]
    仅允许 BASE 分数（先求稳）
    """
    slots = []
    base_ids = list(vocab.param_base_ids().values())
    for i, site in enumerate(ms["sites"]):
        if site["params"] == "-" or not isinstance(site["params"], dict):
            continue
        order = 0
        for axis in ("u", "v", "w"):
            if axis in site["params"]:
                slots.append((f"SITE{i}_{axis.upper()}_BASE", base_ids, (i, order, axis)))
                order += 1
    return slots

from typing import Dict, Any, List, Tuple
from pymatgen.core import Composition
from dlmcsp.tokenization.tokenizer import MaterialTokenizer, Vocab

def _build_masked_inputs(
    tok: MaterialTokenizer,
    ms: Dict[str, Any],
    lattice_slots: List[Tuple[str, List[int]]],
    param_slots: List[Tuple[str, List[int], Tuple[int, int, str]]],
) -> Tuple[List[int], List[Tuple[int, List[int]]], Dict[str, int]]:
    """
    构造带 <MASK> 的 token 序列，并记录每个槽位对应的 token 位置 + allowed token 列表。
    注意：布局必须和训练时 tokenizer.encode 一致，否则分布不对齐。

    约定（和新 tokenizer 对齐）：
      FORMULA= El1,El2,... ;
      NATOMS= NATOMS_k ;
      SG= SG_x ;
      LATT= <6个BIN,逗号分隔><6个DR,逗号分隔> ;
      SITES= (EL:...,WY:...,PARAM:...)->... ;
    """
    v = tok.vocab
    ids: List[int] = []
    pos_map: Dict[str, int] = {}

    # <BOS>
    ids.append(v.token_id("<BOS>"))

    # -------- FORMULA= （按元素、带逗号） ----------
    ids.append(v.token_id("FORMULA="))
    formula_str = str(ms.get("formula", ""))

    try:
        comp = Composition(formula_str)
    except Exception as e:
        raise ValueError(f"[build_masked_inputs] 无法解析 formula='{formula_str}': {e}")

    # 用 formula unit 计数展开元素（顺序用 Composition 的顺序；to_material_string_v2 也来自 Composition）
    elem_seq: List[str] = []
    for el, amt in comp.items():
        cnt = int(round(float(amt)))
        elem_seq.extend([el.symbol] * cnt)

    if not elem_seq:
        raise ValueError(f"[build_masked_inputs] 解析 formula='{formula_str}' 得到空元素列表")

    for i, elem_sym in enumerate(elem_seq):
        ids.append(_token_id_or_add(v, elem_sym))  # 必须在 element_tokens 里
        if i != len(elem_seq) - 1:
            ids.append(v.token_id(","))

    ids.append(v.token_id(";"))

    # -------- NATOMS= NATOMS_k ; ----------
    ids.append(v.token_id("NATOMS="))
    n_atoms = int(ms["n_atoms"])
    natoms_tok = f"NATOMS_{n_atoms}"
    ids.append(_token_id_or_add(v, natoms_tok))
    ids.append(v.token_id(";"))

    # -------- SG= SG_x ; ----------
    ids.append(v.token_id("SG="))
    sg_int = int(ms["sg"])
    sg_tok = f"SG_{sg_int}"
    ids.append(_token_id_or_add(v, sg_tok))
    ids.append(v.token_id(";"))

    # -------- LATT= （6 BIN + 6 DR，都用 <MASK> 槽位） ----------
    ids.append(v.token_id("LATT="))
    slot_positions: List[Tuple[int, List[int]]] = []

    # 6 bins: a,b,c,alpha,beta,gamma
    for name in ["a", "b", "c", "alpha", "beta", "gamma"]:
        for slot_name, allow in lattice_slots:
            if slot_name == f"LATT_{name}_BIN":
                pos = len(ids)
                ids.append(v.token_id("<MASK>"))
                slot_positions.append((pos, allow))
                if name != "gamma":
                    ids.append(v.token_id(","))
                break

    # 6 dr: a,b,c,alpha,beta,gamma
    for name in ["a", "b", "c", "alpha", "beta", "gamma"]:
        for slot_name, allow in lattice_slots:
            if slot_name == f"LATT_{name}_DR":
                pos = len(ids)
                ids.append(v.token_id("<MASK>"))
                slot_positions.append((pos, allow))
                if name != "gamma":
                    ids.append(v.token_id(","))
                break

    ids.append(v.token_id(";"))

    # -------- SITES= ----------
    ids.append(v.token_id("SITES="))

    # 注意：WY token 采用 SG-aware 形式：WY:SG_194:4f
    for si, site in enumerate(ms["sites"]):
        ids.append(v.token_id("("))

        # EL:
        ids.append(v.token_id("EL:"))
        el_sym = str(site["el"])
        ids.append(_token_id_or_add(v, el_sym))
        ids.append(v.token_id(","))

        # WY:
        ids.append(v.token_id("WY:"))
        wy_str = str(site["wy"])  # e.g. "4f"
        wy_tok = f"WY:{sg_tok}:{wy_str}"
        ids.append(_token_id_or_add(v, wy_tok))
        ids.append(v.token_id(","))

        # PARAM:
        ids.append(v.token_id("PARAM:"))
        params = site.get("params", "-")
        if params == "-" or not isinstance(params, dict):
            ids.append(_token_id_or_add(v, "-"))
        else:
            first = True
            for axis in ("u", "v", "w"):
                if axis not in params:
                    continue
                if not first:
                    ids.append(v.token_id(","))
                first = False
                ids.append(_token_id_or_add(v, f"{axis}:"))
                # 对应 param_slots 里的 <MASK> 槽位（只填 BASE/FINE+DR）
                for slot_name, allow, triple in param_slots:
                    site_idx, order_idx, ax = triple
                    if site_idx == si and ax == axis:
                        pos = len(ids)
                        ids.append(v.token_id("<MASK>"))
                        slot_positions.append((pos, allow))
                        break

        ids.append(v.token_id(")"))
        if si != len(ms["sites"]) - 1:
            ids.append(v.token_id("->"))

    ids.append(v.token_id(";"))

    # <EOS>
    ids.append(v.token_id("<EOS>"))

    return ids, slot_positions, pos_map


def _masked_argmax(logits_row: torch.Tensor, allow: List[int]) -> int:
    mask = torch.full_like(logits_row, fill_value=float("-inf"))
    mask[allow] = 0.0
    v = logits_row + mask
    idx = int(torch.argmax(v).item())
    return idx

# ---- 放在文件顶部：需要的 import ----
import re
import torch
from typing import List, Tuple
from dlmcsp.tokenization.tokenizer import MaterialTokenizer
from dlmcsp.constraints.lattice_tying import lattice_tie_plan

# ---- 完整替换 fill_with_llada ----
@torch.no_grad()
def fill_with_llada(
    model,
    tok: MaterialTokenizer,
    ids: List[int],
    slots: List[Tuple[int, List[int]]],
    device: str = "cuda",
    t: float = 0.15,
    sg_for_lattice_tying: int | None = None,
    vocab_yaml: str | None = None,
    return_score: bool = False,
):
    """
    仅在允许的槽位内填充；若提供 sg_for_lattice_tying，则按晶系硬绑定：
      - masters 先预测（a / c / ...）
      - slaves 复制或固定到 90°/120° 的最近 bin
    该函数与 GT 模式兼容，不做 SG 搜索。
    """
    v = tok.vocab
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]
    cum_logp = 0.0

    # 1) 通过 allow-token 名称正则匹配 lattice 槽位（大小写/前缀鲁棒）
    lat_pos = {k: {"BIN": [], "DR": []} for k in ["a","b","c","alpha","beta","gamma"]}
    PAT = {
        "a":      {"BIN": re.compile(r"(?:^|_)A_BIN_\d+$", re.I),
                   "DR":  re.compile(r"(?:^|_)A_DR_\d+$",  re.I)},
        "b":      {"BIN": re.compile(r"(?:^|_)B_BIN_\d+$", re.I),
                   "DR":  re.compile(r"(?:^|_)B_DR_\d+$",  re.I)},
        "c":      {"BIN": re.compile(r"(?:^|_)C_BIN_\d+$", re.I),
                   "DR":  re.compile(r"(?:^|_)C_DR_\d+$",  re.I)},
        "alpha":  {"BIN": re.compile(r"(?:^|_)ALPHA_BIN_\d+$", re.I),
                   "DR":  re.compile(r"(?:^|_)ALPHA_DR_\d+$",  re.I)},
        "beta":   {"BIN": re.compile(r"(?:^|_)BETA_BIN_\d+$", re.I),
                   "DR":  re.compile(r"(?:^|_)BETA_DR_\d+$",   re.I)},
        "gamma":  {"BIN": re.compile(r"(?:^|_)GAMMA_BIN_\d+$", re.I),
                   "DR":  re.compile(r"(?:^|_)GAMMA_DR_\d+$",  re.I)},
    }
    for slot_idx, (pos, allow) in enumerate(slots):
        if not allow:
            continue
        tokname = v.id_token(allow[0])
        for name in lat_pos.keys():
            for kind in ("BIN","DR"):
                if PAT[name][kind].search(tokname):
                    lat_pos[name][kind].append((slot_idx, pos))
                    break

    # 2) 晶系绑定计划
    tying = None
    if sg_for_lattice_tying is not None:
        if not vocab_yaml:
            raise RuntimeError("fill_with_llada: 需要 vocab_yaml 来执行晶系绑定")
        tying = lattice_tie_plan(int(sg_for_lattice_tying), vocab_yaml)

    def masked_pick(pos, allow):
        nonlocal cum_logp
        x0 = x.clone()
        x0[0, pos] = v.token_id("<MASK>")
        logits = model(x0, torch.full((1,), float(t), device=device))[0, pos]
        mask = torch.full_like(logits, float("-inf"))
        mask[allow] = 0.0
        z = logits + mask
        idx = int(torch.argmax(z).item())
        logp = torch.log_softmax(z, dim=-1)[idx].item()
        cum_logp += float(logp)
        x[0, pos] = idx

    lattice_idx_set = set()
    for name in lat_pos:
        for kind in ("BIN","DR"):
            for (slot_idx, _) in lat_pos[name][kind]:
                lattice_idx_set.add(slot_idx)

    # 3) 先处理 lattice（若绑定）
    if tying:
        # masters
        for name in tying["masters"]:
            arr = lat_pos[name]["BIN"]
            if arr:
                slot_idx, pos = arr[0]
                allow = slots[slot_idx][1]
                masked_pick(pos, allow)
            arrdr = lat_pos[name]["DR"]
            if arrdr:
                slot_idx_dr, pos_dr = arrdr[0]
                allow_dr = slots[slot_idx_dr][1]
                x[0, pos_dr] = int(allow_dr[len(allow_dr)//2])  # DR 置中位≈0
        # slaves
        for name, spec in tying["slaves"]:
            # BIN
            arr = lat_pos[name]["BIN"]
            if arr:
                slot_idx, pos = arr[0]
                allow = slots[slot_idx][1]
                if isinstance(spec, tuple) and spec[0] == "fix":
                    target_bin_idx = spec[1]
                    tokname = f"{name.upper()}_BIN_{target_bin_idx}"
                    tokid = v._tok2id.get(tokname, None)
                    x[0, pos] = tokid if (tokid is not None and tokid in allow) else int(allow[0])
                else:
                    mname = spec  # 复制 master
                    marr = lat_pos[mname]["BIN"]
                    if marr:
                        _, mpos = marr[0]
                        x[0, pos] = int(x[0, mpos].item())
            # DR
            arrdr = lat_pos[name]["DR"]
            if arrdr:
                slot_idx_dr, pos_dr = arrdr[0]
                allow_dr = slots[slot_idx_dr][1]
                x[0, pos_dr] = int(allow_dr[len(allow_dr)//2])
    # 4) 其余槽位按白名单贪心填充
    for slot_idx, (pos, allow) in enumerate(slots):
        if tying and slot_idx in lattice_idx_set:
            continue
        if int(x[0, pos].item()) != v.token_id("<MASK>"):
            continue
        masked_pick(pos, allow)

    out_ids = x[0].tolist()
    return (out_ids, cum_logp) if return_score else out_ids


def decode_ms_from_ids(tok: MaterialTokenizer, ids: List[int], vocab_yaml: str) -> Dict[str, Any]:
    """
    仅用于把 lattice+params token 反量化回数值，形成 ms（便于后续写 CIF）
    假设 ids 来自 _build_masked_inputs 的布局
    """
    v = tok.vocab
    toks = [v.id_token(i) for i in ids]
    # 读取 SG
    sg = None
    for i, t in enumerate(toks):
        if t == "SG=":
            sg_tok = toks[i+1]
            if sg_tok.startswith("SG_"):
                sg = int(sg_tok.split("_")[1])
                break
    assert sg is not None, "未找到 SG"

    # lattice 反量化
    cfg = load_vocab_yaml(vocab_yaml)["lattice_bins"]
    def read_latt(name: str, start_idx: int) -> Tuple[int,int]:
        # 找第一个出现的 name 对应 BIN/DR 的 token 索引并解析
        # 这里简单扫描：BIN token 形如 A_BIN_*, DR token 形如 A_DR_*
        bin_tok_prefix = f"{name.upper()}_BIN_"
        dr_tok_prefix  = f"{name.upper()}_DR_"
        b_idx = d_idx = None
        for j in range(start_idx, len(toks)):
            if toks[j].startswith(bin_tok_prefix):
                b_idx = int(toks[j][len(bin_tok_prefix):])
                break
        for j in range(start_idx, len(toks)):
            if toks[j].startswith(dr_tok_prefix):
                d_idx = int(toks[j][len(dr_tok_prefix):])
                break
        return b_idx, d_idx

    latt = {}
    order = ["a","b","c","alpha","beta","gamma"]
    cur = 0
    # 直接扫描 tokens
    for name in order:
        b_idx = None
        for j,t in enumerate(toks):
            if t.startswith(f"{name.upper()}_BIN_"):
                b_idx = int(t.split("_")[-1]); break
        d_idx = None
        for j,t in enumerate(toks):
            if t.startswith(f"{name.upper()}_DR_"):
                d_idx = int(t.split("_")[-1]); break
        if b_idx is None or d_idx is None:
            # 兜底：给个中值
            b_idx = int(cfg[name]["num_bins"])//2
            d_idx = 0
        val = inv_bin_lattice_scalar(b_idx, d_idx, cfg[name])
        latt[name] = {"bin": b_idx, "dr": d_idx, "value": float(val)}

    # 参数
    sites: List[Dict[str, Any]] = []
    i = 0
    # 粗暴重解析，按编码格式读
    # 如果你要严格可逆解码，建议后续实现 tokenizer.decode；此处只为生成 CIF 服务
    # 我们直接复用 material_string 中的 site 列表结构
    # 简化：从 token 序列里再读一次 EL/WY/PARAM
    cur_site: Dict[str, Any] = None
    k = 0
    while k < len(toks):
        t = toks[k]
        if t == "(":
            cur_site = {"el": None, "wy": None, "params": "-"}
        elif t == "EL:":
            cur_site["el"] = toks[k+1]
        elif t == "WY:":
            wy_tok = toks[k+1]
            # 现在形如 "WY:SG_194:4f"
            assert wy_tok.startswith("WY:")
            parts = wy_tok.split(":")
            wy = parts[-1]            # 取最后一段 "4f"
            cur_site["wy"] = wy
        elif t == "PARAM:":
            # 尝试读取轴标签与 BASE_* token
            params = {}
            m = k+1
            while m < len(toks) and toks[m] != ")" and toks[m] != "->":
                if toks[m] in ("u:", "v:", "w:"):
                    ax = toks[m][0]
                    m += 1
                    tt = toks[m]
                    if tt.startswith("BASE_"):
                        base = tt.split("BASE_")[1]
                        params[ax] = {"mode":"BASE", "base": base}
                    else:
                        # 不认识的，跳过
                        pass
                m += 1
            cur_site["params"] = params if params else "-"
        elif t == ")":
            sites.append(cur_site)
        k += 1

    ms = {
        "formula": "",  # 采样后可自行填
        "n_atoms": sum(int(s["wy"][:-1]) for s in sites),
        "sg": sg,
        "latt": latt,
        "sites": sites,
    }
    return ms
