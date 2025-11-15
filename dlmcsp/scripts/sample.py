#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLaDA-cont 条件采样（离散 + 连续 DOF）：

两种使用模式：

1) 手动指定条件（单个 composition 模式）
   例：
   python -m dlmcsp.scripts.sample \
     --ckpt /path/to/llada_cont.best.pt \
     --vocab /path/to/vocab.yaml \
     --outdir /path/to/outdir \
     --device cuda \
     --formula "GaTe" \
     --spacegroup 194 \
     --wyckoff_letters "4f,4f" \
     --elements "Ga,Te" \
     --num 2 --t 0.15

   输出：
     outdir/sample_manual_000.ms.json
     outdir/sample_manual_000.cif
     outdir/sample_manual_001.ms.json
     outdir/sample_manual_001.cif
     ...

2) 从 GT ms.jsonl 中抽条件（对标 diffcsp++ CSP）
   例：
   python -m dlmcsp.scripts.sample \
     --ckpt /path/to/llada_cont.best.pt \
     --vocab /path/to/vocab.yaml \
     --outdir /path/to/outdir \
     --device cuda \
     --cond_ms /path/to/test.clean.ms.jsonl \
     --out_jsonl /path/to/llada_cont.test_samples.jsonl \
     --t 0.15

   对每条 GT 记录：
     - 只读 formula, sg, sites[*].(wy, el) 作为条件；
     - 用 build_plan_from_gt + build_ms_skeleton 构造 skeleton；
     - 用 LLaDA 填离散 tokens，再用 DOFHead 写回连续 lattice/uvw；
     - ms_to_structure + quick_validate_structure 做几何检查；
     - 写出一条记录：
       {
         "orig_material_id": ...,
         "formula": ...,
         "sg": ...,
         "geom_ok": true/false,
         "geom_why": "...",
         "ms_pred": {...}
       }
"""

from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm

from dlmcsp.models.llada import LLaDA
from dlmcsp.models.dof_head import DOFHead
from dlmcsp.tokenization.tokenizer import MaterialTokenizer
from dlmcsp.sampling.phased_sampler import (
    parse_formula,
    build_ms_skeleton,
    _position_slots_for_lattice,
    _position_slots_for_params,
    _build_masked_inputs,
    fill_with_llada,
)
from dlmcsp.constraints.wyckoff_gt import build_plan_from_gt
from dlmcsp.representation.ms_to_cif import ms_to_structure, ms_to_cif
from dlmcsp.eval.validators import quick_validate_structure


# ========= 工具函数 =========

def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            recs.append(json.loads(ln))
    return recs


def _write_jsonl(path: str, recs: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_back_dof(
    ms: Dict[str, Any],
    ids_filled: List[int],
    h: torch.Tensor,              # [L, H]
    dof_head: DOFHead,
    tok: MaterialTokenizer,
) -> None:
    """
    用 DOFHead 的连续预测写回：
      - 晶格：log(a/b/c) & angle(rad) → a,b,c,alpha,beta,gamma 的 value
      - 参数：Base/Fine token → u/v/w ∈ [0,1)，按 ms["sites"] 顺序填第一个尚未赋值的 param
    """
    id2tok = tok.vocab._id2tok

    # ---- 写 lattice value ----
    for pos, tid in enumerate(ids_filled):
        name = id2tok[tid].upper()

        # 长度：exp(mu)
        if "A_BIN_" in name:
            mu, _ = dof_head.norm_params(h[pos:pos+1, :])
            ms["latt"]["a"]["value"] = float(math.exp(mu.item()))
            continue
        if "B_BIN_" in name:
            mu, _ = dof_head.norm_params(h[pos:pos+1, :])
            ms["latt"]["b"]["value"] = float(math.exp(mu.item()))
            continue
        if "C_BIN_" in name:
            mu, _ = dof_head.norm_params(h[pos:pos+1, :])
            ms["latt"]["c"]["value"] = float(math.exp(mu.item()))
            continue

        # 角度：mu 为弧度
        if "ALPHA_BIN_" in name:
            mu, _ = dof_head.norm_params(h[pos:pos+1, :])
            ms["latt"]["alpha"]["value"] = float(mu.item() * 180.0 / math.pi)
            continue
        if "BETA_BIN_" in name:
            mu, _ = dof_head.norm_params(h[pos:pos+1, :])
            ms["latt"]["beta"]["value"] = float(mu.item() * 180.0 / math.pi)
            continue
        if "GAMMA_BIN_" in name:
            mu, _ = dof_head.norm_params(h[pos:pos+1, :])
            ms["latt"]["gamma"]["value"] = float(mu.item() * 180.0 / math.pi)
            continue

    # ---- 写 u/v/w value ----
    for pos, tid in enumerate(ids_filled):
        name = id2tok[tid].upper()
        if not (name.startswith("BASE_") or name.startswith("FINE_")):
            continue

        # von Mises 均值角 → [0,1) frac
        cos_mu, sin_mu, _ = dof_head.vm_params(h[pos:pos+1, :])
        theta = math.atan2(float(sin_mu.item()), float(cos_mu.item()))
        frac = (theta / (2.0 * math.pi)) % 1.0

        # 按 ms["sites"] 顺序，找到第一个还没有 value 的 u/v/w 填进去
        assigned = False
        for site in ms.get("sites", []):
            p = site.get("params", {})
            if p == "-" or p is None:
                continue
            for ax in ("u", "v", "w"):
                if ax in p:
                    node = p[ax]
                    if isinstance(node, dict):
                        if "value" in node:
                            continue
                        node["value"] = float(frac)
                    else:
                        p[ax] = {"value": float(frac)}
                    assigned = True
                    break
            if assigned:
                break


def _build_skeleton_from_cli(
    formula: str,
    sg: int,
    wyckoff_letters: str,
    elements: str,
    vocab_yaml: str,
) -> Tuple[Dict[str, Any], str, int]:
    """
    从命令行给定的条件构造 ms skeleton（不含连续 value，只包含 LATT 占位 + params 占位）。
    """
    wy_in = [w.strip() for w in wyckoff_letters.split(",") if w.strip()]
    els = [e.strip() for e in elements.split(",") if e.strip()]
    if len(wy_in) != len(els):
        raise ValueError(f"wyckoff_letters 与 elements 数量不一致: {len(wy_in)} vs {len(els)}")

    base_counts, _, pretty = parse_formula(formula)
    Z, natoms, plan = build_plan_from_gt(sg, wy_in, els, base_counts, vocab_yaml)
    template = [(p.wy, p.element) for p in plan]
    ms0 = build_ms_skeleton(pretty, natoms, sg, template, vocab_yaml)
    # formula 最终就用 pretty
    ms0["formula"] = pretty
    return ms0, pretty, natoms


def _build_skeleton_from_gt_ms(
    ms_gt: Dict[str, Any],
    vocab_yaml: str,
) -> Tuple[Dict[str, Any], str, int]:
    """
    从 GT material-string v2（一条记录）抽取条件：
      - formula（reduced）
      - sg
      - sites[*].(wy, el)
    然后走同样的 build_plan_from_gt → build_ms_skeleton。
    """
    formula = ms_gt.get("formula")
    sg = int(ms_gt.get("sg"))
    if formula is None:
        raise ValueError("GT ms 缺少 formula 字段")
    wy_in = [str(s["wy"]) for s in ms_gt.get("sites", [])]
    els = [str(s["el"]) for s in ms_gt.get("sites", [])]
    if len(wy_in) != len(els):
        raise ValueError("GT ms 的 sites 中 wy 与 el 数量不一致")

    base_counts, _, pretty = parse_formula(formula)
    Z, natoms, plan = build_plan_from_gt(sg, wy_in, els, base_counts, vocab_yaml)
    template = [(p.wy, p.element) for p in plan]
    ms0 = build_ms_skeleton(pretty, natoms, sg, template, vocab_yaml)
    # 为了和 GT 一致，可以直接沿用 GT 的 formula 字符串
    ms0["formula"] = formula
    return ms0, formula, natoms


def _load_model_and_tok(
    ckpt_path: str,
    vocab_path: str,
    device: torch.device,
) -> Tuple[LLaDA, DOFHead, MaterialTokenizer]:
    """
    从 ckpt 加载 LLaDA + DOFHead + tokenizer。

    重要约定：**不再**在推理阶段扩展 vocab，只使用 vocab.yaml。
    若 ckpt 中存在 runtime_vocab，只用于 sanity check，不会覆盖当前 vocab。
    """
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ck["cfg"]
    stab = ck["stab"]
    runtime_vocab = ck.get("runtime_vocab", None)

    tok = MaterialTokenizer.from_yaml(vocab_path)
    vb_tokens = tok.vocab._id2tok

    if runtime_vocab is not None:
        # 做一个简单一致性检查，但不覆盖
        rv = list(runtime_vocab)
        if len(rv) != len(vb_tokens) or any(a != b for a, b in zip(rv, vb_tokens)):
            print(
                "[WARN] ckpt.runtime_vocab 与当前 vocab.yaml 不一致；"
                "推理阶段将 **忽略** runtime_vocab，仅使用 vocab.yaml。"
            )
        else:
            print("[INFO] ckpt.runtime_vocab 与 vocab.yaml 一致，安全使用 vocab.yaml。")
    else:
        print("[INFO] ckpt 中无 runtime_vocab 字段，仅使用 vocab.yaml。")

    V = len(vb_tokens)
    model = LLaDA(
        vocab_size=V,
        hidden=cfg["hidden"],
        n_layers=cfg["layers"],
        n_heads=cfg["heads"],
    ).to(device)
    dof_head = DOFHead(
        hidden=cfg["hidden"],
        kappa_max=stab["vm_kappa_max"],
        sigma_min=stab["sigma_min"],
        sigma_max=stab["sigma_max"],
    ).to(device)

    model.load_state_dict(ck["model"])
    dof_head.load_state_dict(ck["dof_head"])
    model.eval()
    dof_head.eval()
    print("[INFO] model & DOFHead loaded.")

    return model, dof_head, tok


# ========= 模式 A：命令行条件采样 =========

def run_manual_mode(args):
    device = torch.device(args.device)
    model, dof_head, tok = _load_model_and_tok(args.ckpt, args.vocab, device)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    ms0, pretty, natoms = _build_skeleton_from_cli(
        formula=args.formula,
        sg=int(args.spacegroup),
        wyckoff_letters=args.wyckoff_letters,
        elements=args.elements,
        vocab_yaml=args.vocab,
    )

    print(f"[INFO] manual skeleton built: formula={pretty}, sg={args.spacegroup}, natoms={natoms}")

    lattice_slots = _position_slots_for_lattice(tok.vocab)
    param_slots = _position_slots_for_params(tok.vocab, ms0)

    for i in range(args.num):
        # 1) 构造带 <MASK> 的 token 序列
        ids, slots, _ = _build_masked_inputs(tok, ms0, lattice_slots, param_slots)

        # 2) 用 LLaDA 填离散 token（带晶系绑定）
        ids_filled = fill_with_llada(
            model,
            tok,
            ids,
            slots,
            device=str(device),
            t=float(args.t),
            sg_for_lattice_tying=int(args.spacegroup),
            vocab_yaml=args.vocab,
            return_score=False,
        )

        # 3) 用 DOFHead 写回连续 DOF
        x = torch.tensor(ids_filled, dtype=torch.long, device=device)[None, :]
        t_tensor = torch.full((1,), float(args.t), device=device)
        _, h = model(x, t_tensor, return_hidden=True)
        h = h[0]  # [L, H]

        ms_pred = json.loads(json.dumps(ms0))  # 深拷贝一个
        _write_back_dof(ms_pred, ids_filled, h, dof_head, tok)

        # 4) 尝试构造 Structure 并导出 CIF
        try:
            struct = ms_to_structure(ms_pred, args.vocab)
            ok, why = quick_validate_structure(struct)
        except Exception as e:
            ok = False
            why = f"ms_to_structure_fail:{type(e).__name__}:{str(e)}"

        tag = f"manual_{i:03d}"
        out_ms = Path(args.outdir) / f"sample_{tag}.ms.json"
        out_cif = Path(args.outdir) / f"sample_{tag}.cif"

        with open(out_ms, "w", encoding="utf-8") as f:
            json.dump(ms_pred, f, ensure_ascii=False, indent=2)
        with open(out_cif, "w", encoding="utf-8") as f:
            f.write(ms_to_cif(ms_pred, args.vocab))

        print(f"[OK] {out_ms} {out_cif} | geom_ok={ok} why={why}")


# ========= 模式 B：从 GT ms.jsonl 抽条件采样 =========

def run_cond_ms_mode(args):
    device = torch.device(args.device)
    model, dof_head, tok = _load_model_and_tok(args.ckpt, args.vocab, device)

    gt_recs = _load_jsonl(args.cond_ms)
    n_total = len(gt_recs)
    if args.limit and args.limit > 0:
        gt_recs = gt_recs[: args.limit]
    print(f"[INFO] cond_ms mode: loaded {n_total} GT records, using {len(gt_recs)} for sampling")

    out_recs: List[Dict[str, Any]] = []

    for idx, ms_gt in enumerate(tqdm(gt_recs, desc="cond-sampling")):
        mid = ms_gt.get("material_id")

        try:
            ms0, formula_used, natoms = _build_skeleton_from_gt_ms(ms_gt, args.vocab)
        except Exception as e:
            print(f"[WARN] build_skeleton failed for material_id={mid}: {type(e).__name__}:{e}")
            continue

        lattice_slots = _position_slots_for_lattice(tok.vocab)
        param_slots = _position_slots_for_params(tok.vocab, ms0)

        ids, slots, _ = _build_masked_inputs(tok, ms0, lattice_slots, param_slots)

        sg = int(ms0["sg"])
        ids_filled = fill_with_llada(
            model,
            tok,
            ids,
            slots,
            device=str(device),
            t=float(args.t),
            sg_for_lattice_tying=sg,
            vocab_yaml=args.vocab,
            return_score=False,
        )

        x = torch.tensor(ids_filled, dtype=torch.long, device=device)[None, :]
        t_tensor = torch.full((1,), float(args.t), device=device)
        _, h = model(x, t_tensor, return_hidden=True)
        h = h[0]

        ms_pred = json.loads(json.dumps(ms0))
        _write_back_dof(ms_pred, ids_filled, h, dof_head, tok)

        # 结构级几何检查
        try:
            struct = ms_to_structure(ms_pred, args.vocab)
            ok, why = quick_validate_structure(struct)
        except Exception as e:
            ok = False
            why = f"ms_to_structure_fail:{type(e).__name__}:{str(e)}"

        rec_out = {
            "orig_material_id": mid,
            "formula": ms_gt.get("formula", formula_used),
            "sg": sg,
            "geom_ok": ok,
            "geom_why": why,
            "ms_pred": ms_pred,
        }
        out_recs.append(rec_out)

    if args.out_jsonl:
        _write_jsonl(args.out_jsonl, out_recs)
        print(f"[DONE] wrote {len(out_recs)} records to {args.out_jsonl}")
    else:
        print(f"[DONE] sampled {len(out_recs)} records (no out_jsonl specified)")


# ========= 主入口 =========

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="训练好的 LLaDA-cont ckpt 路径（*.pt）")
    ap.add_argument("--vocab", required=True, help="vocab.yaml 路径")
    ap.add_argument("--outdir", required=True, help="输出目录（manual 模式下写 ms/cif）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--t", type=float, default=0.15, help="LLaDA time step 用于采样")

    # --- 模式 A：手动指定条件 ---
    ap.add_argument("--formula", type=str, default="", help="manual 模式下的化学式，例如 'GaTe'")
    ap.add_argument("--spacegroup", type=int, default=0, help="manual 模式下的空间群号")
    ap.add_argument("--wyckoff_letters", type=str, default="", help="手动 Wyckoff，例如 '4f,4f'")
    ap.add_argument("--elements", type=str, default="", help="手动元素列表，例如 'Ga,Te'")
    ap.add_argument("--num", type=int, default=1, help="manual 模式下每个条件采样多少个结构")

    # --- 模式 B：从 ms.jsonl 抽条件 ---
    ap.add_argument("--cond_ms", type=str, default="", help="若提供，则从该 ms.jsonl 中抽条件（formula+sg+wy+el）")
    ap.add_argument("--limit", type=int, default=0, help="cond_ms 模式下最多使用前 N 条 GT")
    ap.add_argument("--out_jsonl", type=str, default="", help="cond_ms 模式下输出预测 jsonl 路径")

    args = ap.parse_args()

    cond_mode = bool(args.cond_ms)

    if cond_mode:
        run_cond_ms_mode(args)
    else:
        if not (args.formula and args.spacegroup and args.wyckoff_letters and args.elements):
            raise SystemExit(
                "manual 模式需要同时指定 --formula, --spacegroup, --wyckoff_letters, --elements；"
                "或者改用 --cond_ms 走 GT 条件模式。"
            )
        run_manual_mode(args)


if __name__ == "__main__":
    main()
