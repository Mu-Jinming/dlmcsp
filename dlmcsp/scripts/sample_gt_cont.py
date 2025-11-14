#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, math
from pathlib import Path

import torch

from dlmcsp.models.llada import LLaDA
from dlmcsp.models.dof_head import DOFHead
from dlmcsp.tokenization.tokenizer import MaterialTokenizer
from dlmcsp.sampling.phased_sampler import (
    _build_masked_inputs, _position_slots_for_lattice, _position_slots_for_params,
    build_ms_skeleton, decode_ms_from_ids, fill_with_llada
)
from dlmcsp.constraints.wyckoff_gt import build_plan_from_gt
from dlmcsp.representation.ms_to_cif import ms_to_cif

def parse_formula(formula: str):
    from pymatgen.core import Composition
    comp = Composition(formula)
    el_counts = {el.symbol: int(round(float(amount))) for el, amount in comp.items()}
    natoms = int(sum(el_counts.values()))
    pretty = comp.get_reduced_formula_and_factor()[0]
    return el_counts, natoms, pretty

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--formula", required=True)
    ap.add_argument("--spacegroup", type=int, required=True)
    ap.add_argument("--wyckoff_letters", required=True)   # e.g. "4f,4f"
    ap.add_argument("--elements", required=True)          # e.g. "Ga,Te"
    ap.add_argument("--num", type=int, default=1)
    ap.add_argument("--t", type=float, default=0.15)
    args = ap.parse_args()

    device = args.device
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.ckpt, map_location=device)
    tok = MaterialTokenizer.from_yaml(args.vocab)
    if "runtime_vocab" in ck:
        tok.vocab._tok2id.clear(); tok.vocab._id2tok.clear()
        for tkn in ck["runtime_vocab"]:
            tok.vocab._add(tkn)
    cfg = ck["cfg"]
    model = LLaDA(vocab_size=len(tok.vocab._id2tok),
                  hidden=cfg["hidden"], n_layers=cfg["layers"], n_heads=cfg["heads"]).to(device)
    model.load_state_dict(ck["model"], strict=True); model.eval()

    dof = DOFHead(hidden=cfg["hidden"]).to(device)
    dof.load_state_dict(ck["dof_head"], strict=True); dof.eval()

    wy_in = [w.strip() for w in args.wyckoff_letters.split(",")]
    els   = [e.strip() for e in args.elements.split(",")]
    base_counts, _, pretty = parse_formula(args.formula)

    Z, natoms, plan = build_plan_from_gt(args.spacegroup, wy_in, els, base_counts, args.vocab)
    template = [(p.wy, p.element) for p in plan]
    ms0 = build_ms_skeleton(pretty, natoms, args.spacegroup, template, args.vocab)

    for i in range(args.num):
        lattice_slots = _position_slots_for_lattice(tok.vocab)
        param_slots   = _position_slots_for_params(tok.vocab, ms0)
        ids, slots, _ = _build_masked_inputs(tok, ms0, lattice_slots, param_slots)

        # 先离散填充（含晶系 tying）
        ids_filled = fill_with_llada(model, tok, ids, slots, device=device, t=args.t,
                                     sg_for_lattice_tying=args.spacegroup, vocab_yaml=args.vocab, return_score=False)
        ms = decode_ms_from_ids(tok, ids_filled, args.vocab)
        ms["formula"] = pretty

        # 连续头预测 value 写回
        x = torch.tensor(ids_filled, dtype=torch.long, device=device)[None,:]
        t = torch.full((1,), float(args.t), device=device)
        _, h = model(x, t, return_hidden=True)
        h = h[0]
        id2tok = tok.vocab._id2tok

        # 写 lattice 值
        for pos, tid in enumerate(ids_filled):
            name = id2tok[tid].upper()
            if "A_BIN_" in name:
                mu, _ = dof.norm_params(h[pos:pos+1, :]); ms["latt"]["a"]["value"] = float(math.exp(mu.item())); continue
            if "B_BIN_" in name:
                mu, _ = dof.norm_params(h[pos:pos+1, :]); ms["latt"]["b"]["value"] = float(math.exp(mu.item())); continue
            if "C_BIN_" in name:
                mu, _ = dof.norm_params(h[pos:pos+1, :]); ms["latt"]["c"]["value"] = float(math.exp(mu.item())); continue
            if "ALPHA_BIN_" in name:
                mu, _ = dof.norm_params(h[pos:pos+1, :]); ms["latt"]["alpha"]["value"] = float(mu.item()*180.0/math.pi); continue
            if "BETA_BIN_" in name:
                mu, _ = dof.norm_params(h[pos:pos+1, :]); ms["latt"]["beta"]["value"]  = float(mu.item()*180.0/math.pi); continue
            if "GAMMA_BIN_" in name:
                mu, _ = dof.norm_params(h[pos:pos+1, :]); ms["latt"]["gamma"]["value"] = float(mu.item()*180.0/math.pi); continue

        # 写 u/v/w 值：按 BASE_/FINE_ 出现顺序依次写到第一个尚未赋值的 param 上
        for pos, tid in enumerate(ids_filled):
            name = id2tok[tid].upper()
            if name.startswith("BASE_") or name.startswith("FINE_"):
                cos_mu, sin_mu, _ = dof.vm_params(h[pos:pos+1, :])
                theta = math.atan2(float(sin_mu.item()), float(cos_mu.item()))
                uvw = (theta / (2*math.pi)) % 1.0
                for site in ms["sites"]:
                    p = site.get("params", {})
                    for kk in ("u","v","w"):
                        if kk in p and "value" not in p[kk]:
                            p[kk]["value"] = uvw
                            break
                    else:
                        continue
                    break

        out_ms = Path(args.outdir) / f"sample_gtc_{i:03d}.ms.json"
        out_cif = Path(args.outdir) / f"sample_gtc_{i:03d}.cif"
        with open(out_ms, "w", encoding="utf-8") as f:
            json.dump(ms, f, ensure_ascii=False, indent=2)
        with open(out_cif, "w", encoding="utf-8") as f:
            f.write(ms_to_cif(ms, args.vocab))
        print(f"[OK] {out_ms} {out_cif}")

if __name__ == "__main__":
    main()
