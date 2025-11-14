# -*- coding: utf-8 -*-
# dlmcsp/scripts/sample_gt.py
from __future__ import annotations
import argparse, json
from pathlib import Path

import torch
from pymatgen.core import Composition

from dlmcsp.models.llada import LLaDA
from dlmcsp.tokenization.tokenizer import MaterialTokenizer
from dlmcsp.sampling.phased_sampler import (
    _build_masked_inputs, _position_slots_for_lattice, _position_slots_for_params,
    build_ms_skeleton, decode_ms_from_ids
)
from dlmcsp.constraints.wyckoff_gt import build_plan_from_gt
from dlmcsp.sampling.phased_sampler import fill_with_llada
from dlmcsp.representation.ms_to_cif import ms_to_cif


def parse_formula(formula: str):
    comp = Composition(formula)
    el_counts = {el.symbol: int(round(float(amount))) for el, amount in comp.items()}
    natoms = int(sum(el_counts.values()))
    pretty = comp.get_reduced_formula_and_factor()[0]
    return el_counts, natoms, pretty


def load_model_and_tokenizer(ckpt_path: str, vocab_yaml: str, device: str):
    # 与你现有 sample_comp.py 对齐：从 ckpt["cfg"]恢复结构
    ck = torch.load(ckpt_path, map_location=device)
    tok = MaterialTokenizer.from_yaml(vocab_yaml)
    # 若训练时保存了 runtime_vocab，则复现顺序，防止错位
    if "runtime_vocab" in ck:
        tok.vocab._tok2id.clear()
        tok.vocab._id2tok.clear()
        for t in ck["runtime_vocab"]:
            tok.vocab._add(t)
    cfg = ck["cfg"]
    model = LLaDA(
        vocab_size=len(tok.vocab._id2tok),
        hidden=cfg["hidden"], n_layers=cfg["layers"], n_heads=cfg["heads"]
    ).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--formula", required=True, help="e.g., 'GaTe' or 'Na3MnCoNiO6'")
    ap.add_argument("--spacegroup", type=int, required=True, help="e.g., 194")
    ap.add_argument("--wyckoff_letters", required=True,
                    help="comma-separated or compact letters. e.g. '4f,4f' or 'ff' for SG=194")
    ap.add_argument("--elements", required=True, help="comma-separated, one per Wy entry, e.g., 'Ga,Te'")
    ap.add_argument("--num", type=int, default=1)
    args = ap.parse_args()

    device = args.device
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    model, tok = load_model_and_tokenizer(args.ckpt, args.vocab, device=device)

    # 解析输入
    if "," in args.wyckoff_letters:
        wy_in = [w.strip() for w in args.wyckoff_letters.split(",")]
    else:
        wy_in = [args.wyckoff_letters.strip()]
    els = [e.strip() for e in args.elements.split(",")]
    base_counts, _, pretty = parse_formula(args.formula)

    # 从 GT 构造 plan / 计算 Z 与 natoms
    Z, natoms, plan = build_plan_from_gt(args.spacegroup, wy_in, els, base_counts, args.vocab)
    template = [(p.wy, p.element) for p in plan]

    # material-string 骨架（SG/模板锁定；参数占位）
    ms0 = build_ms_skeleton(pretty, natoms, args.spacegroup, template, args.vocab)

    for i in range(args.num):
        lattice_slots = _position_slots_for_lattice(tok.vocab)
        param_slots   = _position_slots_for_params(tok.vocab, ms0)
        ids, slots, _ = _build_masked_inputs(tok, ms0, lattice_slots, param_slots)

        # 晶系绑定 + 贪心填槽（只填 lattice/param）
        ids_filled = fill_with_llada(
            model, tok, ids, slots, device=device, t=0.15,
            sg_for_lattice_tying=args.spacegroup, vocab_yaml=args.vocab, return_score=False
        )
        ms_out = decode_ms_from_ids(tok, ids_filled, args.vocab)
        ms_out["formula"] = pretty

        ms_path = Path(args.outdir) / f"sample_gt_{i:03d}.ms.json"
        with open(ms_path, "w", encoding="utf-8") as f:
            json.dump(ms_out, f, ensure_ascii=False, indent=2)

        cif_txt = ms_to_cif(ms_out, args.vocab)
        cif_path = Path(args.outdir) / f"sample_gt_{i:03d}.cif"
        with open(cif_path, "w", encoding="utf-8") as f:
            f.write(cif_txt)

        print(f"[OK] Z={Z} natoms={natoms}  {ms_path}  {cif_path}")


if __name__ == "__main__":
    main()
