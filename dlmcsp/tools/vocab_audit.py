# -*- coding: utf-8 -*-
"""
vocab_audit.py
- 审计并(可选)修复 vocab.yaml:
  * SG_1..SG_230 是否齐全且无重复
  * wy_tokens: 键与 sg_tokens 一致，值全为小写且形如 r'^\d+[a-z]$'
  * special_tokens/element_tokens 无重复
  * lattice_bins: alpha/beta/gamma 范围为 [30.0,150.0]
用法：
  python -m dlmcsp.tools.vocab_audit --yaml /path/to/vocab.yaml
  python -m dlmcsp.tools.vocab_audit --yaml /path/to/vocab.yaml --autofix-lower --enforce-angle-range --out /path/to/vocab.cleaned.yaml
"""
from __future__ import annotations
import argparse, re, sys, copy
from typing import Dict, Any, List

import yaml

WY_RE = re.compile(r"^\d+[a-z]$")

def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _save_yaml(obj: Dict[str,Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)

def _uniq(seq: List[str]) -> bool:
    return len(seq) == len(set(seq))

def audit_vocab(vb: Dict[str,Any]) -> Dict[str, Any]:
    rep: Dict[str, Any] = {"errors": [], "warnings": [], "stats": {}}

    # 1) SG tokens
    sg = vb.get("sg_tokens", [])
    if not sg:
        rep["errors"].append("missing sg_tokens")
    else:
        # 期望 SG_1..SG_230
        expect = [f"SG_{i}" for i in range(1, 231)]
        if sg != expect:
            # 允许乱序但要求齐全
            if set(sg) != set(expect):
                rep["errors"].append("sg_tokens not equal to SG_1..SG_230")
            else:
                rep["warnings"].append("sg_tokens order differs from SG_1..SG_230")
        if not _uniq(sg):
            rep["errors"].append("sg_tokens duplicated")

    # 2) wy_tokens
    wy = vb.get("wy_tokens", {})
    if not isinstance(wy, dict) or not wy:
        rep["errors"].append("missing wy_tokens dict")
    else:
        # 键覆盖
        sg_set = set(sg) if sg else set([f"SG_{i}" for i in range(1,231)])
        wy_keys = set(wy.keys())
        missing = sg_set - wy_keys
        extra   = wy_keys - sg_set
        if missing:
            rep["errors"].append(f"wy_tokens missing keys: {sorted(missing)[:10]}{'...' if len(missing)>10 else ''}")
        if extra:
            rep["warnings"].append(f"wy_tokens has extra keys: {sorted(extra)[:10]}{'...' if len(extra)>10 else ''}")

        invalid_case = []
        invalid_form = []
        empty_groups = []
        total_wy = 0
        for k, v in wy.items():
            if not isinstance(v, list) or len(v) == 0:
                empty_groups.append(k); continue
            total_wy += len(v)
            for s in v:
                if not isinstance(s, str):
                    invalid_form.append((k, str(s))); continue
                if s != s.lower():
                    invalid_case.append((k, s))
                if not WY_RE.match(s.lower()):
                    invalid_form.append((k, s))
        if empty_groups:
            rep["warnings"].append(f"{len(empty_groups)} SGs have empty wy list (e.g., {empty_groups[:5]})")
        if invalid_case:
            rep["warnings"].append(f"{len(invalid_case)} wy entries not lowercase (e.g., {invalid_case[:5]})")
        if invalid_form:
            rep["errors"].append(f"{len(invalid_form)} wy entries not match '^\\d+[a-z]$' (e.g., {invalid_form[:5]})")

        rep["stats"]["total_wy_entries"] = total_wy

    # 3) special / element tokens
    sp = vb.get("special_tokens", [])
    el = vb.get("element_tokens", [])
    if not _uniq(sp):
        rep["errors"].append("special_tokens has duplicates")
    if not _uniq(el):
        rep["errors"].append("element_tokens has duplicates")
    rep["stats"]["num_elements"] = len(el)

    # 4) lattice_bins 范围
    lb = vb.get("lattice_bins", {})
    for ang in ("alpha","beta","gamma"):
        if ang not in lb:
            rep["errors"].append(f"lattice_bins missing {ang}")
            continue
        rng = lb[ang].get("range")
        if not isinstance(rng, list) or len(rng) != 2:
            rep["errors"].append(f"{ang} range malformatted")
            continue
        lo, hi = float(rng[0]), float(rng[1])
        if not (abs(lo-30.0)<1e-6 and abs(hi-150.0)<1e-6):
            rep["warnings"].append(f"{ang} range is [{lo},{hi}] (recommended [30.0,150.0])")

    return rep

def autofix(vb: Dict[str,Any], fix_lower: bool, enforce_angle: bool) -> Dict[str,Any]:
    vb2 = copy.deepcopy(vb)
    changed = False

    # 小写 wy
    if fix_lower and "wy_tokens" in vb2 and isinstance(vb2["wy_tokens"], dict):
        for k, lst in vb2["wy_tokens"].items():
            if isinstance(lst, list):
                new = [s.lower() if isinstance(s,str) else s for s in lst]
                if new != lst:
                    vb2["wy_tokens"][k] = new
                    changed = True

    # 强制角度域
    if enforce_angle and "lattice_bins" in vb2:
        for ang in ("alpha","beta","gamma"):
            if ang in vb2["lattice_bins"]:
                node = vb2["lattice_bins"][ang]
                if node.get("range") != [30.0, 150.0]:
                    node["range"] = [30.0, 150.0]
                    changed = True

    return vb2, changed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True, help="path to vocab.yaml")
    ap.add_argument("--out", default="", help="optional path to save cleaned yaml")
    ap.add_argument("--autofix_lower", action="store_true", help="auto lower-case wy letters")
    ap.add_argument("--enforce_angle_range", action="store_true", help="force alpha/beta/gamma range to [30,150]")
    args = ap.parse_args()

    vb = _load_yaml(args.yaml)
    rep = audit_vocab(vb)

    print("=== VOCAB AUDIT ===")
    for k,v in rep["stats"].items():
        print(f"{k}: {v}")
    if rep["warnings"]:
        print("\nWarnings:")
        for w in rep["warnings"]:
            print(" -", w)
    if rep["errors"]:
        print("\nErrors:")
        for e in rep["errors"]:
            print(" -", e)

    if args.autofix_lower or args.enforce_angle_range:
        vb2, changed = autofix(vb, args.autofix_lower, args.enforce_angle_range)
        if changed and args.out:
            _save_yaml(vb2, args.out)
            print(f"\n[FIX] saved cleaned yaml to {args.out}")
        elif changed and not args.out:
            print("\n[FIX] changes were made but --out not provided; nothing saved.")
        else:
            print("\n[FIX] no changes necessary.")
    else:
        print("\n(no autofix requested)")

    # 以非 0 退出仅在有 errors 时
    if rep["errors"]:
        sys.exit(2)

if __name__ == "__main__":
    main()
