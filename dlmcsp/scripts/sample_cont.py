#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
采样连续 DOF + 结构级评估：

- 从 ms.jsonl 读取样本（与训练用格式一致）；
- 用训练好的 LLaDA + DOFHead 前向，取连续 DOF 的预测均值；
- 写回新的 ms_pred（修改 latt[*].value 和 sites[*].params[*].value）；
- 用 ms_to_structure + quick_validate_structure 做几何验证；
- 打印重构误差统计。

注意：
- 这里只做“重构”而不是完整生成：离散 token 不变，只重写连续 DOF。
"""

from __future__ import annotations
import argparse, json, math, os, copy
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from dlmcsp.models.llada import LLaDA
from dlmcsp.models.dof_head import DOFHead
from dlmcsp.tokenization.tokenizer import MaterialTokenizer
from dlmcsp.representation.ms_to_cif import ms_to_structure
from dlmcsp.eval.validators import quick_validate_structure


# ========= 复用训练脚本里的 dataset + DOF 收集逻辑 =========

class MsJsonlDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.recs: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                obj = json.loads(ln)
                assert "tokens" in obj and "latt" in obj and "sites" in obj
                self.recs.append(obj)
        if not self.recs:
            raise ValueError(f"empty dataset: {path}")

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        return self.recs[i]


def _flatten_uvw_values(ms: Dict[str, Any]) -> List[float]:
    vals: List[float] = []
    for s in ms.get("sites", []):
        p = s.get("params", {})
        if p == "-" or p is None:
            continue
        for k in ("u", "v", "w"):
            if k in p and isinstance(p[k], dict) and "value" in p[k]:
                vals.append(float(p[k]["value"]) % 1.0)
    return vals


def _collect_positions_and_values(
    ids: List[int],
    ms: Dict[str, Any],
    id2tok: List[str],
):
    """
    与训练脚本一致：根据 token 序列和 ms，收集：
    - vm_pos: 与 u/v/w 对应的 token 位置（BASE_ / FINE_）
    - vm_theta: 目标角度 2πv
    - ga_pos: 与 lattice DOF 对应的 token 位置（A_BIN_ / B_BIN_ / ...）
    - ga_x: 对应的 log(a/b/c) 或 angle(rad)
    """
    vm_pos: List[int] = []
    vm_theta: List[float] = []
    ga_pos: List[int] = []
    ga_x: List[float] = []

    uvw_vals = _flatten_uvw_values(ms)
    uvw_i = 0

    latt = ms["latt"]
    a_val = float(latt["a"]["value"]) if "value" in latt["a"] else None
    b_val = float(latt["b"]["value"]) if "value" in latt["b"] else None
    c_val = float(latt["c"]["value"]) if "value" in latt["c"] else None
    al_val = float(latt["alpha"]["value"]) if "value" in latt["alpha"] else None
    be_val = float(latt["beta"]["value"]) if "value" in latt["beta"] else None
    gm_val = float(latt["gamma"]["value"]) if "value" in latt["gamma"] else None

    for pos, tid in enumerate(ids):
        name = id2tok[tid].upper()

        # 晶格长度：log(a/b/c)
        if "A_BIN_" in name and a_val is not None and a_val > 0:
            ga_pos.append(pos)
            ga_x.append(math.log(a_val))
            continue
        if "B_BIN_" in name and b_val is not None and b_val > 0:
            ga_pos.append(pos)
            ga_x.append(math.log(b_val))
            continue
        if "C_BIN_" in name and c_val is not None and c_val > 0:
            ga_pos.append(pos)
            ga_x.append(math.log(c_val))
            continue

        # 晶格角：弧度
        if "ALPHA_BIN_" in name and al_val is not None:
            ga_pos.append(pos)
            ga_x.append(math.radians(al_val))
            continue
        if "BETA_BIN_" in name and be_val is not None:
            ga_pos.append(pos)
            ga_x.append(math.radians(be_val))
            continue
        if "GAMMA_BIN_" in name and gm_val is not None:
            ga_pos.append(pos)
            ga_x.append(math.radians(gm_val))
            continue

        # Wy 参数：u/v/w → 角度
        if name.startswith("BASE_") or name.startswith("FINE_"):
            if uvw_i < len(uvw_vals):
                v = uvw_vals[uvw_i]
                uvw_i += 1
                vm_pos.append(pos)
                vm_theta.append(2 * math.pi * v)
            continue

    return vm_pos, vm_theta, ga_pos, ga_x


# ========= 把预测值写回 ms =========

def _write_back_uvw(ms: Dict[str, Any], new_uvw: List[float]) -> None:
    """
    按 _flatten_uvw_values 的顺序，把新的 [0,1) DOF 写回 ms["sites"][*]["params"][u/v/w]["value"]。
    """
    uvw_i = 0
    for site in ms.get("sites", []):
        p = site.get("params", {})
        if p == "-" or p is None:
            continue
        for k in ("u", "v", "w"):
            if k in p:
                if uvw_i >= len(new_uvw):
                    return
                v = float(new_uvw[uvw_i]) % 1.0
                uvw_i += 1
                node = p[k]
                if isinstance(node, dict):
                    node["value"] = v
                else:
                    p[k] = {"value": v}


def _write_back_lattice(ms: Dict[str, Any], new_ga: List[float], ids: List[int], id2tok: List[str]) -> None:
    """
    根据 token 顺序，把预测的 ga_x 写回 latt.value：
    - A/B/C: exp(x)
    - α/β/γ: degrees(x)
    顺序与 _collect_positions_and_values 中的 append 一一对应。
    """
    latt = ms["latt"]
    names = ["a", "b", "c", "alpha", "beta", "gamma"]

    ga_i = 0
    for tid in ids:
        if ga_i >= len(new_ga):
            break
        name = id2tok[tid].upper()
        x = float(new_ga[ga_i])

        if "A_BIN_" in name:
            latt["a"]["value"] = math.exp(x)
            ga_i += 1
        elif "B_BIN_" in name:
            latt["b"]["value"] = math.exp(x)
            ga_i += 1
        elif "C_BIN_" in name:
            latt["c"]["value"] = math.exp(x)
            ga_i += 1
        elif "ALPHA_BIN_" in name:
            latt["alpha"]["value"] = math.degrees(x)
            ga_i += 1
        elif "BETA_BIN_" in name:
            latt["beta"]["value"] = math.degrees(x)
            ga_i += 1
        elif "GAMMA_BIN_" in name:
            latt["gamma"]["value"] = math.degrees(x)
            ga_i += 1

    # 防御：如果有某个维度没被覆盖，就保持原值不动


# ========= 单个样本的前向 + 写回 + 误差统计 =========

def reconstruct_ms_with_dof(
    ms: Dict[str, Any],
    tok: MaterialTokenizer,
    model: LLaDA,
    dof_head: DOFHead,
    device: torch.device,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    输入一个 ms（含 tokens / latt / sites），输出：
    - ms_pred: 写回连续 DOF 后的新 ms
    - metrics: 一些简单的重构误差指标
    """
    ms_pred = copy.deepcopy(ms)
    tokens: List[int] = ms["tokens"]
    ids = torch.tensor(tokens, dtype=torch.long, device=device)[None, :]  # [1,L]

    id2tok = tok.vocab._id2tok

    with torch.no_grad():
        B, L = ids.shape
        t = torch.rand(B, device=device)
        logits, h = model(ids, t, return_hidden=True)  # 不用 logits，只要 h

    vm_pos, vm_theta_true, ga_pos, ga_x_true = _collect_positions_and_values(tokens, ms, id2tok)

    metrics: Dict[str, float] = {}

    # ---- VM: Wyckoff u/v/w ----
    if vm_pos:
        pos_vm = torch.tensor(vm_pos, dtype=torch.long, device=device)
        h_vm = h[0, pos_vm, :]  # [N_vm, H]
        cos_mu, sin_mu, kappa = dof_head.vm_params(h_vm)
        theta_pred = torch.atan2(sin_mu, cos_mu)  # [-pi, pi]
        theta_pred = (theta_pred + 2 * math.pi) % (2 * math.pi)  # [0, 2pi)
        v_pred = (theta_pred / (2 * math.pi)).cpu().tolist()

        # 写回
        _write_back_uvw(ms_pred, v_pred)

        # 误差（基于 v ∈ [0,1)）
        v_true = [th / (2 * math.pi) for th in vm_theta_true]
        assert len(v_true) == len(v_pred)
        diffs = []
        for vt, vp in zip(v_true, v_pred):
            # 周期考虑：min(|vt-vp|, 1-|vt-vp|)
            d = abs(vt - vp)
            d = min(d, 1.0 - d)
            diffs.append(d)
        if diffs:
            metrics["uvw_mae"] = float(sum(diffs) / len(diffs))

    # ---- GA: lattice a/b/c/alpha/beta/gamma ----
    if ga_pos:
        pos_ga = torch.tensor(ga_pos, dtype=torch.long, device=device)
        h_ga = h[0, pos_ga, :]
        mu, sigma = dof_head.norm_params(h_ga)
        x_pred = mu.cpu().tolist()
        _write_back_lattice(ms_pred, x_pred, tokens, id2tok)

        # 误差统计：分拆 length / angle
        # 真实 x_true 已经是 log(length) 或 angle(rad)
        # 这里按收集顺序，对 lengths/angles 分开统计
        length_err = []
        angle_err_deg = []
        for x_t, tok_id in zip(ga_x_true, [tokens[p] for p in ga_pos]):
            name = id2tok[tok_id].upper()
            idx = ga_pos.index(ga_pos[0])  # 只为类型判断，不用 index 位置
            # 太麻烦就直接一锅端：log-space MAE + angle(rad) MAE -> 不细分
        diffs_ga = [abs(float(a) - float(b)) for a, b in zip(ga_x_true, x_pred)]
        if diffs_ga:
            metrics["ga_mae"] = float(sum(diffs_ga) / len(diffs_ga))

    return ms_pred, metrics


# ========= 主流程 =========

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="val ms.jsonl 路径")
    ap.add_argument("--vocab", required=True, help="vocab.yaml 路径（与预处理时一致）")
    ap.add_argument("--ckpt", required=True, help="训练得到的 .best.pt 路径")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="最多评估多少条样本（0=全量）")
    ap.add_argument("--out_ms", default="", help="可选：把预测后的 ms 写出到这个 jsonl")
    args = ap.parse_args()

    device = torch.device(args.device)

    # ---- 1) 加载 ckpt + tokenizer ----
    ck = torch.load(args.ckpt, map_location=device)
    cfg = ck["cfg"]
    stab = ck["stab"]
    runtime_vocab = ck.get("runtime_vocab", None)

    tok = MaterialTokenizer.from_yaml(args.vocab)
    if runtime_vocab is not None:
        # 覆盖成训练时的 runtime vocab
        tok.vocab._id2tok = list(runtime_vocab)
        tok.vocab._tok2id = {s: i for i, s in enumerate(tok.vocab._id2tok)}
        print(f"[INFO] tokenizer vocab overridden by runtime_vocab, size={len(tok.vocab._id2tok)}")

    V = len(tok.vocab._id2tok)

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

    # ---- 2) 读数据 ----
    ds = MsJsonlDataset(args.data)
    n_total = len(ds)
    print(f"[INFO] dataset size={n_total}")

    limit = args.limit if args.limit and args.limit > 0 else n_total

    out_f = None
    if args.out_ms:
        out_dir = os.path.dirname(os.path.abspath(args.out_ms))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        out_f = open(args.out_ms, "w", encoding="utf-8")

    # ---- 3) 循环评估 ----
    n_eval = 0
    n_geom_ok = 0
    sum_uvw_mae = 0.0
    n_uvw = 0
    sum_ga_mae = 0.0
    n_ga = 0

    for i in range(n_total):
        if n_eval >= limit:
            break

        ms = ds[i]
        ms_pred, metrics = reconstruct_ms_with_dof(ms, tok, model, dof_head, device)

        # 结构级几何检查
        try:
            struct = ms_to_structure(ms_pred, args.vocab)
            ok, why = quick_validate_structure(struct)
        except Exception as e:
            ok = False
            why = f"ms_to_structure_fail:{type(e).__name__}:{str(e)}"

        if ok:
            n_geom_ok += 1

        if "uvw_mae" in metrics:
            sum_uvw_mae += metrics["uvw_mae"]
            n_uvw += 1
        if "ga_mae" in metrics:
            sum_ga_mae += metrics["ga_mae"]
            n_ga += 1

        if out_f is not None:
            rec = {
                "orig_material_id": ms.get("material_id"),
                "formula": ms.get("formula"),
                "sg": ms.get("sg"),
                "geom_ok": ok,
                "geom_why": why,
                "metrics": metrics,
                "ms_pred": ms_pred,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if (n_eval + 1) % 20 == 0:
            print(f"[EVAL] processed {n_eval+1}/{limit}, geom_ok={n_geom_ok}")

        n_eval += 1

    if out_f is not None:
        out_f.close()

    # ---- 4) 总结 ----
    print("========== SUMMARY ==========")
    print(f"evaluated samples : {n_eval}")
    print(f"geom_ok ratio     : {n_geom_ok}/{n_eval} = {n_geom_ok / max(1,n_eval):.3f}")
    if n_uvw > 0:
        print(f"mean uvw MAE      : {sum_uvw_mae / n_uvw:.5f} (in fractional coord, mod 1)")
    if n_ga > 0:
        print(f"mean GA x MAE     : {sum_ga_mae / n_ga:.5f} (log(length)/angle(rad) space)")


if __name__ == "__main__":
    main()
