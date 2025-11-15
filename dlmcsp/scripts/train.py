#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, math, time, os
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dlmcsp.models.llada import LLaDA
from dlmcsp.models.dof_head import DOFHead, vm_nll, gauss_nll
from dlmcsp.tokenization.tokenizer import MaterialTokenizer


# =========================
# token 类型 & 掩码概率（用于 CE）
# =========================

TOKEN_TYPE_P = {
    "LATT_BIN":   0.60,
    "LATT_DR":    0.00,   # 先不训 DR
    "PARAM_BASE": 0.70,
    "PARAM_FINE": 0.50,
    "PARAM_DR":   0.00,

    "SG":       0.00,
    "WY":       0.00,
    "EL":       0.00,
    "FORMULA":  0.00,
    "PUNCT":    0.00,
    "OTHER":    0.00,
}

PUNCTS = {
    "FORMULA=", "NATOMS=", "SG=", "LATT=", "SITES=",
    "EL:", "WY:", "PARAM:", "u:", "v:", "w:",
    "(", ")", "->", ";", ","
}


def classify_token(tok_str: str) -> str:
    s = tok_str
    up = s.upper()

    # 晶格离散 bin / dr
    if "_BIN_" in up:
        if any(x in up for x in ("A_BIN_", "B_BIN_", "C_BIN_", "ALPHA_BIN_", "BETA_BIN_", "GAMMA_BIN_")):
            return "LATT_BIN"
    if "_DR_" in up:
        if any(x in up for x in ("A_DR_", "B_DR_", "C_DR_", "ALPHA_DR_", "BETA_DR_", "GAMMA_DR_")):
            return "LATT_DR"
        return "PARAM_DR"

    # Wyckoff 参数
    if up.startswith("BASE_"):
        return "PARAM_BASE"
    if up.startswith("FINE_"):
        return "PARAM_FINE"

    # 条件 / 结构性 token
    if up.startswith("SG_") or s == "SG=":
        return "SG"
    if s.startswith("WY:") or s == "WY:":
        return "WY"
    if s in PUNCTS:
        return "PUNCT"

    return "OTHER"


# ---------- 数据集 ----------

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
        for k in ("u", "v", "w"):
            if k in p and "value" in p[k]:
                vals.append(float(p[k]["value"]) % 1.0)
    return vals


def _collect_positions_and_values(
    ids: List[int],
    ms: Dict[str, Any],
    id2tok: List[str]
):
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


def collate(batch: List[Dict[str, Any]], tok: MaterialTokenizer):
    PAD_id = tok.vocab.token_id("<PAD>") if "<PAD>" in tok.vocab._tok2id else None
    if PAD_id is None:
        raise RuntimeError("[collate] <PAD> must be defined in vocab.yaml")
    id2tok = tok.vocab._id2tok

    B = len(batch)
    maxL = max(len(x["tokens"]) for x in batch)
    ids = torch.full((B, maxL), PAD_id, dtype=torch.long)

    vm_pos_list: List[List[int]] = []
    vm_theta_list: List[List[float]] = []
    ga_pos_list: List[List[int]] = []
    ga_x_list: List[List[float]] = []

    for b, rec in enumerate(batch):
        seq = rec["tokens"]
        # 这里假定所有 token id 都 < vocab_size；否则说明 vocab / 预处理不一致
        if max(seq) >= len(id2tok):
            raise ValueError(
                f"[collate] sample {b} contains token id {max(seq)} "
                f">= vocab_size {len(id2tok)}; 请检查 vocab.yaml 与预处理是否一致"
            )
        ids[b, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        vm_pos, vm_theta, ga_pos, ga_x = _collect_positions_and_values(seq, rec, id2tok)
        vm_pos_list.append(vm_pos)
        vm_theta_list.append(vm_theta)
        ga_pos_list.append(ga_pos)
        ga_x_list.append(ga_x)

    return ids, vm_pos_list, vm_theta_list, ga_pos_list, ga_x_list


# ---------- 公共：构造 CE mask 概率表 ----------

def build_mask_probs(vocab, device: torch.device) -> Tuple[torch.Tensor, Dict[str, int]]:
    V = len(vocab._id2tok)
    p_by_id = torch.zeros(V, dtype=torch.float32, device=device)
    type_count = {k: 0 for k in TOKEN_TYPE_P.keys()}
    for i, s in enumerate(vocab._id2tok):
        t = classify_token(s)
        p_by_id[i] = float(TOKEN_TYPE_P.get(t, 0.0))
        if t in type_count:
            type_count[t] += 1
    print("[INFO] token type counts:", {k: v for k, v in type_count.items() if v > 0})
    return p_by_id, type_count


# ---------- 验证函数 ----------

@torch.no_grad()
def evaluate(
    model: LLaDA,
    dof: DOFHead,
    dl: DataLoader,
    vocab,
    p_by_id: torch.Tensor,
    PAD_id: int,
    MASK_id: int,
    lam: float,
    reg_kappa: float,
    reg_sigma_inv: float,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    dof.eval()

    ce = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.0)

    total_loss = total_ce = total_vm = total_ga = 0.0
    total_regK = total_regS = 0.0
    n_batches = 0

    for ids, vm_pos_list, vm_theta_list, ga_pos_list, ga_x_list in dl:
        ids = ids.to(device)
        B, L = ids.shape

        probs = p_by_id[ids]          # [B,L]
        probs[ids == PAD_id] = 0.0
        randu = torch.rand_like(probs)
        mask_bool = randu < probs

        inputs = ids.clone()
        inputs[mask_bool] = MASK_id

        labels = ids.clone()
        labels[~mask_bool] = -100
        labels[ids == PAD_id] = -100

        t = torch.rand(B, device=device)

        logits, h = model(inputs, t, return_hidden=True)

        V = logits.size(-1)
        loss_ce = ce(logits.view(-1, V), labels.view(-1))

        # 连续 NLL
        loss_vm = torch.zeros((), device=device)
        loss_ga = torch.zeros((), device=device)
        n_vm = 0
        n_ga = 0
        reg_k = torch.zeros((), device=device)
        reg_s = torch.zeros((), device=device)

        for b in range(B):
            if vm_pos_list[b]:
                pos = torch.tensor(vm_pos_list[b], device=device, dtype=torch.long)
                theta = torch.tensor(vm_theta_list[b], device=device, dtype=torch.float32)
                cos_mu, sin_mu, kappa = dof.vm_params(h[b, pos, :])
                loss_vm = loss_vm + vm_nll(theta, cos_mu, sin_mu, kappa).mean()
                reg_k = reg_k + kappa.mean()
                n_vm += 1
            if ga_pos_list[b]:
                pos = torch.tensor(ga_pos_list[b], device=device, dtype=torch.long)
                x = torch.tensor(ga_x_list[b], device=device, dtype=torch.float32)
                mu, sigma = dof.norm_params(h[b, pos, :])
                loss_ga = loss_ga + gauss_nll(x, mu, sigma).mean()
                reg_s = reg_s + (1.0 / sigma).mean()
                n_ga += 1

        if n_vm > 0:
            loss_vm = loss_vm / n_vm
            reg_k = reg_k / n_vm
        if n_ga > 0:
            loss_ga = loss_ga / n_ga
            reg_s = reg_s / n_ga

        loss_cont = loss_vm + loss_ga + reg_kappa * reg_k + reg_sigma_inv * reg_s
        loss = loss_ce + lam * loss_cont

        total_loss += float(loss.item())
        total_ce += float(loss_ce.item())
        total_vm += float(loss_vm.item())
        total_ga += float(loss_ga.item())
        total_regK += float(reg_k.item())
        total_regS += float(reg_s.item())
        n_batches += 1

    model.train()
    dof.train()

    if n_batches == 0:
        return {
            "loss": 0.0,
            "ce": 0.0,
            "vm": 0.0,
            "ga": 0.0,
            "regK": 0.0,
            "regS": 0.0,
        }

    return {
        "loss": total_loss / n_batches,
        "ce": total_ce / n_batches,
        "vm": total_vm / n_batches,
        "ga": total_ga / n_batches,
        "regK": total_regK / n_batches,
        "regS": total_regS / n_batches,
    }


# ---------- 训练主函数 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_data", required=True)
    ap.add_argument("--val_data", required=False,
                    default="/home/jmmu/dlmcsp/dlmcsp/data/mp_20/val.clean.ms.jsonl")
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--save", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--lam", type=float, default=0.5)             # 连续项权重
    ap.add_argument("--label_smoothing", type=float, default=0.05)

    # 数值防护超参
    ap.add_argument("--vm_kappa_max", type=float, default=100.0)
    ap.add_argument("--sigma_min", type=float, default=0.02)
    ap.add_argument("--sigma_max", type=float, default=10.0)
    ap.add_argument("--reg_kappa", type=float, default=1e-3)
    ap.add_argument("--reg_sigma_inv", type=float, default=1e-3)
    args = ap.parse_args()

    device = torch.device(args.device)

    # 1) 数据集
    train_ds = MsJsonlDataset(args.train_data)
    print(f"[INFO] train size={len(train_ds)}")

    val_ds = None
    if args.val_data is not None and os.path.exists(args.val_data):
        val_ds = MsJsonlDataset(args.val_data)
        print(f"[INFO] val size={len(val_ds)}")
    else:
        print(f"[WARN] val_data not found: {args.val_data}; skip validation")

    # 2) tokenizer & vocab（只使用 vocab.yaml，不再扩表）
    tok = MaterialTokenizer.from_yaml(args.vocab)

    # 3) DataLoader
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=lambda b: collate(b, tok),
        drop_last=True,
    )
    val_dl = None
    if val_ds is not None:
        val_dl = DataLoader(
            val_ds,
            batch_size=args.batch,
            shuffle=False,
            collate_fn=lambda b: collate(b, tok),
            drop_last=False,
        )

    # 4) 模型 & 优化器
    V = len(tok.vocab._id2tok)
    vocab = tok.vocab
    PAD_id = vocab.token_id("<PAD>")
    MASK_id = vocab.token_id("<MASK>")

    p_by_id, _ = build_mask_probs(vocab, device=device)

    model = LLaDA(
        vocab_size=V,
        hidden=args.hidden,
        n_layers=args.layers,
        n_heads=args.heads,
    ).to(device)
    dof = DOFHead(
        hidden=args.hidden,
        kappa_max=args.vm_kappa_max,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
    ).to(device)

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(dof.parameters()),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    ce = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=args.label_smoothing)

    global_step = 0
    best_val_loss = float("inf")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        dof.train()
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_vm = 0.0
        epoch_ga = 0.0
        n_batches = 0

        for ids, vm_pos_list, vm_theta_list, ga_pos_list, ga_x_list in train_dl:
            global_step += 1
            ids = ids.to(device)
            B, L = ids.shape

            # ---------- masked LM 输入 & CE 标签 ----------
            probs = p_by_id[ids]   # [B,L]
            probs[ids == PAD_id] = 0.0

            randu = torch.rand_like(probs)
            mask_bool = randu < probs

            inputs = ids.clone()
            inputs[mask_bool] = MASK_id

            labels = ids.clone()
            labels[~mask_bool] = -100
            labels[ids == PAD_id] = -100

            t = torch.rand(B, device=device)

            logits, h = model(inputs, t, return_hidden=True)
            VocabSize = logits.size(-1)
            loss_ce = ce(logits.view(-1, VocabSize), labels.view(-1))

            # 连续 NLL
            loss_vm = torch.zeros((), device=device)
            loss_ga = torch.zeros((), device=device)
            n_vm = 0
            n_ga = 0
            reg_k = torch.zeros((), device=device)
            reg_s = torch.zeros((), device=device)

            for b in range(B):
                if vm_pos_list[b]:
                    pos = torch.tensor(vm_pos_list[b], device=device, dtype=torch.long)
                    theta = torch.tensor(vm_theta_list[b], device=device, dtype=torch.float32)
                    cos_mu, sin_mu, kappa = dof.vm_params(h[b, pos, :])
                    loss_vm = loss_vm + vm_nll(theta, cos_mu, sin_mu, kappa).mean()
                    reg_k = reg_k + kappa.mean()
                    n_vm += 1
                if ga_pos_list[b]:
                    pos = torch.tensor(ga_pos_list[b], device=device, dtype=torch.long)
                    x = torch.tensor(ga_x_list[b], device=device, dtype=torch.float32)
                    mu, sigma = dof.norm_params(h[b, pos, :])
                    loss_ga = loss_ga + gauss_nll(x, mu, sigma).mean()
                    reg_s = reg_s + (1.0 / sigma).mean()
                    n_ga += 1

            if n_vm > 0:
                loss_vm = loss_vm / n_vm
                reg_k = reg_k / n_vm
            if n_ga > 0:
                loss_ga = loss_ga / n_ga
                reg_s = reg_s / n_ga

            loss_cont = loss_vm + loss_ga + args.reg_kappa * reg_k + args.reg_sigma_inv * reg_s
            loss = loss_ce + args.lam * loss_cont

            if not torch.isfinite(loss):
                print(
                    f"[WARN] non-finite loss. "
                    f"ce={loss_ce.item():.4f} vm={loss_vm.item():.4f} ga={loss_ga.item():.4f} "
                    f"regK={reg_k.item():.4f} regS={reg_s.item():.4f}"
                )
                opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(dof.parameters()), 1.0)
            opt.step()

            epoch_loss += float(loss.item())
            epoch_ce += float(loss_ce.item())
            epoch_vm += float(loss_vm.item())
            epoch_ga += float(loss_ga.item())
            n_batches += 1

            if global_step % 50 == 0:
                print(
                    f"[TRAIN] epoch {epoch} step {global_step} | "
                    f"loss {loss.item():.4f} | ce {loss_ce.item():.4f} "
                    f"| vm {loss_vm.item():.4f} | ga {loss_ga.item():.4f} "
                    f"| B {B} L {L}"
                )

        if n_batches > 0:
            print(
                f"[TRAIN] epoch {epoch} summary | "
                f"loss {epoch_loss/n_batches:.4f} | ce {epoch_ce/n_batches:.4f} "
                f"| vm {epoch_vm/n_batches:.4f} | ga {epoch_ga/n_batches:.4f}"
            )

        # ---------- 验证 ----------
        if val_dl is not None:
            val_metrics = evaluate(
                model=model,
                dof=dof,
                dl=val_dl,
                vocab=vocab,
                p_by_id=p_by_id,
                PAD_id=PAD_id,
                MASK_id=MASK_id,
                lam=args.lam,
                reg_kappa=args.reg_kappa,
                reg_sigma_inv=args.reg_sigma_inv,
                device=device,
            )
            print(
                f"[VAL] epoch {epoch} | "
                f"loss {val_metrics['loss']:.4f} | ce {val_metrics['ce']:.4f} "
                f"| vm {val_metrics['vm']:.4f} | ga {val_metrics['ga']:.4f} "
                f"| regK {val_metrics['regK']:.4f} | regS {val_metrics['regS']:.4f}"
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                ck_best = {
                    "cfg": {
                        "hidden": args.hidden,
                        "layers": args.layers,
                        "heads": args.heads,
                        "vocab_size": V,
                    },
                    "model": model.state_dict(),
                    "dof_head": dof.state_dict(),
                    "stab": {
                        "vm_kappa_max": args.vm_kappa_max,
                        "sigma_min": args.sigma_min,
                        "sigma_max": args.sigma_max,
                    },
                }
                best_path = args.save.replace(".pt", ".best.pt")
                torch.save(ck_best, best_path)
                print(f"[CKPT] best (val) saved to {best_path}")

        # 每个 epoch 也存一个普通 ckpt
        ck = {
            "cfg": {
                "hidden": args.hidden,
                "layers": args.layers,
                "heads": args.heads,
                "vocab_size": V,
            },
            "model": model.state_dict(),
            "dof_head": dof.state_dict(),
            "stab": {
                "vm_kappa_max": args.vm_kappa_max,
                "sigma_min": args.sigma_min,
                "sigma_max": args.sigma_max,
            },
        }
        torch.save(ck, args.save)
        print(f"[CKPT] epoch {epoch} saved to {args.save}")

    dt = (time.time() - t0) / 60.0
    print(f"[DONE] epochs={args.epochs} steps={global_step} time={dt:.1f} min")


if __name__ == "__main__":
    main()
