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
        probs = p_by_id[ids]
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
        loss_vm = torch.zeros((), device=device)
        loss_ga = torch.zeros((), device=device)
        n_vm, n_ga = 0, 0
        reg_k = torch.zeros((), device=device)
        reg_s = torch.zeros((), device=device)
        for b in range(B):
            if vm_pos_list[b]:
                pos = torch.tensor(vm_pos_list[b], device=device, dtype=torch.long)
                theta = torch.tensor(vm_theta_list[b], device=device, dtype=torch.float32)
                cos_mu, sin_mu, kappa = dof.vm_params(h[b, pos, :])
                loss_vm += vm_nll(theta, cos_mu, sin_mu, kappa).mean()
                reg_k += kappa.mean()
                n_vm += 1
            if ga_pos_list[b]:
                pos = torch.tensor(ga_pos_list[b], device=device, dtype=torch.long)
                x = torch.tensor(ga_x_list[b], device=device, dtype=torch.float32)
                mu, sigma = dof.norm_params(h[b, pos, :])
                loss_ga += gauss_nll(x, mu, sigma).mean()
                reg_s += (1.0 / sigma).mean()
                n_ga += 1
        if n_vm > 0: loss_vm /= n_vm; reg_k /= n_vm
        if n_ga > 0: loss_ga /= n_ga; reg_s /= n_ga
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
        return {"loss": 0.0, "ce": 0.0, "vm": 0.0, "ga": 0.0, "regK": 0.0, "regS": 0.0}
    return {
        "loss": total_loss / n_batches, "ce": total_ce / n_batches,
        "vm": total_vm / n_batches, "ga": total_ga / n_batches,
        "regK": total_regK / n_batches, "regS": total_regS / n_batches,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_data", required=True)
    ap.add_argument("--val_data", required=False, default="/home/jmmu/dlmcsp/dlmcsp/data/mp_20/val.clean.ms.jsonl")
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--save", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=20)
    # +++ NEW: 使用 max_steps 参数来更精确地控制训练总长度 +++
    ap.add_argument("--max_steps", type=int, default=0, help="若大于0, 则覆盖 epochs 参数, 以总步数控制训练")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=8)
    # +++ MODIFIED: lam 参数的说明，强调其重要性 +++
    ap.add_argument("--lam", type=float, default=0.5, help="连续项损失的权重 (关键超参, 可能需要设为10, 50, 甚至更高来平衡梯度)")
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--vm_kappa_max", type=float, default=100.0)
    ap.add_argument("--sigma_min", type=float, default=0.02)
    ap.add_argument("--sigma_max", type=float, default=10.0)
    ap.add_argument("--reg_kappa", type=float, default=1e-3)
    ap.add_argument("--reg_sigma_inv", type=float, default=1e-3)
    args = ap.parse_args()

    device = torch.device(args.device)
    train_ds = MsJsonlDataset(args.train_data)
    print(f"[INFO] 训练集大小={len(train_ds)}")
    val_ds = None
    if args.val_data and os.path.exists(args.val_data):
        val_ds = MsJsonlDataset(args.val_data)
        print(f"[INFO] 验证集大小={len(val_ds)}")
    else:
        print(f"[WARN] 验证集未找到: {args.val_data}; 将跳过验证")
    tok = MaterialTokenizer.from_yaml(args.vocab)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=lambda b: collate(b, tok), drop_last=True)
    val_dl = None
    if val_ds:
        val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=lambda b: collate(b, tok), drop_last=False)
    
    V = len(tok.vocab._id2tok)
    vocab = tok.vocab
    PAD_id = vocab.token_id("<PAD>")
    MASK_id = vocab.token_id("<MASK>")
    p_by_id, _ = build_mask_probs(vocab, device=device)

    model = LLaDA(vocab_size=V, hidden=args.hidden, n_layers=args.layers, n_heads=args.heads).to(device)
    dof = DOFHead(hidden=args.hidden, kappa_max=args.vm_kappa_max, sigma_min=args.sigma_min, sigma_max=args.sigma_max).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(dof.parameters()), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    
    # +++ NEW: 添加学习率调度器 +++
    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = args.epochs * len(train_dl)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=args.lr * 0.1)
    print(f"[INFO] 启用余弦退火学习率调度器, T_max = {total_steps} 步")

    ce = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=args.label_smoothing)
    global_step, best_val_loss, t0 = 0, float("inf"), time.time()
    
    # +++ MODIFIED: 训练循环逻辑, 优先使用 max_steps +++
    training_complete = False
    for epoch in range(1, args.epochs + 1):
        if training_complete: break
        
        model.train()
        dof.train()
        epoch_loss, epoch_ce, epoch_vm, epoch_ga, n_batches = 0.0, 0.0, 0.0, 0.0, 0

        for ids, vm_pos_list, vm_theta_list, ga_pos_list, ga_x_list in train_dl:
            if args.max_steps > 0 and global_step >= args.max_steps:
                training_complete = True
                break
            
            global_step += 1
            ids = ids.to(device)
            B, L = ids.shape
            probs = p_by_id[ids]
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
            loss_ce = ce(logits.view(-1, logits.size(-1)), labels.view(-1))

            loss_vm, loss_ga = torch.zeros((), device=device), torch.zeros((), device=device)
            n_vm, n_ga = 0, 0
            reg_k, reg_s = torch.zeros((), device=device), torch.zeros((), device=device)
            for b in range(B):
                if vm_pos_list[b]:
                    pos = torch.tensor(vm_pos_list[b], device=device, dtype=torch.long)
                    theta = torch.tensor(vm_theta_list[b], device=device, dtype=torch.float32)
                    cos_mu, sin_mu, kappa = dof.vm_params(h[b, pos, :])
                    loss_vm += vm_nll(theta, cos_mu, sin_mu, kappa).mean()
                    reg_k += kappa.mean()
                    n_vm += 1
                if ga_pos_list[b]:
                    pos = torch.tensor(ga_pos_list[b], device=device, dtype=torch.long)
                    x = torch.tensor(ga_x_list[b], device=device, dtype=torch.float32)
                    mu, sigma = dof.norm_params(h[b, pos, :])
                    loss_ga += gauss_nll(x, mu, sigma).mean()
                    reg_s += (1.0 / sigma).mean()
                    n_ga += 1
            if n_vm > 0: loss_vm /= n_vm; reg_k /= n_vm
            if n_ga > 0: loss_ga /= n_ga; reg_s /= n_ga
            loss_cont = loss_vm + loss_ga + args.reg_kappa * reg_k + args.reg_sigma_inv * reg_s
            loss = loss_ce + args.lam * loss_cont

            if not torch.isfinite(loss):
                print(f"[WARN] 在第 {global_step} 步出现非有限损失. 跳过此步...")
                opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            
            # +++ NEW: 梯度范数监控 (每100步打印一次) +++
            if global_step % 100 == 0:
                grad_discrete_norm = torch.norm(torch.stack([p.grad.norm(2) for p in model.parameters() if p.grad is not None]))
                grad_continuous_norm = torch.norm(torch.stack([p.grad.norm(2) for p in dof.parameters() if p.grad is not None]))
                print(f"[诊断] 步 {global_step} | 离散梯度范数: {grad_discrete_norm:.4f} | 连续梯度范数: {grad_continuous_norm:.4f}")

            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(dof.parameters()), 1.0)
            opt.step()
            scheduler.step()

            epoch_loss += float(loss.item())
            epoch_ce += float(loss_ce.item())
            epoch_vm += float(loss_vm.item())
            epoch_ga += float(loss_ga.item())
            n_batches += 1
            
            # +++ MODIFIED: 分离式、信息更丰富的日志 +++
            if global_step % 50 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"[训练] 周期 {epoch} 步 {global_step}/{total_steps} | 学习率 {current_lr:.2e} | "
                    f"总loss {loss.item():.4f} | CE loss {loss_ce.item():.4f} | Cont loss {loss_cont.item():.4f} "
                    f"| VM nll {loss_vm.item():.4f} | GA nll {loss_ga.item():.4f}"
                )

        if n_batches > 0:
            print(
                f"[训练] 周期 {epoch} 总结 | "
                f"平均loss {epoch_loss/n_batches:.4f} | 平均CE {epoch_ce/n_batches:.4f} "
                f"| 平均VM {epoch_vm/n_batches:.4f} | 平均GA {epoch_ga/n_batches:.4f}"
            )

        if val_dl:
            val_metrics = evaluate(model, dof, val_dl, vocab, p_by_id, PAD_id, MASK_id, args.lam, args.reg_kappa, args.reg_sigma_inv, device)
            print(
                f"[验证] 周期 {epoch} | "
                f"loss {val_metrics['loss']:.4f} | ce {val_metrics['ce']:.4f} "
                f"| vm {val_metrics['vm']:.4f} | ga {val_metrics['ga']:.4f} "
                f"| regK {val_metrics['regK']:.4f} | regS {val_metrics['regS']:.4f}"
            )
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                ck_best = {
                    "cfg": {"hidden": args.hidden, "layers": args.layers, "heads": args.heads, "vocab_size": V},
                    "model": model.state_dict(),
                    "dof_head": dof.state_dict(),
                    "stab": {"vm_kappa_max": args.vm_kappa_max, "sigma_min": args.sigma_min, "sigma_max": args.sigma_max},
                }
                best_path = args.save.replace(".pt", ".best.pt")
                torch.save(ck_best, best_path)
                print(f"[保存] 验证集最优模型已保存至 {best_path}")

        ck = {
            "cfg": {"hidden": args.hidden, "layers": args.layers, "heads": args.heads, "vocab_size": V},
            "model": model.state_dict(),
            "dof_head": dof.state_dict(),
            "stab": {"vm_kappa_max": args.vm_kappa_max, "sigma_min": args.sigma_min, "sigma_max": args.sigma_max},
        }
        torch.save(ck, args.save)
        print(f"[保存] 周期 {epoch} 模型已保存至 {args.save}")

    dt = (time.time() - t0) / 60.0
    print(f"[完成] 总步数={global_step} 训练耗时={dt:.1f} 分钟")

if __name__ == "__main__":
    main()