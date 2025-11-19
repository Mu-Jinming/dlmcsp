#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, os, time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dlmcsp.models.llada import LLaDA
from dlmcsp.tokenization.tokenizer import MaterialTokenizer

# -----------------------------
# 基本设置
# -----------------------------
PUNCTS = {
    "FORMULA=", "NATOMS=", "SG=", "LATT=", "SITES=",
    "EL:", "WY:", "PARAM:", "u:", "v:", "w:",
    "(", ")", "->", ";", ","
}

# 只在这些类型上做预测 / 评估
PREDICT_TYPES = {"LATT_BIN", "PARAM_BASE", "PARAM_FINE"}


def classify_token(tok_str: str) -> str:
    """按照字符串规则分类 token 类型"""
    s = tok_str
    up = s.upper()

    # 晶格常数 / 角度离散 bin
    if "_BIN_" in up:
        if any(x in up for x in ("A_BIN_", "B_BIN_", "C_BIN_",
                                 "ALPHA_BIN_", "BETA_BIN_", "GAMMA_BIN_")):
            return "LATT_BIN"

    # DR 类型（这里只做统计，训练不预测）
    if "_DR_" in up:
        if any(x in up for x in ("A_DR_", "B_DR_", "C_DR_",
                                 "ALPHA_DR_", "BETA_DR_", "GAMMA_DR_")):
            return "LATT_DR"
        return "PARAM_DR"

    if up.startswith("BASE_"):
        return "PARAM_BASE"
    if up.startswith("FINE_"):
        return "PARAM_FINE"

    if up.startswith("SG_") or s == "SG=":
        return "SG"
    if s.startswith("WY:") or s == "WY:":
        return "WY"
    if s in PUNCTS:
        return "PUNCT"
    return "OTHER"


class MsJsonlDataset(Dataset):
    def __init__(self, path: str):
        self.recs: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                obj = json.loads(ln)
                assert "tokens" in obj
                self.recs.append(obj)
        if not self.recs:
            raise ValueError(f"empty dataset: {path}")

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        return self.recs[i]


def collate(batch: List[Dict[str, Any]], tok: MaterialTokenizer):
    PAD_id = tok.vocab.token_id("<PAD>")
    B = len(batch)
    maxL = max(len(r["tokens"]) for r in batch)
    ids = torch.full((B, maxL), PAD_id, dtype=torch.long)
    for b, rec in enumerate(batch):
        seq = rec["tokens"]
        ids[b, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    return ids


def build_token_types(vocab) -> Tuple[List[str], Dict[str, int], Dict[str, int]]:
    """为每个 token id 生成类型，并统计"""
    id2tok = vocab._id2tok
    types: List[str] = []
    type_count: Dict[str, int] = {}
    for s in id2tok:
        t = classify_token(s)
        types.append(t)
        type_count[t] = type_count.get(t, 0) + 1

    all_types = sorted(type_count.keys())
    type_id_map = {tname: idx for idx, tname in enumerate(all_types)}

    print("[INFO] token type counts:", {k: v for k, v in type_count.items() if v > 0})
    print("[INFO] token type id map:", type_id_map)
    return types, type_count, type_id_map


def build_id_is_predict(token_types: List[str]) -> torch.Tensor:
    """id -> 是否属于需要预测的类型"""
    V = len(token_types)
    arr = torch.zeros(V, dtype=torch.bool)
    for tid in range(V):
        if token_types[tid] in PREDICT_TYPES:
            arr[tid] = True
    return arr


def build_id2type_id(token_types: List[str], type_id_map: Dict[str, int]) -> torch.Tensor:
    """id -> type_id（用于 token-type embedding）"""
    V = len(token_types)
    id2type = torch.zeros(V, dtype=torch.long)
    for tid in range(V):
        tname = token_types[tid]
        id2type[tid] = type_id_map.get(tname, 0)
    return id2type


def make_type_id_tensor(ids: torch.Tensor, id2type_id: torch.Tensor) -> torch.Tensor:
    """直接查表：ids -> type_ids"""
    return id2type_id[ids]


def apply_mask_and_get_labels(
    ids: torch.Tensor,
    PAD_id: int,
    MASK_id: int,
    id_is_predict: torch.Tensor,
):
    """
    核心：只在 PREDICT_TYPES 上做随机 mask，mask 概率 = t ~ U(0,1)
    其他 token 不训练、不回传梯度。
    """
    device = ids.device
    B, L = ids.shape

    # t ~ U(0,1) per sample
    t = torch.rand(B, device=device)  # [B]

    # PREDICT_TYPES & 非 PAD 的位置才可能被 mask
    is_predict = id_is_predict[ids] & (ids != PAD_id)   # [B,L]

    # 独立 Bernoulli(t_b) 采样
    randu = torch.rand(B, L, device=device)
    mask_bool = (randu < t.view(B, 1)) & is_predict     # [B,L]

    inputs = ids.clone()
    inputs[mask_bool] = MASK_id

    labels = ids.clone()
    labels[~mask_bool] = -100
    labels[ids == PAD_id] = -100

    if is_predict.any():
        mask_ratio_predict = (mask_bool.float().sum() / is_predict.float().sum()).item()
    else:
        mask_ratio_predict = 0.0
    mask_ratio_all = mask_bool.float().mean().item()
    t_mean = t.mean().item()

    return inputs, labels, t, t_mean, mask_ratio_predict, mask_ratio_all


@torch.no_grad()
def evaluate(
    model: LLaDA,
    dl: DataLoader,
    PAD_id: int,
    MASK_id: int,
    id_is_predict: torch.Tensor,
    id2type_id: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    ce = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.0)

    total_loss = 0.0
    total_ce = 0.0
    n_batches = 0

    for ids in dl:
        ids = ids.to(device)
        inputs, labels, t, _, _, _ = apply_mask_and_get_labels(
            ids, PAD_id, MASK_id, id_is_predict
        )
        type_ids = make_type_id_tensor(ids, id2type_id)

        logits = model(inputs, t, token_type_ids=type_ids)
        V = logits.size(-1)
        loss_ce = ce(logits.view(-1, V), labels.view(-1))
        loss = loss_ce

        total_loss += float(loss.item())
        total_ce += float(loss_ce.item())
        n_batches += 1

    model.train()
    if n_batches == 0:
        return {"loss": 0.0, "ce": 0.0}
    return {
        "loss": total_loss / n_batches,
        "ce": total_ce / n_batches,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_data", required=True)
    ap.add_argument("--val_data", required=False, default="")
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--save", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=5e-5)  # 建议先用 5e-5 稳一点
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
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
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda b: collate(b, tok),
    )
    val_dl = None
    if val_ds:
        val_dl = DataLoader(
            val_ds,
            batch_size=args.batch,
            shuffle=False,
            drop_last=False,
            collate_fn=lambda b: collate(b, tok),
        )

    vocab = tok.vocab
    V = len(vocab._id2tok)
    PAD_id = vocab.token_id("<PAD>")
    MASK_id = vocab.token_id("<MASK>")

    token_types, type_count, type_id_map = build_token_types(vocab)
    id_is_predict = build_id_is_predict(token_types).to(device)
    id2type_id = build_id2type_id(token_types, type_id_map).to(device)

    n_token_types = len(type_id_map)

    model = LLaDA(
        vocab_size=V,
        hidden=args.hidden,
        n_layers=args.layers,
        n_heads=args.heads,
        dropout=args.dropout,
        max_len=4096,
        n_token_types=n_token_types,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = args.epochs * len(train_dl)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps, eta_min=args.lr * 0.1
    )
    print(f"[INFO] 使用 CosineAnnealingLR, T_max={total_steps}")

    ce = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=args.label_smoothing)

    global_step = 0
    best_val_loss = float("inf")
    t0 = time.time()
    training_done = False

    for epoch in range(1, args.epochs + 1):
        if training_done:
            break

        model.train()
        epoch_loss = 0.0
        epoch_ce = 0.0
        n_batches = 0

        for ids in train_dl:
            if args.max_steps > 0 and global_step >= args.max_steps:
                training_done = True
                break

            global_step += 1
            ids = ids.to(device)

            inputs, labels, t, t_mean, mask_ratio_predict, mask_ratio_all = apply_mask_and_get_labels(
                ids, PAD_id, MASK_id, id_is_predict
            )
            type_ids = make_type_id_tensor(ids, id2type_id)

            logits = model(inputs, t, token_type_ids=type_ids)
            V_ = logits.size(-1)
            loss_ce = ce(logits.view(-1, V_), labels.view(-1))
            loss = loss_ce

            if not torch.isfinite(loss):
                print(f"[WARN] step {global_step} loss 非有限, 跳过")
                opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()

            epoch_loss += float(loss.item())
            epoch_ce += float(loss_ce.item())
            n_batches += 1

            if global_step % 50 == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"[训练] epoch {epoch} step {global_step}/{total_steps} | "
                    f"lr {lr_now:.2e} | loss {loss.item():.4f} | CE {loss_ce.item():.4f} | "
                    f"t={t_mean:.4f} | mask_ratio_pred={mask_ratio_predict:.4f} | "
                    f"mask_ratio_all={mask_ratio_all:.4f}"
                )

        if n_batches > 0:
            print(
                f"[训练] epoch {epoch} 总结 | "
                f"mean loss {epoch_loss/n_batches:.4f} | mean CE {epoch_ce/n_batches:.4f}"
            )

        # 验证 & 保存
        if val_dl:
            val_metrics = evaluate(
                model, val_dl,
                PAD_id, MASK_id,
                id_is_predict,
                id2type_id,
                device,
            )
            print(
                f"[验证] epoch {epoch} | "
                f"loss {val_metrics['loss']:.4f} | ce {val_metrics['ce']:.4f}"
            )
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                ck_best = {
                    "cfg": {
                        "hidden": args.hidden,
                        "layers": args.layers,
                        "heads": args.heads,
                        "dropout": args.dropout,
                        "vocab_size": V,
                        "n_token_types": n_token_types,
                    },
                    "model": model.state_dict(),
                    "type_id_map": type_id_map,
                }
                best_path = args.save.replace(".pt", ".best.pt")
                torch.save(ck_best, best_path)
                print(f"[保存] best ckpt -> {best_path}")

        ck = {
            "cfg": {
                "hidden": args.hidden,
                "layers": args.layers,
                "heads": args.heads,
                "dropout": args.dropout,
                "vocab_size": V,
                "n_token_types": n_token_types,
            },
            "model": model.state_dict(),
            "type_id_map": type_id_map,
        }
        torch.save(ck, args.save)
        print(f"[保存] epoch {epoch} ckpt -> {args.save}")

    dt = (time.time() - t0) / 60.0
    print(f"[完成] total steps={global_step}, 用时 {dt:.1f} min")


if __name__ == "__main__":
    main()
