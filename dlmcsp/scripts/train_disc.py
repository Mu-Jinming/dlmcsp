#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, os, time, math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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

# 只在这些类型上做「最终预测 / 评估」
PREDICT_TYPES = {"LATT_BIN", "PARAM_BASE", "PARAM_FINE"}

# 参与训练（mask & 反传梯度）的 token 类型：
# 把 SG / WY 也放进来，让模型真正学上下文结构。
TRAIN_TYPES = PREDICT_TYPES | {"SG", "WY"}


def classify_token(tok_str: str) -> str:
    """按照字符串规则分类 token 类型"""
    s = tok_str
    up = s.upper()

    # 晶格常数 / 角度离散 bin
    if "_BIN_" in up:
        if any(x in up for x in ("A_BIN_", "B_BIN_", "C_BIN_",
                                 "ALPHA_BIN_", "BETA_BIN_", "GAMMA_BIN_")):
            return "LATT_BIN"

    # DR 类型（这里只做统计，不训练）
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

    print("[INFO] token type counts:",
          {k: v for k, v in type_count.items() if v > 0})
    print("[INFO] token type id map:", type_id_map)
    return types, type_count, type_id_map


def build_id_is_predict(token_types: List[str]) -> torch.Tensor:
    """id -> 是否属于需要「最终预测 / 评估」的类型 (LATT_BIN / PARAM_BASE / PARAM_FINE)"""
    V = len(token_types)
    arr = torch.zeros(V, dtype=torch.bool)
    for tid in range(V):
        if token_types[tid] in PREDICT_TYPES:
            arr[tid] = True
    return arr


def build_id_is_train(token_types: List[str]) -> torch.Tensor:
    """id -> 是否参与训练 (mask & 反传梯度) 的类型"""
    V = len(token_types)
    arr = torch.zeros(V, dtype=torch.bool)
    for tid in range(V):
        if token_types[tid] in TRAIN_TYPES:
            arr[tid] = True
    return arr


def build_id_mask_for_type(token_types: List[str], type_name: str) -> torch.Tensor:
    """id -> 是否属于某个具体类型"""
    V = len(token_types)
    arr = torch.zeros(V, dtype=torch.bool)
    for tid in range(V):
        if token_types[tid] == type_name:
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
    id_is_train: torch.Tensor,
    id_is_predict: torch.Tensor,
    t: torch.Tensor | None = None,
    # 类型级别 mask 频率调制
    id_is_latt_bin: torch.Tensor | None = None,
    id_is_param_base: torch.Tensor | None = None,
    id_is_param_fine: torch.Tensor | None = None,
    id_is_sg: torch.Tensor | None = None,
    id_is_wy: torch.Tensor | None = None,
    mask_scale_skel: float = 1.0,
    mask_scale_fine: float = 1.0,
    mask_scale_sym: float = 1.0,
):
    """
    核心：
      - 只在 TRAIN_TYPES 上做随机 mask 并回传梯度（SG/WY 也训练）；
      - 评价 / 日志仍然主要关注 PREDICT_TYPES；
      - t: 每个样本的「基准」平均 mask 概率（[B]），由外部 curriculum 控制；
      - mask_scale_*: 按类型放大 / 缩小 mask 概率，实现“先骨架、后细节”的 curriculum。
    """
    device = ids.device
    B, L = ids.shape

    # TRAIN / PREDICT 位置 (bool [B,L])
    is_train   = id_is_train[ids]   & (ids != PAD_id)
    is_predict = id_is_predict[ids] & (ids != PAD_id)

    # 如果外部没给 t，就自己采一个 U(0,1)
    if t is None:
        t = torch.rand(B, device=device)

    # 基准 mask 概率矩阵 [B,L]
    p_base = t.view(B, 1).expand(B, L)
    p_mask = p_base.clone()

    # skeleton = LATT_BIN + PARAM_BASE
    if id_is_latt_bin is not None and id_is_param_base is not None and mask_scale_skel != 1.0:
        is_skel = (id_is_latt_bin[ids] | id_is_param_base[ids])
        p_mask[is_skel] = p_mask[is_skel] * mask_scale_skel

    # fine
    if id_is_param_fine is not None and mask_scale_fine != 1.0:
        is_fine = id_is_param_fine[ids]
        p_mask[is_fine] = p_mask[is_fine] * mask_scale_fine

    # SG/WY 视作对称结构 token，单独控制
    if (id_is_sg is not None or id_is_wy is not None) and mask_scale_sym != 1.0:
        is_sym = torch.zeros_like(ids, dtype=torch.bool)
        if id_is_sg is not None:
            is_sym |= id_is_sg[ids]
        if id_is_wy is not None:
            is_sym |= id_is_wy[ids]
        p_mask[is_sym] = p_mask[is_sym] * mask_scale_sym

    # 概率裁剪到 [0,1]
    p_mask = p_mask.clamp(min=0.0, max=1.0)

    # 独立 Bernoulli(p_mask) 采样，只对 TRAIN_TYPES 位置生效
    randu = torch.rand(B, L, device=device)
    mask_bool = (randu < p_mask) & is_train  # [B,L]

    # 避免某些样本一个训练位置都没 mask（导致这个样本无梯度）
    for b in range(B):
        if is_train[b].any() and not mask_bool[b].any():
            cand_pos = torch.nonzero(is_train[b], as_tuple=False)
            idx = torch.randint(0, cand_pos.size(0), (1,), device=device)
            mask_bool[b, cand_pos[idx]] = True

    inputs = ids.clone()
    inputs[mask_bool] = MASK_id

    labels = ids.clone()
    labels[~mask_bool] = -100
    labels[ids == PAD_id] = -100

    # 日志统计
    if is_predict.any():
        mr_predict = (
            (mask_bool & is_predict).float().sum() /
            is_predict.float().sum()
        ).item()
    else:
        mr_predict = 0.0

    if is_train.any():
        mr_train = (
            (mask_bool & is_train).float().sum() /
            is_train.float().sum()
        ).item()
    else:
        mr_train = 0.0

    mr_all = mask_bool.float().mean().item()
    t_mean = t.mean().item()

    return inputs, labels, t, t_mean, mr_predict, mr_train, mr_all


@torch.no_grad()
def evaluate(
    model: LLaDA,
    dl: DataLoader,
    PAD_id: int,
    MASK_id: int,
    id_is_train: torch.Tensor,
    id_is_predict: torch.Tensor,
    id2type_id: torch.Tensor,
    device: torch.device,
    mask_gamma: float = 1.0,
) -> Dict[str, float]:
    """
    验证时也用 masked LM objective，t 固定来自一个“中等噪声”的分布，
    避免跟训练 curriculum 纠缠在一起。
    """
    model.eval()
    ce = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.0)

    total_loss = 0.0
    total_ce = 0.0
    n_batches = 0

    for ids in dl:
        ids = ids.to(device)
        B = ids.size(0)

        # t_eval ∈ [0.3, 0.7]，带 gamma 形状，稳定一点
        u = torch.rand(B, device=device)
        t_min, t_max = 0.3, 0.7
        t_eval = t_min + (t_max - t_min) * (u ** mask_gamma)

        inputs, labels, t_used, _, _, _, _ = apply_mask_and_get_labels(
            ids, PAD_id, MASK_id,
            id_is_train, id_is_predict,
            t=t_eval,
        )
        type_ids = make_type_id_tensor(ids, id2type_id)

        logits = model(inputs, t_used, token_type_ids=type_ids)
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


def build_lr_scheduler(
    opt: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    eta_min_ratio: float = 0.1,
):
    """
    Warmup + Cosine LR:
      - 前 warmup_steps 线性升到 1.0 * lr
      - 之后余弦退火到 eta_min_ratio * lr
    """
    if warmup_steps < 0:
        warmup_steps = 0
    if warmup_steps >= total_steps:
        warmup_steps = max(1, int(0.1 * total_steps))

    def lr_lambda(step: int):
        # step 从 0 开始计数（PyTorch 内部）
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        if step >= total_steps:
            return eta_min_ratio

        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    """
    根据 preset 设置默认超参。
    显式传入的 cli 参数优先级最高（即 args.xxx 非 None 不会被覆盖）。
    """
    preset = (args.preset or "base").lower()

    for k in ["hidden", "layers", "heads",
              "dropout", "lr", "label_smoothing",
              "batch", "epochs"]:
        if getattr(args, k) is None:
            setattr(args, k, None)

    if preset == "base":
        # 类似“论文级 base”配置
        args.hidden           = args.hidden           if args.hidden           is not None else 768
        args.layers           = args.layers           if args.layers           is not None else 12
        args.heads            = args.heads            if args.heads            is not None else 12
        args.dropout          = args.dropout          if args.dropout          is not None else 0.1
        args.lr               = args.lr               if args.lr               is not None else 3e-4
        args.label_smoothing  = args.label_smoothing  if args.label_smoothing  is not None else 0.02
        args.batch            = args.batch            if args.batch            is not None else 24
        args.epochs           = args.epochs           if args.epochs           is not None else 50

    elif preset == "large":
        # 更重一点的版本，注意显存
        args.hidden           = args.hidden           if args.hidden           is not None else 1024
        args.layers           = args.layers           if args.layers           is not None else 16
        args.heads            = args.heads            if args.heads            is not None else 16
        args.dropout          = args.dropout          if args.dropout          is not None else 0.1
        args.lr               = args.lr               if args.lr               is not None else 3e-4
        args.label_smoothing  = args.label_smoothing  if args.label_smoothing  is not None else 0.02
        args.batch            = args.batch            if args.batch            is not None else 16
        args.epochs           = args.epochs           if args.epochs           is not None else 60

    elif preset == "custom":
        # 完全尊重用户传参；如果仍然 None，则给出兜底 default
        pass
    else:
        raise ValueError(f"Unknown preset: {preset}")

    # 对仍为 None 的字段做兜底（防止你传了一半）
    if args.hidden          is None: args.hidden = 512
    if args.layers          is None: args.layers = 12
    if args.heads           is None: args.heads = 8
    if args.dropout         is None: args.dropout = 0.0
    if args.lr              is None: args.lr = 5e-5
    if args.label_smoothing is None: args.label_smoothing = 0.05
    if args.batch           is None: args.batch = 24
    if args.epochs          is None: args.epochs = 20

    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_data", required=True)
    ap.add_argument("--val_data", required=False, default="")
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--save", required=True)
    ap.add_argument("--device", default="cuda")

    # 预设档位：base / large / custom
    ap.add_argument(
        "--preset",
        type=str,
        default="base",
        choices=["base", "large", "custom"],
        help="超参 preset；base/large 为推荐配置，custom 完全用手动参数"
    )

    # 这些参数默认 None，由 preset 决定默认值；显式传参优先级更高
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--heads", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--label_smoothing", type=float, default=None)

    ap.add_argument("--max_steps", type=int, default=0)

    # warmup 相关
    ap.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="LR warmup 步数；>0 时优先，0 则使用 warmup_ratio",
    )
    ap.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.06,
        help="若 warmup_steps=0，则 warmup_steps = total_steps * warmup_ratio"
    )

    # mask 噪声形状控制
    ap.add_argument(
        "--mask_gamma",
        type=float,
        default=2.0,
        help="控制 full-noise 段 t 分布的 gamma, >1 偏向低噪声 (小 t)",
    )

    args = ap.parse_args()
    args = apply_preset(args)  # 应用 preset，填好默认值

    device = torch.device(args.device)

    # ---------------- 数据集 ----------------
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
        num_workers=4,
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
    id_is_train   = build_id_is_train(token_types).to(device)
    id2type_id    = build_id2type_id(token_types, type_id_map).to(device)

    # 类型级 bool mask（按 id）
    id_is_latt_bin   = build_id_mask_for_type(token_types, "LATT_BIN").to(device)
    id_is_param_base = build_id_mask_for_type(token_types, "PARAM_BASE").to(device)
    id_is_param_fine = build_id_mask_for_type(token_types, "PARAM_FINE").to(device)
    id_is_sg         = build_id_mask_for_type(token_types, "SG").to(device)
    id_is_wy         = build_id_mask_for_type(token_types, "WY").to(device)

    n_token_types = len(type_id_map)

    # ---------------- 模型 ----------------
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

    # ---------------- Scheduler：Warmup + Cosine ----------------
    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = args.epochs * len(train_dl)

    if args.warmup_steps > 0:
        warmup_steps = args.warmup_steps
    else:
        warmup_steps = int(total_steps * args.warmup_ratio)

    eta_min_ratio = 0.1
    scheduler = build_lr_scheduler(
        opt,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        eta_min_ratio=eta_min_ratio,
    )
    print(
        f"[INFO] 使用 Warmup+Cosine LR, total_steps={total_steps}, "
        f"warmup_steps={warmup_steps}, "
        f"base_lr={args.lr:.2e}, eta_min={args.lr * eta_min_ratio:.2e}"
    )

    global_step = 0
    best_val_loss = float("inf")
    t0 = time.time()
    training_done = False

    # curriculum 超参
    # t 相关：easy 段 & full 段
    t_easy_min, t_easy_max = 0.05, 0.20
    t_full_min, t_full_max = 0.05, 0.95
    max_easy_ratio = 0.7  # 训练早期最多 70% sample 走“低噪声”

    for epoch in range(1, args.epochs + 1):
        if training_done:
            break

        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for ids in train_dl:
            if args.max_steps > 0 and global_step >= total_steps:
                training_done = True
                break

            global_step += 1
            ids = ids.to(device)
            B = ids.size(0)

            # --------- 训练进度 [0,1] ---------
            progress = min(1.0, global_step / max(1, total_steps))

            # --------- t curriculum: teacher-forcing → full noise ---------
            # 早期 p_easy 大，后期逐渐降低到 0
            p_easy = max_easy_ratio * max(0.0, 1.0 - progress)  # [0, max_easy_ratio]

            # easy 段：低噪声 t ∈ [0.05, 0.2]
            t_easy = t_easy_min + (t_easy_max - t_easy_min) * torch.rand(B, device=device)

            # full 段：更广的 t ∈ [0.05, 0.95]，用 gamma 拉形
            u_full = torch.rand(B, device=device)
            t_full = t_full_min + (t_full_max - t_full_min) * (u_full ** args.mask_gamma)

            choose_easy = (torch.rand(B, device=device) < p_easy)
            t_batch = torch.where(choose_easy, t_easy, t_full)

            # --------- 类型级 curriculum：先骨架，后细节 ---------
            # mask 频率：早期 skeleton 多 mask，fine 少；后期反过来
            if progress < 0.5:
                # 阶段 1：骨架优先
                mask_scale_skel = 1.3
                mask_scale_fine = 0.5  # 原来 0.3，太容易泄露，抬高一点
            elif progress < 0.8:
                # 阶段 2：平衡过渡
                alpha = (progress - 0.5) / 0.3  # 0~1
                mask_scale_skel = 1.3 - 0.4 * alpha   # 1.3 -> 0.9
                mask_scale_fine = 0.5 + 0.5 * alpha   # 0.5 -> 1.0
            else:
                # 阶段 3：细节优先
                alpha = (progress - 0.8) / 0.2        # 0~1
                mask_scale_skel = 0.9                 # 骨架不再继续减，固定 ≳1 的附近
                mask_scale_fine = 1.0 + 0.5 * alpha   # 1.0 -> 1.5

            # 对 FINE 的 mask scaling 加下限，避免长期几乎不 mask 造成“抄答案”
            mask_scale_fine = max(mask_scale_fine, 0.6)

            # SG/WY 一直当作“结构标签”，mask 频率稍低一点，不让模型老是把它们盖掉
            mask_scale_sym = 0.7

            (
                inputs,
                labels,
                t_used,
                t_mean,
                mask_ratio_predict,
                mask_ratio_train,
                mask_ratio_all,
            ) = apply_mask_and_get_labels(
                ids,
                PAD_id,
                MASK_id,
                id_is_train,
                id_is_predict,
                t=t_batch,
                id_is_latt_bin=id_is_latt_bin,
                id_is_param_base=id_is_param_base,
                id_is_param_fine=id_is_param_fine,
                id_is_sg=id_is_sg,
                id_is_wy=id_is_wy,
                mask_scale_skel=mask_scale_skel,
                mask_scale_fine=mask_scale_fine,
                mask_scale_sym=mask_scale_sym,
            )
            type_ids = make_type_id_tensor(ids, id2type_id)

            logits = model(inputs, t_used, token_type_ids=type_ids)
            V_ = logits.size(-1)

            # --------- 类型感知 loss reweight：先骨架、后细节 ---------
            # skeleton = LATT_BIN + PARAM_BASE
            pos_is_latt_bin   = id_is_latt_bin[ids]
            pos_is_param_base = id_is_param_base[ids]
            pos_is_param_fine = id_is_param_fine[ids]
            pos_is_sg         = id_is_sg[ids]
            pos_is_wy         = id_is_wy[ids]

            # 进度相关权重（先算“基线”，然后对 w_skel 做下限裁剪）：
            if progress < 0.4:
                # 阶段 1：强推骨架，细节基本不管
                phase = progress / 0.4  # 0~1
                w_skel_base = 1.8 - 0.4 * phase   # 1.8 -> 1.4
                w_fine      = 0.2 + 0.6 * phase   # 0.2 -> 0.8
            elif progress < 0.8:
                # 阶段 2：骨架和细节拉平到 ~1
                phase = (progress - 0.4) / 0.4
                w_skel_base = 1.4 - 0.4 * phase   # 1.4 -> 1.0
                w_fine      = 0.8 + 0.2 * phase   # 0.8 -> 1.0
            else:
                # 阶段 3：细节优先，但骨架惩罚不低于 1.0
                phase = (progress - 0.8) / 0.2
                w_skel_base = 1.0 - 0.3 * phase   # 1.0 -> 0.7 (基线)
                w_fine      = 1.0 + 1.0 * phase   # 1.0 -> 2.0

            # 硬约束：骨架 loss 权重永远 >= 1.0，防止后期为了 FINE 而“牺牲” lattice
            w_skel = max(w_skel_base, 1.0)

            # SG/WY 一直给个中等权重，防止完全忽略
            w_sym = 0.7

            ce_raw = F.cross_entropy(
                logits.view(-1, V_),
                labels.view(-1),
                ignore_index=-100,
                reduction="none",
                label_smoothing=args.label_smoothing,
            ).view(B, -1)

            valid = (labels != -100)

            w = torch.ones_like(ce_raw)
            # skeleton: LATT_BIN + PARAM_BASE
            w[pos_is_latt_bin | pos_is_param_base] *= w_skel
            # fine
            w[pos_is_param_fine] *= w_fine
            # SG/WY
            w[pos_is_sg | pos_is_wy] *= w_sym

            # 归一化加权 loss
            w_valid = w * valid
            loss_ce = (ce_raw * w_valid).sum() / (w_valid.sum() + 1e-8)
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
            n_batches += 1

            if global_step % 50 == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"[训练] epoch {epoch} step {global_step}/{total_steps} | "
                    f"lr {lr_now:.2e} | loss {loss.item():.4f} | "
                    f"t_mean={t_mean:.4f} | "
                    f"mask_pred={mask_ratio_predict:.4f} | "
                    f"mask_train={mask_ratio_train:.4f} | "
                    f"mask_all={mask_ratio_all:.4f} | "
                    f"p_easy={p_easy:.3f} | "
                    f"w_skel={w_skel:.2f} | w_fine={w_fine:.2f} | "
                    f"mask_scale_skel={mask_scale_skel:.2f} | mask_scale_fine={mask_scale_fine:.2f}"
                )

        if n_batches > 0:
            print(
                f"[训练] epoch {epoch} 总结 | "
                f"mean loss {epoch_loss/n_batches:.4f}"
            )

        # ---------------- 验证 & 保存 ----------------
        if val_dl:
            val_metrics = evaluate(
                model,
                val_dl,
                PAD_id,
                MASK_id,
                id_is_train,
                id_is_predict,
                id2type_id,
                device,
                mask_gamma=args.mask_gamma,
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
                        "max_len": 4096,
                        "preset": args.preset,
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
                "max_len": 4096,
                "preset": args.preset,
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
