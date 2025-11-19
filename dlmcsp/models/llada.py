# -*- coding: utf-8 -*-
# dlmcsp/models/llada.py
from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------- Rotary Embedding ----------------------

class Rotary(nn.Module):
    """
    RoPE: 只对最后一维 (head_dim) 做旋转, 输入形状 [..., T, D]
    在 Block 中我们用 [B, h, T, d]
    """
    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        self.dim = dim

        # 经典 inv_freq 定义
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()  # [T]
        freqs = torch.einsum("t,f->tf", t, inv_freq)  # [T, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)       # [T, dim]
        # 缓存为 [1,1,T,dim]
        self.register_buffer("cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., T, D]
        """
        T = x.size(-2)
        D = x.size(-1)
        if D != self.dim:
            raise ValueError(f"RoPE dim mismatch: got D={D}, expected {self.dim}")

        cos = self.cos[:, :, :T, :D].to(x.device)  # [1,1,T,D]
        sin = self.sin[:, :, :T, :D].to(x.device)

        # 按偶/奇位拆分
        x_even = x[..., 0::2]  # [..., T, D/2]
        x_odd  = x[..., 1::2]

        cos_even = cos[..., 0::2]
        cos_odd  = cos[..., 1::2]
        sin_even = sin[..., 0::2]
        sin_odd  = sin[..., 1::2]

        # 经典旋转
        xe = x_even * cos_even - x_odd * sin_odd
        xo = x_even * sin_even + x_odd * cos_odd
        return torch.stack((xe, xo), dim=-1).flatten(-2)  # [..., T, D]


# ---------------------- Transformer Block ----------------------

class Block(nn.Module):
    def __init__(
        self,
        n_heads: int,
        hidden: int,
        dropout: float = 0.0,
        max_len: int = 4096,
    ):
        super().__init__()
        assert hidden % n_heads == 0, "hidden must be divisible by n_heads"
        self.n_heads = n_heads
        self.hidden = hidden
        self.head_dim = hidden // n_heads

        self.qkv = nn.Linear(hidden, hidden * 3, bias=False)
        self.proj = nn.Linear(hidden, hidden, bias=False)

        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

        self.mlp = nn.Sequential(
            nn.Linear(hidden, 4 * hidden),
            nn.GELU(),
            nn.Linear(4 * hidden, hidden),
        )

        self.rot = Rotary(self.head_dim, max_len=max_len)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,  # FULL attention 时传 None
    ) -> torch.Tensor:
        """
        x: [B,T,H]
        attn_mask: 可选 [1,1,T,T]，1 表示可见，0 表示 mask
        """
        B, T, H = x.size()
        h = self.n_heads
        d = self.head_dim

        # pre-norm
        qkv = self.qkv(self.norm1(x))  # [B,T,3H]
        q, k, v = qkv.chunk(3, dim=-1)  # [B,T,H] each

        # [B,T,H] -> [B,h,T,d]
        q = q.view(B, T, h, d).transpose(1, 2)  # [B,h,T,d]
        k = k.view(B, T, h, d).transpose(1, 2)
        v = v.view(B, T, h, d).transpose(1, 2)

        # RoPE
        q = self.rot(q)
        k = self.rot(k)

        # Attention: [B,h,T,d] x [B,h,d,T] -> [B,h,T,T]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(d)
        if attn_mask is not None:
            att = att.masked_fill(attn_mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # 输出: [B,h,T,T] x [B,h,T,d] -> [B,T,H]
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, H)
        x = x + self.proj(out)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------- LLaDA Backbone ----------------------

class LLaDA(nn.Module):
    """
    LLaDA 骨干 (离散版):
      - 输入: token_ids [B,T], 连续时间 t [B]
      - 输出: logits [B,T,V]，可选隐藏态 h [B,T,H]

    设计选择:
      - 全局 self-attention (non-causal), 适配 masked infilling
      - RoPE + 无显式 pos embedding (位置信息只注入到 q,k)
      - 可选 token type embedding (建议开启)
      - embedding tying: out.weight = token_emb.weight
    """
    def __init__(
        self,
        vocab_size: int,
        hidden: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        dropout: float = 0.0,
        max_len: int = 4096,
        n_token_types: int = 0,   # 若 >0, 启用 type embedding
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, hidden)

        if n_token_types > 0:
            self.type_emb = nn.Embedding(n_token_types, hidden)
        else:
            self.type_emb = None

        # time embedding: 标准 MLP on scalar t
        self.t_proj = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        self.blocks = nn.ModuleList([
            Block(n_heads, hidden, dropout, max_len)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, vocab_size, bias=False)

        # weight tying
        self.out.weight = self.token_emb.weight

    def forward(
        self,
        input_ids: torch.Tensor,      # [B,T]
        t: torch.Tensor,              # [B]
        token_type_ids: Optional[torch.Tensor] = None,  # [B,T] 或 None
        return_hidden: bool = False,
    ):
        B, T = input_ids.size()
        if T > self.max_len:
            raise ValueError(f"sequence too long: {T}>{self.max_len}")

        # token embedding
        x = self.token_emb(input_ids)  # [B,T,H]

        # type embedding (optional)
        if (self.type_emb is not None) and (token_type_ids is not None):
            x = x + self.type_emb(token_type_ids)

        # time embedding: broadcast 到 [B,T,H]
        t = t.view(B, 1, 1)  # [B,1,1]
        t_emb = self.t_proj(t)  # [B,1,H]
        x = x + t_emb          # broadcast over T

        # full attention: 不再用因果下三角掩码
        attn_mask = None

        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)

        h = self.norm(x)
        logits = self.out(h)
        return (logits, h) if return_hidden else logits
