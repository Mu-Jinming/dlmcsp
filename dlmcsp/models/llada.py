# -*- coding: utf-8 -*-
# dlmcsp/models/llada.py
from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Rotary(nn.Module):
    """
    RoPE：适配任意形状 [..., T, D]，但我们在 Block 中传入 [B, h, T, d]
    缓存 cos/sin 形状为 [1, 1, max_len, D]，可广播到 [B, h, T, d]
    """
    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()  # [T]
        freqs = torch.einsum("t,f->tf", t, inv_freq)  # [T, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)       # [T, dim]
        # 缓存为 [1,1,T,dim]
        self.register_buffer("cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., T, D]，最后两维分别是序列长度与 head_dim
        """
        T = x.size(-2)
        D = x.size(-1)
        if D != self.dim:
            raise ValueError(f"RoPE dim mismatch: got D={D}, expected {self.dim}")
        cos = self.cos[:, :, :T, :D].to(x.device)  # [1,1,T,D]
        sin = self.sin[:, :, :T, :D].to(x.device)
        x_even = x[..., :, 0::2] if x.dim() == 5 else x[..., 0::2]  # 兼容性，不用到 5D
        x_odd  = x[..., :, 1::2] if x.dim() == 5 else x[..., 1::2]
        # 为了广播，把 cos/sin 的偶/奇也切
        cos_e = cos[..., 0::2]
        cos_o = cos[..., 1::2]
        sin_e = sin[..., 0::2]
        sin_o = sin[..., 1::2]
        # 经典旋转： [x_even * cos - x_odd * sin, x_even * sin + x_odd * cos]
        xe = x_even * cos_e - x_odd * sin_o
        xo = x_even * sin_e + x_odd * cos_o
        return torch.stack((xe, xo), dim=-1).flatten(-2)  # 拼回 last dim


class Block(nn.Module):
    def __init__(self, n_heads: int, hidden: int, dropout: float = 0.0, max_len: int = 4096):
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

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B,T,H] -> qkv -> [B,h,T,d]，RoPE -> 注意力 -> [B,T,H]
        attn_mask: [1,1,T,T] (下三角)
        """
        B, T, H = x.size()
        h = self.n_heads
        d = self.head_dim

        qkv = self.qkv(self.norm1(x))  # [B,T,3H]
        q, k, v = qkv.chunk(3, dim=-1)  # [B,T,H] each
        # [B,T,H] -> [B,T,h,d] -> [B,h,T,d]
        q = q.view(B, T, h, d).transpose(1, 2)
        k = k.view(B, T, h, d).transpose(1, 2)
        v = v.view(B, T, h, d).transpose(1, 2)

        # RoPE on last dim per head
        q = self.rot(q)
        k = self.rot(k)

        # 注意力: [B,h,T,d] x [B,h,d,T] -> [B,h,T,T]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(d)
        if attn_mask is not None:
            att = att.masked_fill(attn_mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # 输出: [B,h,T,T] x [B,h,T,d] -> [B,h,T,d] -> [B,T,H]
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, H)
        x = x + self.proj(out)
        x = x + self.mlp(self.norm2(x))
        return x


class LLaDA(nn.Module):
    """
    语言扩散骨干：
      - 输入：token ids [B,T]，扩散时间 t [B]
      - 输出：logits [B,T,V]；可选返回隐藏态 h [B,T,H]
    """
    def __init__(self, vocab_size: int, hidden: int = 512, n_layers: int = 8, n_heads: int = 8,
                 dropout: float = 0.0, max_len: int = 4096):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden)
        self.pos = nn.Embedding(max_len, hidden)  # 可保留与 RoPE 共存（不冲突）
        self.t_proj = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.blocks = nn.ModuleList([Block(n_heads, hidden, dropout, max_len) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, input_ids: torch.Tensor, t: torch.Tensor, return_hidden: bool = False):
        B, T = input_ids.size()
        if T > self.max_len:
            raise ValueError(f"sequence too long: {T}>{self.max_len}")
        x = self.emb(input_ids)  # [B,T,H]
        pos_ids = torch.arange(T, device=input_ids.device)
        x = x + self.pos(pos_ids)[None, :, :]
        tt = t.view(B, 1, 1)
        x = x + self.t_proj(tt)

        # 下三角掩码 [1,1,T,T] 广播到 [B,h,T,T]
        attn_mask = torch.tril(torch.ones(T, T, device=input_ids.device)).unsqueeze(0).unsqueeze(1)
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
        h = self.norm(x)
        logits = self.out(h)
        return (logits, h) if return_hidden else logits
