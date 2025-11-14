# -*- coding: utf-8 -*-
# dlmcsp/models/dof_head.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _log_i0_stable(x: torch.Tensor) -> torch.Tensor:
    """
    稳定的 log(I0(x))：
      - |x| < 3.75：采用 Amos 多项式近似 I0，然后取 log
      - |x| >= 3.75：采用渐近展开 log I0(x) ≈ x - 0.5*log(2πx) + log(1 + 1/(8x) + 9/(128x^2))
    """
    x = x.abs()
    small = x < 3.75
    y = (x / 3.75) ** 2

    # 小 x：I0 多项式
    i0_small = 1.0 + y * (
        3.5156229
        + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813))))
    )
    log_small = torch.log(i0_small + 1e-12)

    # 大 x：渐近展开
    # I0(x) ~ exp(x)/sqrt(2πx) * (1 + 1/(8x) + 9/(128x^2))
    eps = (1.0 / (8.0 * x.clamp_min(1e-6))) + (9.0 / (128.0 * x.clamp_min(1e-6) ** 2))
    log_large = x - 0.5 * torch.log(2 * torch.tensor(math.pi, device=x.device) * x.clamp_min(1e-6)) + torch.log1p(eps)

    return torch.where(small, log_small, log_large)


def vm_nll(theta: torch.Tensor, cos_mu: torch.Tensor, sin_mu: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """von Mises 的负对数似然（稳定版）"""
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    k = kappa.squeeze(-1)
    return -k * (cos_t * cos_mu + sin_t * sin_mu) + math.log(2 * math.pi) + _log_i0_stable(k)


def gauss_nll(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """高斯负对数似然"""
    z = (x - mu) / sigma
    return 0.5 * z * z + torch.log(sigma) + 0.5 * math.log(2 * math.pi)


class DOFHead(nn.Module):
    """
    连续自由度头：
      - 周期参数（u/v/w，映射到角度）：von Mises
      - 非周期（log a/b/c、角度弧度）：Gaussian
    提供 κ 与 σ 的数值稳定裁剪。
    """
    def __init__(self, hidden: int, kappa_max: float = 100.0, sigma_min: float = 0.02, sigma_max: float = 10.0):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU())
        self.vm = nn.Linear(hidden, 3)   # cos, sin, kappa_raw
        self.nm = nn.Linear(hidden, 2)   # mu, log_sigma
        self.kappa_max = float(kappa_max)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

    def vm_params(self, h: torch.Tensor):
        z = self.base(h)
        out = self.vm(z)
        cos_mu = torch.tanh(out[..., 0])
        sin_mu = torch.tanh(out[..., 1])
        kappa = F.softplus(out[..., 2]) + 1e-3
        # 归一化方向向量
        norm = torch.clamp(torch.sqrt(cos_mu**2 + sin_mu**2), min=1e-6)
        cos_mu = cos_mu / norm
        sin_mu = sin_mu / norm
        # κ 裁剪稳定
        kappa = torch.clamp(kappa, max=self.kappa_max)
        return cos_mu, sin_mu, kappa

    def norm_params(self, h: torch.Tensor):
        z = self.base(h)
        mu, log_sigma = torch.chunk(self.nm(z), 2, dim=-1)
        sigma = F.softplus(log_sigma) + 1e-3
        sigma = torch.clamp(sigma, min=self.sigma_min, max=self.sigma_max)
        return mu.squeeze(-1), sigma.squeeze(-1)
