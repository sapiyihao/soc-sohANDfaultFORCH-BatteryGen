# -*- coding: utf-8 -*-
"""DNN-MTL 单电池包适配模型。

忠实还原 DNN-MTLV1.3 的核心结构，并将"8 支路交叉注意力"适配为"时序自注意力"：
  - 共享 TCN 时序编码器（4 层空洞因果卷积）
  - 浅层出口：GAP(TCN)  -> 时序特征（供 SOC 分支，快变量）
  - 深层出口：时序自注意力 -> GAP -> 时序特征（供 SOH 分支，慢变量/长期退化）
  - 共享上下文编码器 + 共享投影层 FC(192->256)
  - Star Operation 结构乘积耦合
  - 可学习不确定性参数 log_sigma_soc / log_sigma_soh
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation):
        super().__init__()
        pad = dilation * (kernel - 1)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=pad)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        y = self.conv(x)
        y = y[..., :x.shape[-1]]          # 因果裁剪：去掉未来侧 padding
        y = self.act(self.bn(y))
        res = x if self.proj is None else self.proj(x)
        return y + res


class TCN(nn.Module):
    def __init__(self, c_in, channels, layers, kernel, dilations):
        super().__init__()
        self.blocks = nn.ModuleList()
        in_ch = c_in
        for i in range(layers):
            d = dilations[i] if i < len(dilations) else 1
            self.blocks.append(TCNBlock(in_ch, channels, kernel, d))
            in_ch = channels
        self.out_ch = channels

    def forward(self, x):                 # x: [B, c_in, W]
        for b in self.blocks:
            x = b(x)
        return x                          # [B, out_ch, W]


class TemporalAttention(nn.Module):
    """对时间步做多头自注意力，作为"深层出口"（类比 8 支路交叉注意力）。"""
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):                 # x: [B, W, dim]
        a, _ = self.mha(x, x, x)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ffn(x))
        return x


class DNNMTL(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tcn = TCN(cfg.c_in, cfg.tcn_channels, cfg.tcn_layers, cfg.tcn_kernel, cfg.tcn_dilations)
        # 时序特征头（浅层 SOC / 深层 SOH 共享权重）
        self.temporal_head = nn.Sequential(
            nn.Linear(cfg.tcn_channels, cfg.tcn_out), nn.LayerNorm(cfg.tcn_out), nn.GELU())
        self.attn = TemporalAttention(cfg.tcn_channels, cfg.n_attn_heads, cfg.dropout)
        # 上下文编码器
        self.ctx_enc = nn.Sequential(
            nn.Linear(cfg.c_ctx, cfg.ctx_hidden), nn.LayerNorm(cfg.ctx_hidden), nn.GELU(),
            nn.Linear(cfg.ctx_hidden, cfg.ctx_out), nn.LayerNorm(cfg.ctx_out), nn.GELU())
        # 共享投影层 [tcn_out + ctx_out] -> fusion_dim
        self.proj = nn.Sequential(
            nn.Linear(cfg.tcn_out + cfg.ctx_out, cfg.fusion_dim),
            nn.LayerNorm(cfg.fusion_dim), nn.GELU(), nn.Dropout(cfg.dropout))
        # 分支第一层 FC
        self.soc_branch = nn.Linear(cfg.fusion_dim, cfg.branch_dim)
        # SOH 分支：深层窗口特征 Z' + 循环级特征（容量是整车级慢变量，需循环级统计量）
        self.cycle_enc = nn.Sequential(
            nn.Linear(cfg.c_cycle, 32), nn.GELU(), nn.Linear(32, 32))
        self.soh_branch = nn.Linear(cfg.fusion_dim + 32, cfg.branch_dim)
        self.soh_cycle_only = bool(getattr(cfg, "soh_cycle_only", 0))
        # Star Operation
        self.star = nn.Sequential(nn.Linear(cfg.branch_dim, cfg.branch_dim), nn.GELU())
        # 输出头
        self.soc_head = nn.Sequential(nn.Linear(cfg.branch_dim, 64), nn.GELU(), nn.Linear(64, 2))
        self.soh_head = nn.Sequential(nn.Linear(cfg.branch_dim, 64), nn.GELU(), nn.Linear(64, 1))
        # 不确定性自适应加权参数
        self.log_sigma_soc = nn.Parameter(torch.zeros(1))
        self.log_sigma_soh = nn.Parameter(torch.zeros(1))
        # Thevenin 等效电路参数（log 空间保证物理正性）：欧姆内阻 R0、极化内阻 R1、时间常数 τ
        self.log_r0 = nn.Parameter(torch.tensor(math.log(max(getattr(cfg, "r0_init", 0.0005), 1e-8))))
        self.log_r1 = nn.Parameter(torch.tensor(math.log(max(getattr(cfg, "r1_init", 0.0005), 1e-8))))
        self.log_tau = nn.Parameter(torch.tensor(math.log(max(getattr(cfg, "tau_init", 60.0), 1e-8))))

    def forward(self, x, ctx, cycle_feat=None):
        # x: [B, W, c_in]; ctx: [B, c_ctx]; cycle_feat: [B, c_cycle] 或 None
        h = self.tcn(x.transpose(1, 2))                    # [B, ch, W]

        # 浅层时序特征（SOC 快变量）
        f_shallow = self.temporal_head(h.mean(dim=2))      # [B, tcn_out]
        # 深层时序特征（SOH 慢变量，经时序自注意力）
        h_att = self.attn(h.transpose(1, 2))               # [B, W, ch]
        f_deep = self.temporal_head(h_att.mean(dim=1))     # [B, tcn_out]

        f_ctx = self.ctx_enc(ctx)                          # [B, ctx_out]

        z_shallow = self.proj(torch.cat([f_shallow, f_ctx], dim=1))   # [B, fusion]
        z_deep = self.proj(torch.cat([f_deep, f_ctx], dim=1))         # [B, fusion]

        f_soc = F.gelu(self.soc_branch(z_shallow))         # [B, branch]
        # SOH 分支：循环级特征（必选）+ 深层窗口特征（可选，soh_cycle_only=1 时置零）
        if cycle_feat is None:
            cycle_emb = torch.zeros(x.shape[0], 32, device=x.device)
        else:
            cycle_emb = self.cycle_enc(cycle_feat)         # [B, 32]
        z_deep_in = torch.zeros_like(z_deep) if self.soh_cycle_only else z_deep
        f_soh = F.gelu(self.soh_branch(torch.cat([z_deep_in, cycle_emb], dim=1)))   # [B, branch]

        f_star = self.star(f_soc * f_soh)                  # Star Operation
        f_soc = f_soc + f_star
        f_soh = f_soh + f_star

        soc = torch.sigmoid(self.soc_head(f_soc))          # [B, 2] = (soc_start, soc_end)
        # 容量法 SOH 可因新电池/标称容量偏差略高于 1.0，不再用 [0,1] 硬截断。
        soh01 = torch.sigmoid(self.soh_head(f_soh))
        soh = self.cfg.soh_min + (self.cfg.soh_max - self.cfg.soh_min) * soh01  # [B, 1]
        return soc, soh

    def sigma_soc(self):
        return torch.exp(self.log_sigma_soc)

    def sigma_soh(self):
        return torch.exp(self.log_sigma_soh)

    @property
    def r0(self):
        return torch.exp(self.log_r0)

    @property
    def r1(self):
        return torch.exp(self.log_r1)

    @property
    def tau(self):
        return torch.exp(self.log_tau)
