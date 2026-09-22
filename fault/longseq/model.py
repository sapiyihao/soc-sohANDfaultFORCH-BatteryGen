"""面向真实时间轴的纯 PyTorch 选择性状态空间网络。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SelectiveSSM(nn.Module):
    """输入相关的 Δ/B/C 与门控扫描；分块并行实现，避免512次Python循环。

    该层是可在Windows/PyTorch直接运行的 Mamba 风格实现，不等同于
    mamba-ssm 官方 fused CUDA kernel；算法/效率比较时须注明。
    """

    def __init__(self, d_model: int, d_state: int = 8, kernel_size: int = 5, chunk_size: int = 32):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.chunk_size = chunk_size
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model, padding=kernel_size - 1)
        self.delta_proj = nn.Linear(d_model, d_model)
        self.bc_proj = nn.Linear(d_model, 2 * d_state)
        self.a_log = nn.Parameter(torch.linspace(-2.0, -0.3, d_state).repeat(d_model, 1))
        self.skip = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[:, :, :length].transpose(1, 2)
        u = F.silu(u).float()
        # 扫描在float32中完成，避免AMP下前缀积和累积和溢出。
        delta = (0.01 + 0.09 * torch.sigmoid(self.delta_proj(u))).float()
        b_t, c_t = self.bc_proj(u).float().chunk(2, dim=-1)
        a_rate = self.a_log.float().exp()
        state = torch.zeros(batch, self.d_model, self.d_state, device=x.device, dtype=torch.float32)
        outputs = []
        for start in range(0, length, self.chunk_size):
            end = min(start + self.chunk_size, length)
            valid = mask[:, start:end, None, None]
            decay = torch.exp(-delta[:, start:end, :, None] * a_rate[None, None])
            decay = torch.where(valid, decay, torch.ones_like(decay))
            drive = delta[:, start:end, :, None] * b_t[:, start:end, None, :] * u[:, start:end, :, None]
            drive = drive * valid
            prefix = torch.cumprod(decay, dim=1)
            states = prefix * (state[:, None] + torch.cumsum(drive / prefix.clamp_min(1e-12), dim=1))
            state = states[:, -1]
            y = (states * c_t[:, start:end, None, :]).sum(dim=-1)
            y = y + self.skip.float()[None, None, :] * u[:, start:end]
            outputs.append(y)
        y = torch.cat(outputs, dim=1).to(gate.dtype) * F.silu(gate)
        return self.out_proj(y) * mask.unsqueeze(-1)


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state)
        self.dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.SiLU(),
                                nn.Dropout(dropout), nn.Linear(2 * d_model, d_model))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.ssm(self.norm(x), mask))
        x = x + self.dropout(self.ff(self.ff_norm(x)))
        return x * mask.unsqueeze(-1)


class ConvStem(nn.Module):
    """用局部一维卷积替代逐时间点线性映射，保持序列长度不变。"""

    def __init__(self, in_features: int, d_model: int, kernel_size: int = 5):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("Conv Stem的kernel_size必须为正奇数")
        self.conv = nn.Conv1d(
            in_features,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.activation(self.norm(x))


class MultiScaleConvStem(nn.Module):
    """并行提取突变、中短期变化和平缓趋势，再门控融合到统一隐藏维度。"""

    def __init__(self, in_features: int, d_model: int,
                 kernel_sizes: tuple[int, ...] = (3, 7, 15)):
        super().__init__()
        if not kernel_sizes or any(kernel < 1 or kernel % 2 == 0 for kernel in kernel_sizes):
            raise ValueError("多尺度卷积核必须为非空的正奇数序列")
        self.branches = nn.ModuleList([
            nn.Conv1d(in_features, d_model, kernel, padding=kernel // 2)
            for kernel in kernel_sizes
        ])
        self.fuse = nn.Conv1d(len(kernel_sizes) * d_model, d_model, kernel_size=1)
        self.residual = nn.Linear(in_features, d_model)
        self.gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels_first = x.transpose(1, 2)
        multi_scale = torch.cat(
            [F.silu(branch(channels_first)) for branch in self.branches],
            dim=1,
        )
        multi_scale = self.fuse(multi_scale).transpose(1, 2)
        residual = self.residual(x)
        gate = self.gate(torch.cat([multi_scale, residual], dim=-1))
        return self.activation(self.norm(residual + gate * multi_scale))


class TemporalFAF(nn.Module):
    """沿时间轴进行频率自适应滤波，输入输出均为[B,T,D]。

    这里将SE-Mamba中的FAF改造成适合一维车辆时间序列的版本：频域门控由
    当前样本的频谱幅值动态生成，同时保留相位。FFT固定使用float32，以避免
    混合精度训练时cuFFT的数值/长度限制。
    """

    def __init__(self, d_model: int, expansion: int = 2):
        super().__init__()
        hidden = expansion * d_model
        self.norm = nn.LayerNorm(d_model)
        self.filter_net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )
        # 初始门控为1，刚开始训练时FAF近似恒等频率滤波。
        nn.init.zeros_(self.filter_net[-1].weight)
        nn.init.zeros_(self.filter_net[-1].bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        output_dtype = x.dtype
        valid = mask.unsqueeze(-1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            normalized = self.norm(x.float()) * valid
            spectrum = torch.fft.rfft(normalized, dim=1, norm="ortho")
            magnitude = torch.log1p(spectrum.abs())
            # 每个频率点都根据全部隐藏通道的幅值生成自适应通道权重。
            frequency_gate = 2.0 * torch.sigmoid(self.filter_net(magnitude))
            filtered = torch.fft.irfft(
                spectrum * frequency_gate,
                n=x.shape[1],
                dim=1,
                norm="ortho",
            )
        return filtered.to(output_dtype) * valid


class LongSequenceMamba(nn.Module):
    """[B,T,7] -> 四类车辆故障logits。"""

    def __init__(self, d_model: int = 32, d_state: int = 8, layers: int = 2,
                 dropout: float = 0.1, in_features: int = 7, classes: int = 4):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(in_features, d_model), nn.LayerNorm(d_model), nn.SiLU())
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state, dropout) for _ in range(layers)])
        self.pool_score = nn.Linear(d_model, 1)
        self.classifier = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model),
                                        nn.SiLU(), nn.Dropout(dropout), nn.Linear(d_model, classes))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.input(x) * mask.unsqueeze(-1)
        for block in self.blocks:
            h = block(h, mask)
        valid = mask.unsqueeze(-1)
        mean = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        maximum = h.masked_fill(~valid, -1e4).max(dim=1).values
        attention = self.pool_score(h).squeeze(-1).masked_fill(~mask, -1e4).softmax(dim=1)
        pooled = (h * attention.unsqueeze(-1)).sum(dim=1)
        return self.classifier(torch.cat([mean, maximum, pooled], dim=-1))


class ConvStemMamba(LongSequenceMamba):
    """仅将基线输入映射替换为Conv Stem，用于受控消融实验。"""

    def __init__(self, d_model: int = 32, d_state: int = 8, layers: int = 2,
                 dropout: float = 0.1, in_features: int = 7, classes: int = 4,
                 stem_kernel_size: int = 5):
        super().__init__(d_model, d_state, layers, dropout, in_features, classes)
        self.input = ConvStem(in_features, d_model, stem_kernel_size)


class ConvStemFAFMamba(LongSequenceMamba):
    """Conv Stem + Mamba + 时序FAF，并采用直接残差相加融合。"""

    def __init__(self, d_model: int = 32, d_state: int = 8, layers: int = 2,
                 dropout: float = 0.1, in_features: int = 7, classes: int = 4,
                 stem_kernel_size: int = 5):
        super().__init__(d_model, d_state, layers, dropout, in_features, classes)
        self.input = ConvStem(in_features, d_model, stem_kernel_size)
        self.faf = TemporalFAF(d_model)
        self.fusion_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask.unsqueeze(-1)
        stem_features = self.input(x) * valid
        mamba_features = stem_features
        for block in self.blocks:
            mamba_features = block(mamba_features, mask)
        frequency_features = self.faf(mamba_features, mask)
        h = self.fusion_norm(stem_features + mamba_features + frequency_features) * valid

        mean = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        maximum = h.masked_fill(~valid, -1e4).max(dim=1).values
        attention = self.pool_score(h).squeeze(-1).masked_fill(~mask, -1e4).softmax(dim=1)
        pooled = (h * attention.unsqueeze(-1)).sum(dim=1)
        return self.classifier(torch.cat([mean, maximum, pooled], dim=-1))


class MultiScaleHierarchicalMamba(nn.Module):
    """多尺度局部特征、三层Mamba、五路池化与故障分层双头。"""

    is_hierarchical = True

    def __init__(self, d_model: int = 32, d_state: int = 8, layers: int = 3,
                 dropout: float = 0.1, in_features: int = 7):
        super().__init__()
        self.input = MultiScaleConvStem(in_features, d_model)
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state, dropout) for _ in range(layers)])
        self.pool_score = nn.Linear(d_model, 1)
        pooled_features = 5 * d_model
        shared_features = 2 * d_model
        self.shared = nn.Sequential(
            nn.LayerNorm(pooled_features),
            nn.Linear(pooled_features, shared_features),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.detection_head = nn.Sequential(
            nn.LayerNorm(shared_features),
            nn.Linear(shared_features, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )
        self.type_head = nn.Sequential(
            nn.LayerNorm(shared_features),
            nn.Linear(shared_features, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask.unsqueeze(-1)
        h = self.input(x) * valid
        for block in self.blocks:
            h = block(h, mask)

        count = valid.sum(dim=1).clamp_min(1)
        mean = (h * valid).sum(dim=1) / count
        variance = ((h - mean.unsqueeze(1)).square() * valid).sum(dim=1) / count
        standard_deviation = torch.sqrt(variance.clamp_min(1e-6))
        maximum = h.masked_fill(~valid, -1e4).max(dim=1).values
        last_index = mask.long().sum(dim=1).sub(1).clamp_min(0)
        last = h[torch.arange(h.shape[0], device=h.device), last_index]
        attention = self.pool_score(h).squeeze(-1).masked_fill(~mask, -1e4).softmax(dim=1)
        attended = (h * attention.unsqueeze(-1)).sum(dim=1)
        return self.shared(torch.cat([mean, maximum, standard_deviation, last, attended], dim=-1))

    def forward_with_heads(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = self.encode(x, mask)
        detection_logits = self.detection_head(features)
        type_logits = self.type_head(features)
        detection_log_prob = F.log_softmax(detection_logits, dim=-1)
        type_log_prob = F.log_softmax(type_logits, dim=-1)
        # P(正常)=P(正常)；P(故障类型)=P(故障)×P(该故障类型|故障)。
        combined_log_prob = torch.cat([
            detection_log_prob[:, :1],
            detection_log_prob[:, 1:] + type_log_prob,
        ], dim=-1)
        return combined_log_prob, detection_logits, type_logits

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        combined_log_prob, _, _ = self.forward_with_heads(x, mask)
        return combined_log_prob
