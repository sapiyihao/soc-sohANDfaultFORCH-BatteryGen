# -*- coding: utf-8 -*-
"""DNN-MTL 适配实现 —— 配置与命令行解析。

本实现把 DNN-MTLV1.3 的"8 支路并联 + 交叉注意力"适配为
"单电池包 SOC/SOH 联合估计"：
  - 输入第1路：滑动窗口 [W, C_in]（电压/电流/温度时序）
  - 输入第2路：上下文标量 [C_ctx]（SOC_BMS / 工况 / SOH 锚点 / 温度 / 进度）
  - 浅层出口 Z -> SOC 分支；深层出口 Z'（时序自注意力）-> SOH 分支
  - Star Operation 结构乘积耦合 + 不确定性自适应加权
"""
from dataclasses import dataclass, field, asdict
import argparse
import json
import os


@dataclass
class Config:
    # ---- 数据 ----
    data_root: str = ""              # 原始数据根目录；空值默认为工作区根目录的 dataset/
    output_root: str = ""            # 实验输出根目录；空值默认为 soc-soh/experiments/
    chemistry: str = "LFP"          # LFP | NCM | BOTH
    n_vehicles: int = 80            # 正常车辆数量上限（train+val+test 总和）
    include_low_capacity: int = 0   # 额外纳入低容量(老化)故障车，给 SOH 真实分化信号
    include_high_resistance: int = 0  # 额外纳入高内阻(老化)故障车
    val_ratio: float = 0.125        # 验证集占车辆比例
    test_ratio: float = 0.125       # 测试集占车辆比例
    window_len: int = 128           # 滑动窗口长度（10s 采样，128 ≈ 21 分钟）
    stride: int = 16                # 滑动步长（采样点数）
    min_cycle_dsoc: float = 0.20    # 构造 SOH 标签所需的最小充电 ΔSOC
    cap_clip: tuple = (0.7, 1.3)    # 相对 C_nom 的容量裁剪，过滤离群循环
    soh_per_cycle: int = 0          # 1=逐循环标签, 0=车辆级稳健中位数（因果窗口 SOH 默认）
    causal_window_soh: int = 1      # 1=仅使用窗口末端之前的循环前缀特征
    soh_all_modes: int = 1          # 1=充放电窗口均使用车辆级 SOH 监督
    soh_min: float = 0.6            # SOH 输出下界
    soh_max: float = 1.3            # SOH 输出上界
    max_cycle_hours: float = 4.0    # 已经过时间归一化上限

    # ---- 特征通道 ----
    # 时序窗口 7 通道: v_sum_cell, v_max, v_min, current, t_max, t_min, v_spread
    c_in: int = 7
    # 因果上下文 5 维: mode, t_mean, elapsed, window_current, voltage_spread
    c_ctx: int = 5
    # 循环级特征 4 维: 循环安时/C_nom, ΔSOC, 时长(h), 平均电流/100（仅供 SOH 分支）
    c_cycle: int = 4
    soh_cycle_only: int = 0        # 1=SOH 分支只用循环级特征; 0=深层窗口特征 Z' + 循环级特征(默认, exp012 更优)
    use_soc_bms_ctx: bool = False   # 是否把 BMS SOC 作为"弱参考"上下文输入（False=纯 V/I/T 估计）
    smooth_soc_label: bool = True   # 高质量 SOC 标签：安时积分平滑 + 首尾 BMS 锚点漂移修正
    infer_n_cells: int = 0          # 1=按每个 CSV 的 VOLT_n 列数识别串数；0=使用 LFP=124/NCM=96

    # ---- 模型 ----
    tcn_channels: int = 64
    tcn_layers: int = 4
    tcn_kernel: int = 5
    tcn_dilations: tuple = (1, 2, 4, 8)
    tcn_out: int = 128
    ctx_hidden: int = 32
    ctx_out: int = 64
    fusion_dim: int = 256
    branch_dim: int = 128
    n_attn_heads: int = 4
    dropout: float = 0.1

    # ---- 损失 ----
    lambda_physics: float = 0.1
    lambda_consistency: float = 0.0   # 0 = 关闭 OCV 电压一致性损失
    ocv_n_bins: int = 128
    ocv_i_thresh: float = 15.0        # 构造 OCV 曲线用的低电流阈值(A)
    r0_init: float = 0.0005           # 欧姆内阻 R0 初值(Ω, 每串/单体级)
    r1_init: float = 0.0005           # 极化内阻 R1 初值(Ω, 每串/单体级)
    tau_init: float = 60.0            # 极化时间常数 τ=R1·C1 初值(s)
    sample_dt: float = 10.0           # 采样间隔(s)，Thevenin 极化递推用

    # ---- 训练 ----
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    log_every: int = 20
    val_every: int = 1
    early_stop_patience: int = 8

    # ---- 分阶段训练（DNN-MTL §5.2）----
    staged: int = 1               # 1=分阶段(默认), 0=端到端单阶段
    stage1_epochs: int = 20       # 阶段1: SOC 预训练
    stage2_epochs: int = 20       # 阶段2: SOH 预训练
    stage3_epochs: int = 20       # 阶段3: 联合微调
    stage1_lr: float = 1e-3
    stage2_lr: float = 5e-4
    stage3_lr: float = 1e-4
    stage2_freeze_shared: int = 1  # 1=阶段2 完全冻结共享编码器(仅训注意力+SOH分支), 0=共享编码器微调

    # ---- 实验记录 ----
    exp_name: str = "exp001_baseline"
    note: str = ""

    @property
    def model_config(self) -> dict:
        keys = ["c_in", "c_ctx", "window_len", "tcn_channels", "tcn_layers",
                "tcn_kernel", "tcn_dilations", "tcn_out", "ctx_hidden", "ctx_out",
                "fusion_dim", "branch_dim", "n_attn_heads", "dropout"]
        return {k: getattr(self, k) for k in keys}


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(description="DNN-MTL (适配) 训练")
    p.add_argument("--data_root", type=str, default="")
    p.add_argument("--output_root", type=str, default="")
    p.add_argument("--chemistry", choices=["LFP", "NCM", "BOTH"], default="LFP")
    p.add_argument("--n_vehicles", type=int, default=80)
    p.add_argument("--include_low_capacity", type=int, default=0, choices=[0, 1])
    p.add_argument("--include_high_resistance", type=int, default=0, choices=[0, 1])
    p.add_argument("--window_len", type=int, default=128)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tcn_channels", type=int, default=64)
    p.add_argument("--fusion_dim", type=int, default=256)
    p.add_argument("--lambda_physics", type=float, default=0.1)
    p.add_argument("--lambda_consistency", type=float, default=0.0)
    p.add_argument("--r0_init", type=float, default=0.0005)
    p.add_argument("--r1_init", type=float, default=0.0005)
    p.add_argument("--tau_init", type=float, default=60.0)
    p.add_argument("--stage2_freeze_shared", type=int, default=1, choices=[0, 1])
    p.add_argument("--soh_per_cycle", type=int, default=0, choices=[0, 1])
    p.add_argument("--soh_cycle_only", type=int, default=0, choices=[0, 1])
    p.add_argument("--causal_window_soh", type=int, default=1, choices=[0, 1])
    p.add_argument("--soh_all_modes", type=int, default=1, choices=[0, 1])
    p.add_argument("--soh_min", type=float, default=0.6)
    p.add_argument("--soh_max", type=float, default=1.3)
    p.add_argument("--use_soc_bms_ctx", action="store_true", default=False)
    p.add_argument("--infer_n_cells", type=int, default=0, choices=[0, 1])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--staged", type=int, default=1, choices=[0, 1], help="1=分阶段训练(默认), 0=端到端")
    p.add_argument("--stage1_epochs", type=int, default=20)
    p.add_argument("--stage2_epochs", type=int, default=20)
    p.add_argument("--stage3_epochs", type=int, default=20)
    p.add_argument("--exp_name", type=str, default="exp001_baseline")
    p.add_argument("--note", type=str, default="")
    args = p.parse_args(argv)
    cfg = Config(**{k: v for k, v in vars(args).items() if v is not None})
    return cfg


def to_dict(cfg: Config) -> dict:
    d = asdict(cfg)
    d["cap_clip"] = list(d["cap_clip"])
    d["tcn_dilations"] = list(d["tcn_dilations"])
    return d


def save_config(cfg: Config, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_dict(cfg), f, ensure_ascii=False, indent=2)
