# -*- coding: utf-8 -*-
"""数据模块：解析正常车辆 CSV、按充电循环构造 SOH 标签、滑动窗口、归一化、按车划分。

关键约定（来自数据探查）：
  - 采样间隔 10s；CHARGE_STATUS: 1=充电, 3=放电；SUM_CURRENT: 负=充电, 正=放电
  - SOC 列为 BMS SOC（0-100），作为 SOC 标签与"弱参考"上下文
  - SOH 标签 = 充电循环安时积分外推容量 / C_nom；C_nom 取训练集充电容量中位数
"""
import os
import re
import random
import functools
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(CODE_ROOT)
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)
DEFAULT_DATA_ROOT = os.path.join(WORKSPACE_ROOT, "dataset")
CHEM_CELLS = {"LFP": 124, "NCM": 96}
# 每串电压物理范围（用于清洗 OCV 坏点）
CHEM_VRANGE = {"LFP": (2.5, 3.7), "NCM": (2.8, 4.3)}

# 时序窗口 7 通道（顺序固定）
FEAT_NAMES = ["v_sum_cell", "v_max", "v_min", "current", "t_max", "t_min", "v_spread"]


def is_charge_file(fname):
    """按文件名判定充电/放电（注意 'discharge' 含 'charge' 子串）。"""
    return "discharge" not in fname and "charge" in fname


def resolve_data_root(data_root=""):
    """解析数据根目录。默认使用工作区根目录的 dataset/ 目录。"""
    root = data_root or DEFAULT_DATA_ROOT
    return os.path.abspath(os.path.expanduser(root))


def infer_n_cells_from_columns(columns, default):
    """根据 VOLT_1、VOLT_2 ... 列识别当前文件串数，无法识别时回退到化学体系默认值。"""
    indices = []
    for name in columns:
        m = re.fullmatch(r"VOLT_(\d+)", str(name))
        if m:
            indices.append(int(m.group(1)))
    return max(indices) if indices else int(default)


@functools.lru_cache(maxsize=256)
def infer_n_cells_from_csv(path, default):
    columns = pd.read_csv(path, nrows=0).columns
    return infer_n_cells_from_columns(columns, default)


def get_n_cells(path, chem, infer=False):
    default = CHEM_CELLS[chem]
    return infer_n_cells_from_csv(path, default) if infer else default


def list_vehicle_dirs(chemistry, categories=("normal",), data_root=""):
    """返回 [(vin_dir, chem, category), ...]，按 chemistry 与类别筛选车辆。"""
    root = resolve_data_root(data_root)
    chems = [chemistry] if chemistry != "BOTH" else ["LFP", "NCM"]
    out = []
    for c in chems:
        for cat in categories:
            ndir = os.path.join(root, c, cat)
            if os.path.isdir(ndir):
                for vin in sorted(os.listdir(ndir), key=lambda s: int(s.split("_")[1])):
                    d = os.path.join(ndir, vin)
                    if os.path.isdir(d):
                        out.append((d, c, cat))
    return out


@functools.lru_cache(maxsize=16)
def _load_cycle_cached(path):
    df = pd.read_csv(path)
    return df


def load_cycle_df(path):
    return _load_cycle_cached(path)


@functools.lru_cache(maxsize=16)
def _load_arrays_cached(path, n_cells):
    """缓存单文件全量数组，供滑动窗口切片复用。"""
    df = pd.read_csv(path)
    feats = build_features(df, n_cells)                       # (T, 7) float32
    soc = df["SOC"].values.astype(np.float32) / 100.0
    t = df["TIME"].values.astype(np.float32)
    i = df["SUM_CURRENT"].values.astype(np.float32)
    tmax = df["MAX_TEMP"].values.astype(np.float32)
    tmin = df["MIN_TEMP"].values.astype(np.float32)
    return feats, soc, t, i, tmax, tmin


def charge_capacity_ah(df):
    """充电段安时积分外推满充容量(Ah)。返回 capacity 或 None。"""
    m = df["CHARGE_STATUS"] == 1
    d = df[m]
    if len(d) < 10:
        return None
    t = d["TIME"].values
    i = d["SUM_CURRENT"].values
    dt = np.diff(t)
    ah = np.sum(np.abs(i[:-1]) * dt) / 3600.0
    dsoc = (d["SOC"].iloc[-1] - d["SOC"].iloc[0]) / 100.0
    if abs(dsoc) < 1e-6:
        return None
    return ah / abs(dsoc)


def smooth_soc_array(soc_bms, t, i, c_nom):
    """高质量 SOC 标签：安时积分（平滑）+ 首尾 BMS 锚点线性漂移修正。

    用电流积分把 BMS 的 1% 整数阶梯标签"打散"成平滑、单调、与电流一致的连续标签；
    首尾锚定到 BMS 值，消除安时积分的累积漂移。返回 [0,1] 的 float32 数组。
    """
    soc_bms = np.asarray(soc_bms, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    i = np.asarray(i, dtype=np.float64)
    dt = np.diff(t)
    # 充入电量(Ah)：i<0 为充电，-i 为正
    delta_ah = np.concatenate([[0.0], np.cumsum(-i[:-1] * dt) / 3600.0])
    soc_coulomb = soc_bms[0] + delta_ah / max(c_nom, 1e-6)
    span = t[-1] - t[0]
    if span > 0:
        drift = soc_bms[-1] - soc_coulomb[-1]
        frac = (t - t[0]) / span
        soc_coulomb = soc_coulomb + drift * frac
    return np.clip(soc_coulomb, 0.0, 1.0).astype(np.float32)


def compute_c_nom(vehicle_dirs, min_dsoc=0.20):
    """从训练车辆充电循环的中位数容量估算标称容量 C_nom。"""
    caps = []
    for vdir, _ in vehicle_dirs:
        for f in sorted(os.listdir(vdir)):
            if is_charge_file(f) and f.endswith(".csv"):
                df = load_cycle_df(os.path.join(vdir, f))
                m = df["CHARGE_STATUS"] == 1
                if int(m.sum()) < 2:
                    continue
                dsoc = (df["SOC"][m].iloc[-1] - df["SOC"][m].iloc[0]) / 100.0
                if dsoc >= min_dsoc:
                    c = charge_capacity_ah(df)
                    if c is not None:
                        caps.append(c)
    return float(np.median(caps)) if caps else None


def build_vehicle_soh(vehicle_dirs, c_nom_per_chem, min_dsoc, cap_clip, per_cycle=True):
    """构造 SOH 标签。返回 {vin_dir: {filename: soh}}。

    per_cycle=True：每个有效充电循环一个 SOH（=该循环容量/C_nom），与循环级特征一致；
    放电/无效循环用该车充电循环 SOH 中位数填充。
    per_cycle=False：每台车一个车辆级 SOH（充电循环容量中位数/C_nom）。
    """
    lo, hi = cap_clip
    res = {}
    for vdir, chem, _cat in vehicle_dirs:
        c_nom = c_nom_per_chem[chem]
        caps = []
        per_file_cycle = {}
        for f in sorted(os.listdir(vdir)):
            if not (is_charge_file(f) and f.endswith(".csv")):
                continue
            df = load_cycle_df(os.path.join(vdir, f))
            m = df["CHARGE_STATUS"] == 1
            if int(m.sum()) < 2:
                continue
            dsoc = (df["SOC"][m].iloc[-1] - df["SOC"][m].iloc[0]) / 100.0
            if dsoc < min_dsoc:
                continue
            c = charge_capacity_ah(df)
            if c is not None and lo * c_nom <= c <= hi * c_nom:
                caps.append(c)
                per_file_cycle[f] = float(np.clip(c / c_nom, 0.0, 1.5))
        # 无有有效容量循环时使用 NaN，由 Dataset 显式屏蔽 SOH 监督，
        # 避免把“无标签”误当成 SOH=1.0。
        soh_veh = float(np.clip(np.median(caps) / c_nom, lo, hi)) if caps else np.nan
        per_file = {}
        for f in os.listdir(vdir):
            if not f.endswith(".csv"):
                continue
            per_file[f] = per_file_cycle.get(f, soh_veh) if per_cycle else soh_veh
        res[vdir] = per_file
    return res


class FeatureStats:
    """训练集逐通道均值/标准差（z-score）。"""
    def __init__(self):
        self.n = 0
        self.sum = None
        self.sumsq = None

    def _add(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.sum is None:
            self.sum = np.zeros(x.shape[-1], dtype=np.float64)
            self.sumsq = np.zeros(x.shape[-1], dtype=np.float64)
        self.sum += x.sum(axis=0)
        self.sumsq += (x * x).sum(axis=0)
        self.n += x.shape[0]

    def compute(self, vehicle_dirs, infer_n_cells=False):
        for vdir, chem, _cat in vehicle_dirs:
            for f in sorted(os.listdir(vdir)):
                if not f.endswith(".csv"):
                    continue
                path = os.path.join(vdir, f)
                df = load_cycle_df(path)
                if len(df) < 2:
                    continue
                n_cells = get_n_cells(path, chem, infer_n_cells)
                feats = build_features(df, n_cells)   # (T, 7)
                self._add(feats)
        mean = self.sum / self.n
        var = self.sumsq / self.n - mean ** 2
        std = np.sqrt(np.clip(var, 1e-6, None))
        return mean, std


def build_features(df, n_cells):
    """由原始 DataFrame 构造 (T, 7) 特征矩阵。"""
    v_sum_cell = df["SUM_VOLTAGE"].values / n_cells
    v_max = df["MAX_CELL_VOLT"].values
    v_min = df["MIN_CELL_VOLT"].values
    cur = df["SUM_CURRENT"].values
    t_max = df["MAX_TEMP"].values
    t_min = df["MIN_TEMP"].values
    v_spread = v_max - v_min
    return np.stack([v_sum_cell, v_max, v_min, cur, t_max, t_min, v_spread], axis=1).astype(np.float32)


def build_ocv_curve(vehicle_dirs, n_bins=128, infer_n_cells=False):
    """拟合每串 OCV(SOC) 曲线（清洗坏点版）。返回 (soc_grid[n_bins], ocv[n_bins])。

    采用"充放电半和法"：同一 SOC 处 OCV ≈ (V_charge_median + V_discharge_median)/2，
    一阶抵消充电(+IR+极化)与放电(-IR-极化)的电压偏移，比"低电流点"法更稳，
    能自然剔除电流过零瞬态造成的坏点（如低电流点法出现的 0.6V 异常）。
    再叠加：物理电压范围裁剪 → 空箱线性插值 → 轻平滑。
    """
    grid = np.linspace(0.0, 1.0, n_bins)
    bins_c = [[] for _ in range(n_bins)]
    bins_d = [[] for _ in range(n_bins)]
    for vdir, chem, _cat in vehicle_dirs:
        for f in sorted(os.listdir(vdir)):
            if not f.endswith(".csv"):
                continue
            path = os.path.join(vdir, f)
            df = load_cycle_df(path)
            if len(df) < 2:
                continue
            n_cells = get_n_cells(path, chem, infer_n_cells)
            v_cell = df["SUM_VOLTAGE"].values / n_cells
            soc = df["SOC"].values / 100.0
            status = df["CHARGE_STATUS"].values
            idx = np.clip((soc * (n_bins - 1)).astype(int), 0, n_bins - 1)
            for k, v, s in zip(idx, v_cell, status):
                if s == 1:
                    bins_c[int(k)].append(float(v))
                elif s == 3:
                    bins_d[int(k)].append(float(v))
    ocv = np.full(n_bins, np.nan)
    vmin, vmax = 2.0, 4.6  # 宽松整体物理界，再按化学精剪
    for k in range(n_bins):
        c = float(np.median(bins_c[k])) if len(bins_c[k]) >= 3 else np.nan
        d = float(np.median(bins_d[k])) if len(bins_d[k]) >= 3 else np.nan
        if not np.isnan(c) and not np.isnan(d):
            ocv[k] = (c + d) / 2.0
        elif not np.isnan(c):
            ocv[k] = c
        elif not np.isnan(d):
            ocv[k] = d
    # 物理范围裁剪 + 剔除明显离群（与邻域中值偏差过大）
    ocv = np.clip(ocv, vmin, vmax)
    # 线性插值填补空箱
    xs = np.where(~np.isnan(ocv))[0]
    if len(xs) == 0:
        return grid, np.full(n_bins, 3.6, dtype=np.float32)
    if len(xs) < n_bins:
        ocv = np.interp(np.arange(n_bins), xs, ocv[xs])
    # 中值滤波剔除孤立毛刺，再轻平滑（edge 填充，避免零填充把两端拖向 0）
    ocv = _median_filter(ocv, 5)
    ocv = _moving_average(ocv, 5)
    # 保序回归强制单调递增（OCV 物理上随 SOC 单调，消除噪声抖动）
    from sklearn.isotonic import IsotonicRegression
    ocv = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(np.arange(n_bins), ocv)
    return grid, ocv.astype(np.float32)


def _median_filter(x, k):
    """一维中值滤波，k 为奇数窗口。"""
    if k % 2 == 0:
        k += 1
    x = np.asarray(x, dtype=np.float64)
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.median(xp[i:i + k])
    return out


def _moving_average(x, k):
    """一维滑动平均（edge 填充，保持长度不变）。"""
    x = np.asarray(x, dtype=np.float64)
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    w = np.ones(k) / k
    return np.convolve(xp, w, mode="valid")


def cycle_features_array(soc_bms, t, i, c_nom, is_charge):
    """循环级特征（供 SOH 分支）：[循环安时/C_nom, ΔSOC, 时长(h), 平均电流/100]。

    SOH(容量)是整车级慢变量，单窗口 V/I/T 推断不出；这些循环级统计量编码了容量信息。
    """
    dt = np.diff(t)
    if is_charge:
        ah = float(np.sum(np.maximum(-i[:-1], 0.0) * dt) / 3600.0)
        dsoc = float(soc_bms[-1] - soc_bms[0])
    else:
        ah = float(np.sum(np.maximum(i[:-1], 0.0) * dt) / 3600.0)
        dsoc = float(soc_bms[0] - soc_bms[-1])
    duration_h = float((t[-1] - t[0]) / 3600.0)
    mean_i = float(np.mean(np.abs(i)))
    return np.array([ah / max(c_nom, 1e-6), max(dsoc, 0.0), duration_h, mean_i / 100.0], dtype=np.float32)


def causal_prefix_features(soc_bms, t, i, c_nom, mode, end):
    """构造截至当前窗口末端的循环前缀特征。

    返回 [累计Ah/C_nom, 累计ΔSOC, 已经过时间(h), 运行平均|I|/100]。
    标签可以由离线完整循环构造，但模型输入不得读取 end 之后的任何点。
    """
    end = int(end)
    t_prefix = t[:end + 1]
    i_prefix = i[:end + 1]
    if len(t_prefix) < 2:
        ah = 0.0
    else:
        dt = np.diff(t_prefix)
        flow = np.maximum(-i_prefix[:-1], 0.0) if mode == 1 else np.maximum(i_prefix[:-1], 0.0)
        ah = float(np.sum(flow * dt) / 3600.0)
    dsoc = float(abs(soc_bms[end] - soc_bms[0]))
    elapsed_h = float(max(t[end] - t[0], 0.0) / 3600.0)
    mean_i = float(np.mean(np.abs(i_prefix)))
    return np.array([ah / max(c_nom, 1e-6), dsoc, elapsed_h, mean_i / 100.0], dtype=np.float32)


class BatteryDataset(Dataset):
    """滑动窗口样本集。样本 = 单文件内一段窗口（不跨文件/循环）。"""

    def __init__(self, vehicle_dirs, soh_map, c_nom_per_chem, stats, cfg):
        self.cfg = cfg
        self.mean = stats[0].astype(np.float32)
        self.std = stats[1].astype(np.float32)
        self.samples = []          # (path, start, mode, soh, n_cells, chem, c_nom)
        self.arrays = {}           # path -> (feats, soc_bms, soc_label, t, i, tmax, tmin) 预载入内存
        self.cycle_feat = {}       # path -> 旧版完整循环特征（仅兼容非因果模式）
        for vdir, chem, _cat in vehicle_dirs:
            c_nom = c_nom_per_chem[chem]
            files = sorted([f for f in os.listdir(vdir) if f.endswith(".csv")])
            for f in files:
                path = os.path.join(vdir, f)
                n_cells = get_n_cells(path, chem, bool(cfg.infer_n_cells))
                feats, soc_bms, t, i, tmax, tmin = _load_arrays_cached(path, n_cells)
                L = feats.shape[0]
                if L < cfg.window_len:
                    continue
                if cfg.smooth_soc_label:
                    soc_label = smooth_soc_array(soc_bms, t, i, c_nom)
                else:
                    soc_label = soc_bms
                self.arrays[path] = (feats, soc_bms, soc_label, t, i, tmax, tmin)
                is_chg = is_charge_file(f)
                self.cycle_feat[path] = cycle_features_array(soc_bms, t, i, c_nom, is_chg)
                soh = soh_map.get(vdir, {}).get(f, np.nan)
                mode = 1 if is_chg else 0
                starts = list(range(0, L - cfg.window_len + 1, cfg.stride))
                n_windows = max(len(starts), 1)
                for s in starts:
                    self.samples.append((path, s, mode, soh, n_cells, chem, c_nom, n_windows))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, start, mode, soh, n_cells, chem, c_nom, n_windows = self.samples[idx]
        W = self.cfg.window_len
        feats_all, soc_bms, soc_label, t, i, tmax, tmin = self.arrays[path]
        feats = feats_all[start:start + W]                     # (W, 7)
        feats = (feats - self.mean[None, :]) / self.std[None, :]

        # 上下文 5 维：只使用当前窗口可见量，不再输入 SOH 标签或最终循环进度。
        e = start + W - 1
        t_mean = float((tmax[e] + tmin[e]) / 2.0)
        # t_mean 用温度通道统计量归一化（取 t_max/t_min 均值/标准差的平均）
        t_mean_z = (t_mean - float((self.mean[4] + self.mean[5]) / 2.0)) / max(float((self.std[4] + self.std[5]) / 2.0), 1e-6)
        elapsed_h = float(max(t[e] - t[0], 0.0) / 3600.0)
        elapsed_norm = min(elapsed_h / max(float(self.cfg.max_cycle_hours), 1e-6), 1.0)
        current_mean_z = float(feats[:, 3].mean())
        spread_end_z = float(feats[-1, 6])
        ctx = np.array([mode, t_mean_z, elapsed_norm, current_mean_z, spread_end_z], dtype=np.float32)

        # 标签（高质量：平滑安时积分 SOC）
        y_soc_start = float(soc_label[start])
        y_soc_end = float(soc_label[e])
        label_valid = bool(np.isfinite(soh))
        y_soh = float(soh) if label_valid else 1.0
        soh_valid = 1.0 if label_valid and (bool(self.cfg.soh_all_modes) or mode == 1) else 0.0

        # 物理损失独立锚点：窗口内充电安时积分（Ah），仅充电样本有意义
        dt = np.diff(t[start:start + W])
        i_w = i[start:start + W]
        coulomb_ah = float(np.sum(np.maximum(-i_w[:-1], 0.0) * dt) / 3600.0)

        return {
            "x": torch.from_numpy(np.ascontiguousarray(feats)),
            "ctx": torch.from_numpy(ctx.copy()),
            "soc_start": torch.tensor(y_soc_start),
            "soc_end": torch.tensor(y_soc_end),
            "soh": torch.tensor(y_soh),
            "soh_valid": torch.tensor(soh_valid),
            # 长循环会产生更多重叠窗口；此权重使每个循环的 SOH 总贡献近似相等。
            "soh_weight": torch.tensor(soh_valid / n_windows),
            "physics_valid": torch.tensor(1.0 if label_valid and mode == 1 else 0.0),
            "coulomb_ah": torch.tensor(coulomb_ah),
            "c_nom": torch.tensor(c_nom),
            "n_cells": torch.tensor(float(n_cells)),
            "v_sum_raw": torch.tensor(float(feats_all[e, 0] * n_cells)),
            "i_raw": torch.tensor(float(i[e])),
            "i_seq": torch.from_numpy(i_w.astype(np.float32).copy()),
            "cycle_feat": torch.from_numpy(
                causal_prefix_features(soc_bms, t, i, c_nom, mode, e)
                if bool(self.cfg.causal_window_soh) else self.cycle_feat[path].copy()),
        }


def make_splits(cfg, seed=42):
    """按车辆划分 train/val/test，返回三组 vehicle_dirs 及每化学 C_nom。"""
    rng = random.Random(seed)
    # 正常车辆（受 n_vehicles 上限约束）+ 可选故障/老化车辆（全部纳入）
    data_root = resolve_data_root(cfg.data_root)
    normal = list_vehicle_dirs(cfg.chemistry, ["normal"], data_root)
    if cfg.n_vehicles < len(normal):
        normal = normal[:cfg.n_vehicles]
    vehicle_dirs = list(normal)
    if cfg.include_low_capacity:
        vehicle_dirs += list_vehicle_dirs(cfg.chemistry, ["low_capacity"], data_root)
    if cfg.include_high_resistance:
        vehicle_dirs += list_vehicle_dirs(cfg.chemistry, ["high_resistance"], data_root)
    if not vehicle_dirs:
        raise FileNotFoundError(
            f"未在数据根目录 {data_root!r} 下找到 chemistry={cfg.chemistry} 的车辆数据")
    rng.shuffle(vehicle_dirs)
    n = len(vehicle_dirs)
    n_val = int(round(n * cfg.val_ratio))
    n_test = int(round(n * cfg.test_ratio))
    n_train = n - n_val - n_test
    train = vehicle_dirs[:n_train]
    val = vehicle_dirs[n_train:n_train + n_val]
    test = vehicle_dirs[n_train + n_val:]
    # C_nom 只从训练集"正常"车辆估计（避免老化车拉低标称容量）
    c_nom_per_chem = {}
    for chem in (["LFP", "NCM"] if cfg.chemistry == "BOTH" else [cfg.chemistry]):
        train_normal = [(d, c) for d, c, cat in train if c == chem and cat == "normal"]
        c_nom = compute_c_nom(train_normal, cfg.min_cycle_dsoc)
        c_nom_per_chem[chem] = c_nom if c_nom else 1.0
    return train, val, test, c_nom_per_chem


def build_datasets(cfg):
    train, val, test, c_nom_per_chem = make_splits(cfg, cfg.seed)
    # 归一化统计量只从训练集估计（按化学分别统计，取平均尺度；单化学时即该化学）
    stats = FeatureStats()
    # 训练集可能混合化学，但 v_sum_cell 已按单体外推、cell 电压同尺度，整体统计即可
    mean, std = stats.compute(train, bool(cfg.infer_n_cells))
    soh_train = build_vehicle_soh(train, c_nom_per_chem, cfg.min_cycle_dsoc, cfg.cap_clip, bool(cfg.soh_per_cycle))
    soh_val = build_vehicle_soh(val, c_nom_per_chem, cfg.min_cycle_dsoc, cfg.cap_clip, bool(cfg.soh_per_cycle))
    soh_test = build_vehicle_soh(test, c_nom_per_chem, cfg.min_cycle_dsoc, cfg.cap_clip, bool(cfg.soh_per_cycle))
    ds_train = BatteryDataset(train, soh_train, c_nom_per_chem, (mean, std), cfg)
    ds_val = BatteryDataset(val, soh_val, c_nom_per_chem, (mean, std), cfg)
    ds_test = BatteryDataset(test, soh_test, c_nom_per_chem, (mean, std), cfg)
    meta = {
        "n_train_veh": len(train), "n_val_veh": len(val), "n_test_veh": len(test),
        "n_train": len(ds_train), "n_val": len(ds_val), "n_test": len(ds_test),
        "c_nom_per_chem": c_nom_per_chem,
        "feature_mean": mean.tolist(), "feature_std": std.tolist(),
        "data_root": resolve_data_root(cfg.data_root),
        "infer_n_cells": bool(cfg.infer_n_cells),
    }
    # OCV(SOC) 曲线（仅从训练集拟合；充放电半和法清洗坏点）
    ocv = None
    if cfg.lambda_consistency > 0:
        soc_grid, ocv_curve = build_ocv_curve(train, cfg.ocv_n_bins, bool(cfg.infer_n_cells))
        ocv = {"soc_grid": soc_grid.tolist(), "ocv": ocv_curve.tolist()}
        meta["ocv"] = ocv
    return ds_train, ds_val, ds_test, meta, ocv
