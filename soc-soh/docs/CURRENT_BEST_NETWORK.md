# 当前推荐网络完整说明：Causal-Window DNN-MTL

> 本文件是当前项目中“最好且推荐继续演进的网络”的统一描述性文件，也是后续智能体理解现状时应优先阅读的网络文档。
>
> 文档依据：当前 `dnn_mtl/` 实际代码、exp016/exp017 固化代码快照、实验配置、数据元信息、测试指标和因果分析结果。
>
> 状态日期：2026-09-07。

---

## 1. “目前最好”的准确含义

当前推荐主网络是 **Causal-Window DNN-MTL（因果窗口 SOC/SOH 多任务网络）**，对应：

- LFP：`experiments/exp016_lfp_causal_window_soh/`
- NCM：`experiments/exp017_ncm_causal_window_soh/`

两轮实验采用同一套模型结构和因果窗口方案，主要区别是：

- 化学体系不同；
- 训练数据和归一化参数不同；
- NCM 开启了 Thevenin/OCV 电压一致性损失，LFP 未开启；
- 两个体系分别保存独立权重，不能混用。

这里的“最好”表示它是当前最符合项目目标的版本：

1. 每个滑动窗口都输出一个 SOH；
2. 不把当前 SOH 标签或完整循环统计量直接输入网络；
3. SOH 特征只读取窗口末端及此前的数据；
4. 同时估计窗口起点 SOC、窗口终点 SOC 和窗口 SOH；
5. 保留物理一致性约束、任务耦合和分阶段训练；
6. 已在 LFP、NCM 两种体系上完成训练与独立车辆测试。

“当前推荐主网络”不等于所有单项指标的历史最优：

- LFP 的历史最低 SOC MAE 来自 exp014（1.12%），但它不是当前因果逐窗口 SOH 口径；
- exp016 的 LFP SOC MAE 为 1.68%，SOH MAE 为 2.43%；
- exp017 的 NCM SOC MAE 为 4.56%，SOH MAE 为 5.05%；
- exp016/017 的价值主要在于修正了 SOH 输入泄漏问题，并实现了真正的逐窗口输出。

因此，后续若继续开发逐窗口 SOH、适配 Wenzhou 数据或做在线部署，应以 exp016/017 架构为基线，而不是退回 exp014/015。

---

## 2. 代码和实验资产

### 2.1 当前活动代码

```text
H:\中汽研-华为\dnn_mtl\
├── config.py          # 配置、命令行参数
├── data.py            # 数据划分、标签、特征、窗口、OCV
├── model.py           # 网络结构
├── losses.py          # 多任务损失、物理损失、OCV一致性损失
├── train.py           # 三阶段训练、验证、测试、结果保存
├── analyze_causal.py  # 因果窗口分组分析
└── explore_data.py    # 数据探查
```

### 2.2 可复现代码快照

```text
H:\中汽研-华为\experiments\exp016_lfp_causal_window_soh\code\
H:\中汽研-华为\experiments\exp017_ncm_causal_window_soh\code\
```

当前 `dnn_mtl/` 中的 `model.py`、`data.py`、`losses.py`、`config.py`、`train.py` 与 exp017 快照的 SHA-256 均一致。exp017 快照可以视为当前代码的可复现固化版本。

### 2.3 最优权重

```text
LFP: H:\中汽研-华为\experiments\exp016_lfp_causal_window_soh\best_model.pt
NCM: H:\中汽研-华为\experiments\exp017_ncm_causal_window_soh\best_model.pt
```

`best_model.pt` 保存：

```python
{
    "model_state": model.state_dict(),
    "cfg": 完整实验配置,
    "test_metrics": 测试集指标,
    "best_epoch": 最优轮次,
}
```

加载权重时应优先读取权重文件内的 `cfg`，不要只依赖 `config.py` 的默认值。

---

## 3. 任务定义

网络面向单个车辆电池包/试验对象，接收当前窗口和截至窗口末端的因果历史统计，完成三个回归输出：

```text
输入：
  x           [B, W, 7]  当前滑动窗口时序
  ctx         [B, 5]     当前窗口上下文
  cycle_feat  [B, 4]     截至窗口末端的循环前缀统计

输出：
  SOC_start   [B, 1]     窗口第一个采样点 SOC
  SOC_end     [B, 1]     窗口最后一个采样点 SOC
  SOH         [B, 1]     当前窗口所属车辆的健康状态
```

默认配置：

```text
采样间隔：10 s
窗口长度：W = 128
窗口时间跨度：约 21.3 min
滑动步长：16 点，即约 160 s
每个窗口：独立输出 SOC_start、SOC_end、SOH
```

当前实现不会输出窗口中间每一帧的 SOC 序列。所谓“每窗口输出一个 SOH”，是每移动一次窗口进行一次前向计算并输出一个标量 SOH。

---

## 4. 输入数据的完整定义

### 4.1 时序输入 `x: [B, 128, 7]`

通道顺序固定为：

| 索引 | 特征名 | 原始来源/计算 | 物理含义 |
|---:|---|---|---|
| 0 | `v_sum_cell` | `SUM_VOLTAGE / n_cells` | 电池包总电压折算的平均每串电压 |
| 1 | `v_max` | `MAX_CELL_VOLT` | 最高单体电压 |
| 2 | `v_min` | `MIN_CELL_VOLT` | 最低单体电压 |
| 3 | `current` | `SUM_CURRENT` | 电池包总电流 |
| 4 | `t_max` | `MAX_TEMP` | 最高温度 |
| 5 | `t_min` | `MIN_TEMP` | 最低温度 |
| 6 | `v_spread` | `v_max - v_min` | 单体电压极差 |

串数由化学体系固定：

```text
LFP: 124 串
NCM: 96 串
```

现有车辆数据电流约定：

```text
SUM_CURRENT < 0：充电
SUM_CURRENT > 0：放电
CHARGE_STATUS == 1：充电
CHARGE_STATUS == 3：放电
```

7 个通道均采用训练集均值和标准差做 Z-score：

```text
x_normalized = (x - feature_mean) / feature_std
```

归一化统计只从训练车辆计算，验证集和测试集不参与。

### 4.2 上下文 `ctx: [B, 5]`

| 索引 | 特征 | 构造方法 | 是否因果 |
|---:|---|---|---|
| 0 | `mode` | 充电=1，放电=0 | 是 |
| 1 | `t_mean_z` | 窗口末点最高/最低温度均值，再按温度统计量归一化 | 是 |
| 2 | `elapsed_norm` | 从当前 CSV/循环起点到窗口末端的时间，除以 4 h 并截断到 1 | 是 |
| 3 | `window_current` | 当前窗口归一化电流通道的均值 | 是 |
| 4 | `voltage_spread` | 窗口末点归一化电压极差 | 是 |

配置中 `use_soc_bms_ctx=False`，因此 BMS SOC 不作为这 5 维上下文中的直接通道。

### 4.3 SOH 分支的循环前缀 `cycle_feat: [B, 4]`

开启 `causal_window_soh=1` 后，四维特征只统计到当前窗口末端：

| 索引 | 特征 | 定义 |
|---:|---|---|
| 0 | `partial_ah / C_nom` | 循环起点到窗口末端的充入或放出 Ah，除以标称容量 |
| 1 | `partial_dsoc` | 循环起点 BMS SOC 与窗口末端 BMS SOC 的绝对差 |
| 2 | `elapsed_h` | 循环起点到窗口末端的小时数 |
| 3 | `mean_abs_current / 100` | 循环前缀平均绝对电流除以 100 |

充电前缀电量：

```text
Q_prefix = Σ max(-I, 0) · Δt / 3600
```

放电前缀电量：

```text
Q_prefix = Σ max(I, 0) · Δt / 3600
```

关键边界：

- 这四维特征没有读取窗口末端以后的采样点；
- 但 `partial_dsoc` 直接使用了截至窗口末端的 BMS SOC；
- 因此当前 SOH 方案是 **BMS SOC 辅助的因果 SOH 估计**；
- 它不是严格意义上的纯 V/I/T SOH 估计；
- 若后续移除 `partial_dsoc`，必须重新训练并单独报告消融结果。

---

## 5. 标签构造

### 5.1 SOC 标签

原始 BMS SOC 是 0–100 的整数阶梯信号。网络监督目标不是直接使用粗糙阶梯，而是采用：

```text
安时积分平滑 + 循环首尾 BMS 锚点线性漂移修正
```

计算过程：

1. 从循环起始 BMS SOC 出发；
2. 根据电流和时间积分得到连续 SOC；
3. 用循环末端 BMS SOC 与积分末值的偏差生成线性漂移项；
4. 把漂移项分摊到整个循环；
5. 将结果裁剪到 `[0, 1]`。

公式可写为：

```text
ΔAh(t) = Σ[-I(k) · Δt(k)] / 3600
SOC_cc(t) = SOC_bms(0) + ΔAh(t) / C_nom
drift = SOC_bms(T) - SOC_cc(T)
SOC_label(t) = clip(SOC_cc(t) + drift · (t-t0)/(T-t0), 0, 1)
```

其中现有数据中充电电流为负，因此使用 `-I`。

需要准确理解：SOC 标签生成使用了完整 CSV 的末端 BMS SOC 做离线漂移修正；这是监督标签处理，不进入模型输入。网络推理输入仍然只到窗口末端，但训练目标属于离线生成的高质量标签。

### 5.2 SOH 标签

先对每个有效充电 CSV 估算满容量：

```text
Q_cycle = ∫|I|dt / ΔSOC_bms
```

有效循环要求：

- 充电状态采样点不少于 2；
- `ΔSOC_bms >= 0.20`；
- 容量位于 `[0.7·C_nom, 1.3·C_nom]`。

标称容量 `C_nom` 只由训练集中正常车辆的有效充电容量中位数估计：

```text
LFP C_nom = 152.93961329899713 Ah
NCM C_nom = 130.05982905982904 Ah
```

exp016/017 设置 `soh_per_cycle=0`，所以 SOH 是车辆级稳健标签：

```text
SOH_vehicle = median(Q_cycle of vehicle) / C_nom
```

结果裁剪到 `[0.7, 1.3]`。同一车辆的充电和放电窗口共享该车辆 SOH 标签；模型输出范围设置为 `[0.6, 1.3]`，给预测留出边界余量。

若一辆车没有有效容量循环，则 SOH 标签记为 NaN，训练时通过 mask 排除，而不是错误地设为 1.0。

标签口径的限制：

- SOH 是车辆级容量中位数，不表达同一车辆内部随时间逐循环缓慢变化；
- 该标签可能使用了相对当前窗口更晚的充电循环来确定车辆健康水平；
- 这属于离线监督标签定义，不是输入泄漏，但不能将当前结果表述为“完全在线获得真实 SOH 标签”；
- Wenzhou 长周期退化数据具备逐循环容量，后续可改成真正的循环级 SOH 监督。

---

## 6. 网络总体数据流

```text
时序窗口 x [B,128,7]
        │
        ▼
4层因果 TCN [B,64,128]
        │
        ├─────────────── 浅层路径 ─────────────────┐
        │                                           │
        │     时间均值池化 → 共享 temporal_head     │
        │                 → f_shallow [B,128]       │
        │                                           ▼
        │                                  与 ctx_emb [B,64] 拼接
        │                                           │
        │                                  共享 proj 192→256
        │                                           │
        │                                  z_shallow [B,256]
        │                                           │
        │                                  SOC branch 256→128
        │                                           │
        │                                      f_soc [B,128]
        │
        └─────────────── 深层路径 ─────────────────┐
                                                    │
                    4头时间自注意力 + FFN           │
                    → 时间均值池化                   │
                    → 共享 temporal_head             │
                    → f_deep [B,128]                 │
                                                    ▼
                                           与 ctx_emb [B,64] 拼接
                                                    │
                                           共享 proj 192→256
                                                    │
                                           z_deep [B,256]
                                                    │
cycle_feat [B,4] → cycle_enc 4→32 ────────────────┤
                                                    │
                                           SOH branch 288→128
                                                    │
                                               f_soh [B,128]

f_soc ⊙ f_soh → Star Linear(128→128)+GELU → f_star
       │                                      │
       ├── f_soc' = f_soc + f_star            │
       └── f_soh' = f_soh + f_star ───────────┘

f_soc' → 128→64→2 → Sigmoid → [SOC_start, SOC_end]
f_soh' → 128→64→1 → Sigmoid → 线性映射到[0.6,1.3] → SOH
```

---

## 7. 各网络模块的精确定义

### 7.1 四层残差因果 TCN

配置：

```text
输入通道：7
隐藏通道：64
层数：4
卷积核：5
膨胀率：[1, 2, 4, 8]
激活：GELU
归一化：BatchNorm1d
连接：残差连接
```

每个 TCNBlock：

```text
Conv1d → 因果裁剪 → BatchNorm → GELU → 加残差
```

卷积先在两侧形式上加入 `dilation × (kernel-1)` padding，再裁掉输出右端，使每个位置不访问其后的输入。首层通过 `1×1 Conv1d` 将 7 通道投影到 64 通道，其余层直接使用恒等残差。

理论感受野：

```text
1 + (kernel-1) × Σdilation
= 1 + 4 × (1+2+4+8)
= 61 个采样点
≈ 610 s
≈ 10.2 min
```

深层注意力随后可以在整个 128 点可见窗口内整合信息。

### 7.2 共享时间特征头

```text
Linear(64→128) → LayerNorm(128) → GELU
```

浅层和深层路径共用同一个 `temporal_head`，有利于把两种时间表示映射到统一空间，也减少参数量。

### 7.3 时间自注意力深层出口

```text
MultiheadAttention(dim=64, heads=4)
→ 残差 + LayerNorm
→ FFN: Linear(64→128) + GELU + Linear(128→64)
→ 残差 + LayerNorm
```

注意力没有使用内部 causal mask，因此窗口内各位置可以彼此关注。不过所有位置都属于推理时已经完整获得的当前窗口，没有读取窗口末端以后的数据。网络是在窗口结束时输出结果，而不是在窗口中每个时间点同步输出。

### 7.4 上下文编码器

```text
Linear(5→32) → LayerNorm → GELU
→ Linear(32→64) → LayerNorm → GELU
```

### 7.5 共享融合投影

```text
Concat(time_feature[128], context[64]) = 192
→ Linear(192→256)
→ LayerNorm
→ GELU
→ Dropout(0.1)
```

浅层 SOC 路径与深层 SOH 路径共享同一个投影模块。

### 7.6 SOC 分支

```text
z_shallow [256]
→ Linear(256→128)
→ GELU
→ f_soc [128]
```

### 7.7 循环前缀编码器与 SOH 分支

```text
cycle_feat [4]
→ Linear(4→32)
→ GELU
→ Linear(32→32)
→ cycle_emb [32]

Concat(z_deep[256], cycle_emb[32]) = 288
→ Linear(288→128)
→ GELU
→ f_soh [128]
```

`soh_cycle_only=0`，因此 SOH 同时使用深层窗口表征和循环前缀统计。exp013 已验证，仅使用循环统计而丢弃深层窗口表征会使 SOH 结果变差。

### 7.8 Star Operation

先做任务特征逐元素乘法：

```text
f_interaction = f_soc ⊙ f_soh
f_star = GELU(W_star · f_interaction + b_star)
```

再分别残差注入两个任务分支：

```text
f_soc' = f_soc + f_star
f_soh' = f_soh + f_star
```

作用是显式建模 SOC 与 SOH 的耦合：可用容量由 SOH 决定，而短期 SOC 变化和长期容量退化又共享电压、电流、温度及倍率信息。

### 7.9 输出头

SOC：

```text
Linear(128→64) → GELU → Linear(64→2) → Sigmoid
输出范围：[0,1]
```

SOH：

```text
Linear(128→64) → GELU → Linear(64→1) → Sigmoid
SOH = 0.6 + (1.3-0.6) × SigmoidOutput
输出范围：[0.6,1.3]
```

---

## 8. 参数规模

在指定 `py312` 环境中以当前配置实例化，实测总参数量为 **263,752（0.264 M）**。

| 模块 | 参数量 |
|---|---:|
| TCN | 64,960 |
| temporal_head | 8,576 |
| TemporalAttention | 33,472 |
| context encoder | 2,496 |
| shared projection | 49,920 |
| SOC branch | 32,896 |
| cycle encoder | 1,216 |
| SOH branch | 36,992 |
| Star Operation | 16,512 |
| SOC head | 8,386 |
| SOH head | 8,321 |
| 两个任务不确定性参数 | 2 |
| Thevenin 参数 R0、R1、τ | 3 |
| **合计** | **263,752** |

以 float32 只计算裸参数约 1.01 MiB；实际 `.pt` 文件还包含键名、配置和序列化开销。早期 V1.3 文档中的“量化后约 122 KB”是设计目标，不是当前已验证产物。

---

## 9. 损失函数

### 9.1 SOC 监督损失

窗口起点和终点各计算 Huber，再取均值：

```text
L_SOC = 0.5 × [Huber(SOC_start_pred, SOC_start_true)
             + Huber(SOC_end_pred, SOC_end_true)]
```

Huber `delta=0.1`。

### 9.2 SOH 监督损失

```text
L_SOH = weighted_mean(Huber(SOH_pred, SOH_true))
```

每个有效车辆/循环可能生成不同数量的重叠窗口。代码为每个窗口赋予：

```text
soh_weight = soh_valid / windows_per_csv
```

这使每个 CSV/循环对 SOH 损失的总贡献近似一致，避免长循环仅因窗口多而支配训练。

### 9.3 不确定性自适应任务加权

模型学习 `log_sigma_soc` 和 `log_sigma_soh`：

```text
L_SOC_weighted = L_SOC / (2σ_SOC²) + log(σ_SOC)
L_SOH_weighted = L_SOH / (2σ_SOH²) + log(σ_SOH)
L_task = L_SOC_weighted + L_SOH_weighted
```

这让训练自动调整 SOC 与 SOH 的相对尺度。由于包含 `log(σ)`，`L_task` 和总损失出现负值是正常现象，不能用“损失小于零”判断实现错误。

### 9.4 充电库仑物理约束

只对有效充电窗口启用：

```text
ΔSOC_pred = clamp(SOC_end - SOC_start, min=0)
Q_pred = ΔSOC_pred × SOH_pred × C_nom
Q_measured = Σ max(-I,0) × Δt / 3600
L_physics = Huber(Q_pred, Q_measured)
```

默认权重：

```text
lambda_physics = 0.1
```

当前公式只适配“充电电流为负、充电 SOC 上升”的旧数据。应用到 Wenzhou 放电数据时必须改成模式感知的有符号公式，不能直接复用。

### 9.5 NCM Thevenin/OCV 电压一致性

exp017 启用：

```text
lambda_consistency = 0.5
```

exp016 关闭：

```text
lambda_consistency = 0.0
```

从训练车辆构建 128 点 OCV(SOC) 曲线：

1. 按 SOC 分箱；
2. 对同一 SOC 的充电和放电每串电压分别取中位数；
3. 两者均存在时取半和，降低极化偏差；
4. 插值空箱；
5. 中值滤波和移动平均；
6. 通过 Isotonic Regression 强制单调递增。

一阶 RC 极化：

```text
α = exp(-Δt/τ)
V_p[k+1] = αV_p[k] + I[k]R1(1-α)
V_pred = OCV(SOC_end) + I_end·R0 + V_p
L_consistency = Huber(V_measured_per_cell, V_pred)
```

`R0`、`R1`、`τ` 在 log 空间学习，确保取值为正。exp017 最终值：

```text
R0  = 3.60e-5 Ω
R1  = 3.43e-5 Ω
τ   = 178.66 s
```

实现限制：`losses.py` 目前把 Thevenin 递推步长硬编码为 10 s，尽管配置存在 `sample_dt`。改变采样周期时必须同步修改实现。

### 9.6 总损失

联合阶段：

```text
L_total = L_task
        + lambda_physics × L_physics
        + lambda_consistency × L_consistency
```

当前尚未实现跨循环 SOH 单调退化损失；旧文档中的 `L_monotonicity` 仍是待办项。

---

## 10. 三阶段训练方法

优化器统一采用 AdamW：

```text
weight_decay = 1e-5
gradient clipping = 1.0
batch_size = 256
seed = 42
```

### 阶段 1：SOC 预训练

```text
epoch 1–20
learning rate = 1e-3
loss mode = soc
```

冻结：

```text
SOH branch、SOH head、Star、TemporalAttention、σ_soh、cycle_enc
```

主要训练：TCN、共享时间头、上下文编码器、共享投影、SOC 分支、SOC 输出头和 `σ_soc`。NCM 同时受 OCV 一致性项约束。

### 阶段 2：SOH 预训练

```text
epoch 21–40
learning rate = 5e-4
loss mode = soh
```

配置 `stage2_freeze_shared=1`，冻结：

```text
TCN、temporal_head、ctx_enc、proj
SOC branch、SOC head、Star、σ_soc
```

主要训练 TemporalAttention、cycle_enc、SOH branch、SOH head 和 `σ_soh`。这样避免 SOH 训练破坏第一阶段已学到的 SOC 共享表示。

### 阶段 3：联合微调

```text
epoch 41–60（最多）
learning rate = 1e-4
loss mode = joint
全部模块解冻
early_stop_patience = 8
```

只在第三阶段保存最优权重。当前最优模型选择指标是：

```text
validation SOC_end MAE
```

这一点是当前方案的重要局限：模型虽然是多任务网络，但 checkpoint 选择没有直接考虑 SOH。未来应比较 SOC/SOH 组合评分、Pareto 选择或分别保存最佳 SOC 与最佳 SOH checkpoint。

---

## 11. 数据划分与防泄漏机制

数据按车辆目录划分，而不是随机划分窗口：

```text
train / validation / test = 75% / 12.5% / 12.5%
```

实际两轮均为：

```text
训练车辆：172
验证车辆：29
测试车辆：29
```

正常车辆先受 `n_vehicles=200` 限制，随后额外纳入低容量车辆；随机种子为 42。训练集、验证集和测试集分别构造 SOH 映射，但 `C_nom` 与时序特征归一化参数只从训练集估计。

已采取的防泄漏措施：

- 同一车辆不会跨 train/val/test；
- 输入不包含当前 SOH 标签；
- 输入不包含完整循环最终进度；
- 因果前缀不读取窗口末端后的点；
- 归一化均值和标准差只来自训练车辆；
- OCV 曲线只用训练车辆拟合；
- `C_nom` 只由训练集正常车辆估计。

仍需诚实标注的监督定义：

- 平滑 SOC 标签使用完整循环末端 BMS SOC；
- 车辆级 SOH 标签使用该车辆多个有效充电循环的中位容量；
- SOH 前缀特征中的 `partial_dsoc` 使用当前可见的 BMS SOC；
- 因此网络输入是时间因果的，但标签是离线构造的，SOH 也不是纯 V/I/T。

---

## 12. 实验配置差异

| 项目 | exp016 LFP | exp017 NCM |
|---|---:|---:|
| 网络结构 | 相同 | 相同 |
| 参数量 | 263,752 | 263,752 |
| 串数 | 124 | 96 |
| 训练样本 | 216,597 | 61,472 |
| 验证样本 | 37,723 | 12,165 |
| 测试样本 | 37,669 | 9,871 |
| `C_nom` | 152.94 Ah | 130.06 Ah |
| `lambda_physics` | 0.1 | 0.1 |
| `lambda_consistency` | 0.0 | 0.5 |
| OCV/Thevenin | 关闭 | 开启 |
| 最优 epoch | 48 | 54 |
| 训练运行时长 | 2793.2 s | 956.8 s |

NCM 样本显著少于 LFP，而且电压、温度、电流分布不同，因此不能用 LFP 的归一化统计或权重直接替代 NCM 模型。

---

## 13. 测试结果

所有 MAE/RMSE 都是在 `[0,1]` 或 SOH 比率尺度上计算。例如 `0.0243` 表示约 2.43 个百分点。

### 13.1 总体结果

| 指标 | exp016 LFP | exp017 NCM |
|---|---:|---:|
| SOC_end MAE | **1.675%** | **4.559%** |
| SOC_end RMSE | 2.536% | 6.317% |
| SOC_start MAE | 1.667% | 4.420% |
| SOH MAE | **2.431%** | **5.052%** |
| SOH RMSE | 3.660% | 7.106% |
| 有效 SOH 测试窗口 | 34,534 | 9,871 |
| 学得 `σ_soc` | 0.0523 | 0.0736 |
| 学得 `σ_soh` | 0.0627 | 0.1012 |

### 13.2 与常数 SOH 基线比较

常数基线始终输出训练车辆 SOH 中位数：

| 体系 | 模型 SOH MAE | 常数基线 MAE | 结论 |
|---|---:|---:|---|
| LFP | **2.43%** | 9.44% | 明显优于常数基线 |
| NCM | **5.05%** | 8.29% | 有提升，但空间仍大 |

### 13.3 按充放电模式

| 体系 | 充电 SOH MAE | 放电 SOH MAE |
|---|---:|---:|
| LFP | 1.80% | 4.24% |
| NCM | 3.21% | 6.45% |

当前网络在放电窗口上的 SOH 明显更弱。部署时宜将窗口预测与历史 SOH 状态平滑结合，并在后续数据适配中重点改善放电表征。

### 13.4 按循环前缀信息量

| 体系 | `partial ΔSOC < 10%` | `10%–20%` | `>=20%` |
|---|---:|---:|---:|
| LFP SOH MAE | 4.48% | 2.66% | **1.22%** |
| NCM SOH MAE | 6.29% | 4.91% | **3.15%** |

误差随已观测循环片段增长而下降，符合容量估计的信息规律：循环刚开始时累计 Ah 和 ΔSOC 太少，SOH 可辨识性不足。

---

## 14. 推理语义与部署要求

### 14.1 单次推理时机

网络应在获得完整的 128 点窗口后运行。每向前滑动一个部署步长，就产生：

```text
SOC_start(t-W+1)
SOC_end(t)
SOH(t)
```

其中真正对应当前时刻的是 `SOC_end(t)` 和 `SOH(t)`。`SOC_start` 是利用整个可见窗口回归窗口起点状态，并不是在窗口起点当时在线产生的预测。

### 14.2 推理所需状态

除 128 点环形缓冲区外，还要维护从当前充/放电 CSV 或循环起点至窗口末端的：

- 累计充/放电 Ah；
- 起始 BMS SOC；
- 已历时间；
- 运行平均绝对电流。

若循环边界识别错误，`cycle_feat` 会发生错误累积，影响 SOH。

### 14.3 推荐的 SOH 输出平滑

SOH 是慢变量，不应在部署层跟随每个窗口大幅跳变。可采用：

- 指数滑动平均；
- 仅在累计 ΔSOC 达到阈值后提高当前窗口权重；
- 按模型不确定性或前缀长度自适应融合；
- 对异常窗口做限幅和变化率约束。

上述是部署后处理建议，当前训练代码尚未实现。

### 14.4 化学体系选择

- LFP 数据使用 exp016 配置与权重；
- NCM 数据使用 exp017 配置与权重；
- 不要只按文件路径选择权重，应由可靠的电池体系配置决定；
- 当前没有训练 `chemistry=BOTH` 的统一最佳 checkpoint。

---

## 15. 当前网络的优势

1. **逐窗口 SOH**：满足每个可用窗口输出健康状态的需求。
2. **时间因果输入**：不读取窗口终点后的样本或完整循环最终统计量。
3. **多时间尺度**：浅层 TCN 表征 SOC 快变量，注意力深层出口表征 SOH 慢变量。
4. **显式任务耦合**：Star Operation 让 SOC/SOH 特征相互作用。
5. **物理引导**：容量变化与库仑积分一致；NCM 还加入 OCV/Thevenin 约束。
6. **任务自动加权**：学习两个不确定性参数，降低固定权重调参负担。
7. **分阶段训练**：先建立 SOC 表征，再训练 SOH，最后联合微调。
8. **车辆级隔离**：数据集按车辆拆分，避免同车重叠窗口跨集合。
9. **轻量**：仅 0.264 M 参数，适合后续量化和边缘部署研究。
10. **完整实验追踪**：配置、元数据、历史、指标、权重、图表和代码快照均已保存。

---

## 16. 当前网络的局限与风险

### 16.1 SOH 并非纯 V/I/T

`partial_dsoc` 来自 BMS SOC。虽然 `use_soc_bms_ctx=False`，SOH 分支仍间接使用 BMS SOC。发表或汇报时必须称为“BMS SOC 辅助的因果 SOH 估计”。

### 16.2 SOH 标签是车辆级常数

同一车辆的所有窗口共享一个稳健中位 SOH，网络尚不能学习车辆内部逐循环退化轨迹。Wenzhou 数据应解决这一问题。

### 16.3 checkpoint 按 SOC 选择

最优权重以验证 SOC MAE 为唯一依据，可能错过 SOH 更优的轮次。

### 16.4 放电 SOH 较弱

两种化学体系的放电 SOH 均明显差于充电。当前物理损失也只用于充电。

### 16.5 TCN 使用 BatchNorm

训练与推理批分布变化、很小 batch 或域迁移时，BatchNorm 统计可能成为不稳定来源。Wenzhou 小对象迁移可评估 LayerNorm/GroupNorm。

### 16.6 注意力不是严格逐点 causal mask

深层注意力会看完整当前窗口。它对“窗口末端输出”是因果的，但不能直接宣称窗口内每个位置都能实时因果输出。

### 16.7 物理模型绑定旧数据约定

库仑损失默认负电流充电，OCV 损失默认已知串数，Thevenin 步长硬编码为 10 s。这些约定对 Wenzhou 均不能原样使用。

### 16.8 缺少真正的不确定性输出

`σ_soc` 和 `σ_soh` 是全局可学习的任务损失尺度，不是每个样本的预测置信区间。早期设计文档中的逐样本置信区间尚未实现。

### 16.9 未完成部署验证

当前没有经过 ONNX 导出、TensorRT、量化、剪枝、端侧延迟和数值一致性验证。

---

## 17. 与早期 DNN-MTL V1.3 的关系

早期文档描述的是“8 条并联支路同时输入、支路嵌入、跨支路注意力、每 100 ms 输出”的设想。当前实际网络已经做了以下适配：

| 早期 V1.3 | 当前 exp016/017 |
|---|---|
| 8 支路并行输入 | 单车辆/单电池包窗口 |
| 跨支路自注意力 | 窗口内时间自注意力 |
| 单点 SOC/SOH | SOC_start + SOC_end + 每窗口 SOH |
| 6 通道 | 7 通道，增加电压极差 |
| SOH 锚点上下文 | 已移除 SOH 标签锚点 |
| 10 Hz、W=128 约 12.8 s | 0.1 Hz、W=128 约 21.3 min |
| 约 1.22 M 参数 | 0.264 M 参数 |
| 每样本置信区间设想 | 仅全局任务不确定性参数 |

因此早期 `DNN-MTLV1.3/DNN-MTL.md` 和 `DNN-MTL-method.md` 只能用于理解设计来源，不能作为当前代码的接口说明。本文件和 exp016/017 快照才是现状依据。

---

## 18. 与 Wenzhou 数据集的关系

Wenzhou 数据集的完整说明位于：

```text
H:\中汽研-华为\wenzhou\WENZHOU_DATASET_HANDOFF.md
```

当前网络不能直接读取 Wenzhou XLSX，原因包括：

- Wenzhou 没有温度；
- 没有最大/最小单体电压和电压极差；
- 只有单路端电压、电流；
- 电流单位为 mA；
- 正电流为充电、负电流为放电，与当前项目相反；
- Cell 约 1 s、Pack 约 0.5 s，采样周期不同；
- Pack 串并联拓扑无法从字段确定；
- SOC 需根据 Detail 容量和循环最终放电容量构造；
- SOH 可以使用逐循环放电容量衰减率，而不是车辆级常数。

建议保留以下网络核心：

- TCN；
- 时间注意力；
- 浅层 SOC/深层 SOH 双出口；
- Star Operation；
- SOC/SOH 多任务头；
- 三阶段训练框架。

必须修改：

- `WenzhouDataset` 和输入通道；
- 电流单位及符号；
- 窗口和重采样配置；
- SOC/SOH 标签；
- 放电物理损失；
- OCV/串数处理；
- 对象级 Cell→Pack 迁移划分。

在 Wenzhou 实验完成前，不能把 exp016/017 指标当作 Wenzhou 上的预期性能。

---

## 19. 推荐的下一步改进优先级

### P0：Wenzhou 数据适配与真实逐循环 SOH

- 新增只读 Wenzhou 索引和 Dataset；
- 以 `(object_id, segment_id, cycle_number)` 为循环主键；
- 优先使用恒流放电段；
- 每个窗口继承所在循环的 SOH；
- 按对象留一 Pack 测试；
- 保留 exp016/017 和当前代码快照。

### P1：消除 BMS SOC 辅助依赖

至少做两组严格消融：

```text
A. 当前版本：partial Ah + partial ΔSOC + elapsed + mean|I|
B. 纯因果 V/I：partial Ah 或仅 V/I 导出量，不使用 BMS SOC
```

若保留 partial Ah，应称为库仑辅助；若连 partial Ah 也移除，才更接近纯端到端 V/I SOH。

### P1：改进 checkpoint 选择

同时保存：

- `best_soc_model.pt`；
- `best_soh_model.pt`；
- `best_joint_model.pt`。

联合评分应事先固定，避免看测试集后调权。

### P1：放电物理约束

实现 mode-aware 公式：

```text
charge:    (SOC_end - SOC_start) × SOH × C_nom ≈ Q_charge
discharge: (SOC_start - SOC_end) × SOH × C_nom ≈ Q_discharge
```

### P2：SOH 时序稳定性

- 同一对象内按循环排序训练；
- 引入合理的退化平滑/单调先验，但允许容量恢复和噪声；
- 输出窗口置信度或异方差，而不是只有全局 `σ`。

### P2：部署工程

- 推理脚本和状态管理；
- ONNX 导出；
- FP16/INT8 精度对比；
- 端侧延迟、内存和漂移检测；
- 循环边界识别和异常输入处理。

---

## 20. 复现实验的环境与命令口径

指定 Python：

```text
D:\enviroment\anaconda\envs\py312\python.exe
```

已核实：

```text
Python 3.12.13
PyTorch 2.5.1+cu121
CUDA available: True
GPU: NVIDIA GeForce RTX 3070 Laptop GPU
```

训练前必须检查：

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

历史实验已使用 `device="cuda"`。新实验应沿用项目的编号、结果目录和代码快照模式，保存：

```text
config.json
data_meta.json
train_history.csv
metrics.json
best_model.pt
loss_curves.png
soc_scatter.png
README.md
code/
```

---

## 21. 后续智能体阅读顺序

1. 本文件：理解当前推荐网络全貌；
2. `dnn_mtl/model.py`：核对实际前向结构；
3. `dnn_mtl/data.py`：理解标签、因果前缀和数据边界；
4. `dnn_mtl/losses.py`：理解物理假设；
5. `dnn_mtl/train.py`：理解冻结策略和 checkpoint 选择；
6. `experiments/exp016_lfp_causal_window_soh/README.md`：LFP 结果；
7. `experiments/exp017_ncm_causal_window_soh/README.md`：NCM 结果；
8. `wenzhou/WENZHOU_DATASET_HANDOFF.md`：新数据集适配要求。

若代码与本文未来发生冲突，应以某个已固化实验目录中的 `config.json + code/ + best_model.pt` 三者组合为可复现依据，并及时同步更新本文和版本说明。

---

## 22. 最简摘要

当前推荐网络是一个 0.264 M 参数的因果窗口 DNN-MTL：它用 128×7 的 V/I/T 窗口、5 维当前上下文和 4 维循环因果前缀，通过残差因果 TCN、时间自注意力、SOC/SOH 双分支与 Star Operation，一次输出窗口起点 SOC、终点 SOC和一个 SOH。训练采用 SOC→SOH→联合三阶段，以及库仑物理约束；NCM 额外使用 Thevenin/OCV 约束。当前 LFP/NCM SOH MAE 分别为 2.43%/5.05%，均优于常数基线。它已经避免读取窗口后的输入，但 SOH 前缀仍使用 BMS SOC，标签也由离线完整数据构造，因此准确定位应是“BMS SOC 辅助的因果逐窗口 SOH 网络”。
