# LFP / NCM 车辆电池数据集说明与使用交接

> 本文件只描述当前用于训练 DNN-MTL 的原有 LFP/NCM 车辆 CSV 数据集，不包含 Wenzhou。

> 本文以当前实际加载代码 `dnn_mtl/data.py`、exp016/exp017 的配置、元数据和实验结果为准。

## 1. 快速结论

当前数据集由两种化学体系的车辆运行数据组成：

```text
LFP：磷酸铁锂车辆电池数据
NCM：三元锂车辆电池数据
```

数据以“化学体系 → 车辆类别 → VIN 车辆 → 充/放电 CSV”的方式组织。每个 CSV 是一段充电或放电过程，当前代码按 10 s 采样使用。

当前网络实际依赖以下原始字段：

```text
TIME
SUM_VOLTAGE
MAX_CELL_VOLT
MIN_CELL_VOLT
SUM_CURRENT
MAX_TEMP
MIN_TEMP
SOC
CHARGE_STATUS
```

数据包含总电压、最高/最低单体电压、总电流、最高/最低温度、BMS SOC、充放电状态和时间。网络将其构造成 7 通道窗口：

```text
[平均每串电压, 最高单体电压, 最低单体电压,
 总电流, 最高温度, 最低温度, 单体电压极差]
```

每个窗口输出 `SOC_start`、`SOC_end` 和一个 `SOH`。

## 2. 数据目录结构

```text
H:\中汽研-华为\
├── LFP\
│   ├── normal\vin_xxx\*.csv
│   ├── low_capacity\vin_xxx\*.csv
│   └── high_resistance\vin_xxx\*.csv
├── NCM\
│   ├── normal\vin_xxx\*.csv
│   ├── low_capacity\vin_xxx\*.csv
│   └── high_resistance\vin_xxx\*.csv
├── dnn_mtl\
└── experiments\
```

```text
chemistry ∈ {LFP, NCM}
category  ∈ {normal, low_capacity, high_resistance}
```

每个 `vin_xxx` 是一个独立车辆对象，也是训练、验证、测试的最小划分单位。同一 VIN 的任何 CSV 和窗口都不能跨数据集合。

典型文件名：

```text
vin_101_charge_1.csv
vin_101_discharge_1.csv
```

判断充电文件时必须先排除 `discharge`，因为该单词本身包含 `charge`：

```python
def is_charge_file(fname):
    return "discharge" not in fname and "charge" in fname
```

当前 Dataset 把每个 CSV 当作一段独立充电或放电过程，窗口不会跨 CSV。

## 3. 当前加载器要求的原始字段

下表是 `dnn_mtl/data.py` 实际读取的必需字段。原始 CSV 即使还有其他列，也不是当前网络的必需输入。

| 字段 | 当前代码中的解释 | 当前使用方式 |
|---|---|---|
| `TIME` | 时间，按秒差计算 | 采样间隔、Ah、已历时间、SOC 平滑 |
| `SUM_VOLTAGE` | 电池包总电压，按 V 使用 | 除以串数得到平均每串电压 |
| `MAX_CELL_VOLT` | 最高单体电压，按 V 使用 | 时序输入 |
| `MIN_CELL_VOLT` | 最低单体电压，按 V 使用 | 时序输入和电压极差 |
| `SUM_CURRENT` | 电池包总电流，按 A 使用 | 时序输入、安时积分、物理损失 |
| `MAX_TEMP` | 最高温度 | 时序输入和上下文 |
| `MIN_TEMP` | 最低温度 | 时序输入和上下文 |
| `SOC` | BMS SOC，按 0–100 解释 | SOC 标签、容量和因果 SOH 前缀 |
| `CHARGE_STATUS` | 充放电状态码 | 充放电筛选和 OCV 拟合 |

### 3.1 原始文件还包含逐单体电压

对 LFP、NCM 的 `vin_101_charge_1.csv` 实际抽查确认，典型 CSV 还包含完整的逐单体电压列：

```text
LFP：VOLT_1 ... VOLT_124
NCM：VOLT_1 ... VOLT_96
```

因此典型原始表结构为：

```text
9 个公共字段
+ LFP 124 个逐单体电压字段，共 133 列
或
+ NCM 96 个逐单体电压字段，共 105 列
```

这些逐单体列是数据集中真实存在的附加信息，可用于单体一致性、异常单体、排序统计或更丰富的电压表征研究。但当前 exp016/017 **没有直接把 `VOLT_1...VOLT_N` 输入网络**，只使用：

```text
SUM_VOLTAGE
MAX_CELL_VOLT
MIN_CELL_VOLT
MAX_CELL_VOLT - MIN_CELL_VOLT
```

后续若启用完整单体电压向量，应创建新实验并重新设计输入维度、缺失单体处理、归一化和跨单体建模，不能把它视为当前 7 通道模型已经使用的特征。

当前实现的单位口径：

```text
TIME          : s
SUM_VOLTAGE   : V
MAX_CELL_VOLT : V
MIN_CELL_VOLT : V
SUM_CURRENT   : A
MAX_TEMP      : 通常按 °C 使用
MIN_TEMP      : 通常按 °C 使用
SOC           : %，即 0–100
```

代码按 `Ah = Σ(I × Δt) / 3600` 积分，没有将电流除以 1000，因此 `SUM_CURRENT` 必须以 A 进入当前流程。更换数据来源时需重新核实单位。

## 4. 电流方向与状态码

```text
SUM_CURRENT < 0：充电
SUM_CURRENT > 0：放电
CHARGE_STATUS == 1：充电
CHARGE_STATUS == 3：放电
```

```text
充电 Ah = Σ max(-I, 0) × Δt / 3600
放电 Ah = Σ max( I, 0) × Δt / 3600
```

文件模式主要由文件名确定，状态码还用于筛选充电点、放电点和拟合 OCV。若文件名与内部状态码不一致，应视为数据质量问题。

## 5. LFP 与 NCM 的差异

| 项目 | LFP | NCM |
|---|---:|---:|
| 化学体系 | 磷酸铁锂 | 三元锂 |
| 当前代码假设串数 | 124 | 96 |
| 总电压折算 | `SUM_VOLTAGE / 124` | `SUM_VOLTAGE / 96` |
| 每串合理电压范围（清洗参考） | 2.5–3.7 V | 2.8–4.3 V |
| 当前最佳实验 | exp016 | exp017 |
| OCV/Thevenin 一致性 | 关闭 | 开启 |

两种体系不能混用模型权重、`C_nom`、特征均值和标准差、OCV 曲线或串数配置。它们可以使用同一网络代码，但必须加载各自实验配置。

## 6. 网络实际使用的 7 通道特征

```python
FEAT_NAMES = [
    "v_sum_cell",
    "v_max",
    "v_min",
    "current",
    "t_max",
    "t_min",
    "v_spread",
]
```

```python
v_sum_cell = SUM_VOLTAGE / n_cells
v_max      = MAX_CELL_VOLT
v_min      = MIN_CELL_VOLT
current    = SUM_CURRENT
t_max      = MAX_TEMP
t_min      = MIN_TEMP
v_spread   = MAX_CELL_VOLT - MIN_CELL_VOLT
```

```text
完整 CSV：features [T,7]
单个窗口：x [128,7]
合批以后：x [B,128,7]
```

7 个通道使用训练集 Z-score：

```text
x_norm = (x - feature_mean_train) / feature_std_train
```

验证集和测试集必须复用训练集统计。exp016/exp017 的具体统计保存在各自的 `data_meta.json`；重新划分车辆后必须重新计算。

## 7. 窗口构造

```text
采样间隔：10 s
window_len：128
stride：16
窗口覆盖时间：约 21.3 min
相邻窗口起点间隔：约 2.67 min
```

每个 CSV 内从索引 0 开始滑窗。长度不足 128 的 CSV 被跳过。窗口不跨 CSV，且大量窗口相互重叠。因此真正的独立泛化单位是 VIN，不是窗口数量。

## 8. 每个窗口的 5 维上下文

| 索引 | 特征 | 当前定义 |
|---:|---|---|
| 0 | `mode` | 充电=1，放电=0 |
| 1 | `t_mean_z` | 窗口末点最高/最低温度均值，按训练温度统计归一化 |
| 2 | `elapsed_norm` | CSV 起点至窗口末点的时间 / 4 h，最大截断到 1 |
| 3 | `window_current` | 当前窗口归一化电流通道的均值 |
| 4 | `voltage_spread` | 窗口末点归一化电压极差 |

```text
单样本 ctx：[5]
合批 ctx：[B,5]
```

当前配置 `use_soc_bms_ctx=False`，所以 BMS SOC 不直接作为这 5 维上下文输入。

## 9. SOH 分支的 4 维因果前缀

```text
causal_window_soh = 1
cycle_feat = [
  partial_ah / C_nom,
  partial_dsoc,
  elapsed_h,
  mean_abs_current / 100,
]
```

| 特征 | 说明 |
|---|---|
| `partial_ah / C_nom` | CSV 起点到窗口末端累计的充入/放出容量比例 |
| `partial_dsoc` | CSV 起始 BMS SOC 与窗口末端 BMS SOC 的绝对差 |
| `elapsed_h` | CSV 起点到窗口末端的小时数 |
| `mean_abs_current / 100` | 截至窗口末端的平均绝对电流缩放值 |

这些特征不读取窗口末端之后的数据，因此在时间上是因果的。但 `partial_dsoc` 使用截至当前时刻的 BMS SOC，所以当前 SOH 网络应称为“BMS SOC 辅助的因果逐窗口 SOH 估计”，不是完全纯粹的 V/I/T SOH。

## 10. SOC 标签构造

原始 `SOC` 是 0–100 的 BMS SOC，先除以 100。当前 `smooth_soc_label=True`，采用“安时积分平滑 + CSV 首尾 BMS SOC 锚点漂移修正”：

1. 以 CSV 第一个 BMS SOC 为起点；
2. 对电流积分得到连续 SOC；
3. 计算积分末值与最后一个 BMS SOC 的偏差；
4. 将偏差按时间线性分摊；
5. 裁剪到 `[0,1]`。

```text
ΔAh(t) = cumulative_sum[-I(k) × Δt(k)] / 3600
SOC_cc(t) = SOC_bms(0) + ΔAh(t) / C_nom
drift = SOC_bms(T) - SOC_cc(T)
SOC_label(t) = clip(SOC_cc(t) + drift × (t-t0)/(T-t0), 0, 1)
```

窗口标签：

```text
soc_start = SOC_label[window_start]
soc_end   = SOC_label[window_end]
```

标签生成使用完整 CSV 的末端 BMS SOC 做离线漂移修正，但末端信息不进入窗口输入。因此模型输入是因果的，SOC 监督标签属于离线校正标签。

## 11. 标称容量 C_nom

`C_nom` 不是写死的厂家值，而是从训练集正常车辆的有效充电 CSV 估计。

```text
Q_cycle = 充电 Ah / ΔSOC_bms
C_nom = median(Q_cycle of valid normal training files)
```

有效条件：

```text
充电状态点数 >= 2
ΔSOC_bms >= 0.20
```

exp016/017 得到：

```text
LFP C_nom = 152.93961329899713 Ah
NCM C_nom = 130.05982905982904 Ah
```

不能用验证集、测试集或低容量车辆计算 `C_nom`。

## 12. SOH 标签构造

先对有效充电 CSV 估算：

```text
Q_estimated = ∫|I|dt / ΔSOC_bms
SOH = Q_estimated / C_nom
```

过滤规则：

```text
ΔSOC_bms >= 0.20
0.7 × C_nom <= Q_estimated <= 1.3 × C_nom
```

exp016/017 设置 `soh_per_cycle=0`，每辆车构造一个稳健标签：

```text
SOH_vehicle = median(valid Q_estimated of vehicle) / C_nom
```

同一车辆的充电和放电窗口共享该标签。没有有效容量循环时，标签为 NaN，并通过 mask 排除，不能用 1.0 假标签代替。

该标签表示车辆整体容量健康水平，不能表达同一车辆内部逐循环退化轨迹；它也可能综合该车辆多个充电文件。当前方案适合车辆健康分层，不等同于逐循环在线 SOH 真值。

## 13. 每个 Dataset 样本的完整接口

`BatteryDataset.__getitem__()` 返回：

```python
{
    "x":             Tensor[W, 7],
    "ctx":           Tensor[5],
    "soc_start":     scalar Tensor,
    "soc_end":       scalar Tensor,
    "soh":           scalar Tensor,
    "soh_valid":     scalar Tensor,
    "soh_weight":    scalar Tensor,
    "physics_valid": scalar Tensor,
    "coulomb_ah":    scalar Tensor,
    "c_nom":         scalar Tensor,
    "n_cells":       scalar Tensor,
    "v_sum_raw":     scalar Tensor,
    "i_raw":         scalar Tensor,
    "i_seq":         Tensor[W],
    "cycle_feat":    Tensor[4],
}
```

| 键 | 用途 |
|---|---|
| `x` | TCN 的 7 通道时序输入 |
| `ctx` | 上下文编码器输入 |
| `soc_start`, `soc_end` | SOC 双输出监督 |
| `soh` | 每窗口 SOH 监督 |
| `soh_valid` | 是否存在有效 SOH 标签 |
| `soh_weight` | 平衡不同 CSV 的窗口数量 |
| `physics_valid` | 是否启用当前窗口的充电物理损失 |
| `coulomb_ah` | 当前窗口实测充电 Ah |
| `c_nom` | 当前化学体系标称容量 |
| `n_cells` | 串数，用于总电压折算 |
| `v_sum_raw` | 窗口末端总电压 |
| `i_raw` | 窗口末端原始电流 |
| `i_seq` | 当前窗口原始电流序列，用于 RC 极化 |
| `cycle_feat` | 截至窗口末端的 4 维因果前缀 |

DataLoader 合批后：

```text
x          [B,128,7]
ctx        [B,5]
cycle_feat [B,4]
i_seq      [B,128]
标量字段    [B]
```

## 14. SOH 窗口权重

长 CSV 会产生更多重叠窗口。若窗口等权，长循环会在 SOH 损失中占据过大权重。当前代码设置：

```text
soh_weight = soh_valid / windows_per_csv
```

所以同一个 CSV 的所有窗口权重之和约为 1。它平衡的是 CSV/循环，不是车辆；若车辆间 CSV 数量差异很大，车辆层面仍可能不完全均衡。

## 15. 训练、验证、测试划分

划分函数：

```text
dnn_mtl/data.py::make_splits
```

默认比例：

```text
train      = 75%
validation = 12.5%
test       = 12.5%
```

执行顺序：

1. 枚举所选体系的 `normal` VIN；
2. 超过 `n_vehicles` 时先按 VIN 数字顺序截取；
3. 根据配置追加 `low_capacity` 或 `high_resistance` VIN；
4. 用固定随机种子打乱车辆列表；
5. 按车辆数量划分 train/val/test；
6. 在各车辆集合内部加载 CSV 和切窗。

exp016/017 使用：

```text
n_vehicles = 200
include_low_capacity = 1
include_high_resistance = 0
seed = 42
```

两轮最终均为：

```text
train = 172 VIN
val   = 29 VIN
test  = 29 VIN
total = 230 VIN
```

必须禁止：

- 先生成所有窗口再随机拆分；
- 把同一 VIN 的不同 CSV 分到不同集合；
- 用全数据计算归一化参数；
- 用验证或测试车辆估计 `C_nom`；
- 用测试数据拟合 OCV。

## 16. exp016 LFP 数据统计

实验目录：

```text
H:\中汽研-华为\experiments\exp016_lfp_causal_window_soh
```

| 集合 | VIN 数 | 窗口数 |
|---|---:|---:|
| train | 172 | 216,597 |
| validation | 29 | 37,723 |
| test | 29 | 37,669 |

其他关键数据：

```text
有效 SOH 测试窗口：34,534
C_nom：152.93961329899713 Ah
SOC_end MAE：1.675%
SOH MAE：2.431%
常数 SOH 基线 MAE：9.443%
```

## 17. exp017 NCM 数据统计

实验目录：

```text
H:\中汽研-华为\experiments\exp017_ncm_causal_window_soh
```

| 集合 | VIN 数 | 窗口数 |
|---|---:|---:|
| train | 172 | 61,472 |
| validation | 29 | 12,165 |
| test | 29 | 9,871 |

其他关键数据：

```text
有效 SOH 测试窗口：9,871
C_nom：130.05982905982904 Ah
SOC_end MAE：4.559%
SOH MAE：5.052%
常数 SOH 基线 MAE：8.285%
```

NCM 窗口数明显少于 LFP，这是当前 NCM 泛化性能较弱的重要背景。

## 18. NCM OCV 数据生成

当 `lambda_consistency > 0` 时，只使用训练车辆构造 128 点 OCV(SOC) 曲线：

1. 将 SOC `[0,1]` 分为 128 个箱；
2. 按 `CHARGE_STATUS` 分开收集充电和放电每串电压；
3. 同一 SOC 箱分别取充电、放电电压中位数；
4. 两侧都有数据时取半和，近似抵消充放电极化；
5. 缺失箱线性插值；
6. 使用中值滤波和滑动平均；
7. 通过保序回归强制 OCV 随 SOC 单调不降。

曲线保存在：

```text
experiments/exp017_ncm_causal_window_soh/data_meta.json
```

## 19. 数据质量检查清单

正式训练前，每个 CSV 至少检查：

- [ ] 文件可读取且不是空表；
- [ ] 9 个必需字段完整；
- [ ] 必需字段可转换为数值；
- [ ] `TIME` 无大规模重复、倒退或异常间隔；
- [ ] 中位采样周期接近当前 10 s 假设；
- [ ] 电压值与化学体系合理范围一致；
- [ ] `MAX_CELL_VOLT >= MIN_CELL_VOLT`；
- [ ] 电压极差非负；
- [ ] 温度范围合理；
- [ ] SOC 位于 0–100；
- [ ] 充电文件电流主要为负；
- [ ] 放电文件电流主要为正；
- [ ] `CHARGE_STATUS` 与文件名模式基本一致；
- [ ] 容量标签所用 `ΔSOC >= 0.20`；
- [ ] 容量通过 `[0.7,1.3] × C_nom` 过滤；
- [ ] train/val/test 不存在重复 VIN；
- [ ] 归一化、`C_nom`、OCV 只来自训练集；
- [ ] 窗口没有跨 CSV；
- [ ] 无效 SOH 使用 mask，不使用 1.0 假标签。

## 20. 最小 CSV 检查示例

以下代码只读检查单个 CSV：

```python
from pathlib import Path
import pandas as pd

required = {
    "TIME",
    "SUM_VOLTAGE",
    "MAX_CELL_VOLT",
    "MIN_CELL_VOLT",
    "SUM_CURRENT",
    "MAX_TEMP",
    "MIN_TEMP",
    "SOC",
    "CHARGE_STATUS",
}

csv_path = Path(r"H:\中汽研-华为\LFP\normal\vin_xxx\example.csv")
df = pd.read_csv(csv_path)

missing = required - set(df.columns)
if missing:
    raise ValueError(f"缺少字段: {sorted(missing)}")

dt = pd.to_numeric(df["TIME"], errors="coerce").diff().dropna()
print("rows:", len(df))
print("median_dt_s:", dt.median())
print("soc_range:", df["SOC"].min(), df["SOC"].max())
print("current_range_A:", df["SUM_CURRENT"].min(), df["SUM_CURRENT"].max())
print("status:", sorted(df["CHARGE_STATUS"].dropna().unique().tolist()))
```

示例路径只是模板，实际文件应从目录枚举。

## 21. 常见错误及后果

### 21.1 随机拆分窗口

同一车辆、同一循环的重叠窗口进入训练和测试，指标会严重虚高。

### 21.2 把 discharge 识别为 charge

模式、电流积分方向和物理损失会全部错误。

### 21.3 忽略电流符号

会造成充电 Ah 为 0 或 SOC 变化方向相反。

### 21.4 把原始 SOC 当成 0–1

原始 SOC 按 0–100 使用，进入模型标签前需除以 100。

### 21.5 用总电压直接对比单体 OCV

`SUM_VOLTAGE` 必须先除以串数：LFP=124，NCM=96。

### 21.6 用全体车辆估计 C_nom

会引入测试信息，而且低容量车辆会拉低标称容量。

### 21.7 声称当前 SOH 是纯 V/I/T

`cycle_feat.partial_dsoc` 使用 BMS SOC，这种说法不准确。

### 21.8 把窗口数当成独立样本数

窗口高度重叠，独立泛化单位是 VIN。

### 21.9 混用 LFP/NCM 权重和归一化

两种体系的串数、容量、电压分布和 OCV 不同。

### 21.10 覆盖 exp016/017

这两轮是当前基线。新工作必须创建新实验编号并保存新代码快照。

## 22. 能支持和不能直接支持的任务

### 当前可以支持

- 基于窗口的 SOC 起点/终点估计；
- 每窗口车辆级 SOH 估计；
- LFP 与 NCM 分体系训练；
- 正常与低容量车辆健康分层；
- 充电/放电模式对比；
- 电压、电流、温度特征消融；
- NCM OCV/Thevenin 一致性研究；
- 按 VIN 的独立泛化评估。

### 当前不能直接严谨支持

- 同一车辆内部真正的逐循环 SOH 退化曲线；
- 不使用 BMS SOC 的纯 V/I/T SOH 结论；
- 单体级独立 SOC/SOH 输出；
- 8 条真实并联支路交叉注意力验证；
- 已验证的逐样本置信区间；
- Wenzhou Cell→Pack 迁移结论；
- 不同采样率直接混合而不重采样；
- 未知串数电池包直接部署。

## 23. 直接使用流程

1. 阅读本文、`CURRENT_BEST_NETWORK.md`、`dnn_mtl/data.py` 和 `dnn_mtl/config.py`；
2. 使用指定 Python 环境；
3. 抽查 CSV 字段、单位、符号和时间间隔；
4. 通过 `make_splits()` 按 VIN 划分；
5. 只用训练集计算归一化、`C_nom` 和 OCV；
6. 创建 Dataset 后检查一个 batch 的形状和有限值；
7. 在 CUDA 上完成前向与反向冒烟测试；
8. 新实验使用新编号，不覆盖 exp016/017。

指定环境：

```text
D:\enviroment\anaconda\envs\py312\python.exe
Python 3.12.13
PyTorch 2.5.1+cu121
GPU: NVIDIA GeForce RTX 3070 Laptop GPU
```

## 24. 相关权威文件

```text
当前网络总说明：
H:\中汽研-华为\CURRENT_BEST_NETWORK.md

数据加载实现：
H:\中汽研-华为\dnn_mtl\data.py

配置定义：
H:\中汽研-华为\dnn_mtl\config.py

LFP 当前基线：
H:\中汽研-华为\experiments\exp016_lfp_causal_window_soh\

NCM 当前基线：
H:\中汽研-华为\experiments\exp017_ncm_causal_window_soh\
```

实验目录中的以下文件共同构成可复现依据：

```text
README.md
config.json
data_meta.json
metrics.json
causal_analysis.json
best_model.pt
code/
```

## 25. 给下一位智能体的最短接手提示

```text
本项目当前使用 LFP/NCM 车辆 CSV 数据集。数据按 chemistry/category/vin/charge-or-discharge.csv 组织；采样周期 10 s；负电流充电、正电流放电；SOC 为 0–100 的 BMS SOC。当前网络输入为 128×7：[总电压/串数、最高单体电压、最低单体电压、总电流、最高温度、最低温度、电压极差]，另有 5 维因果上下文和 4 维循环前缀。SOC 标签由电流积分和平滑 BMS SOC 构造；SOH 是以训练集正常车容量中位数定义 C_nom 后，每辆车有效充电容量中位数/C_nom。数据必须按 VIN 划分，归一化、C_nom、OCV 只能由训练集拟合。当前 LFP/NCM 基线分别是 exp016/exp017，新工作不得覆盖它们。
```

## 26. 文档维护规则

本文描述的是 exp016/exp017 已实际使用的数据接口和标签口径，不是早期 DNN-MTL V1.3 的设想，也不是 Wenzhou 数据格式。

如果后续修改以下任一项，应同步更新本文并创建新实验编号：

- 原始必需字段；
- 电流符号或单位；
- 串数；
- 采样周期；
- 窗口长度或步长；
- SOC 平滑方法；
- `C_nom` 定义；
- SOH 标签口径；
- VIN 划分；
- 输入特征顺序；
- Dataset 返回接口。
