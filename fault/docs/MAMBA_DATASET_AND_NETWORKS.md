# 数据集与网络结构简表

## 1. 数据集基本信息
仓库：

https://github.com/CH-BatteryGen/dataset-warehouse
原文：
[Battery Fault: A Comprehensive Dataset and Benchmark for Battery Fault Diagnosis](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0d781fa5f639bf2caf728a68e9678362-Abstract-Conference.html)

![[Pasted image 20260922100745.png]]

### 数据规模

| 项目 | 内容 |
|---|---|
| 数据目录 | 工程根目录下的 `dataset`（即 `fault/../dataset`） |
| 化学体系 | LFP、NCM |
| 车辆数 | 1,000 辆 |
| 原始 CSV | 19,478 个 |
| 有效序列 | 19,471 条 |
| 有效时间点 | 15,569,262 个 |
| 缓存窗口 | 56,083 个 |
| 采样间隔 | 10 s |
| 数据划分 | 按 VIN 划分，训练/验证/测试 = 70%/15%/15% |

四个类别：

```text
0：normal（正常）
1：low_capacity（低容量）
2：high_resistance（高内阻）
3：self_discharge（自放电）
```

### 网络输入

每个样本为一个连续序列窗口：

```text
输入尺寸：[B, 512, 7]
窗口长度：512 个时间点，约 85.3 min
滑动步长：256 个时间点
短序列：右侧补零，并使用 mask 屏蔽
```

7个输入通道：

```text
1. 平均单体电压
2. 最高单体电压
3. 最低单体电压
4. 电池包电流
5. 最高温度
6. 最低温度
7. 单体电压极差
```

各化学体系分别使用训练车辆计算均值和标准差；验证集、测试集不参与归一化统计。

---

## 2. 当前故障分类 Mamba 网络

当前推荐版本是 **MultiScale-Hierarchical Mamba**：

```text
输入 [B,512,7]
    ↓
多尺度 Conv Stem
  ├─ Conv1D，kernel=3
  ├─ Conv1D，kernel=7
  └─ Conv1D，kernel=15
  拼接 → 1×1 Conv → 门控残差融合 → 32维
    ↓ [B,512,32]
Mamba Block × 3
    ↓ [B,512,32]
Mean Pooling      → [B,32]
Max Pooling       → [B,32]
Std Pooling       → [B,32]
Last Pooling      → [B,32]
Attention Pooling → [B,32]
    ↓ 拼接
特征 [B,160]
    ↓
共享层：160 → 64
    ├─ 检测头：正常 / 故障
    └─ 类型头：低容量 / 高内阻 / 自放电
              ↓
       组合为四分类概率 [B,4]
```

每个 Mamba Block：

```text
LayerNorm
  ↓
Selective SSM
  ├─ 输入投影与门控
  ├─ Depthwise Conv1D，kernel=5
  ├─ 输入相关的 Δ、B、C
  └─ 状态空间扫描，d_state=8
  ↓
残差连接
  ↓
LayerNorm → FFN（32 → 64 → 32）
  ↓
残差连接
```

模型参数量约55,094。当前LFP测试结果：

```text
车辆准确率：98.67%
车辆 Macro-F1：97.52%
窗口 Macro-F1：82.89%
```

原ConvStem-Mamba保留为23,781参数的轻量基线，其窗口Macro-F1为76.86%。FAF直接相加版本结果较低，不作为推荐网络。

---

## 3. `CURRENT_BEST_NETWORK.md` 中的网络

该文件描述的是 **Causal-Window DNN-MTL**，用于SOC/SOH估计，不是故障四分类网络。该架构同时适用于 **LFP和NCM**；两种化学体系使用相同的主体结构，但分别预处理、分别归一化、分别训练并保存权重，模型权重不能混用。

其中，原文件顶部的“长安300辆车”实现是该架构在96S1P NCM数据上的具体版本；历史实验exp016和exp017分别对应LFP与NCM。通用网络结构如下：

```text
时序输入 x [B,128,7]
    ↓
四层残差因果 TCN：[B,128,7] → [B,128,64]
    │
    ├── 浅层路径
    │     时间均值池化：[B,128,64] → [B,64]
    │     Linear + LayerNorm + GELU：64 → 128
    │     与上下文特征拼接并投影：192 → 256
    │     SOC分支：256 → 128，得到 f_soc [B,128]
    │
    └── 深层路径
          4头时间自注意力 + FFN：[B,128,64]
          时间均值池化：[B,128,64] → [B,64]
          Linear + LayerNorm + GELU：64 → 128
          与上下文特征拼接并投影：192 → 256
          SOH分支：256 → 128，得到 f_soh [B,128]

f_soc、f_soh
    ↓
门控 Star 任务交互
    ├── f_soc ⊙ f_soh
    ├── Linear(128→128) + GELU
    └── 经过可学习门控后，残差注入两个任务分支
    ↓
    ├── SOC输出头：128 → 64 → 1 → Sigmoid
    │                 输出 SOC_end [B,1]，范围[0,1]
    │
    └── SOH输出头：128 → 64 → 1 → Sigmoid + 区间映射
                      输出 SOH [B,1]，范围[0.6,1.3]
```

时序输入同样包含7个电池特征：平均/最高/最低单体电压、电流、最高/最低温度和电压极差。

因果上下文 `ctx [B,6]` 包含：

```text
1. 充电/放电工况
2. 窗口平均温度
3. 当前过程已经持续的时间
4. 窗口平均电流
5. 单体电压极差
6. 截至当前时刻的累计充电或放电 Ah / C_nom
```

上下文先编码为 `ctx_emb [B,64]`，再分别与浅层、深层时序特征拼接。浅层路径主要服务于变化较快的SOC，深层注意力路径主要服务于变化较慢的SOH；Star模块负责在两个任务之间交换有用信息。

两种化学体系的主要区别：

```text
LFP：使用LFP车辆拟合归一化参数、标称容量和LFP模型权重
NCM：使用NCM车辆拟合归一化参数、标称容量和NCM模型权重
长安NCM实例：C_nom = 155 Ah，输入仍为[B,128,7]
```

`C_nom`表示对应数据集的标称容量，155 Ah只适用于当前长安NCM实例，不是LFP和所有NCM数据的通用常数。

训练分为三个阶段：

```text
SOC单任务训练 10轮
→ SOH单任务训练 10轮
→ SOC/SOH联合训练 30轮
```

两套网络的任务不同：

```text
ConvStem-Mamba：正常、低容量、高内阻、自放电四分类
Causal-Window DNN-MTL：SOC_end与SOH双回归
```
