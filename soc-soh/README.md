# LFP/NCM SOC-SOH 双输出网络

本目录包含 SOC/SOH 联合估计网络的代码、训练/验证脚本、环境清单和说明文档。

> Git 仓库不分发原始 CSV、预处理数据、模型权重或训练输出。

## 目录

```text
soc-soh/
├── README.md
├── code/
│   └── dnn_mtl/
│       ├── config.py
│       ├── data.py
│       ├── model.py
│       ├── losses.py
│       ├── train.py
│       ├── smoke_test.py
│       ├── retrain_both.py
│       └── verify_pretrained.py
├── docs/
└── environment/
```

## 输入与输出

- 时序窗口：`[B, 128, 7]`。
- 7个通道：平均单体电压、最高单体电压、最低单体电压、电流、最高温度、最低温度、单体压差。
- 因果上下文：5维。
- SOH循环前缀统计：4维。
- 每个窗口输出：`SOC_start`、`SOC_end` 和 `SOH`。

详细结构和标签口径见 `docs/CURRENT_BEST_NETWORK.md`。

## 数据目录

默认从仓库根目录的 `dataset/` 读取：

```text
dataset/
├── LFP/
│   ├── normal/vin_xxx/*.csv
│   ├── low_capacity/vin_xxx/*.csv
│   └── high_resistance/vin_xxx/*.csv
└── NCM/
    ├── normal/vin_xxx/*.csv
    ├── low_capacity/vin_xxx/*.csv
    └── high_resistance/vin_xxx/*.csv
```

也可使用 `--data_root <path>` 指向仓库外部的数据目录。数据集不应提交到 Git。

## 冒烟测试

```powershell
cd soc-soh/code/dnn_mtl
python smoke_test.py --data_root D:\path\to\dataset
```

冒烟测试使用少量车辆和1个epoch，验证数据读取、完整网络前向/反向、评估与产物保存。

## 正式重训

```powershell
cd soc-soh/code/dnn_mtl
python retrain_both.py
```

训练输出默认保存到 `soc-soh/experiments/`，该目录已被 `.gitignore` 排除。当数据中存在多种串数时，建议使用 `infer_n_cells=1`。

## 环境

依赖见 `environment/requirements.txt`。原实验环境为 Python 3.12、PyTorch 2.5.1+cu121 和 CUDA GPU。
