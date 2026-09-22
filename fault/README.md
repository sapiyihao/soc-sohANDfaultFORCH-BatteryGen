# 电池故障检测 Mamba 可运行包

本目录包含LFP/NCM长序列故障检测代码、处理后缓存和当前最佳LFP模型。原始CSV数据集位于工程根目录的 `../dataset`。

## 目录

```text
fault/
├── longseq/                         数据集读取与网络结构
├── data/processed_v2/               已处理缓存，可直接训练
├── experiments/current_best_lfp/    当前最佳LFP权重和实验结果
├── docs/                            数据与网络说明
├── .vscode/launch.json              VS Code启动配置
├── train.py                         训练入口
├── evaluate.py                      已训练模型评估入口
├── inspect_dataset.py               缓存检查
├── prepare_dataset.py               可选的重新预处理入口
├── repair_duplicate_split.py        重复序列防泄漏处理
└── requirements.txt                 Python依赖
```

## 环境

当前验证环境：

```text
Python 3.12
PyTorch 2.5.1+cu121
CUDA GPU：NVIDIA GeForce RTX 3070 Laptop GPU
```

本机可直接使用：

```powershell
conda activate py312
cd "H:\中汽研数据集研究\fault"
python -m pip install -r requirements.txt
```

## 检查缓存

```powershell
python inspect_dataset.py --cache data/processed_v2
```

## 从原始数据重新预处理

`prepare_dataset.py` 默认读取工程根目录下的 `dataset`，因此无需再指定旧机器路径：

```powershell
python prepare_dataset.py
```

如需使用其他位置的数据集，仍可通过 `--source` 显式指定。脚本拒绝覆盖已有输出目录。

## 评估当前最佳模型

```powershell
python evaluate.py
```

默认评估：

```text
模型：MultiScale-Hierarchical Mamba
体系：LFP
权重：experiments/current_best_lfp/best.pt
缓存：data/processed_v2
```

## 训练

冒烟测试：

```powershell
python train.py --cache data/processed_v2 --chemistry LFP --model multiscale_hierarchical --batch-size 64 --smoke-test
```

正式训练：

```powershell
python train.py --cache data/processed_v2 --chemistry LFP --model multiscale_hierarchical --batch-size 64 --epochs 24 --learning-rate 3e-4 --seed 42
```

也可以直接在VS Code中打开本目录，按F5选择对应启动项。

## 输入与输出

```text
输入：[B,512,7]
通道：平均单体电压、最高/最低单体电压、电流、最高/最低温度、电压极差
类别：正常、低容量、高内阻、自放电
```

缓存按VIN划分训练、验证和测试，窗口不会跨CSV；LFP和NCM分别归一化并分别训练。

## 当前最佳LFP结果

```text
车辆准确率：98.67%
车辆Macro-F1：97.52%
窗口Macro-F1：82.89%
模型参数量：55,094
```

结果来自单次固定划分。故障测试车辆数量较少，正式结论还需要多随机种子或按VIN交叉验证。
