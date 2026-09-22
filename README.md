# soc-sohANDfaultFORCH-BatteryGen

本仓库用于管理电池 SOC/SOH 联合估计与故障相关代码。

当前 SOC/SOH 工程位于 [`soc-soh/`](soc-soh/)。仓库只保存代码、配置说明和文档，不保存原始 CSV 数据、预处理数据、模型权重或训练输出。

本地运行时，可将数据放在仓库根目录的 `dataset/`，或通过 `--data_root` 指向仓库外部的数据目录。
