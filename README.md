# soc-sohANDfaultFORCH-BatteryGen

本仓库用于管理电池 SOC/SOH 联合估计与故障相关代码。

当前包含两个独立工程：

- [`soc-soh/`](soc-soh/)：SOC/SOH 联合估计；
- [`fault/`](fault/)：LFP/NCM 长序列四分类故障诊断。

仓库只保存代码、配置说明和文档，不保存原始 CSV 数据、预处理缓存、模型权重或训练输出。

本地运行时，可将数据放在仓库根目录的 `dataset/`。两个工程也都支持通过命令行参数指定仓库外部的数据目录，具体用法见各自的 README。
