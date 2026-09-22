# 包内容与来源

## 运行代码

```text
train.py
evaluate.py
inspect_dataset.py
prepare_dataset.py
repair_duplicate_split.py
longseq/data.py
longseq/model.py
```

## 数据

`data/processed_v2` 是从工程根目录 `../dataset` 中的原始CSV一次性构建的二进制缓存。缓存包含完整的训练、验证、测试VIN划分、归一化参数、窗口索引和7通道序列。

## 模型

`experiments/current_best_lfp` 保存多尺度分层Mamba的第7轮最佳权重、24轮训练记录、测试摘要和结果图。

## 文档

```text
docs/MAMBA_DATASET_AND_NETWORKS.md
docs/ORIGINAL_CODE2_README.md
docs/LFP_FAULT_DETAILS.xlsx
```

故障等级Excel仅作为元数据保留；当前四分类训练不读取故障等级。
