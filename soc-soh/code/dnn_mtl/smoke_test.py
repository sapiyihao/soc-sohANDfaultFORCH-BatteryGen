# -*- coding: utf-8 -*-
"""DNN-MTL 端到端冒烟测试：小数据、完整网络、1 epoch，并验证实验产物落盘。"""
import argparse
import json
import os

from config import Config
from train import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="", help="数据集根目录；空值使用默认 dataset/")
    parser.add_argument("--output_root", default="", help="冒烟实验输出目录")
    args = parser.parse_args()
    cfg = Config(
        data_root=args.data_root,
        output_root=args.output_root,
        chemistry="LFP",
        n_vehicles=8,
        include_low_capacity=0,
        include_high_resistance=0,
        window_len=128,
        stride=512,
        infer_n_cells=1,
        staged=0,
        epochs=1,
        batch_size=64,
        device="auto",
        lambda_consistency=0.0,
        early_stop_patience=0,
        exp_name="smoke_lfp",
        note="根目录 dataset 数据路径与实验保存端到端冒烟测试。",
    )
    exp_dir, metrics, meta = run(cfg)
    required = [
        "best_model.pt",
        "config.json",
        "data_meta.json",
        "metrics.json",
        "train_history.csv",
        "loss_curves.png",
        "soc_scatter.png",
        "README.md",
    ]
    missing = [name for name in required if not os.path.isfile(os.path.join(exp_dir, name))]
    if missing:
        raise RuntimeError(f"冒烟测试缺少实验产物: {missing}")
    print(json.dumps({
        "status": "ok",
        "exp_dir": exp_dir,
        "artifacts": required,
        "n_train": meta["n_train"],
        "n_val": meta["n_val"],
        "n_test": meta["n_test"],
        "test": metrics["test"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
