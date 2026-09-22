# -*- coding: utf-8 -*-
"""使用真实串数识别口径，顺序重新训练 LFP 和 NCM 正式模型。"""
import gc
import json
import os

import torch

from config import Config
from train import run, DEFAULT_EXP_ROOT


def make_config(chemistry):
    is_ncm = chemistry == "NCM"
    return Config(
        chemistry=chemistry,
        n_vehicles=200,
        include_low_capacity=1,
        include_high_resistance=0,
        infer_n_cells=1,
        window_len=128,
        stride=16,
        batch_size=256,
        device="cuda" if torch.cuda.is_available() else "cpu",
        staged=1,
        stage1_epochs=20,
        stage2_epochs=20,
        stage3_epochs=20,
        stage1_lr=1e-3,
        stage2_lr=5e-4,
        stage3_lr=1e-4,
        stage2_freeze_shared=1,
        lambda_physics=0.1,
        lambda_consistency=0.5 if is_ncm else 0.0,
        early_stop_patience=8,
        seed=42,
        exp_name=f"retrain_{chemistry.lower()}_infer_cells",
        note=(
            f"{chemistry}正式重训：使用工作区 dataset，按每个CSV的VOLT_n列"
            "自动识别真实串数；其余口径对齐exp016/exp017。"
        ),
    )


def main():
    results = []
    for chemistry in ("LFP", "NCM"):
        print(f"[retrain] start {chemistry}", flush=True)
        exp_dir, metrics, meta = run(make_config(chemistry))
        results.append({
            "chemistry": chemistry,
            "exp_dir": exp_dir,
            "metrics": metrics,
            "data": {
                "n_train": meta["n_train"],
                "n_val": meta["n_val"],
                "n_test": meta["n_test"],
                "n_train_veh": meta["n_train_veh"],
                "n_val_veh": meta["n_val_veh"],
                "n_test_veh": meta["n_test_veh"],
                "c_nom_per_chem": meta["c_nom_per_chem"],
                "infer_n_cells": meta["infer_n_cells"],
            },
        })
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(DEFAULT_EXP_ROOT, exist_ok=True)
    out_path = os.path.join(DEFAULT_EXP_ROOT, "retrain_both_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    print(f"[retrain] summary -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
