# -*- coding: utf-8 -*-
"""在完整独立测试集上复核 exp016/exp017 预训练权重。"""
import argparse
import gc
import json
import os

import torch
from torch.utils.data import DataLoader

import data as D
from config import Config
from model import DNNMTL
from train import evaluate, get_device


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "experiments", "pretrained_verification")


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw["cap_clip"] = tuple(raw["cap_clip"])
    raw["tcn_dilations"] = tuple(raw["tcn_dilations"])
    return Config(**raw)


def verify_one(exp_dir):
    cfg = load_config(os.path.join(exp_dir, "config.json"))
    # 旧权重必须保持旧实验的固定串数预处理口径。
    cfg.infer_n_cells = 0
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = get_device(cfg)

    print(f"[verify] build full datasets: {os.path.basename(exp_dir)}", flush=True)
    ds_train, ds_val, ds_test, meta, ocv_dict = D.build_datasets(cfg)
    print(
        f"[verify] samples train/val/test={len(ds_train)}/{len(ds_val)}/{len(ds_test)}, "
        f"vehicles={meta['n_train_veh']}/{meta['n_val_veh']}/{meta['n_test_veh']}",
        flush=True,
    )
    del ds_train, ds_val
    gc.collect()

    checkpoint = torch.load(
        os.path.join(exp_dir, "best_model.pt"), map_location=device, weights_only=False)
    model = DNNMTL(cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    loader = DataLoader(
        ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    ocv = None if ocv_dict is None else torch.tensor(ocv_dict["ocv"], device=device)
    current, _pred, _true = evaluate(
        model, loader, device, cfg.lambda_physics, cfg.lambda_consistency, ocv)
    reference = checkpoint["test_metrics"]

    comparison = {}
    for key in sorted(set(reference) | set(current)):
        ref = reference.get(key)
        cur = current.get(key)
        if isinstance(ref, (int, float)) and isinstance(cur, (int, float)):
            comparison[key] = {
                "reference": float(ref),
                "current": float(cur),
                "abs_diff": abs(float(cur) - float(ref)),
            }
    return {
        "experiment": os.path.basename(exp_dir),
        "device": str(device),
        "data_root": meta["data_root"],
        "infer_n_cells": False,
        "samples": {
            "train": meta["n_train"], "val": meta["n_val"], "test": meta["n_test"]},
        "vehicles": {
            "train": meta["n_train_veh"], "val": meta["n_val_veh"],
            "test": meta["n_test_veh"]},
        "comparison": comparison,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_root", default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    experiments = ["LFP_exp016", "NCM_exp017"]
    results = []
    for name in experiments:
        results.append(verify_one(os.path.join(os.path.abspath(args.models_root), name)))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    os.makedirs(os.path.abspath(args.output), exist_ok=True)
    out_path = os.path.join(os.path.abspath(args.output), "verification_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    print(f"[verify] saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
