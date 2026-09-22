# -*- coding: utf-8 -*-
"""因果窗口 SOH 实验分组审计。

输出正常/低容量、充/放电、循环早/中/后段 MAE，并与“恒输出训练车辆 SOH 中位数”比较。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

import data as D
from config import Config
from model import DNNMTL


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw["cap_clip"] = tuple(raw["cap_clip"])
    raw["tcn_dilations"] = tuple(raw["tcn_dilations"])
    return Config(**raw)


def group_mae(pred, target, mask):
    n = int(mask.sum())
    return {"n": n, "mae": float(np.mean(np.abs(pred[mask] - target[mask]))) if n else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("exp_dir")
    args = p.parse_args()
    exp_dir = os.path.abspath(args.exp_dir)
    cfg = load_config(os.path.join(exp_dir, "config.json"))
    cfg.device = "cuda"

    ds_train, _ds_val, ds_test, _meta, _ocv = D.build_datasets(cfg)
    ckpt = torch.load(os.path.join(exp_dir, "best_model.pt"), map_location="cuda", weights_only=False)
    model = DNNMTL(cfg).cuda()
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    preds, targets, valids, modes, dsocs, low_caps = [], [], [], [], [], []
    loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    offset = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].cuda(non_blocking=True)
            ctx = batch["ctx"].cuda(non_blocking=True)
            cycle_feat = batch["cycle_feat"].cuda(non_blocking=True)
            _soc, soh = model(x, ctx, cycle_feat)
            b = x.shape[0]
            preds.append(soh[:, 0].cpu().numpy())
            targets.append(batch["soh"].numpy())
            valids.append(batch["soh_valid"].numpy().astype(bool))
            modes.append(ctx[:, 0].cpu().numpy().astype(np.int64))
            dsocs.append(cycle_feat[:, 1].cpu().numpy())
            for sample in ds_test.samples[offset:offset + b]:
                low_caps.append("low_capacity" in sample[0])
            offset += b

    pred = np.concatenate(preds)
    target = np.concatenate(targets)
    valid = np.concatenate(valids)
    mode = np.concatenate(modes)
    dsoc = np.concatenate(dsocs)
    low_cap = np.asarray(low_caps, dtype=bool)
    # 训练集常数基线按车去重，避免长循环/长文件改变中位数。
    vehicle_soh = {}
    for sample in ds_train.samples:
        path, _start, _mode, soh = sample[:4]
        if np.isfinite(soh):
            vehicle_soh[os.path.dirname(path)] = float(soh)
    constant = float(np.median(list(vehicle_soh.values())))
    constant_pred = np.full_like(target, constant)

    groups = {
        "all": valid,
        "normal": valid & ~low_cap,
        "low_capacity": valid & low_cap,
        "charge": valid & (mode == 1),
        "discharge": valid & (mode == 0),
        "early_dsoc_lt_0.10": valid & (dsoc < 0.10),
        "middle_dsoc_0.10_0.20": valid & (dsoc >= 0.10) & (dsoc < 0.20),
        "late_dsoc_ge_0.20": valid & (dsoc >= 0.20),
    }
    result = {
        "experiment": os.path.basename(exp_dir),
        "device": str(next(model.parameters()).device),
        "constant_baseline": constant,
        "groups": {
            name: {
                "model": group_mae(pred, target, mask),
                "constant": group_mae(constant_pred, target, mask),
            }
            for name, mask in groups.items()
        },
    }
    out_path = os.path.join(exp_dir, "causal_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
