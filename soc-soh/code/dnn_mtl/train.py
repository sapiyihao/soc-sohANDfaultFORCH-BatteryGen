# -*- coding: utf-8 -*-
"""训练 / 评估 / 结果保存。实验结果统一存入 experiments/<exp_name>/，不删除历史结果。"""
import os
import json
import time
import math
import csv
import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config, save_config, to_dict
import data as D
from model import DNNMTL
from losses import compute_losses

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(ROOT)
DEFAULT_EXP_ROOT = os.path.join(PROJECT_ROOT, "experiments")


def resolve_exp_dir(exp_name, output_root=""):
    exp_root = os.path.abspath(os.path.expanduser(output_root or DEFAULT_EXP_ROOT))
    base = os.path.join(exp_root, exp_name)
    if not os.path.isdir(base):
        return base
    i = 2
    while os.path.isdir(f"{base}_{i}"):
        i += 1
    return f"{base}_{i}"


def get_device(cfg):
    if cfg.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg.device


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device, lambda_physics, lambda_consistency, ocv):
    model.eval()
    n = 0
    loss_acc = {}
    se_end = 0.0
    ae_end = 0.0
    ae_start = 0.0
    se_soh = 0.0
    ae_soh = 0.0
    n_soh = 0
    all_end = []
    all_true = []
    for batch in loader:
        batch = to_device(batch, device)
        _, ld, soc, soh = compute_losses(model, batch, lambda_physics, lambda_consistency, ocv)
        for k, v in ld.items():
            loss_acc[k] = loss_acc.get(k, 0.0) + v * batch["x"].shape[0]
        B = batch["x"].shape[0]
        n += B
        p_end = soc[:, 1]
        p_start = soc[:, 0]
        t_end = batch["soc_end"]
        t_start = batch["soc_start"]
        err = (p_end - t_end)
        se_end += (err ** 2).sum().item()
        ae_end += err.abs().sum().item()
        ae_start += (p_start - t_start).abs().sum().item()
        v = batch["soh_valid"]
        if v.sum() > 0:
            e = (soh[:, 0] - batch["soh"]) * v
            se_soh += (e ** 2).sum().item()
            ae_soh += e.abs().sum().item()
            n_soh += v.sum().item()
        all_end.append(p_end.cpu().numpy())
        all_true.append(t_end.cpu().numpy())
    m = {}
    for k, v in loss_acc.items():
        m[k] = v / max(n, 1)
    m["soc_mae"] = ae_end / max(n, 1)
    m["soc_rmse"] = math.sqrt(se_end / max(n, 1))
    m["soc_start_mae"] = ae_start / max(n, 1)
    m["soh_mae"] = ae_soh / max(n_soh, 1)
    m["soh_rmse"] = math.sqrt(se_soh / max(n_soh, 1))
    m["n_soh"] = int(n_soh)
    pred_all = np.concatenate(all_end)
    true_all = np.concatenate(all_true)
    return m, pred_all, true_all


def train_one_epoch(model, loader, opt, device, lambda_physics, lambda_consistency, ocv, mode="joint"):
    model.train()
    n = 0
    loss_acc = {}
    for batch in loader:
        batch = to_device(batch, device)
        opt.zero_grad(set_to_none=True)
        total, ld, _, _ = compute_losses(model, batch, lambda_physics, lambda_consistency, ocv, mode)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        B = batch["x"].shape[0]
        n += B
        for k, v in ld.items():
            loss_acc[k] = loss_acc.get(k, 0.0) + v * B
    return {k: v / max(n, 1) for k, v in loss_acc.items()}


def freeze_by_prefix(model, frozen_prefixes):
    """按参数名前缀冻结/解冻。frozen_prefixes 为空则全部解冻。"""
    for name, p in model.named_parameters():
        p.requires_grad = not any(name.startswith(pfx) for pfx in frozen_prefixes)


SHARED_PREFIXES = ["tcn", "temporal_head", "ctx_enc", "proj"]


def make_optimizer(model, stage, cfg):
    """构造优化器。阶段带 shared_lr 时对共享编码器用更低学习率（微调），任务分支用 stage.lr。"""
    if stage.get("shared_lr") is not None:
        shared = [p for n, p in model.named_parameters()
                  if p.requires_grad and any(n.startswith(pfx) for pfx in SHARED_PREFIXES)]
        task = [p for n, p in model.named_parameters()
                if p.requires_grad and not any(n.startswith(pfx) for pfx in SHARED_PREFIXES)]
        return torch.optim.AdamW([
            {"params": shared, "lr": stage["shared_lr"]},
            {"params": task, "lr": stage["lr"]},
        ], weight_decay=cfg.weight_decay)
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=stage["lr"], weight_decay=cfg.weight_decay)


def make_stages(cfg):
    """按配置构造训练阶段序列（DNN-MTL §5.2 的 3 阶段，或端到端单阶段）。"""
    if cfg.staged:
        stage2_freeze = ["soc_branch", "soc_head", "star", "log_sigma_soc"]
        stage2 = {"name": "stage2_soh", "epochs": cfg.stage2_epochs, "lr": cfg.stage2_lr, "mode": "soh",
                  "stop_metric": "soh_mae", "patience": 0}
        if cfg.stage2_freeze_shared:
            # 完全冻结共享编码器，仅训练 循环编码器 + SOH 分支 + σ_soh（避免污染 SOC 特征）
            stage2["freeze"] = stage2_freeze + list(SHARED_PREFIXES)
        else:
            # 共享编码器用更低学习率微调（对齐设计 1e-4 vs 5e-4）
            stage2["freeze"] = stage2_freeze
            stage2["shared_lr"] = cfg.stage2_lr * 0.2
        return [
            {"name": "stage1_soc", "epochs": cfg.stage1_epochs, "lr": cfg.stage1_lr, "mode": "soc",
             "freeze": ["soh_branch", "soh_head", "star", "attn", "log_sigma_soh", "cycle_enc"],
             "stop_metric": "soc_mae", "patience": 0},
            stage2,
            {"name": "stage3_joint", "epochs": cfg.stage3_epochs, "lr": cfg.stage3_lr, "mode": "joint",
             "freeze": [], "stop_metric": "soc_mae", "patience": cfg.early_stop_patience},
        ]
    return [
        {"name": "joint", "epochs": cfg.epochs, "lr": cfg.lr, "mode": "joint",
         "freeze": [], "stop_metric": "soc_mae", "patience": cfg.early_stop_patience},
    ]


def write_history(history_path, rows):
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_results(exp_dir, history_rows, pred_all, true_all, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 损失曲线
    epochs = [r["epoch"] for r in history_rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    tr_l = [r.get("train_l_total", np.nan) for r in history_rows]
    va_l = [r.get("val_l_total", np.nan) for r in history_rows]
    axes[0].plot(epochs, tr_l, label="train L_total")
    axes[0].plot(epochs, va_l, label="val L_total")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend(); axes[0].set_title("Loss")
    tr_soc = [r.get("train_l_soc", np.nan) for r in history_rows]
    va_soc = [r.get("val_soc_mae", np.nan) for r in history_rows]
    axes[1].plot(epochs, tr_soc, label="train L_soc")
    ax2 = axes[1].twinx()
    ax2.plot(epochs, va_soc, label="val SOC MAE", color="orange")
    axes[1].set_xlabel("epoch"); axes[1].legend(loc="upper left"); ax2.legend(loc="upper right")
    axes[1].set_title("SOC")
    fig.tight_layout(); fig.savefig(os.path.join(exp_dir, "loss_curves.png"), dpi=110); plt.close(fig)

    # SOC 散点
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true_all, pred_all, s=3, alpha=0.3)
    ax.plot([0, 1], [0, 1], "r--", lw=1)
    ax.set_xlabel("true SOC"); ax.set_ylabel("pred SOC")
    mae = np.mean(np.abs(pred_all - true_all))
    ax.set_title(f"SOC test (MAE={mae:.4f})")
    fig.tight_layout(); fig.savefig(os.path.join(exp_dir, "soc_scatter.png"), dpi=110); plt.close(fig)


def write_readme(exp_dir, cfg, metrics, meta):
    note = cfg.note or "（本轮为基线：单电池包 DNN-MTL，纯 V/I/T 输入估计 SOC/SOH。）"
    chem = cfg.chemistry
    c_nom = meta.get("c_nom_per_chem", {})
    lines = [
        f"# {cfg.exp_name}",
        "",
        f"- 化学体系：`{chem}`",
        f"- 生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 本轮代码特性",
        "",
        note,
        "",
        "## 模型结构（DNN-MTL 单电池包适配）",
        "",
        "- 输入第1路：滑动窗口 `[W=window_len, 7]`（v_sum_cell, v_max, v_min, current, t_max, t_min, v_spread）",
        "- 输入第2路：因果上下文 5 维（mode, t_mean, elapsed, window_current, voltage_spread）",
        "- SOH 输入仅含截至当前窗口末端的前缀统计（partial Ah/ΔSOC/elapsed/mean|I|）",
        "- 不输入当前 SOH 标签，不使用完整循环未来信息",
        "- 共享 TCN（4 层空洞因果卷积，膨胀率 [1,2,4,8]）",
        "- 浅层出口（GAP）→ SOC 分支；深层出口（时序自注意力→GAP）→ SOH 分支",
        "- 共享上下文编码器 + 共享投影层 FC(192→256)",
        "- Star Operation 结构乘积耦合 + 可学习不确定性 σ_soc/σ_soh",
        f"- 参数量：{metrics.get('n_params', 0)/1e6:.3f}M",
        "",
        "## 损失",
        "",
        f"- L_SOC / L_SOH = Huber（SOH {'充放电窗口均监督' if cfg.soh_all_modes else '仅充电窗口监督'}）",
        "- 不确定性自适应加权：L_task = L_SOC/(2σ_soc²) + L_SOH/(2σ_soh²) + log σ_soc + log σ_soh",
        f"- L_physics（仅充电）= Huber(ΔSOC_pred·SOH_pred·C_nom, ∫|I|dt)，λ={cfg.lambda_physics}",
        f"- L_consistency = Huber(V_measured, OCV(SOC)·N+I·R0)，λ={cfg.lambda_consistency}",
        "- 未启用：L_monotonicity（留待后续轮次）",
        "",
        "## 数据",
        "",
        f"- 训练/验证/测试样本：{meta.get('n_train')} / {meta.get('n_val')} / {meta.get('n_test')}",
        f"- 训练/验证/测试车辆：{meta.get('n_train_veh')} / {meta.get('n_val_veh')} / {meta.get('n_test_veh')}",
        f"- C_nom（标称容量 Ah）：{c_nom}",
        "",
        "## 关键超参数",
        "",
        f"- window_len={cfg.window_len}, stride={cfg.stride}, batch_size={cfg.batch_size}",
        f"- lr={cfg.lr}, weight_decay={cfg.weight_decay}, seed={cfg.seed}",
        f"- 训练方式：{'分阶段(SOC→SOH→联合)' if cfg.staged else '端到端单阶段'}",
        f"- tcn_channels={cfg.tcn_channels}, fusion_dim={cfg.fusion_dim}, branch_dim={cfg.branch_dim}, dropout={cfg.dropout}",
        f"- causal_window_soh={cfg.causal_window_soh}, soh_all_modes={cfg.soh_all_modes}, soh_per_cycle={cfg.soh_per_cycle}",
        f"- SOH 输出范围=[{cfg.soh_min}, {cfg.soh_max}], smooth_soc_label={cfg.smooth_soc_label}",
        "",
        "## 结果",
        "",
        "| 指标 | 值 |",
        "|------|----|",
        f"| 最优 epoch | {metrics.get('best_epoch')} |",
        f"| 验证集最佳 SOC MAE | {metrics.get('best_val_soc_mae'):.5f} |",
    ]
    t = metrics.get("test", {})
    for k, v in t.items():
        if isinstance(v, (int, float)):
            lines.append(f"| test {k} | {v:.5f} |" if isinstance(v, float) else f"| test {k} | {v} |")
    lines += [
        f"| 运行时长 | {metrics.get('runtime_sec')}s |",
        "",
        "## 结论与下一步",
        "",
        "（待补充）",
        "",
    ]
    with open(os.path.join(exp_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run(cfg: Config):
    seed_all(cfg.seed)
    device = get_device(cfg)
    exp_dir = resolve_exp_dir(cfg.exp_name, cfg.output_root)
    os.makedirs(exp_dir, exist_ok=True)
    t0 = time.time()

    print(f"[data] building datasets (chemistry={cfg.chemistry}, n_vehicles={cfg.n_vehicles}) ...")
    ds_train, ds_val, ds_test, meta, ocv_dict = D.build_datasets(cfg)
    print(f"[data] train={meta['n_train']} val={meta['n_val']} test={meta['n_test']}  "
          f"C_nom={meta['c_nom_per_chem']}")

    ocv_tensor = None
    if ocv_dict is not None:
        ocv_tensor = torch.tensor(ocv_dict["ocv"], device=device)
        print(f"[data] OCV curve built ({len(ocv_dict['ocv'])} bins, "
              f"range {min(ocv_dict['ocv']):.3f}~{max(ocv_dict['ocv']):.3f} V/cell)")

    model = DNNMTL(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.3f}M")

    train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, drop_last=False)
    val_loader = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    history_rows = []
    best_val_mae = float("inf")
    best_epoch = -1
    best_state = None
    epoch_counter = 0
    stages = make_stages(cfg)
    n_stages = len(stages)

    for si, stage in enumerate(stages):
        freeze_by_prefix(model, stage["freeze"])
        opt = make_optimizer(model, stage, cfg)
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[stage] {stage['name']} epochs={stage['epochs']} lr={stage['lr']} mode={stage['mode']} "
              f"trainable={n_tr}")
        bad_epochs = 0
        track_best = (si == n_stages - 1)   # 仅在最后一个阶段保存最优模型

        for ep in range(1, stage["epochs"] + 1):
            epoch_counter += 1
            tr = train_one_epoch(model, train_loader, opt, device, cfg.lambda_physics,
                                 cfg.lambda_consistency, ocv_tensor, stage["mode"])
            val_m, _, _ = evaluate(model, val_loader, device, cfg.lambda_physics,
                                   cfg.lambda_consistency, ocv_tensor)
            row = {"epoch": epoch_counter, "stage": stage["name"]}
            for k, v in tr.items():
                row[f"train_{k}"] = v
            for k, v in val_m.items():
                row[f"val_{k}"] = v
            history_rows.append(row)

            if epoch_counter % cfg.log_every == 0 or ep == stage["epochs"]:
                print(f"[epoch {epoch_counter:3d} {stage['name']}] tr_l_total={tr['l_total']:.4f} "
                      f"val_soc_mae={val_m['soc_mae']:.5f} val_soc_rmse={val_m['soc_rmse']:.5f} "
                      f"val_soh_mae={val_m['soh_mae']:.5f} sigma_soc={tr['sigma_soc']:.3f} "
                      f"sigma_soh={tr['sigma_soh']:.3f}")

            if track_best:
                if val_m["soc_mae"] < best_val_mae - 1e-6:
                    best_val_mae = val_m["soc_mae"]
                    best_epoch = epoch_counter
                    bad_epochs = 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    bad_epochs += 1
                if stage["patience"] and bad_epochs >= stage["patience"]:
                    print(f"[early stop] {stage['name']} no improvement for {stage['patience']} epochs "
                          f"at epoch {epoch_counter}")
                    break

    # 恢复最优模型评估测试集
    if best_state is not None:
        model.load_state_dict(best_state)
    test_m, pred_test, true_test = evaluate(model, test_loader, device, cfg.lambda_physics, cfg.lambda_consistency, ocv_tensor)

    # 保存结果
    save_config(cfg, os.path.join(exp_dir, "config.json"))
    with open(os.path.join(exp_dir, "data_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=float)
    write_history(os.path.join(exp_dir, "train_history.csv"), history_rows)
    torch.save({"model_state": model.state_dict(), "cfg": to_dict(cfg),
                "test_metrics": test_m, "best_epoch": best_epoch},
               os.path.join(exp_dir, "best_model.pt"))
    metrics = {
        "best_epoch": best_epoch,
        "best_val_soc_mae": best_val_mae,
        "test": test_m,
        "n_params": n_params,
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(exp_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    plot_results(exp_dir, history_rows, pred_test, true_test, cfg)
    write_readme(exp_dir, cfg, metrics, meta)

    print(f"[done] test_soc_mae={test_m['soc_mae']:.5f} test_soc_rmse={test_m['soc_rmse']:.5f} "
          f"test_soh_mae={test_m['soh_mae']:.5f}  best_epoch={best_epoch}")
    print(f"[save] results -> {exp_dir}")
    return exp_dir, metrics, meta


if __name__ == "__main__":
    from config import parse_args
    _cfg = parse_args()
    run(_cfg)
