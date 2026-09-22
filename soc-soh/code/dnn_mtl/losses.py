# -*- coding: utf-8 -*-
"""损失函数：任务 Huber + 不确定性自适应加权 + 物理耦合（库仑计数）+ 电压一致性（OCV）。

忠实还原 DNN-MTLV1.3 的损失体系（单电池包适配版）：
  L_task = L_SOC/(2σ_soc²) + L_SOH/(2σ_soh²) + log σ_soc + log σ_soh
  L_physics(仅充电)   = Huber( ΔSOC_pred · SOH_pred · C_nom , ∫|I|dt )
  L_consistency       = Huber( V_measured , OCV(SOC_pred)·N_cells + I·R0 )
  L_total = L_task + λ_physics·L_physics + λ_consistency·L_consistency

注：L_monotonicity 留待后续轮次（需时序排序批）。
"""
import torch
import torch.nn.functional as F

HUBER_DELTA = 0.1


def huber(pred, target, delta=HUBER_DELTA):
    return F.smooth_l1_loss(pred, target, reduction="none", beta=delta)


def masked_mean(loss, mask):
    """按 mask 求平均；若全为 0 返回 0 张量（保持可导）。"""
    n = mask.sum()
    # mask 也可以是小于 1 的样本权重，只有权重和为 0 才表示无有效样本。
    if n <= 0:
        return (loss * 0.0).sum()
    return (loss * mask).sum() / n


def ocv_lookup(soc, ocv):
    """对 OCV 曲线做可微线性插值。soc: [B,1] ∈ [0,1]; ocv: [n_bins] 1D。返回 [B,1]。"""
    n = ocv.shape[0] - 1
    x = torch.clamp(soc, 0.0, 1.0) * n
    idx0 = x.floor().long()
    idx1 = (idx0 + 1).clamp(max=n)
    frac = (x - idx0.float()).clamp(0.0, 1.0)
    v0 = ocv[idx0]
    v1 = ocv[idx1]
    return v0 * (1.0 - frac) + v1 * frac


def compute_losses(model, batch, lambda_physics=0.1, lambda_consistency=0.0, ocv=None, mode="joint"):
    """返回 (total_loss, loss_dict, soc, soh)。

    mode: "soc" 只算 SOC 任务损失(阶段1)；"soh" 只算 SOH+物理(阶段2)；"joint" 全量(阶段3/端到端)。
    """
    x = batch["x"]
    ctx = batch["ctx"]
    cycle_feat = batch.get("cycle_feat")
    soc, soh = model(x, ctx, cycle_feat)               # soc: [B,2], soh: [B,1]

    soc_start = soc[:, 0:1]
    soc_end = soc[:, 1:2]
    y_start = batch["soc_start"].unsqueeze(1)
    y_end = batch["soc_end"].unsqueeze(1)
    y_soh = batch["soh"].unsqueeze(1)
    soh_valid = batch["soh_valid"].unsqueeze(1)
    soh_weight = batch.get("soh_weight", batch["soh_valid"]).unsqueeze(1)
    physics_valid = batch.get("physics_valid", batch["soh_valid"]).unsqueeze(1)
    c_nom = batch["c_nom"].unsqueeze(1)
    coulomb = batch["coulomb_ah"].unsqueeze(1)

    l_soc = (huber(soc_start, y_start).mean() + huber(soc_end, y_end).mean()) * 0.5
    l_soh = masked_mean(huber(soh, y_soh), soh_weight)

    sigma_soc = model.sigma_soc()
    sigma_soh = model.sigma_soh()

    # 物理耦合损失（仅充电样本）：预测容量变化 vs 库仑计数
    dsoc_pred = torch.clamp(soc_end - soc_start, min=0.0)
    pred_ah = dsoc_pred * soh * c_nom
    l_physics = masked_mean(huber(pred_ah, coulomb), physics_valid)

    # 电压一致性损失（Thevenin：OCV + I·R0 + RC 极化 V_p，按每串电压计算）
    if lambda_consistency > 0 and ocv is not None:
        n_cells = batch["n_cells"].unsqueeze(1)
        v_cell_meas = batch["v_sum_raw"].unsqueeze(1) / n_cells
        i_raw = batch["i_raw"].unsqueeze(1)                 # 窗口末电流
        i_seq = batch["i_seq"]                              # [B, W] 原始电流序列
        ocv_val = ocv_lookup(soc_end, ocv)                  # 每串 OCV [B,1]

        # 一阶 RC 极化：V_p[k+1] = V_p[k]·α + I[k]·R1·(1-α)，α=exp(-dt/τ)
        tau = model.tau
        r1 = model.r1
        dt = 10.0                                          # 本数据集采样间隔固定 10s
        alpha = torch.exp(-dt / tau)                       # 标量
        W = i_seq.shape[1]
        k = torch.arange(W, device=i_seq.device, dtype=torch.float32)
        w = alpha ** (W - 1 - k)                           # [W] 权重 α^(W-1-k)
        vp = r1 * (1.0 - alpha) * (i_seq * w[None, :]).sum(dim=1, keepdim=True)   # [B,1]

        v_cell_pred = ocv_val + i_raw * model.r0 + vp      # 每串电压（欧姆 + 极化）
        l_consistency = huber(v_cell_meas, v_cell_pred).mean()
    else:
        l_consistency = (soc_end * 0.0).sum()

    l_soc_w = l_soc / (2 * sigma_soc ** 2) + torch.log(sigma_soc)
    l_soh_w = l_soh / (2 * sigma_soh ** 2) + torch.log(sigma_soh)
    if mode == "soc":
        total = l_soc_w + lambda_consistency * l_consistency
    elif mode == "soh":
        total = l_soh_w + lambda_physics * l_physics
    else:  # joint
        total = l_soc_w + l_soh_w + lambda_physics * l_physics + lambda_consistency * l_consistency

    return total, {
        "l_soc": l_soc.item(),
        "l_soh": l_soh.item(),
        "l_physics": l_physics.item(),
        "l_consistency": l_consistency.item(),
        "l_task": (l_soc_w + l_soh_w).item(),
        "l_total": total.item(),
        "sigma_soc": sigma_soc.item(),
        "sigma_soh": sigma_soh.item(),
        "r0": model.r0.item(),
        "r1": model.r1.item(),
        "tau": model.tau.item(),
    }, soc, soh
