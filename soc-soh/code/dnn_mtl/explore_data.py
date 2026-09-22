# -*- coding: utf-8 -*-
"""数据探查(修正版)：用 CHARGE_STATUS 判定工况，精确统计满充/满放容量。"""
import os
import glob
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def capacity_ah(df, status):
    m = df["CHARGE_STATUS"] == status
    d = df[m]
    if len(d) < 10:
        return None
    t = d["TIME"].values
    i = d["SUM_CURRENT"].values
    dt = np.diff(t)
    ah = np.sum(np.abs(i[:-1]) * dt) / 3600.0
    dsoc = (d["SOC"].iloc[-1] - d["SOC"].iloc[0]) / 100.0
    if abs(dsoc) < 0.02:
        return None
    return ah / abs(dsoc)


def main():
    for chem in ["LFP", "NCM"]:
        normal_dir = os.path.join(ROOT, chem, "normal")
        vins = sorted(os.listdir(normal_dir))[:40]
        chg_caps, dis_caps = [], []
        chg_dsoc, dis_dsoc = [], []
        for vin in vins:
            for f in glob.glob(os.path.join(normal_dir, vin, "*.csv")):
                df = pd.read_csv(f)
                if len(df) < 10:
                    continue
                c1 = capacity_ah(df, 1)
                c3 = capacity_ah(df, 3)
                if c1:
                    chg_caps.append(c1)
                    chg_dsoc.append((df[df["CHARGE_STATUS"] == 1]["SOC"].iloc[-1]
                                     - df[df["CHARGE_STATUS"] == 1]["SOC"].iloc[0]))
                if c3:
                    dis_caps.append(c3)
                    dis_dsoc.append((df[df["CHARGE_STATUS"] == 3]["SOC"].iloc[-1]
                                     - df[df["CHARGE_STATUS"] == 3]["SOC"].iloc[0]))
        print("=" * 60)
        print(f"CHEMISTRY = {chem}")
        for name, caps, dsocs in [("CHARGE", chg_caps, chg_dsoc), ("DISCHARGE", dis_caps, dis_dsoc)]:
            c = np.array(caps)
            d = np.array(dsocs)
            if len(c):
                print(f"  {name}: n={len(c)}  capacity(Ah) median={np.median(c):.1f} mean={c.mean():.1f} std={c.std():.1f} "
                      f"p5={np.percentile(c,5):.1f} p95={np.percentile(c,95):.1f}")
                print(f"          ΔSOC: median={np.median(d):.1f}%  min={d.min():.1f}%  max={d.max():.1f}%")
            else:
                print(f"  {name}: none")


if __name__ == "__main__":
    main()
