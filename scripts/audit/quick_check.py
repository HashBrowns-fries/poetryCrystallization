#!/usr/bin/env python3
"""
quick_check.py
快速检查 PERMANOVA 数字关系（不做置换，只看数学一致性）
"""

import json
import numpy as np
from collections import Counter

np.random.seed(42)

BASE = "/home/chenhao/poetryCrystallization"
POETS = f"{BASE}/data/processed/poet_poems.json"
GENRE_SRC = f"{BASE}/data/processed/poet_genre_by_source.json"
DIST_NPY = f"{BASE}/data/processed/poet_distances.npy"

print("快速诊断：PERMANOVA 数字关系")
print("=" * 80)

# 加载数据
poets = json.load(open(POETS, encoding="utf-8"))
gsrc = json.load(open(GENRE_SRC, encoding="utf-8"))
D = np.load(DIST_NPY).astype(np.float64)
n = len(D)

ri, ci = np.triu_indices(n, k=1)
d_sq = D[ri, ci] ** 2
SS_T = float(d_sq.sum())

print(f"n = {n}, SS_Total = {SS_T:.4f}\n")

# 标签
def dom_genre(name):
    g = gsrc.get(name, {})
    s = g.get("shi", 0); c = g.get("ci", 0); q = g.get("qu", 0); f = g.get("fu", 0)
    t = s + c + q + f
    if t == 0: return "shi"
    if c / t > 0.25: return "ci"
    if q / t > 0.25: return "qu"
    return "shi"

dyn_map = {"周":"Other","汉":"Other","晋":"Other","南北朝":"Other","隋":"Other",
           "唐":"Tang","五代":"Other","宋":"Song","元":"Yuan","明":"Ming",
           "清":"Qing","近代":"Modern","其他":"Other","当代":"Other","未知":"Other"}

dyn_lbl = np.array([dyn_map.get(p.get("dynasty","?"),"Other") for p in poets])
genre_lbl = np.array([dom_genre(p["name"]) for p in poets])
g_id = np.array([{"shi":0,"ci":1,"qu":2}[g] for g in genre_lbl])
dyn_id = np.array([{"Other":0,"Tang":1,"Song":2,"Yuan":3,"Ming":4,"Qing":5,"Modern":6}[d]
                   for d in dyn_lbl])

print(f"体裁: {dict(Counter(genre_lbl))}")
print(f"朝代: {dict(Counter(dyn_lbl))}\n")

# SS 函数
def ss_within(labels):
    total = 0.0
    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]
        if len(idx) < 2: continue
        m = np.isin(ri, idx) & np.isin(ci, idx)
        total += d_sq[m].sum()
    return total

def ss_between(labels):
    return SS_T - ss_within(labels)

# 单因素
ss_genre = ss_between(g_id)
ss_dynasty = ss_between(dyn_id)
R2_genre = ss_genre / SS_T
R2_dynasty = ss_dynasty / SS_T

print("单因素 PERMANOVA:")
print(f"  体裁:   SS_b = {ss_genre:.4f}, R² = {R2_genre:.6f}")
print(f"  朝代:   SS_b = {ss_dynasty:.4f}, R² = {R2_dynasty:.6f}\n")

# 双因素
joint_labels = np.array([f"{g}_{d}" for g, d in zip(g_id, dyn_id)])
ss_joint = ss_between(joint_labels)
R2_joint = ss_joint / SS_T

ss_genre_cond = ss_joint - ss_dynasty
ss_dynasty_cond = ss_joint - ss_genre
ss_interaction = max(0, ss_joint - ss_genre - ss_dynasty)

R2_genre_cond = ss_genre_cond / SS_T
R2_dynasty_cond = ss_dynasty_cond / SS_T
R2_interaction = ss_interaction / SS_T
R2_residual = 1 - R2_joint

print("双因素 PERMANOVA:")
print(f"  SS_joint = {ss_joint:.4f}, R²_joint = {R2_joint:.6f}")
print(f"  SS_genre|dynasty = {ss_genre_cond:.4f}, R²_genre|dynasty = {R2_genre_cond:.6f}")
print(f"  SS_dynasty|genre = {ss_dynasty_cond:.4f}, R²_dynasty|genre = {R2_dynasty_cond:.6f}")
print(f"  SS_interaction = {ss_interaction:.4f}, R²_interaction = {R2_interaction:.6f}")
print(f"  R²_residual = {R2_residual:.6f}\n")

# F 统计量
k_genre = len(np.unique(g_id))
k_dynasty = len(np.unique(dyn_id))
k_joint = len(np.unique(joint_labels))

df_genre = k_genre - 1
df_dynasty = k_dynasty - 1
df_joint = k_joint - 1
df_residual = n - k_joint

ss_residual = SS_T - ss_joint
MS_residual = ss_residual / df_residual
MS_genre_cond = ss_genre_cond / df_genre
MS_dynasty_cond = ss_dynasty_cond / df_dynasty

F_genre = MS_genre_cond / MS_residual
F_dynasty = MS_dynasty_cond / MS_residual

print("F 统计量:")
print(f"  df_genre = {df_genre}, df_dynasty = {df_dynasty}, df_residual = {df_residual}")
print(f"  MS_genre|dynasty = {MS_genre_cond:.4f}, MS_residual = {MS_residual:.6f}")
print(f"  F_genre = {F_genre:.2f}")
print(f"  MS_dynasty|genre = {MS_dynasty_cond:.4f}")
print(f"  F_dynasty = {F_dynasty:.2f}\n")

# 核心矛盾
print("=" * 80)
print("关键矛盾:")
print(f"  论文报告: R²_dynasty|genre = 0.733, p = 0.283")
print(f"  实际计算: R²_dynasty|genre = {R2_dynasty_cond:.6f}")
print(f"            F_dynasty = {F_dynasty:.2f}")
print(f"  若 F = {F_dynasty:.2f}，按正态近似，p << 0.001")
print(f"  p = 0.283 必然来自错误的置换检验\n")

# Bootstrap 矛盾
print("Bootstrap 矛盾:")
print(f"  genre_dominance.json 报告:")
print(f"    point_R2 = 0.1243 (这不是任何已知的 R²!)")
print(f"    ci_low = 0.1057, ci_high = 0.1431")
print(f"  论文报告:")
print(f"    R² = 0.014, Bootstrap CI = [0.106, 0.143]")
print(f"  实际计算:")
print(f"    R²_genre (单因素) = {R2_genre:.6f}")
print(f"    R²_genre|dynasty (条件) = {R2_genre_cond:.6f}")
print(f"  三个数字都不一致!\n")

# 读取 genre_dominance.json
try:
    gd = json.load(open(f"{BASE}/data/processed/genre_dominance.json"))
    print(f"genre_dominance.json 中的数字:")
    print(f"  exp1_genre_permanova.R2 = {gd['exp1_genre_permanova']['R2']}")
    print(f"  exp2_two_factor.R2_genre_cond = {gd['exp2_two_factor']['R2_genre_cond']}")
    print(f"  bootstrap_genre_R2.point_R2 = {gd['bootstrap_genre_R2']['point_R2']}")
    print(f"  bootstrap_genre_R2.median_R2 = {gd['bootstrap_genre_R2']['median_R2']}")
    print(f"  bootstrap_genre_R2.ci = [{gd['bootstrap_genre_R2']['ci_low']}, {gd['bootstrap_genre_R2']['ci_high']}]")
except:
    pass

print("\n" + "=" * 80)
print("结论: 论文中的数字来源混乱，需要全面重写报告")
