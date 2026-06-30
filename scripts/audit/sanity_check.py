#!/usr/bin/env python3
"""
sanity_check.py
对 permanova_full_audit.py 中的关键函数做小规模健全性检查
"""
import sys
sys.path.insert(0, '/home/chenhao/poetryCrystallization/scripts/audit')

import numpy as np

# 模拟一个小数据集：100 个样本，2 个因素，已知结构
np.random.seed(42)
n = 100

# 因素 A：3 组（30, 30, 40）
A = np.array([0]*30 + [1]*30 + [2]*40)
# 因素 B：4 组（25 each），与 A 部分相关
B_base = np.array([0]*25 + [1]*25 + [2]*25 + [3]*25)
B = B_base.copy()

# 构造嵌入：A 强信号，B 弱信号
X = np.random.randn(n, 5)
X[A == 0, 0] += 3
X[A == 1, 0] -= 3
X[B == 0, 1] += 0.5
X[B == 1, 1] -= 0.5

# 距离矩阵
from sklearn.metrics.pairwise import cosine_distances
D = cosine_distances(X)
ri, ci_idx = np.triu_indices(n, k=1)
d_sq = D[ri, ci_idx] ** 2
SS_T = float(d_sq.sum())

def ss_within(labels):
    total = 0.0
    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]
        if len(idx) < 2: continue
        m = np.isin(ri, idx) & np.isin(ci_idx, idx)
        total += d_sq[m].sum()
    return total

def ss_between(labels):
    return SS_T - ss_within(labels)

ss_A = ss_between(A)
ss_B = ss_between(B)
joint = A.astype(np.int64) * 10 + B.astype(np.int64)
ss_J = ss_between(joint)
ss_A_cond = ss_J - ss_B
ss_B_cond = ss_J - ss_A

print(f"SS_Total = {SS_T:.4f}")
print(f"SS_A (强) = {ss_A:.4f}, R²_A = {ss_A/SS_T:.4f}")
print(f"SS_B (弱) = {ss_B:.4f}, R²_B = {ss_B/SS_T:.4f}")
print(f"SS_joint = {ss_J:.4f}, R²_joint = {ss_J/SS_T:.4f}")
print(f"SS_A|B = {ss_A_cond:.4f}, R²_A|B = {ss_A_cond/SS_T:.4f}")
print(f"SS_B|A = {ss_B_cond:.4f}, R²_B|A = {ss_B_cond/SS_T:.4f}")

# 置换函数（精简版）
def perm_unrestricted(target, fixed, ss_obs, n_perm=199, seed=42):
    rng = np.random.default_rng(seed)
    fixed_ss = ss_between(fixed)
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(target)
        joint_perm = perm.astype(np.int64) * 100 + fixed.astype(np.int64)
        ss_joint_perm = ss_between(joint_perm)
        ss_target_cond_perm = ss_joint_perm - fixed_ss
        if ss_target_cond_perm >= ss_obs:
            cnt += 1
    return cnt / n_perm

def perm_within_strata(target, strata, ss_obs, n_perm=199, seed=42):
    rng = np.random.default_rng(seed)
    fixed_ss = ss_between(strata)
    strata_groups = {}
    for s in np.unique(strata):
        strata_groups[s] = np.where(strata == s)[0]
    cnt = 0
    for _ in range(n_perm):
        perm = target.copy()
        for s, idx in strata_groups.items():
            if len(idx) < 2: continue
            sub_perm = rng.permutation(perm[idx])
            perm[idx] = sub_perm
        joint_perm = perm.astype(np.int64) * 100 + strata.astype(np.int64)
        ss_joint_perm = ss_between(joint_perm)
        ss_target_cond_perm = ss_joint_perm - fixed_ss
        if ss_target_cond_perm >= ss_obs:
            cnt += 1
    return cnt / n_perm

# 健全性：A 应显著，B 可能边界
print("\n--- 健全性检查 ---")
p_A_unrestr = perm_unrestricted(A, B, ss_A_cond, n_perm=199, seed=42)
p_B_unrestr = perm_unrestricted(B, A, ss_B_cond, n_perm=199, seed=42)
p_A_strata = perm_within_strata(A, B, ss_A_cond, n_perm=199, seed=42)
p_B_strata = perm_within_strata(B, A, ss_B_cond, n_perm=199, seed=42)

print(f"A | B (unrestricted)        p = {p_A_unrestr:.4f}  (期望显著, A 是强信号)")
print(f"B | A (unrestricted)        p = {p_B_unrestr:.4f}  (期望弱)")
print(f"A | B (within-strata)       p = {p_A_strata:.4f}  (期望显著)")
print(f"B | A (within-strata)       p = {p_B_strata:.4f}  (期望弱)")

# 完全无关变量：应该不显著
np.random.seed(99)
C = np.random.choice([0,1,2], size=n)
ss_C = ss_between(C)
joint_AC = A.astype(np.int64) * 10 + C.astype(np.int64)
ss_AC = ss_between(joint_AC)
ss_C_cond = ss_AC - ss_A
print(f"\nC (随机) | A,  SS_C|A = {ss_C_cond:.4f}, R²_C|A = {ss_C_cond/SS_T:.4f}")
p_C = perm_unrestricted(C, A, ss_C_cond, n_perm=199, seed=42)
print(f"C | A (unrestricted)        p = {p_C:.4f}  (期望 ≈ uniform，不显著)")
