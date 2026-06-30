#!/usr/bin/env python3
"""
sanity_check_v2.py
用 McArdle & Anderson 2001 的 PCoA-投影 SS 重新实现 PERMANOVA
关键：SS_X = trace((H_X - H_grand) @ G @ (H_X - H_grand))
这绕开了 sub-cell 计数问题。
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

np.random.seed(42)
n = 100
A = np.array([0]*30 + [1]*30 + [2]*40)
B = np.array([0]*25 + [1]*25 + [2]*25 + [3]*25)
X = np.random.randn(n, 5)
X[A == 0, 0] += 3
X[A == 1, 0] -= 3
X[B == 0, 1] += 0.5
X[B == 1, 1] -= 0.5

D = cosine_distances(X)
# Gower 中心化
A_mat = -0.5 * D**2
H = np.eye(n) - np.ones((n,n))/n
G = H @ A_mat @ H
SS_T = np.trace(G)

print(f"SS_T (PCoA framework) = {SS_T:.4f}")

def design_matrix(labels):
    """One-hot 编码（不去截距）"""
    levels = np.unique(labels)
    X_mat = np.zeros((len(labels), len(levels)))
    for i, lvl in enumerate(levels):
        X_mat[:, i] = (labels == lvl).astype(float)
    return X_mat

def hat_matrix(X_design):
    """投影矩阵 H_X = X(X'X)^{-1}X'"""
    return X_design @ np.linalg.pinv(X_design.T @ X_design) @ X_design.T

def ss_proj(H_proj, G):
    """SS via PCoA projection"""
    return np.trace(H_proj @ G @ H_proj)

# 单因素
H_grand = np.ones((n,n))/n  # 投影到 grand mean
H_A = hat_matrix(design_matrix(A))
H_B = hat_matrix(design_matrix(B))

# 联合设计矩阵：[XA | XB]（共线性需要伪逆）
joint_design = np.hstack([design_matrix(A), design_matrix(B)])
H_AB = hat_matrix(joint_design)

SS_A = ss_proj(H_A - H_grand, G)
SS_B = ss_proj(H_B - H_grand, G)
SS_AB = ss_proj(H_AB - H_grand, G)
SS_resid = SS_T - SS_AB

# 纯效应
SS_A_cond = SS_AB - SS_B
SS_B_cond = SS_AB - SS_A

R2_A_cond = SS_A_cond / SS_T
R2_B_cond = SS_B_cond / SS_T

print(f"SS_A = {SS_A:.4f}, R²_A = {SS_A/SS_T:.4f}")
print(f"SS_B = {SS_B:.4f}, R²_B = {SS_B/SS_T:.4f}")
print(f"SS_AB (joint) = {SS_AB:.4f}, R²_AB = {SS_AB/SS_T:.4f}")
print(f"SS_A|B = {SS_A_cond:.4f}, R²_A|B = {R2_A_cond:.4f}")
print(f"SS_B|A = {SS_B_cond:.4f}, R²_B|A = {R2_B_cond:.4f}")
print(f"SS_resid = {SS_resid:.4f}")

# F 统计量
k_A = len(np.unique(A))
k_B = len(np.unique(B))
df_A = k_A - 1
df_B = k_B - 1
df_resid_full = n - k_A - k_B + 1  # 假设无交互且无完全共线

# 实际 df_resid 用矩阵秩
rank_AB = np.linalg.matrix_rank(joint_design)
df_resid_full = n - rank_AB

print(f"\nrank(joint design) = {rank_AB}")
print(f"df_A = {df_A}, df_B = {df_B}, df_resid = {df_resid_full}")
MS_resid = SS_resid / df_resid_full
F_A_cond = (SS_A_cond / df_A) / MS_resid
F_B_cond = (SS_B_cond / df_B) / MS_resid
print(f"F_A|B = {F_A_cond:.2f}, F_B|A = {F_B_cond:.2f}")

# ════════════════════════════════════════════════════════════════════════════
# 置换检验：用 McArdle-Anderson PCoA-SS
# ════════════════════════════════════════════════════════════════════════════
def perm_unrestricted_pcoa(target, fixed, ss_target_cond_obs, G, n_perm=199, seed=42):
    """
    在 PCoA 框架下置换 target，固定 fixed
    重新计算 SS_target | fixed 的零分布
    """
    rng = np.random.default_rng(seed)
    cnt = 0
    H_fixed = hat_matrix(design_matrix(fixed))
    SS_fixed = ss_proj(H_fixed - np.ones((n,n))/n, G)

    for _ in range(n_perm):
        target_perm = rng.permutation(target)
        joint = np.hstack([design_matrix(target_perm), design_matrix(fixed)])
        H_AB_perm = hat_matrix(joint)
        SS_AB_perm = ss_proj(H_AB_perm - np.ones((n,n))/n, G)
        SS_target_cond_perm = SS_AB_perm - SS_fixed
        if SS_target_cond_perm >= ss_target_cond_obs:
            cnt += 1
    return cnt / n_perm

print("\n--- 用 PCoA-SS 重做置换检验 ---")
p_A = perm_unrestricted_pcoa(A, B, SS_A_cond, G, n_perm=199, seed=42)
p_B = perm_unrestricted_pcoa(B, A, SS_B_cond, G, n_perm=199, seed=42)
print(f"A | B (PCoA unrestricted)   p = {p_A:.4f}  (期望显著)")
print(f"B | A (PCoA unrestricted)   p = {p_B:.4f}  (期望相对弱但仍可能显著)")

# 完全无关变量
np.random.seed(99)
C = np.random.choice([0,1,2], size=n)
joint_AC = np.hstack([design_matrix(A), design_matrix(C)])
H_AC = hat_matrix(joint_AC)
SS_AC = ss_proj(H_AC - np.ones((n,n))/n, G)
SS_C_cond = SS_AC - SS_A
print(f"\nSS_C|A = {SS_C_cond:.4f}, R²_C|A = {SS_C_cond/SS_T:.4f}")
p_C = perm_unrestricted_pcoa(C, A, SS_C_cond, G, n_perm=199, seed=42)
print(f"C | A (PCoA unrestricted)   p = {p_C:.4f}  (期望 ≈ uniform)")
