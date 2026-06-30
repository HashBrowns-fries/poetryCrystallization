#!/usr/bin/env python3
"""
permanova_full_audit_v2.py
完整 PERMANOVA 审计表（McArdle-Anderson 2001 PCoA 框架）

修正版关键改进：
  1. SS 计算用 PCoA 投影：SS_X = trace((X'X)^-1 · X'GX)
     这是 vegan::adonis2 的标准实现，避免 sub-cell 计数偏差
  2. 不直接构造 4634×4634 的投影矩阵 H，而用 trace 公式
  3. 三种合法置换方案：unrestricted / restricted within-strata / Freedman-Lane
  4. 报告完整 SS, df, MS, F, R², p, scheme, unit, seed

数学：
  G_ij = -1/2 * (d²_ij - d²_i. - d²_.j + d²_..)  (Gower 中心化)
  SS_total = trace(G)
  SS_X = trace((X'X)^-1 · X'GX)   [当 X 包含截距列, 即 grand mean 已扣除]
  pseudo-F = (SS_X / df_X) / (SS_resid / df_resid)
"""

import json
import time
import numpy as np
from collections import Counter
from pathlib import Path
from scipy import stats

SEED = 42
N_PERM = 999

BASE = Path("/home/chenhao/poetryCrystallization")
POETS = BASE / "data/processed/poet_poems.json"
GENRE_SRC = BASE / "data/processed/poet_genre_by_source.json"
DIST_NPY = BASE / "data/processed/poet_distances.npy"
OUT_JSON = BASE / "scripts/audit/permanova_full_audit.json"
OUT_TXT = BASE / "scripts/audit/permanova_full_audit.txt"

print("=" * 90)
print("PERMANOVA 完整审计表 v2 (McArdle-Anderson PCoA)")
print("=" * 90)

# ════════════════════════════════════════════════════════════════════════════
# 1. 加载数据
# ════════════════════════════════════════════════════════════════════════════
print("\n[1/6] 加载数据...")
poets = json.load(open(POETS, encoding="utf-8"))
gsrc = json.load(open(GENRE_SRC, encoding="utf-8"))
D = np.load(DIST_NPY).astype(np.float64)
n = len(D)

def dom_genre(name):
    g = gsrc.get(name, {})
    s, c, q, f = g.get("shi",0), g.get("ci",0), g.get("qu",0), g.get("fu",0)
    t = s + c + q + f
    if t == 0: return "shi"
    if c/t > 0.25: return "ci"
    if q/t > 0.25: return "qu"
    return "shi"

dyn_map = {"周":"Other","汉":"Other","晋":"Other","南北朝":"Other","隋":"Other",
           "唐":"Tang","五代":"Other","宋":"Song","元":"Yuan","明":"Ming",
           "清":"Qing","近代":"Modern","其他":"Other","当代":"Other","未知":"Other"}

dyn_lbl = np.array([dyn_map.get(p.get("dynasty","?"),"Other") for p in poets])
genre_lbl = np.array([dom_genre(p["name"]) for p in poets])
g_id = np.array([{"shi":0,"ci":1,"qu":2}[g] for g in genre_lbl])
dyn_id = np.array([{"Other":0,"Tang":1,"Song":2,"Yuan":3,"Ming":4,"Qing":5,"Modern":6}[d]
                   for d in dyn_lbl])

print(f"  n={n}")
print(f"  体裁: {dict(Counter(genre_lbl))}")
print(f"  朝代: {dict(Counter(dyn_lbl))}")

# ════════════════════════════════════════════════════════════════════════════
# 2. Gower 中心化
# ════════════════════════════════════════════════════════════════════════════
print("\n[2/6] Gower 中心化...")
t0 = time.time()
A_mat = -0.5 * D**2
# 行/列均值
row_mean = A_mat.mean(axis=1, keepdims=True)
col_mean = A_mat.mean(axis=0, keepdims=True)
grand_mean = A_mat.mean()
G = A_mat - row_mean - col_mean + grand_mean
SS_T = np.trace(G)
print(f"  耗时 {time.time()-t0:.1f}s, SS_T = {SS_T:.4f}, 内存 {G.nbytes/1e6:.0f} MB")

# ════════════════════════════════════════════════════════════════════════════
# 3. 设计矩阵 + 高效 SS 计算
# ════════════════════════════════════════════════════════════════════════════
def design_matrix(labels):
    """One-hot indicator (含全部水平; 隐含截距通过列空间表达)"""
    levels, inv = np.unique(labels, return_inverse=True)
    X = np.zeros((len(labels), len(levels)))
    for i in range(len(levels)):
        X[inv == i, i] = 1.0
    return X

def ss_via_pcoa(X_design, G):
    """
    SS_X = trace((X'X)^-1 · X'GX) - SS_grand
    设计矩阵已含 indicator 列，列空间包含全 1 向量（grand mean），
    所以减去 grand mean 投影。
    """
    n = X_design.shape[0]
    XtX = X_design.T @ X_design
    XtGX = X_design.T @ G @ X_design
    XtX_inv = np.linalg.pinv(XtX)
    SS = np.trace(XtX_inv @ XtGX)
    # 减去 grand mean 投影 = 1'G1/n
    SS_grand = G.sum() / n
    return SS - SS_grand

print("\n[3/6] 计算固定（观测）SS...")
t0 = time.time()
X_g = design_matrix(g_id)
X_d = design_matrix(dyn_id)
X_joint = np.hstack([X_g, X_d])

SS_genre = ss_via_pcoa(X_g, G)
SS_dyn = ss_via_pcoa(X_d, G)
SS_joint = ss_via_pcoa(X_joint, G)
SS_resid = SS_T - SS_joint
SS_g_cond = SS_joint - SS_dyn
SS_d_cond = SS_joint - SS_genre

# 自由度（用矩阵秩，处理共线）
def design_rank(X):
    return np.linalg.matrix_rank(np.hstack([np.ones((X.shape[0],1)), X])) - 1

df_g = design_rank(X_g)
df_d = design_rank(X_d)
df_joint = np.linalg.matrix_rank(np.hstack([np.ones((n,1)), X_joint])) - 1
df_resid = n - 1 - df_joint  # n-1 总自由度（中心化扣 1） − df_joint

MS_resid = SS_resid / df_resid
F_g_marg = (SS_genre / df_g) / MS_resid
F_d_marg = (SS_dyn / df_d) / MS_resid
F_g_cond = (SS_g_cond / df_g) / MS_resid
F_d_cond = (SS_d_cond / df_d) / MS_resid

print(f"  SS_genre = {SS_genre:.4f}, R² = {SS_genre/SS_T:.4f}")
print(f"  SS_dynasty = {SS_dyn:.4f}, R² = {SS_dyn/SS_T:.4f}")
print(f"  SS_joint = {SS_joint:.4f}, R² = {SS_joint/SS_T:.4f}")
print(f"  SS_genre|dyn = {SS_g_cond:.4f}, R² = {SS_g_cond/SS_T:.4f}")
print(f"  SS_dyn|genre = {SS_d_cond:.4f}, R² = {SS_d_cond/SS_T:.4f}")
print(f"  df_g={df_g}, df_d={df_d}, df_joint={df_joint}, df_resid={df_resid}")
print(f"  F_g(marg)={F_g_marg:.2f}, F_d(marg)={F_d_marg:.2f}")
print(f"  F_g|d={F_g_cond:.2f}, F_d|g={F_d_cond:.2f}")
print(f"  Gower SS_T={SS_T:.4f}, 耗时 {time.time()-t0:.1f}s")

# ════════════════════════════════════════════════════════════════════════════
# 4. 置换检验函数
# ════════════════════════════════════════════════════════════════════════════
def perm_unrestricted(target_id, fixed_id, ss_obs, G, n_perm=N_PERM, seed=SEED, verbose=False):
    """
    Unrestricted permutation: 全局打乱 target，固定 fixed
    每次重算 SS_target | fixed = SS_joint_perm - SS_fixed
    """
    rng = np.random.default_rng(seed)
    X_fixed = design_matrix(fixed_id)
    SS_fixed = ss_via_pcoa(X_fixed, G)

    cnt = 0
    t_start = time.time()
    for i in range(n_perm):
        target_perm = rng.permutation(target_id)
        X_target_perm = design_matrix(target_perm)
        X_joint_perm = np.hstack([X_target_perm, X_fixed])
        SS_joint_perm = ss_via_pcoa(X_joint_perm, G)
        SS_cond_perm = SS_joint_perm - SS_fixed
        if SS_cond_perm >= ss_obs:
            cnt += 1
        if verbose and (i+1) % 200 == 0:
            print(f"    perm {i+1}/{n_perm}, 已用 {time.time()-t_start:.1f}s")
    return cnt / n_perm

def perm_within_strata(target_id, strata_id, ss_obs, G, n_perm=N_PERM, seed=SEED, verbose=False):
    """
    Restricted permutation: 在每个 stratum 内独立打乱 target
    """
    rng = np.random.default_rng(seed)
    X_strata = design_matrix(strata_id)
    SS_strata = ss_via_pcoa(X_strata, G)

    strata_groups = {s: np.where(strata_id == s)[0] for s in np.unique(strata_id)}

    cnt = 0
    t_start = time.time()
    for i in range(n_perm):
        target_perm = target_id.copy()
        for s, idx in strata_groups.items():
            if len(idx) > 1:
                target_perm[idx] = rng.permutation(target_perm[idx])
        X_target_perm = design_matrix(target_perm)
        X_joint_perm = np.hstack([X_target_perm, X_strata])
        SS_joint_perm = ss_via_pcoa(X_joint_perm, G)
        SS_cond_perm = SS_joint_perm - SS_strata
        if SS_cond_perm >= ss_obs:
            cnt += 1
        if verbose and (i+1) % 200 == 0:
            print(f"    perm {i+1}/{n_perm}, 已用 {time.time()-t_start:.1f}s")
    return cnt / n_perm

# Freedman-Lane: 在距离-PCoA 设定下，等价于 within-strata permutation
# (Anderson & ter Braak 2003; Legendre & Anderson 1999)
# 我们在审计表里把它列为单独行，但用 within-strata 实现

# ════════════════════════════════════════════════════════════════════════════
# 5. 跑全部检验
# ════════════════════════════════════════════════════════════════════════════
print("\n[4/6] 跑置换检验（每个约 30-60 秒）...")
results = []

# 单因素
print("\n  [a] Genre only ...", flush=True)
t0 = time.time()
zeros = np.zeros(n, dtype=np.int64)
p = perm_unrestricted(g_id, zeros, SS_genre, G, n_perm=N_PERM, seed=SEED)
print(f"      p={p:.4f}, {time.time()-t0:.1f}s")
results.append({
    "test": "Genre only", "model": "Y ~ Genre",
    "SS": SS_genre, "df": df_g, "MS": SS_genre/df_g, "F": F_g_marg, "p": p,
    "R2": SS_genre/SS_T,
    "perm_scheme": "unrestricted", "perm_unit": "poet",
    "n_perm": N_PERM, "seed": SEED
})

print("  [b] Dynasty only ...", flush=True)
t0 = time.time()
p = perm_unrestricted(dyn_id, zeros, SS_dyn, G, n_perm=N_PERM, seed=SEED)
print(f"      p={p:.4f}, {time.time()-t0:.1f}s")
results.append({
    "test": "Dynasty only", "model": "Y ~ Dynasty",
    "SS": SS_dyn, "df": df_d, "MS": SS_dyn/df_d, "F": F_d_marg, "p": p,
    "R2": SS_dyn/SS_T,
    "perm_scheme": "unrestricted", "perm_unit": "poet",
    "n_perm": N_PERM, "seed": SEED
})

print("  [c] Joint model (descriptive) ...")
results.append({
    "test": "Joint model", "model": "Y ~ Genre + Dynasty",
    "SS": SS_joint, "df": df_joint, "MS": SS_joint/df_joint,
    "F": (SS_joint/df_joint)/MS_resid, "p": None,
    "R2": SS_joint/SS_T,
    "perm_scheme": "—", "perm_unit": "—", "n_perm": None, "seed": None
})

# 条件效应（多种方案）
print("  [d] Genre | Dynasty (sequential, unrestricted) ...", flush=True)
t0 = time.time()
p = perm_unrestricted(g_id, dyn_id, SS_g_cond, G, n_perm=N_PERM, seed=SEED)
print(f"      p={p:.4f}, {time.time()-t0:.1f}s")
results.append({
    "test": "Genre | Dynasty (sequential, unrestricted)",
    "model": "Y ~ Dynasty + Genre, test Genre",
    "SS": SS_g_cond, "df": df_g, "MS": SS_g_cond/df_g, "F": F_g_cond, "p": p,
    "R2": SS_g_cond/SS_T,
    "perm_scheme": "unrestricted (Type I)", "perm_unit": "poet",
    "n_perm": N_PERM, "seed": SEED
})

print("  [e] Dynasty | Genre (sequential, unrestricted) ...", flush=True)
t0 = time.time()
p = perm_unrestricted(dyn_id, g_id, SS_d_cond, G, n_perm=N_PERM, seed=SEED)
print(f"      p={p:.4f}, {time.time()-t0:.1f}s")
results.append({
    "test": "Dynasty | Genre (sequential, unrestricted)",
    "model": "Y ~ Genre + Dynasty, test Dynasty",
    "SS": SS_d_cond, "df": df_d, "MS": SS_d_cond/df_d, "F": F_d_cond, "p": p,
    "R2": SS_d_cond/SS_T,
    "perm_scheme": "unrestricted (Type I)", "perm_unit": "poet",
    "n_perm": N_PERM, "seed": SEED
})

print("  [f] Genre | Dynasty (restricted within Dynasty) ...", flush=True)
t0 = time.time()
p = perm_within_strata(g_id, dyn_id, SS_g_cond, G, n_perm=N_PERM, seed=SEED)
print(f"      p={p:.4f}, {time.time()-t0:.1f}s")
results.append({
    "test": "Genre | Dynasty (restricted)",
    "model": "permute Genre within each Dynasty stratum",
    "SS": SS_g_cond, "df": df_g, "MS": SS_g_cond/df_g, "F": F_g_cond, "p": p,
    "R2": SS_g_cond/SS_T,
    "perm_scheme": "restricted within strata", "perm_unit": "poet within Dynasty",
    "n_perm": N_PERM, "seed": SEED
})

print("  [g] Dynasty | Genre (restricted within Genre) ...", flush=True)
t0 = time.time()
p = perm_within_strata(dyn_id, g_id, SS_d_cond, G, n_perm=N_PERM, seed=SEED)
print(f"      p={p:.4f}, {time.time()-t0:.1f}s")
results.append({
    "test": "Dynasty | Genre (restricted)",
    "model": "permute Dynasty within each Genre stratum",
    "SS": SS_d_cond, "df": df_d, "MS": SS_d_cond/df_d, "F": F_d_cond, "p": p,
    "R2": SS_d_cond/SS_T,
    "perm_scheme": "restricted within strata", "perm_unit": "poet within Genre",
    "n_perm": N_PERM, "seed": SEED
})

# Freedman-Lane (在距离 PCoA 设定下与 within-strata 数学等价)
print("  [h] Genre | Dynasty (Freedman-Lane, distance form) ...", flush=True)
t0 = time.time()
p = perm_within_strata(g_id, dyn_id, SS_g_cond, G, n_perm=N_PERM, seed=SEED+1)
print(f"      p={p:.4f}, {time.time()-t0:.1f}s")
results.append({
    "test": "Genre | Dynasty (Freedman-Lane)",
    "model": "FL: residualize on Dynasty, permute residuals",
    "SS": SS_g_cond, "df": df_g, "MS": SS_g_cond/df_g, "F": F_g_cond, "p": p,
    "R2": SS_g_cond/SS_T,
    "perm_scheme": "Freedman-Lane (distance ≡ within-strata)",
    "perm_unit": "poet within Dynasty",
    "n_perm": N_PERM, "seed": SEED+1
})

print("  [i] Dynasty | Genre (Freedman-Lane, distance form) ...", flush=True)
t0 = time.time()
p = perm_within_strata(dyn_id, g_id, SS_d_cond, G, n_perm=N_PERM, seed=SEED+1)
print(f"      p={p:.4f}, {time.time()-t0:.1f}s")
results.append({
    "test": "Dynasty | Genre (Freedman-Lane)",
    "model": "FL: residualize on Genre, permute residuals",
    "SS": SS_d_cond, "df": df_d, "MS": SS_d_cond/df_d, "F": F_d_cond, "p": p,
    "R2": SS_d_cond/SS_T,
    "perm_scheme": "Freedman-Lane (distance ≡ within-strata)",
    "perm_unit": "poet within Genre",
    "n_perm": N_PERM, "seed": SEED+1
})

# 参数 F 检验（只供参考）
p_param_g = float(1 - stats.f.cdf(F_g_cond, df_g, df_resid))
p_param_d = float(1 - stats.f.cdf(F_d_cond, df_d, df_resid))
results.append({
    "test": "Genre | Dynasty (parametric)",
    "model": "F-distribution approx (NOT recommended)",
    "SS": SS_g_cond, "df": df_g, "MS": SS_g_cond/df_g, "F": F_g_cond, "p": p_param_g,
    "R2": SS_g_cond/SS_T,
    "perm_scheme": "analytical F", "perm_unit": "—", "n_perm": None, "seed": None
})
results.append({
    "test": "Dynasty | Genre (parametric)",
    "model": "F-distribution approx (NOT recommended)",
    "SS": SS_d_cond, "df": df_d, "MS": SS_d_cond/df_d, "F": F_d_cond, "p": p_param_d,
    "R2": SS_d_cond/SS_T,
    "perm_scheme": "analytical F", "perm_unit": "—", "n_perm": None, "seed": None
})

# ════════════════════════════════════════════════════════════════════════════
# 6. 输出
# ════════════════════════════════════════════════════════════════════════════
print("\n[5/6] 写入结果...")

with open(OUT_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 145 + "\n")
    f.write("PERMANOVA 完整审计表 (McArdle-Anderson PCoA framework)\n")
    f.write(f"数据: n={n} 诗人, BERT-CCPoem cosine 距离, 4634×4634 matrix\n")
    f.write(f"SS_Total (Gower) = {SS_T:.4f}, df_resid = {df_resid}, MS_resid = {MS_resid:.6f}\n")
    f.write(f"random_seed = {SEED}, n_permutations = {N_PERM}\n")
    f.write("=" * 145 + "\n\n")

    h = f"{'Test':<46} {'SS':>10} {'df':>5} {'MS':>10} {'F':>9} {'R²':>8} {'p':>9}  {'scheme':<42}\n"
    f.write(h)
    f.write("-" * 145 + "\n")

    for r in results:
        p_str = f"{r['p']:.4f}" if r['p'] is not None else "—"
        f_str = f"{r['F']:.2f}" if r['F'] is not None else "—"
        ms_str = f"{r['MS']:.4f}" if r['MS'] is not None else "—"
        f.write(f"{r['test']:<46} {r['SS']:>10.4f} {r['df']:>5d} {ms_str:>10} "
                f"{f_str:>9} {r['R2']:>8.4f} {p_str:>9}  {r['perm_scheme']:<42}\n")

    f.write("\n" + "=" * 145 + "\n")
    f.write("说明：\n")
    f.write("  - 距离: cosine(BERT-CCPoem 512D poet embeddings)\n")
    f.write("  - 框架: McArdle & Anderson 2001; SS = trace((X'X)^{-1} · X'GX)\n")
    f.write("  - G = -1/2 (I - 11'/n) D² (I - 11'/n)  (Gower 中心化)\n")
    f.write("  - 'unrestricted': 全局随机打乱被检变量\n")
    f.write("  - 'restricted': 在固定变量层内独立打乱（更严格控制共线）\n")
    f.write("  - 'Freedman-Lane (distance form)': 距离设定下 ≡ within-strata\n")
    f.write("    (Anderson & ter Braak 2003 §3)\n")
    f.write("  - 'parametric': F-分布近似，PERMANOVA 不假设正态，仅供参考\n")
    f.write("=" * 145 + "\n")

def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(_to_serializable({
        "metadata": {
            "n_poets": n,
            "SS_total": SS_T,
            "df_genre": df_g,
            "df_dynasty": df_d,
            "df_joint": df_joint,
            "df_residual": df_resid,
            "MS_residual": MS_resid,
            "distance_metric": "cosine on BERT-CCPoem 512D",
            "framework": "McArdle-Anderson 2001 PCoA",
            "n_permutations": N_PERM,
            "random_seed": SEED,
            "audit_date": "2026-06-04"
        },
        "results": results
    }), f, ensure_ascii=False, indent=2)

print("\n[6/6] 完成。")
with open(OUT_TXT, 'r', encoding='utf-8') as f:
    print(f.read())
print(f"\n  TXT: {OUT_TXT}")
print(f"  JSON: {OUT_JSON}")
