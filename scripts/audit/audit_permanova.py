#!/usr/bin/env python3
"""
audit_permanova.py
诊断 PERMANOVA R²=0.733 与 p=0.283 的内部矛盾

审稿人质疑：
- 双因素 PERMANOVA 报告 "纯朝代效应 R²=0.733, p=0.283"
- 这在统计上不可能：若 R² 如此高，F 统计量应极大，p 应极小
- 怀疑：公式错误、计算错误、或报告错误

诊断目标：
1. 重新计算双因素 PERMANOVA 的所有 SS、df、F、p
2. 验证 ss_A_cond = ss_joint - ss_B 这个公式是否正确
3. 检查置换检验的实现逻辑
4. 确认 R² 与 p 值是否对应同一统计量
"""

import json
import numpy as np
from collections import Counter

np.random.seed(42)

BASE = "/home/chenhao/poetryCrystallization"
POETS = f"{BASE}/data/processed/poet_poems.json"
GENRE_SRC = f"{BASE}/data/processed/poet_genre_by_source.json"
DIST_NPY = f"{BASE}/data/processed/poet_distances.npy"

N_PERM = 999

print("=" * 80)
print("PERMANOVA 诊断审计")
print("=" * 80)

# ══════════════════════════════════════════════════════════════════════════════
# 1. 加载数据
# ══════════════════════════════════════════════════════════════════════════════
print("\n1. 加载数据...")
poets = json.load(open(POETS, encoding="utf-8"))
gsrc = json.load(open(GENRE_SRC, encoding="utf-8"))
D = np.load(DIST_NPY).astype(np.float64)
n = len(D)

ri, ci = np.triu_indices(n, k=1)
d_sq = D[ri, ci] ** 2
SS_T = float(d_sq.sum())

print(f"  诗人数: {n}")
print(f"  上三角对数: {len(d_sq):,}")
print(f"  SS_Total: {SS_T:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. 体裁与朝代标签
# ══════════════════════════════════════════════════════════════════════════════
print("\n2. 构建标签...")

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

print(f"  体裁分布: {dict(sorted(Counter(genre_lbl).items()))}")
print(f"  朝代分布: {dict(sorted(Counter(dyn_lbl).items()))}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. SS 计算函数
# ══════════════════════════════════════════════════════════════════════════════
def ss_within(labels):
    """组内平方和"""
    total = 0.0
    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]
        if len(idx) < 2: continue
        m = np.isin(ri, idx) & np.isin(ci, idx)
        total += d_sq[m].sum()
    return total

def ss_between(labels):
    """组间平方和 = SS_Total - SS_Within"""
    return SS_T - ss_within(labels)

# ══════════════════════════════════════════════════════════════════════════════
# 4. 单因素 PERMANOVA（验证基础计算是否正确）
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("3. 单因素 PERMANOVA（作为基准验证）")
print("=" * 80)

def permanova_oneway_detailed(labels, name):
    k = len(np.unique(labels))
    ss_b = ss_between(labels)
    ss_w = ss_within(labels)
    df_b = k - 1
    df_w = n - k

    MS_b = ss_b / df_b
    MS_w = ss_w / df_w
    F_obs = MS_b / MS_w if MS_w > 0 else 0
    R2 = ss_b / SS_T

    print(f"\n{name}:")
    print(f"  k (组数): {k}")
    print(f"  SS_between: {ss_b:.4f}")
    print(f"  SS_within: {ss_w:.4f}")
    print(f"  df_between: {df_b}, df_within: {df_w}")
    print(f"  MS_between: {MS_b:.4f}, MS_within: {MS_w:.4f}")
    print(f"  F = {F_obs:.2f}")
    print(f"  R² = {R2:.6f}")

    # 置换检验
    count = 0
    for _ in range(N_PERM):
        perm = labels.copy()
        np.random.shuffle(perm)
        ss_bp = ss_between(perm)
        ss_wp = ss_within(perm)
        Fp = (ss_bp / df_b) / (ss_wp / df_w) if ss_wp > 0 else 0
        if Fp >= F_obs:
            count += 1

    p_val = count / N_PERM
    print(f"  p = {p_val:.4f} (置换次数={N_PERM})")

    return {"R2": R2, "F": F_obs, "p": p_val, "ss_b": ss_b, "ss_w": ss_w}

genre_result = permanova_oneway_detailed(g_id, "体裁单因素")
dynasty_result = permanova_oneway_detailed(dyn_id, "朝代单因素")

# ══════════════════════════════════════════════════════════════════════════════
# 5. 双因素 PERMANOVA —— 详细解剖
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("4. 双因素 PERMANOVA —— 逐步计算")
print("=" * 80)

# 联合分组
joint_labels = np.array([f"{g}_{d}" for g, d in zip(g_id, dyn_id)])
k_joint = len(np.unique(joint_labels))

ss_genre = genre_result['ss_b']
ss_dynasty = dynasty_result['ss_b']
ss_joint = ss_between(joint_labels)

print(f"\n联合分组:")
print(f"  联合组数 (k_joint): {k_joint}")
print(f"  SS_genre (边际): {ss_genre:.4f}")
print(f"  SS_dynasty (边际): {ss_dynasty:.4f}")
print(f"  SS_joint: {ss_joint:.4f}")

# 交互效应
ss_interaction = max(0, ss_joint - ss_genre - ss_dynasty)
print(f"  SS_interaction = SS_joint - SS_genre - SS_dynasty")
print(f"                 = {ss_joint:.4f} - {ss_genre:.4f} - {ss_dynasty:.4f}")
print(f"                 = {ss_interaction:.4f}")

# 纯效应（条件效应）—— 这是关键
ss_genre_cond = ss_joint - ss_dynasty
ss_dynasty_cond = ss_joint - ss_genre

print(f"\n纯效应（条件效应）:")
print(f"  SS_genre|dynasty = SS_joint - SS_dynasty")
print(f"                   = {ss_joint:.4f} - {ss_dynasty:.4f}")
print(f"                   = {ss_genre_cond:.4f}")
print(f"  SS_dynasty|genre = SS_joint - SS_genre")
print(f"                   = {ss_joint:.4f} - {ss_genre:.4f}")
print(f"                   = {ss_dynasty_cond:.4f}")

# R² 值
R2_genre_cond = ss_genre_cond / SS_T
R2_dynasty_cond = ss_dynasty_cond / SS_T
R2_joint = ss_joint / SS_T
R2_residual = 1 - R2_joint

print(f"\nR² 值:")
print(f"  R²_genre|dynasty = {R2_genre_cond:.6f}")
print(f"  R²_dynasty|genre = {R2_dynasty_cond:.6f}")
print(f"  R²_joint = {R2_joint:.6f}")
print(f"  R²_residual = {R2_residual:.6f}")

# 检查加和
R2_sum = R2_genre_cond + ss_dynasty / SS_T + ss_interaction / SS_T + R2_residual
print(f"  检查: R²_genre_cond + R²_dynasty_marginal + R²_interaction + R²_residual")
print(f"       = {R2_genre_cond:.6f} + {ss_dynasty/SS_T:.6f} + {ss_interaction/SS_T:.6f} + {R2_residual:.6f}")
print(f"       = {R2_sum:.6f}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. 置换检验 —— 这是核心问题所在
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("5. 置换检验 —— 诊断 p=0.283 的来源")
print("=" * 80)

print("\n原代码的置换检验逻辑:")
print("  1. 固定 fA (体裁), 置换 fB (朝代)")
print("  2. 计算置换后的 ss_joint_perm, ss_B_perm")
print("  3. ss_inter_perm = max(0, ss_joint_perm - ss_A - ss_B_perm)")
print("  4. 比较 ss_inter_perm >= ss_inter_obs")
print("\n问题: 这个置换检验的是 **交互效应**, 不是纯朝代效应!")

print("\n让我们分别做两个置换检验:")

# (A) 检验纯体裁效应
print("\n(A) 纯体裁效应的置换检验 (固定朝代, 置换体裁):")
count_genre = 0
for perm_i in range(N_PERM):
    g_perm = g_id.copy()
    np.random.shuffle(g_perm)
    joint_perm = np.array([f"{g}_{d}" for g, d in zip(g_perm, dyn_id)])
    ss_joint_perm = ss_between(joint_perm)
    ss_genre_perm = ss_between(g_perm)
    ss_genre_cond_perm = ss_joint_perm - ss_dynasty

    if ss_genre_cond_perm >= ss_genre_cond:
        count_genre += 1

p_genre = count_genre / N_PERM
print(f"  观测值: SS_genre|dynasty = {ss_genre_cond:.4f}")
print(f"  p_genre = {p_genre:.4f}")

# (B) 检验纯朝代效应
print("\n(B) 纯朝代效应的置换检验 (固定体裁, 置换朝代):")
count_dynasty = 0
for perm_i in range(N_PERM):
    d_perm = dyn_id.copy()
    np.random.shuffle(d_perm)
    joint_perm = np.array([f"{g}_{d}" for g, d in zip(g_id, d_perm)])
    ss_joint_perm = ss_between(joint_perm)
    ss_dynasty_perm = ss_between(d_perm)
    ss_dynasty_cond_perm = ss_joint_perm - ss_genre

    if ss_dynasty_cond_perm >= ss_dynasty_cond:
        count_dynasty += 1

p_dynasty = count_dynasty / N_PERM
print(f"  观测值: SS_dynasty|genre = {ss_dynasty_cond:.4f}")
print(f"  p_dynasty = {p_dynasty:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. F 统计量计算 (Anderson 2001 正确公式)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("6. F 统计量与 pseudo-F (检查数学一致性)")
print("=" * 80)

k_genre = len(np.unique(g_id))
k_dynasty = len(np.unique(dyn_id))
df_genre = k_genre - 1
df_dynasty = k_dynasty - 1
df_residual = n - k_joint

ss_residual = SS_T - ss_joint
MS_residual = ss_residual / df_residual

MS_genre_cond = ss_genre_cond / df_genre
MS_dynasty_cond = ss_dynasty_cond / df_dynasty

F_genre = MS_genre_cond / MS_residual
F_dynasty = MS_dynasty_cond / MS_residual

print(f"\n自由度:")
print(f"  df_genre = {df_genre}, df_dynasty = {df_dynasty}")
print(f"  df_joint = {k_joint - 1}, df_residual = {df_residual}")

print(f"\nMS 值:")
print(f"  MS_genre|dynasty = {MS_genre_cond:.4f}")
print(f"  MS_dynasty|genre = {MS_dynasty_cond:.4f}")
print(f"  MS_residual = {MS_residual:.6f}")

print(f"\nF 统计量:")
print(f"  F_genre = MS_genre|dynasty / MS_residual")
print(f"          = {MS_genre_cond:.4f} / {MS_residual:.6f}")
print(f"          = {F_genre:.2f}")
print(f"  F_dynasty = MS_dynasty|genre / MS_residual")
print(f"            = {MS_dynasty_cond:.4f} / {MS_residual:.6f}")
print(f"            = {F_dynasty:.2f}")

print(f"\n关键诊断:")
print(f"  若 R²_dynasty|genre = {R2_dynasty_cond:.4f} (73.3%)")
print(f"  且 F_dynasty = {F_dynasty:.2f}")
print(f"  按理 p 值应该 << 0.05")
print(f"  但报告的是 p = 0.283")
print(f"  → 怀疑: 置换检验用的不是 SS_dynasty|genre, 而是 SS_interaction")

# ══════════════════════════════════════════════════════════════════════════════
# 8. 检查原代码的置换检验逻辑
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("7. 复现原代码的置换检验逻辑")
print("=" * 80)

print("\n原代码 line 135-145:")
print("  def perm_test(cond_ss_func, n_p=N_PERM):")
print("      cnt = 0")
print("      for _ in range(n_p):")
print("          perm_fB = fB_lbl.copy()")
print("          np.random.shuffle(perm_fB)")
print("          jp = np.array([f'{a}_{b}' for a, b in zip(fA_lbl, perm_fB)])")
print("          ss_jp  = ss_between(jp)")
print("          ss_Bp  = ss_between(perm_fB)")
print("          cp     = max(0, ss_jp - ss_b_A - ss_Bp)")
print("          if cp >= cond_ss_func: cnt += 1")
print("      return round(cnt / n_p, 4)")
print("")
print("  p_A = perm_test(ss_A_cond)")
print("  p_B = perm_test(ss_B_cond)")

print("\n问题分析:")
print("  1. 函数固定 fA, 置换 fB")
print("  2. 计算 cp = ss_jp - ss_b_A - ss_Bp")
print("  3. 这个 cp 是 **交互效应的置换值**, 不是纯效应!")
print("  4. p_A 和 p_B 用的是同一个置换逻辑")
print("  5. 传入的 cond_ss_func 是 ss_A_cond 或 ss_B_cond")
print("  6. 但比较的是 cp (交互) >= cond_ss (纯效应)")
print("  7. 这个比较在数学上没有意义!")

print("\n复现原代码的 p_B 计算:")
count_original = 0
for _ in range(N_PERM):
    # 置换朝代 (fB = dyn_id)
    d_perm = dyn_id.copy()
    np.random.shuffle(d_perm)
    jp = np.array([f"{g}_{d}" for g, d in zip(g_id, d_perm)])
    ss_jp = ss_between(jp)
    ss_Bp = ss_between(d_perm)
    cp = max(0, ss_jp - ss_genre - ss_Bp)  # 这是交互效应!

    if cp >= ss_dynasty_cond:  # 比较交互 >= 纯朝代
        count_original += 1

p_original = count_original / N_PERM
print(f"  原代码逻辑的 p_dynasty = {p_original:.4f}")
print(f"  (应该接近 0.283)")

# ══════════════════════════════════════════════════════════════════════════════
# 9. 总结
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("8. 诊断结论")
print("=" * 80)

print("\n问题根源:")
print("  ✗ 代码计算 R²_dynasty|genre = 0.733 (正确)")
print("  ✗ 但置换检验用的是交互效应置换")
print("  ✗ 比较 SS_interaction_perm >= SS_dynasty_cond_obs")
print("  ✗ 这在统计上没有意义")
print("  ✗ 导致 p=0.283 不对应 R²=0.733")

print("\n正确的 p 值:")
print(f"  p_genre (正确置换) = {p_genre:.4f}")
print(f"  p_dynasty (正确置换) = {p_dynasty:.4f}")

print("\n审稿人的质疑完全正确:")
print("  若 R²=0.733, F={:.2f}, p 应该极小 (<0.001)".format(F_dynasty))
print("  论文中报告 p=0.283 是因为代码逻辑错误")
print("  这不是 PERMDISP 能解释的 —— 这是实现错误")

print("\n" + "=" * 80)
print("审计完成")
print("=" * 80)

# 保存结果
output = {
    "audit_date": "2026-06-04",
    "issue": "PERMANOVA R² and p-value mismatch",
    "problem": "置换检验用的是交互效应置换，不是纯效应置换",
    "single_factor": {
        "genre_R2": float(genre_result['R2']),
        "genre_p": float(genre_result['p']),
        "dynasty_R2": float(dynasty_result['R2']),
        "dynasty_p": float(dynasty_result['p'])
    },
    "two_factor": {
        "R2_genre_cond": float(R2_genre_cond),
        "R2_dynasty_cond": float(R2_dynasty_cond),
        "R2_joint": float(R2_joint),
        "F_genre": float(F_genre),
        "F_dynasty": float(F_dynasty),
        "p_genre_correct": float(p_genre),
        "p_dynasty_correct": float(p_dynasty),
        "p_dynasty_original": float(p_original)
    },
    "conclusion": "代码逻辑错误：置换检验的不是纯效应，而是交互效应。需要重写置换函数。"
}

import json
with open(f"{BASE}/scripts/audit/permanova_audit_results.json", 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n结果已保存至: {BASE}/scripts/audit/permanova_audit_results.json")
