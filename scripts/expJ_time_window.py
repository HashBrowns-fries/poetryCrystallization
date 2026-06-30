#!/usr/bin/env python3
"""
expJ_time_window.py
实验 J: 时间窗口稳健性检验

检验"体裁>朝代"效应是否在不同历史时期保持一致：
  1. 唐宋子集（约 1,030 位诗人）
  2. 明清子集（约 1,027 位诗人）

在两个时期上分别运行 PERMANOVA，对比 R²、p 值、Cohen's d。
"""

import json
import numpy as np
import warnings
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import matplotlib as mpl

warnings.filterwarnings('ignore')

# 设置中文字体
mpl.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data"
N_PERM = 999

# ═════════════════════════════════════════════════════════════════════════════
# 1. 加载数据
# ═════════════════════════════════════════════════════════════════════════════

print("=== 加载嵌入和元数据 ===")
embeddings = np.load(DATA_DIR / "processed/poet_embeddings.npy")

# 加载诗人信息（dynasty 从 poet_poems.json 获取）
print("加载 poet_poems.json（含朝代信息）...")
with open(DATA_DIR / "processed/poet_poems.json") as f:
    poets_list = json.load(f)

# 构建 name -> dynasty 映射
name_to_dynasty = {poet["name"]: poet["dynasty"] for poet in poets_list}

# 加载体裁信息（使用 genre_by_source 以复用原论文的体裁分类逻辑）
with open(DATA_DIR / "processed/poet_genre_by_source.json") as f:
    genre_data = json.load(f)

# 体裁分类函数（与原论文一致：ci>25% → ci, qu>25% → qu, 否则 shi）
def dom_genre(name):
    g = genre_data.get(name, {})
    s = g.get("shi", 0)
    c = g.get("ci", 0)
    q = g.get("qu", 0)
    f = g.get("fu", 0)
    t = s + c + q + f
    if t == 0:
        return "shi"
    if c / t > 0.25:
        return "ci"
    if q / t > 0.25:
        return "qu"
    return "shi"

# 构建 poet_info: name -> (genre, dynasty)
poet_info = {}
for poet in poets_list:
    name = poet["name"]
    dynasty = poet.get("dynasty", "unknown")
    genre = dom_genre(name)

    # 只保留有朝代信息的诗人
    if dynasty not in ["unknown", "其他", "当代", None]:
        poet_info[name] = {"genre": genre, "dynasty": dynasty}

print(f"总诗人数: {len(poets_list)}")
print(f"有效标签（genre + dynasty）: {len(poet_info)}\n")

# ═════════════════════════════════════════════════════════════════════════════
# 2. 定义朝代分组
# ═════════════════════════════════════════════════════════════════════════════

TANG_SONG = ["唐", "五代十国", "五代", "宋"]
MING_QING = ["明", "清"]

def group_by_period(poet_info, poets_list):
    """按时期分组诗人，返回 {name: {genre, dynasty, index}}"""
    # 建立 name -> index 映射
    name_to_idx = {p["name"]: i for i, p in enumerate(poets_list)}

    tang_song_poets = {}
    ming_qing_poets = {}

    for name, info in poet_info.items():
        dynasty = info["dynasty"]
        if name not in name_to_idx:
            continue

        entry = {
            "genre": info["genre"],
            "dynasty": dynasty,
            "index": name_to_idx[name]
        }

        if dynasty in TANG_SONG:
            tang_song_poets[name] = entry
        elif dynasty in MING_QING:
            ming_qing_poets[name] = entry

    return tang_song_poets, ming_qing_poets

tang_song_poets, ming_qing_poets = group_by_period(poet_info, poets_list)

print("=== 时间窗口划分 ===")
print(f"唐宋时期: {len(tang_song_poets)} 位诗人")
print(f"明清时期: {len(ming_qing_poets)} 位诗人\n")

# ═════════════════════════════════════════════════════════════════════════════
# 3. PERMANOVA 分析函数（手动实现，与原论文一致）
# ═════════════════════════════════════════════════════════════════════════════

def permanova_manual(D, labels, n_perm=N_PERM, seed=42):
    """
    手动 PERMANOVA（基于距离矩阵 D 与组标签 labels）
    使用 d² 上三角累计 SSB / SSW，与原论文 40_genre_dominance.py 一致
    """
    rng = np.random.default_rng(seed)
    n = len(D)
    ri, ci = np.triu_indices(n, k=1)
    d_sq = D[ri, ci] ** 2
    SS_T = float(d_sq.sum())

    labels = np.asarray(labels)
    uniq = np.unique(labels)
    k = len(uniq)
    if k < 2 or SS_T == 0:
        return {"R2": np.nan, "F": np.nan, "p": np.nan, "k": k}

    def ss_within(lbl):
        total = 0.0
        for u in np.unique(lbl):
            idx = np.where(lbl == u)[0]
            if len(idx) < 2:
                continue
            m = np.isin(ri, idx) & np.isin(ci, idx)
            total += d_sq[m].sum()
        return total

    ss_w = ss_within(labels)
    ss_b = SS_T - ss_w
    df_b = k - 1
    df_w = n - k
    F_obs = (ss_b / df_b) / (ss_w / df_w) if ss_w > 0 and df_w > 0 else 0.0
    R2 = ss_b / SS_T

    # 置换检验
    cnt = 0
    for _ in range(n_perm):
        perm = labels.copy()
        rng.shuffle(perm)
        ss_wp = ss_within(perm)
        ss_bp = SS_T - ss_wp
        Fp = (ss_bp / df_b) / (ss_wp / df_w) if ss_wp > 0 and df_w > 0 else 0.0
        if Fp >= F_obs:
            cnt += 1
    p = cnt / n_perm

    return {"R2": float(R2), "F": float(F_obs), "p": float(p), "k": k,
            "df_b": df_b, "df_w": df_w}


def run_permanova_for_period(poets_dict, embeddings, period_name):
    """
    在给定时期的诗人子集上运行 PERMANOVA
    poets_dict: {name: {genre, dynasty, index}}
    """
    print(f"--- {period_name} ---")

    names = list(poets_dict.keys())
    indices = [poets_dict[n]["index"] for n in names]
    subset_embeddings = embeddings[indices]

    genres = np.array([poets_dict[n]["genre"] for n in names])
    dynasties = np.array([poets_dict[n]["dynasty"] for n in names])

    print(f"  有效诗人数: {len(names)}")
    print(f"  体裁分布: ci={int((genres=='ci').sum())}, "
          f"shi={int((genres=='shi').sum())}, qu={int((genres=='qu').sum())}")
    print(f"  朝代分布: {dict(zip(*np.unique(dynasties, return_counts=True)))}")

    # 计算距离矩阵
    dist_matrix = euclidean_distances(subset_embeddings)

    # PERMANOVA: 体裁效应
    res_genre = permanova_manual(dist_matrix, genres, n_perm=N_PERM)
    r2_genre = res_genre["R2"]
    p_genre = res_genre["p"]
    print(f"  体裁 PERMANOVA: R² = {r2_genre:.4f}, F = {res_genre['F']:.2f}, "
          f"p = {p_genre:.4f}, k = {res_genre['k']}")

    # PERMANOVA: 朝代效应
    if len(np.unique(dynasties)) > 1:
        res_dynasty = permanova_manual(dist_matrix, dynasties, n_perm=N_PERM)
        r2_dynasty = res_dynasty["R2"]
        p_dynasty = res_dynasty["p"]
        print(f"  朝代 PERMANOVA: R² = {r2_dynasty:.4f}, F = {res_dynasty['F']:.2f}, "
              f"p = {p_dynasty:.4f}, k = {res_dynasty['k']}")
    else:
        r2_dynasty, p_dynasty = np.nan, np.nan
        print(f"  朝代种类过少，跳过朝代 PERMANOVA")

    # Cohen's d（组间 vs 组内距离）
    def cohen_d_genre(D, lbl):
        n = len(lbl)
        within_dists = []
        between_dists = []
        for i in range(n):
            for j in range(i+1, n):
                if lbl[i] == lbl[j]:
                    within_dists.append(D[i, j])
                else:
                    between_dists.append(D[i, j])
        if not within_dists or not between_dists:
            return np.nan
        mean_within = np.mean(within_dists)
        mean_between = np.mean(between_dists)
        pooled_std = np.sqrt((np.var(within_dists) + np.var(between_dists)) / 2)
        return (mean_between - mean_within) / pooled_std if pooled_std > 0 else np.nan

    cohens_d = cohen_d_genre(dist_matrix, genres)
    print(f"  Cohen's d (genre): {cohens_d:.4f}\n")

    return {
        "period": period_name,
        "n_poets": len(names),
        "n_ci": int((genres == "ci").sum()),
        "n_shi": int((genres == "shi").sum()),
        "n_qu": int((genres == "qu").sum()),
        "r2_genre": float(r2_genre),
        "p_genre": float(p_genre),
        "F_genre": float(res_genre["F"]),
        "r2_dynasty": float(r2_dynasty) if not np.isnan(r2_dynasty) else None,
        "p_dynasty": float(p_dynasty) if not np.isnan(p_dynasty) else None,
        "cohens_d": float(cohens_d) if not np.isnan(cohens_d) else None,
        "poet_names": names,
        "embeddings_shape": list(subset_embeddings.shape)
    }

# ═════════════════════════════════════════════════════════════════════════════
# 4. 运行两个时期的分析
# ═════════════════════════════════════════════════════════════════════════════

print("=== 运行 PERMANOVA（两个时期）===\n")

results = {}
results["tang_song"] = run_permanova_for_period(
    tang_song_poets, embeddings, "唐宋时期"
)
results["ming_qing"] = run_permanova_for_period(
    ming_qing_poets, embeddings, "明清时期"
)

# ═════════════════════════════════════════════════════════════════════════════
# 5. 对比表格
# ═════════════════════════════════════════════════════════════════════════════

print("=== 时间窗口稳健性对比 ===\n")
header = "{:<12} {:<8} {:<12} {:<12} {:<12}".format("时期", "诗人数", "R2(体裁)", "p(体裁)", "Cohen_d")
print(header)
print("-" * 60)

for key in ["tang_song", "ming_qing"]:
    r = results[key]
    period_name = r["period"]
    n = r["n_poets"]
    r2 = r["r2_genre"]
    p = r["p_genre"]
    d = r["cohens_d"]

    p_str = f"{p:.4f}" if p >= 0.001 else "<0.001"
    d_str = f"{d:.4f}" if d is not None else "N/A"

    print(f"{period_name:<12} {n:<8} {r2:<12.4f} {p_str:<12} {d_str:<12}")

print("\n结论:")
print("  如果两个时期的 R²(体裁) 相近且都显著（p < 0.001），")
print("  则说明'体裁>朝代'效应在不同历史时期保持稳健。\n")

# ═════════════════════════════════════════════════════════════════════════════
# 6. 保存结果
# ═════════════════════════════════════════════════════════════════════════════

output_path = DATA_DIR / "processed/expJ_time_window.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)

print(f"✅ 结果已保存至: {output_path}")

# ═════════════════════════════════════════════════════════════════════════════
# 7. 可视化（可选）：两个时期的 PCA 对比
# ═════════════════════════════════════════════════════════════════════════════

def plot_pca_comparison(results, embeddings):
    """绘制两个时期的 PCA 对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    genre_colors = {"ci": "#E74C3C", "shi": "#3498DB", "qu": "#2ECC71"}

    for idx, (key, period_name) in enumerate([("tang_song", "唐宋"), ("ming_qing", "明清")]):
        ax = axes[idx]
        r = results[key]
        names = r["poet_names"]

        # 从原始embeddings筛选对应诗人
        indices = [i for i, p in enumerate(poets_list) if p["name"] in names]
        subset_embeddings = embeddings[indices]

        # 构建name -> genre映射
        name_to_genre = {n: poet_info[n]["genre"] for n in names}
        genres = [name_to_genre[poets_list[i]["name"]] for i in indices]

        # PCA
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(subset_embeddings)

        # 绘制
        for genre, color in genre_colors.items():
            mask = np.array(genres) == genre
            if mask.sum() > 0:
                ax.scatter(coords[mask, 0], coords[mask, 1],
                          c=color, label=genre.upper(), alpha=0.6, s=20)

        ax.set_title(f"{period_name} (n={r['n_poets']}, R²={r['r2_genre']:.3f})",
                    fontsize=14, fontweight='bold')
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=12)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = BASE / "data/figures/fig_time_window.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"✅ 可视化已保存至: {fig_path}")
    plt.close()

plot_pca_comparison(results, embeddings)

print("\n=== 实验 J 完成 ===")
