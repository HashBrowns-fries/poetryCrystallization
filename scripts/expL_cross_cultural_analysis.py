#!/usr/bin/env python3
"""
expL_cross_cultural_analysis.py
实验 L: 跨文化对照分析

分析：
1. 联合 PCA：中国诗人（ci/shi/qu）+ 西班牙诗人（sonnet）
2. PERMANOVA：文化效应（中国 vs 西班牙）vs 体裁效应（within-culture）
3. 跨文化距离矩阵：sonnet vs ci/shi/qu
4. 可视化：双文化语义空间
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import mannwhitneyu
import matplotlib.font_manager as fm

# 中文字体
for _p in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
           '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc']:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
_cjk = [f.name for f in fm.fontManager.ttflist if 'CJK' in f.name]
if _cjk:
    plt.rcParams['font.sans-serif'] = [_cjk[0]]
plt.rcParams['axes.unicode_minus'] = False

BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data"

# ═══════════════════════════════════════════════════════════════════════════
# 1. 加载数据
# ═══════════════════════════════════════════════════════════════════════════

print("=== 加载中国诗人数据 ===")
# 中国诗人嵌入 (BERT-CCPoem 512D)
chinese_embeddings = np.load(DATA_DIR / "processed/poet_embeddings.npy")
with open(DATA_DIR / "processed/poet_poems.json") as f:
    chinese_poets = json.load(f)

# 体裁标签
with open(DATA_DIR / "processed/poet_genre_by_source.json") as f:
    genre_data = json.load(f)

def dom_genre(name):
    g = genre_data.get(name, {})
    s, c, q, f = g.get("shi", 0), g.get("ci", 0), g.get("qu", 0), g.get("fu", 0)
    t = s + c + q + f
    if t == 0:
        return "shi"
    if c / t > 0.25:
        return "ci"
    if q / t > 0.25:
        return "qu"
    return "shi"

chinese_genres = [dom_genre(p['name']) for p in chinese_poets]
chinese_names = [p['name'] for p in chinese_poets]

print(f"中国诗人数: {len(chinese_poets)}")
print(f"嵌入维度: {chinese_embeddings.shape[1]}")

print("\n=== 加载西班牙诗人数据 ===")
# 西班牙诗人嵌入 (XLM-RoBERTa 768D)
spanish_embeddings = np.load(DATA_DIR / "processed/spanish_poet_embeddings.npy")
with open(DATA_DIR / "processed/spanish_poet_names.json") as f:
    spanish_names = json.load(f)

print(f"西班牙诗人数: {len(spanish_names)}")
print(f"嵌入维度: {spanish_embeddings.shape[1]}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. 维度对齐（PCA降维到相同维度）
# ═══════════════════════════════════════════════════════════════════════════

print("\n=== 维度对齐 ===")
# 将两者都降至相同维度（取 min(中国维度, 西班牙维度, 西班牙样本数-1)）
target_dim = min(chinese_embeddings.shape[1], spanish_embeddings.shape[1], len(spanish_names) - 1)
print(f"目标维度: {target_dim}")

if chinese_embeddings.shape[1] != target_dim:
    pca_cn = PCA(n_components=target_dim)
    chinese_embeddings_aligned = pca_cn.fit_transform(chinese_embeddings)
    print(f"中国诗人：{chinese_embeddings.shape[1]}D → {target_dim}D")
else:
    chinese_embeddings_aligned = chinese_embeddings

if spanish_embeddings.shape[1] != target_dim:
    pca_es = PCA(n_components=target_dim)
    spanish_embeddings_aligned = pca_es.fit_transform(spanish_embeddings)
    print(f"西班牙诗人：{spanish_embeddings.shape[1]}D → {target_dim}D")
else:
    spanish_embeddings_aligned = spanish_embeddings

# ═══════════════════════════════════════════════════════════════════════════
# 3. 联合 PCA 投影
# ═══════════════════════════════════════════════════════════════════════════

print("\n=== 联合 PCA 投影 ===")
# 合并所有诗人
all_embeddings = np.vstack([chinese_embeddings_aligned, spanish_embeddings_aligned])
all_labels = ['中国'] * len(chinese_names) + ['西班牙'] * len(spanish_names)
all_genres = chinese_genres + ['sonnet'] * len(spanish_names)

# PCA 降至 2D
pca_joint = PCA(n_components=2)
pca_coords = pca_joint.fit_transform(all_embeddings)

print(f"联合 PCA 解释方差: PC1={pca_joint.explained_variance_ratio_[0]:.3f}, "
      f"PC2={pca_joint.explained_variance_ratio_[1]:.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. 跨文化距离分析
# ═══════════════════════════════════════════════════════════════════════════

print("\n=== 跨文化距离分析 ===")

# 计算余弦距离
dist_matrix = cosine_distances(all_embeddings)

# 分组索引
n_cn = len(chinese_names)
n_es = len(spanish_names)

cn_indices = np.arange(n_cn)
es_indices = np.arange(n_cn, n_cn + n_es)

# 组内距离
cn_within = dist_matrix[cn_indices][:, cn_indices]
cn_within_dist = cn_within[np.triu_indices_from(cn_within, k=1)]

es_within = dist_matrix[es_indices][:, es_indices]
es_within_dist = es_within[np.triu_indices_from(es_within, k=1)]

# 跨文化距离
cross_dist = dist_matrix[cn_indices][:, es_indices].flatten()

print(f"中国组内距离: {cn_within_dist.mean():.4f} ± {cn_within_dist.std():.4f}")
print(f"西班牙组内距离: {es_within_dist.mean():.4f} ± {es_within_dist.std():.4f}")
print(f"跨文化距离: {cross_dist.mean():.4f} ± {cross_dist.std():.4f}")

# Mann-Whitney U 检验
u_stat, p_val = mannwhitneyu(cross_dist, cn_within_dist)
print(f"\n跨文化 vs 中国组内: U={u_stat:.0f}, p={p_val:.2e}")

# ═══════════════════════════════════════════════════════════════════════════
# 5. 体裁级跨文化距离
# ═══════════════════════════════════════════════════════════════════════════

print("\n=== 体裁级跨文化距离（sonnet vs ci/shi/qu）===")

genre_distances = {}
for genre in ['ci', 'shi', 'qu']:
    genre_indices = [i for i, g in enumerate(chinese_genres) if g == genre]
    if not genre_indices:
        continue

    # sonnet 到该体裁的距离
    dist_to_genre = dist_matrix[es_indices][:, genre_indices].flatten()
    genre_distances[genre] = {
        'mean': dist_to_genre.mean(),
        'std': dist_to_genre.std(),
        'n_pairs': len(dist_to_genre)
    }
    print(f"sonnet → {genre}: {dist_to_genre.mean():.4f} ± {dist_to_genre.std():.4f} (n={len(dist_to_genre)})")

# ═══════════════════════════════════════════════════════════════════════════
# 6. 可视化
# ═══════════════════════════════════════════════════════════════════════════

print("\n=== 生成可视化 ===")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# (a) 文化分组
ax = axes[0]
colors_culture = {'中国': '#E74C3C', '西班牙': '#3498DB'}
for culture in ['中国', '西班牙']:
    mask = np.array([l == culture for l in all_labels])
    ax.scatter(pca_coords[mask, 0], pca_coords[mask, 1],
               c=colors_culture[culture], label=culture, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax.set_xlabel(f"PC1 ({pca_joint.explained_variance_ratio_[0]:.1%})", fontsize=12)
ax.set_ylabel(f"PC2 ({pca_joint.explained_variance_ratio_[1]:.1%})", fontsize=12)
ax.set_title("(a) 跨文化语义空间：中国 vs 西班牙", fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# (b) 体裁细分
ax = axes[1]
colors_genre = {'ci': '#E74C3C', 'shi': '#F39C12', 'qu': '#27AE60', 'sonnet': '#3498DB'}
for genre in ['ci', 'shi', 'qu', 'sonnet']:
    mask = np.array([g == genre for g in all_genres])
    label_map = {'ci': 'ci (词)', 'shi': 'shi (诗)', 'qu': 'qu (曲)', 'sonnet': 'sonnet (十四行诗)'}
    ax.scatter(pca_coords[mask, 0], pca_coords[mask, 1],
               c=colors_genre[genre], label=label_map[genre], alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax.set_xlabel(f"PC1 ({pca_joint.explained_variance_ratio_[0]:.1%})", fontsize=12)
ax.set_ylabel(f"PC2 ({pca_joint.explained_variance_ratio_[1]:.1%})", fontsize=12)
ax.set_title("(b) 体裁细分：ci/shi/qu vs sonnet", fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = BASE / "data/figures/fig_cross_cultural.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_path.with_suffix('.pdf'), bbox_inches='tight')
print(f"✅ 已保存: {fig_path}")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# 7. 保存结果
# ═══════════════════════════════════════════════════════════════════════════

results = {
    'metadata': {
        'n_chinese': n_cn,
        'n_spanish': n_es,
        'embedding_dim': target_dim
    },
    'distances': {
        'chinese_within': {
            'mean': float(cn_within_dist.mean()),
            'std': float(cn_within_dist.std())
        },
        'spanish_within': {
            'mean': float(es_within_dist.mean()),
            'std': float(es_within_dist.std())
        },
        'cross_cultural': {
            'mean': float(cross_dist.mean()),
            'std': float(cross_dist.std())
        },
        'u_test_p': float(p_val)
    },
    'genre_distances': {
        genre: {k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in data.items()}
        for genre, data in genre_distances.items()
    },
    'pca': {
        'explained_variance_ratio': pca_joint.explained_variance_ratio_.tolist()
    }
}

output_path = DATA_DIR / "processed/expL_cross_cultural.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✅ 结果已保存至: {output_path}")
print("\n=== 实验 L 完成 ===")
