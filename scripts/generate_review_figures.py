"""
generate_review_figures.py
按审稿人要求补充的图表：
1. PCA散点图 + 95%置信椭圆（真实数据，区分shi/ci/qu三色）
2. Louvain纯度小提琴图（体裁 vs 朝代，标注随机期望基线）
"""
import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.font_manager as fm

# ── 中文字体 ─────────────────────────────────────────────────
for p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
          "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"]:
    try: fm.fontManager.addfont(p)
    except: pass
cjk = [f.name for f in fm.fontManager.ttflist if "CJK" in f.name]
FONT_CJK = cjk[0] if cjk else None
if FONT_CJK:
    plt.rcParams['font.sans-serif'] = [FONT_CJK]
plt.rcParams['axes.unicode_minus'] = False

BASE = "/home/chenhao/poetryCrystallization"
DATA = f"{BASE}/data/processed"
FIGS = f"{BASE}/data/figures"

# ═════════════════════════════════════════════════════════════════════════
# 1. Load data
# ═════════════════════════════════════════════════════════════════════════
embeddings = np.load(f"{DATA}/poet_embeddings.npy").astype(np.float32)
poets = json.load(open(f"{DATA}/poet_poems.json"))
gsrc  = json.load(open(f"{DATA}/poet_genre_by_source.json"))

def dom_genre(name):
    g = gsrc.get(name, {})
    s, c, q = g.get("shi",0), g.get("ci",0), g.get("qu",0)
    t = s + c + q
    if t == 0: return "shi"
    if c/t > 0.25: return "ci"
    if q/t > 0.25: return "qu"
    return "shi"

genres = np.array([dom_genre(p["name"]) for p in poets])
dynasties = np.array([p.get("dynasty", "未知") for p in poets])

# ═════════════════════════════════════════════════════════════════════════
# 2. PCA computation
# ═════════════════════════════════════════════════════════════════════════
print("Computing PCA...")
from sklearn.decomposition import PCA
pca = PCA(n_components=5, random_state=42)
coords = pca.fit_transform(embeddings)
pc1, pc2 = coords[:, 0], coords[:, 1]
var1 = pca.explained_variance_ratio_[0] * 100
var2 = pca.explained_variance_ratio_[1] * 100
print(f"  PC1: {var1:.1f}%, PC2: {var2:.1f}%")

# ═════════════════════════════════════════════════════════════════════════
# 3. PCA scatter with 95% confidence ellipses
# ═════════════════════════════════════════════════════════════════════════
print("Generating PCA scatter with confidence ellipses...")

GENRE_COLORS = {"shi": "#4472C4", "ci": "#ED7D31", "qu": "#70AD47"}
GENRE_LABELS = {"shi": "shi (诗)", "ci": "ci (词)", "qu": "qu (曲)"}

fig, ax = plt.subplots(figsize=(12, 9))

for g in ["shi", "ci", "qu"]:
    mask = genres == g
    x_full, y_full = pc1[mask], pc2[mask]
    n = mask.sum()
    if n < 3:
        continue

    # Sample shi for visual clarity (too many points)
    if g == "shi" and n > 2000:
        sample_idx = np.random.choice(n, 1500, replace=False)
        x, y = x_full[sample_idx], y_full[sample_idx]
    else:
        x, y = x_full, y_full

    # Scatter — ci and qu on top
    zord = 4 if g in ("ci", "qu") else 2
    size = 25 if g in ("ci", "qu") else 8
    alpha = 0.7 if g in ("ci", "qu") else 0.25
    ax.scatter(x, y, c=GENRE_COLORS[g], s=size, alpha=alpha,
               edgecolors='none', zorder=zord, label=f"{GENRE_LABELS[g]} (n={n})")

    # 95% confidence ellipse (use full data, not sampled)
    cov = np.cov(x_full, y_full)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    # For small n (<30), the chi2 approximation overestimates; cap at 2*sd
    if n >= 30:
        width, height = 2 * np.sqrt(eigenvalues * 5.991)  # chi2(0.95, df=2)
    else:
        width, height = 4 * np.sqrt(eigenvalues)  # ~2*sd for small n
    ellipse = Ellipse(xy=(np.mean(x_full), np.mean(y_full)),
                      width=width, height=height, angle=angle,
                      facecolor=GENRE_COLORS[g], alpha=0.12,
                      edgecolor=GENRE_COLORS[g], linewidth=2, linestyle='-', zorder=3)
    ax.add_patch(ellipse)

# Annotations
ax.set_xlabel(f"PC1 ({var1:.1f}%) — 体裁-历史复合轴", fontsize=13)
ax.set_ylabel(f"PC2 ({var2:.1f}%) — 元曲戏剧轴", fontsize=13)
ax.set_title("语义空间PCA散点图：shi/ci/qu三体95%置信椭圆", fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.15, linestyle='--')

# Inset: variance ratio bar
ax_inset = ax.inset_axes([0.72, 0.68, 0.25, 0.25])
ax_inset.bar(range(5), pca.explained_variance_ratio_[:5] * 100,
             color=['#4472C4', '#ED7D31', '#70AD47', '#A5A5A5', '#A5A5A5'])
ax_inset.set_xticks(range(5))
ax_inset.set_xticklabels(['PC1','PC2','PC3','PC4','PC5'], fontsize=7)
ax_inset.set_ylabel('% Variance', fontsize=7)
ax_inset.set_title('Explained Variance', fontsize=8)

fig.savefig(f"{FIGS}/fig_pca_ellipse.pdf", bbox_inches='tight', facecolor='white')
fig.savefig(f"{FIGS}/fig_pca_ellipse.png", bbox_inches='tight', facecolor='white', dpi=200)
plt.close()
print("  → fig_pca_ellipse.pdf/png")

# ═════════════════════════════════════════════════════════════════════════
# 4. Louvain purity violin plot
# ═════════════════════════════════════════════════════════════════════════
print("Generating Louvain purity violin plot...")

import community as community_louvain
import networkx as nx

# Reconstruct Louvain partition
D = np.load(f"{DATA}/poet_distances_filtered.npy").astype(np.float64)
K = 20
D_sorted = np.argsort(D, axis=1)
knn = D_sorted[:, 1:K+1]
G = nx.Graph()
for i in range(len(poets)):
    for j in knn[i]:
        if i != j:
            w = max(0.0, 1.0 - D[i,j])
            if w > 0.15:
                G.add_edge(i, j, weight=w)

partition = community_louvain.best_partition(G, random_state=42)
from collections import defaultdict, Counter

# Per-community purity
comm_genre = defaultdict(lambda: defaultdict(int))
comm_dyn   = defaultdict(lambda: defaultdict(int))
for i, p in enumerate(poets):
    c = partition[i]
    comm_genre[c][genres[i]] += 1
    comm_dyn[c][dynasties[i]] += 1

def purity(dist):
    t = sum(dist.values())
    return max(dist.values())/t if t > 0 else 0

# Collect per-community purities (≥10 poets)
genre_purities = []
dyn_purities = []
for c in set(partition.values()):
    size = sum(comm_genre[c].values())
    if size >= 10:
        genre_purities.append(purity(comm_genre[c]))
        dyn_purities.append(purity(comm_dyn[c]))

# Load null baseline
lpn = json.load(open(f"{DATA}/louvain_purity_null.json"))
null_mean = lpn.get("null_mean", 0.935)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

# Genre violin
parts_g = ax1.violinplot([genre_purities], positions=[0], showmeans=True,
                          showmedians=True, widths=0.6)
for pc in parts_g['bodies']:
    pc.set_facecolor('#4472C4')
    pc.set_alpha(0.6)
ax1.scatter([0]*len(genre_purities), genre_purities, s=15, alpha=0.4,
            c='#4472C4', edgecolors='white', linewidth=0.5, zorder=3)
ax1.axhline(y=null_mean, color='red', linestyle='--', linewidth=2,
            label=f'随机期望基线 = {null_mean:.3f}')
ax1.axhline(y=np.mean(genre_purities), color='#1a3a6b', linestyle='-', linewidth=1.5,
            label=f'观察均值 = {np.mean(genre_purities):.3f}')
# Net increment annotation
obs_g_mean = np.mean(genre_purities)
increment_g = obs_g_mean - null_mean
ax1.annotate(f'净增量\n+{increment_g:.3f}',
            xy=(0, (null_mean + obs_g_mean)/2),
            fontsize=11, color='red', fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))
ax1.set_ylabel('体裁纯度')
ax1.set_title(f'体裁纯度分布 (n={len(genre_purities)}个社区)', fontweight='bold', fontsize=12)
ax1.legend(fontsize=9, loc='lower left')
ax1.set_ylim(0.25, 1.05)
ax1.set_xticks([])

# Dynasty violin
parts_d = ax2.violinplot([dyn_purities], positions=[0], showmeans=True,
                          showmedians=True, widths=0.6)
for pc in parts_d['bodies']:
    pc.set_facecolor('#ED7D31')
    pc.set_alpha(0.6)
ax2.scatter([0]*len(dyn_purities), dyn_purities, s=15, alpha=0.4,
            c='#ED7D31', edgecolors='white', linewidth=0.5, zorder=3)
ax2.axhline(y=null_mean, color='red', linestyle='--', linewidth=2,
            label=f'随机期望基线 = {null_mean:.3f}')
obs_d_mean = np.mean(dyn_purities)
ax2.axhline(y=obs_d_mean, color='#8B4513', linestyle='-', linewidth=1.5,
            label=f'观察均值 = {obs_d_mean:.3f}')
increment_d = obs_d_mean - null_mean
ax2.annotate(f'净增量\n{increment_d:+.3f}',
            xy=(0, (null_mean + obs_d_mean)/2),
            fontsize=11, color='red' if increment_d > 0 else 'gray',
            fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))
ax2.set_ylabel('朝代纯度')
ax2.set_title(f'朝代纯度分布 (n={len(dyn_purities)}个社区)', fontweight='bold', fontsize=12)
ax2.legend(fontsize=9, loc='lower left')
ax2.set_ylim(0.25, 1.05)
ax2.set_xticks([])

fig.suptitle('Louvain社区纯度对比：体裁 vs 朝代', fontsize=14, fontweight='bold', y=1.01)
fig.text(0.5, -0.02,
         f'随机期望基线来自999次体裁标签shuffle (mean={null_mean:.3f}±{lpn.get("null_std",0.004):.3f})',
         ha='center', fontsize=9, color='gray')

fig.savefig(f"{FIGS}/fig_louvain_violin.pdf", bbox_inches='tight', facecolor='white')
fig.savefig(f"{FIGS}/fig_louvain_violin.png", bbox_inches='tight', facecolor='white', dpi=200)
plt.close()
print("  → fig_louvain_violin.pdf/png")

print("\nDone.")
