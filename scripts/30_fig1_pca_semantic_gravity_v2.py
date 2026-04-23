#!/usr/bin/env python3
"""
fig1_pca_semantic_gravity_v2.py — PCA语义空间：体裁是主轴
(a) PC1-PC2散点（体裁+朝代双标记）+ (b) PC1首轴是ci/shi体裁分化 + (c) 朝代均值趋势
核心改进：强调PC1首要信号是体裁分化，而非朝代分期
"""
import json, numpy as np
import matplotlib.font_manager as fm
import matplotlib; matplotlib.use('Agg')
for _p in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
           '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc']:
    try: fm.fontManager.addfont(_p)
    except: pass
_cjk = [f.name for f in fm.fontManager.ttflist if 'CJK' in f.name]
_font_cjk = _cjk[0] if _cjk else None
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr

if _font_cjk:
    plt.rcParams['font.sans-serif'] = [_font_cjk]
plt.rcParams['axes.unicode_minus'] = False
import matplotlib.patches as mpatches

BASE = "/home/chenhao/poetry-phylogeny"
X    = np.load(f"{BASE}/data/processed/poet_embeddings.npy").astype(np.float64)
poets = json.load(open(f"{BASE}/data/processed/poet_poems.json", encoding="utf-8"))
genres = json.load(open(f"{BASE}/data/processed/poet_genre_by_source.json", encoding="utf-8"))

n = len(poets)
dyn_list = ["唐","宋","元","明","清","近代"]
dyn_colors_map = {"唐":"#4ECDC4","宋":"#FFA07A","元":"#FF6B6B",
                  "明":"#98D8C8","清":"#F7DC6F","近代":"#ABB2B9"}

def dom_genre(name):
    g = genres.get(name, {})
    ci=g.get("ci",0); shi=g.get("shi",0); qu=g.get("qu",0)
    t=ci+shi+qu
    if t==0: return "other"
    if ci/t>0.5: return "ci"
    if qu/t>0.5: return "qu"
    return "shi"

gen_lb = [dom_genre(p["name"]) for p in poets]

# ci_binary: 任意一首词即为ci诗人（与pca_genre_analysis.json一致）
is_ci  = np.array([genres.get(p["name"],{}).get("ci",0) > 0 for p in poets])
is_shi = ~is_ci & np.array([genres.get(p["name"],{}).get("shi",0) > 0 for p in poets])
is_qu  = np.array([genres.get(p["name"],{}).get("qu",0) > 0 for p in poets])
is_oth = ~(is_ci | is_shi | is_qu)

# ── PCA ─────────────────────────────────────────────────────────
from sklearn.decomposition import PCA
X_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
pca  = PCA(n_components=4).fit(X_n)
PC  = pca.transform(X_n)
pc_var = pca.explained_variance_ratio_[:4]

# Spearman相关
dyn_rank = {d: i for i, d in enumerate(dyn_list)}
dyn_arr  = np.array([dyn_rank.get(p.get("dynasty","其他"), -1) for p in poets])
ci_arr   = is_ci.astype(float)

rho_pc1_dyn, _ = spearmanr(PC[:,0], dyn_arr)
rho_pc1_ci,  _ = spearmanr(PC[:,0], ci_arr)
rho_pc2_ci,  _ = spearmanr(PC[:,1], ci_arr)
qu_arr   = np.array([genres.get(p["name"],{}).get("qu",0) > 0 for p in poets]).astype(float)
rho_pc2_qu,  _ = spearmanr(PC[:,1], qu_arr)

# ── 朝代均值PC值 ─────────────────────────────────────────────────
dyn_pc_means = {}
for d in dyn_list:
    m = np.array([p.get("dynasty","其他")==d for p in poets])
    dyn_pc_means[d] = (PC[m,0].mean(), PC[m,1].mean())

# ── 图布局 ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 5.8))
gs  = GridSpec(1, 3, figure=fig, wspace=0.30,
               left=0.06, right=0.97, top=0.88, bottom=0.14)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2])

# ── (a) PC1-PC2 散点图（体裁着色+朝代标记）────────────────────────
# 先画其他（灰色背景）
for i, (m, c, lbl) in enumerate([(is_oth, '#CCCCCC','其他')]):
    if m.sum() > 0:
        ax0.scatter(PC[m,0], PC[m,1], s=4, c=c, alpha=0.2, label=lbl, zorder=1)

# ci=红色，shi=蓝色，qu=绿色（主类别，前景）
for mask, color, lbl in [(is_ci, '#FF4444','词(ci)'),
                           (is_shi, '#4ECDC4','诗(shi)'),
                           (is_qu, '#FFD700','曲(qu)')]:
    if mask.sum() > 0:
        ax0.scatter(PC[mask,0], PC[mask,1], s=6, c=color, alpha=0.55, label=lbl, zorder=3)

ax0.set_xlabel(f"PC1 ({pc_var[0]*100:.1f}%)", fontsize=11)
ax0.set_ylabel(f"PC2 ({pc_var[1]*100:.1f}%)", fontsize=11)
ax0.set_title(f"(a) PCA语义空间（体裁着色）\n"
              f"PC1=体裁轴(ρci={rho_pc1_ci:.3f}) | PC2=曲轴(ρqu={rho_pc2_qu:.3f})",
              fontsize=12, fontweight='bold')
ax0.legend(fontsize=9, markerscale=2.5, loc='upper right')
ax0.grid(True, alpha=0.15)

# 标注象限
ax0.text(0.02, 0.98, "PC1+：庙堂/诗\nPC1-：词/个人",
         transform=ax0.transAxes, fontsize=8, va='top', ha='left',
         bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))

# ── (b) PC1按体裁类内分布箱线图 ─────────────────────────────────
genre_groups = {"词(ci)": PC[is_ci, 0], "诗(shi)": PC[is_shi, 0],
               "曲(qu)": PC[is_qu, 0], "其他": PC[is_oth, 0]}
box_colors   = {"词(ci)":"#FF4444","诗(shi)":"#4ECDC4",
                "曲(qu)":"#FFD700","其他":"#CCCCCC"}

bp = ax1.boxplot([genre_groups[k] for k in ["词(ci)","诗(shi)","曲(qu)","其他"]],
                  labels=["词(ci)","诗(shi)","曲(qu)","其他"],
                  patch_artist=True, notch=False,
                  medianprops=dict(color='black', lw=2))
for patch, key in zip(bp['boxes'], ["词(ci)","诗(shi)","曲(qu)","其他"]):
    patch.set_facecolor(box_colors[key])
    patch.set_alpha(0.7)

ax1.set_ylabel("PC1值", fontsize=11)
ax1.set_title("(b) PC1值按体裁分布\n（PC1首要信号=体裁分化，非朝代分期）",
              fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.25, axis='y')
ax1.axhline(0, color='gray', lw=0.8, ls='--')

# 统计标注
for i, (k, v) in enumerate(genre_groups.items()):
    ax1.text(i, v.max()+0.05, f"n={len(v)}", ha='center', fontsize=8, color=box_colors[k])

# ── (c) PC1朝代均值趋势 ──────────────────────────────────────────
x = np.arange(len(dyn_list))
means1 = [dyn_pc_means[d][0] for d in dyn_list]
means2 = [dyn_pc_means[d][1] for d in dyn_list]

ax2_twin = ax2.twinx()
bars = ax2.bar(x, means1, width=0.4, color=[dyn_colors_map[d] for d in dyn_list],
               alpha=0.75, edgecolor='black', label='PC1均值')
line, = ax2_twin.plot(x, means2, 's--', color='#FF4444', lw=2,
                       markersize=8, label='PC2均值', zorder=5)

ax2.set_xticks(x); ax2.set_xticklabels(dyn_list, fontsize=10)
ax2.set_ylabel("PC1均值", fontsize=11)
ax2_twin.set_ylabel("PC2均值", fontsize=11, color='#FF4444')
ax2_twin.tick_params(axis='y', labelcolor='#FF4444')
ax2.set_title(f"(c) PC1/PC2朝代均值趋势\n(Spearman ρ={rho_pc1_dyn:.3f})",
              fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.25, axis='y')

from matplotlib.lines import Line2D
legend_elems = [
    mpatches.Patch(facecolor='gray', alpha=0.6, label='PC1均值（随朝代递增）'),
    Line2D([0],[0], color='#FF4444', marker='s', lw=2, label='PC2均值'),
]
ax2.legend(handles=legend_elems, fontsize=9, loc='upper left')

# 标注PC1趋势
rho_s, p_s = spearmanr(np.arange(len(dyn_list)), means1)
ax2.text(0.95, 0.05, f"PC1朝代趋势\nρ={rho_s:.3f}",
         transform=ax2.transAxes, ha='right', fontsize=9,
         bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.85))

fig.suptitle("图5  PCA语义空间：PC1首要信号是体裁分化（词/诗/曲）",
             fontsize=14, fontweight='bold', y=0.99)

out = f"{BASE}/data/figures/fig5_pca_semantic_gravity.png"
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✓ fig1_pca_semantic_gravity_v2 saved: {out}")
print(f"  PC1: {pc_var[0]*100:.1f}% | PC2: {pc_var[1]*100:.1f}%")
print(f"  ρ(PC1,ci)={rho_pc1_ci:.3f} | ρ(PC1,dynasty)={rho_pc1_dyn:.3f}")
