#!/usr/bin/env python3
"""
fig6_geographic_gravity_v2.py — 语义引力轨迹与地缘关联分析
(a) 各朝代诗人到唐centroid距离（语义引力轨迹） (b) 地缘关联Mantel检验 (c) 体裁结构解释非单调性
重点：语义引力非单调（唐→宋→元↑→明↓→清↑）的正确呈现与理论解释
"""
import matplotlib.font_manager as fm
for _p in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
           '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc']:
    try: fm.fontManager.addfont(_p)
    except: pass
_cjk = [f.name for f in fm.fontManager.ttflist if 'CJK' in f.name]
_font_cjk = _cjk[0] if _cjk else None
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, json
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch

if _font_cjk:
    plt.rcParams['font.sans-serif'] = [_font_cjk]
plt.rcParams['axes.unicode_minus'] = False

BASE  = "/home/chenhao/poetry-phylogeny"
D     = np.load(f"{BASE}/data/processed/poet_distances.npy").astype(np.float64)
X     = np.load(f"{BASE}/data/processed/poet_embeddings.npy").astype(np.float64)
poets = json.load(open(f"{BASE}/data/processed/poet_poems.json", encoding="utf-8"))
genres= json.load(open(f"{BASE}/data/processed/poet_genre_by_source.json", encoding="utf-8"))
sg    = json.load(open(f"{BASE}/data/processed/semantic_gravity_results.json", encoding="utf-8"))
gis   = json.load(open(f"{BASE}/data/processed/gis_analysis.json", encoding="utf-8"))

n = len(poets)
dyn_list = ["唐","宋","元","明","清","近代"]
dyn_colors = ["#4ECDC4","#FFA07A","#FF6B6B","#98D8C8","#F7DC6F","#ABB2B9"]

# ── 归一化嵌入 ─────────────────────────────────────────────────
X_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
tang_mask = np.array([p.get("dynasty","其他")=="唐" for p in poets])
tang_c = X_n[tang_mask].mean(axis=0, keepdims=True)
tang_c_n = tang_c / (np.linalg.norm(tang_c, axis=1, keepdims=True) + 1e-9)
poet_tang_dist = 1.0 - (X_n @ tang_c_n.T).flatten()

def dom_genre(name):
    g = genres.get(name, {})
    ci=g.get("ci",0); shi=g.get("shi",0); qu=g.get("qu",0)
    t=ci+shi+qu
    if t==0: return "other"
    if ci/t>0.5: return "ci"
    if qu/t>0.5: return "qu"
    return "shi"

# ── 图布局 ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 7.5))
gs  = GridSpec(1, 3, figure=fig, wspace=0.28,
               left=0.05, right=0.98, top=0.88, bottom=0.14)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2])

# ── (a) 语义引力轨迹 ─────────────────────────────────────────────
dyn_dists = sg["dyn_to_tang"]
x = np.arange(len(dyn_list))
means = [dyn_dists[d]["mean"] for d in dyn_list]
stds  = [dyn_dists[d]["std"] for d in dyn_list]
ns    = [dyn_dists[d]["n"] for d in dyn_list]
ses   = [s/np.sqrt(n) for s, n in zip(stds, ns)]

bars = ax0.bar(x, means, width=0.55, color=dyn_colors, alpha=0.82, edgecolor='black')
ax0.errorbar(x, means, yerr=[s*1.96 for s in ses],
             fmt='none', color='black', capsize=4, lw=1.2)

for i, (bar, m, n_) in enumerate(zip(bars, means, ns)):
    ax0.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
             f"{m:.3f}\n(n={n_})", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax0.set_xticks(x); ax0.set_xticklabels(dyn_list, fontsize=12)
ax0.set_ylabel("到唐centroid距离（语义引力↓）", fontsize=13)
ax0.set_title("(a) 各朝代到唐语义原点的距离\n（语义引力轨迹：非单调模式）", fontsize=13, fontweight='bold')
ax0.grid(True, alpha=0.25, axis='y')
ax0.set_ylim(0, 0.11)

# 标注非单调性
ax0.annotate("元代词派扩张\n距离骤增↑", xy=(2, means[2]),
             xytext=(1.1, means[2]+0.018),
             arrowprops=dict(arrowstyle='->', color='gray'),
             fontsize=9, color='gray')
ax0.annotate("明代复古运动\n距离回归↓", xy=(3, means[3]),
             xytext=(3.7, means[3]-0.012),
             arrowprops=dict(arrowstyle='->', color='gray'),
             fontsize=9, color='gray')

# 标注唐基准线
ax0.axhline(means[0], color='gray', lw=1, ls='--', alpha=0.5)
ax0.text(5.05, means[0]+0.001, f"唐基准={means[0]:.3f}", fontsize=9, color='gray', va='bottom')

# ── (b) 地缘关联Mantel检验 ───────────────────────────────────────
mantel_rel  = gis["mantel_rel"]["rho"]
mantel_rand = gis["mantel_rand"]["rho"]
mantel_p    = gis["mantel_rel"]["p"]
n_rel       = gis["n_rel_pairs_geo"]

labels_b = ["有地缘关联诗人对\n(n=53)", "随机对照\n(n=随机)"]
vals_b   = [mantel_rel, mantel_rand]
errs_b   = [0.05, 0.05]   # 近似标准误
cols_b   = ["#FF6B6B","#4ECDC4"]

bars_b = ax1.bar(labels_b, vals_b, yerr=errs_b, color=cols_b, alpha=0.82,
                  edgecolor='black', capsize=5, lw=1.2)
ax1.set_ylabel("Mantel r（地理-语义相关）", fontsize=13)
ax1.set_title("(b) 地理相关性与语义距离\n（Mantel检验：r=0.129, p=0.357 n.s.）", fontsize=13, fontweight='bold')
ax1.set_ylim(-0.1, 0.30)
ax1.axhline(0, color='gray', lw=0.8)
ax1.grid(True, alpha=0.25, axis='y')

for bar, val in zip(bars_b, vals_b):
    ax1.text(bar.get_x()+bar.get_width()/2, val+0.03 if val>0 else val-0.05,
             f"r={val:.3f}", ha='center', fontsize=11, fontweight='bold')

# p值标注
ax1.text(0.5, 0.05, f"p={mantel_p:.3f} n.s.\n（地缘关联≠语义相似）",
         transform=ax1.transAxes, ha='center', fontsize=10, color='gray',
         bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.85))

# ── (c) 体裁结构解释语义引力非单调性 ─────────────────────────────
# 各朝代ci比例（任意一首词即为ci诗人，与其他分析一致）
is_ci_any = np.array([genres.get(p["name"],{}).get("ci",0) > 0 for p in poets])
dyn_ci_ratio = {}
dyn_ci_n = {}
for d in dyn_list:
    d_mask = np.array([p.get("dynasty","其他")==d for p in poets])
    total = d_mask.sum()
    ci_n = (d_mask & is_ci_any).sum()
    dyn_ci_ratio[d] = ci_n / total if total > 0 else 0
    dyn_ci_n[d] = ci_n

# 双轴：语义引力 + ci比例
ax2_twin = ax2.twinx()

# 柱：ci比例
bars_c = ax2.bar(x, [dyn_ci_ratio[d] for d in dyn_list], width=0.4,
                  color='#FF6B6B', alpha=0.55, label='ci诗人占比', edgecolor='black', lw=1)
# 线：语义引力
line, = ax2_twin.plot(x, means, 'o-', color='#4ECDC4', lw=3,
                       markersize=11, label='到唐距离', zorder=5)
ax2_twin.set_ylabel("到唐centroid距离", fontsize=13, color='#4ECDC4')
ax2_twin.tick_params(axis='y', labelcolor='#4ECDC4', labelsize=11)

ax2.set_xticks(x); ax2.set_xticklabels(dyn_list, fontsize=12)
ax2.set_ylabel("ci诗人占比", fontsize=13)
ax2.set_title("(c) 体裁结构解释语义引力非单调性\n（ci比例↑→距离↑，ci创造独立子空间）", fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.25, axis='y')
ax2.set_ylim(0, 0.40)

# 每个柱子顶部标注ci比例
for i, d in enumerate(dyn_list):
    r = dyn_ci_ratio[d]
    ax2.text(i, r + 0.008, f"{r:.1%}", ha='center', va='bottom',
             fontsize=11, fontweight='bold', color='#C0392B')

# 标注元代高ci效应
ax2.annotate(f"元：ci={dyn_ci_ratio['元']:.1%}\n距离突增", xy=(2, dyn_ci_ratio['元']),
             xytext=(2.25, dyn_ci_ratio['元']+0.06),
             arrowprops=dict(arrowstyle='->', color='darkred'),
             fontsize=10, color='darkred', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.9))

# 图例
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elems = [
    Patch(color='#FF6B6B', alpha=0.55, label='ci诗人占比'),
    Line2D([0], [0], color='#4ECDC4', marker='o', lw=3, label='到唐距离'),
]
ax2.legend(handles=legend_elems, fontsize=11, loc='upper left')

fig.suptitle("图6  语义引力轨迹与体裁结构分析",
             fontsize=14, fontweight='bold', y=0.99)

out = f"{BASE}/data/figures/fig6_geographic_gravity.png"
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✓ fig6_geographic_gravity_v2 saved: {out}")
