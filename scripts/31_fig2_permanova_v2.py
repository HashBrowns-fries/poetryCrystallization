#!/usr/bin/env python3
"""
fig2_permanova_v2.py — 体裁先于朝代：PERMANOVA核心证据可视化
(a) 单因素R²对比 + (b) 双因素条件效应分解 + (c) 朝代距离热力图
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
import matplotlib.patches as mpatches
import numpy as np, json
from matplotlib.gridspec import GridSpec

if _font_cjk:
    plt.rcParams['font.sans-serif'] = [_font_cjk]
plt.rcParams['axes.unicode_minus'] = False

BASE = "/home/chenhao/poetry-phylogeny"

# ── 加载正确数据 ──────────────────────────────────────────────────
with open(f"{BASE}/data/processed/genre_dominance.json") as f:
    gd = json.load(f)

poets = json.load(open(f"{BASE}/data/processed/poet_poems.json", encoding="utf-8"))

D = np.load(f"{BASE}/data/processed/poet_distances.npy").astype(np.float64)

dyn_list = ["唐","宋","元","明","清","近代"]
dyn_colors_map = {"唐":"#4ECDC4","宋":"#FFA07A","元":"#FF6B6B",
                  "明":"#98D8C8","清":"#F7DC6F","近代":"#ABB2B9"}

# 正确PERMANOVA数据（来自 genre_dominance.json exp2_two_factor）
r2_single_dyn   = gd["exp1_dynasty_permanova"]["R2"]      # 0.8907
r2_single_genre = gd["exp1_genre_permanova"]["R2"]       # 0.1718
r2_joint        = gd["exp2_two_factor"]["R2_joint"]       # 0.9047
r2_cond_genre   = gd["exp2_two_factor"]["R2_genre_cond"]  # 0.014
r2_cond_dyn     = gd["exp2_two_factor"]["R2_dynasty_cond"]# 0.733
p_cond_genre    = gd["exp2_two_factor"]["p_genre"]       # 0.0 (***)
p_cond_dyn      = gd["exp2_two_factor"]["p_dynasty"]     # 0.2828 (n.s.)
boot_ci         = gd["bootstrap_genre_R2"]

# ── 图布局 ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 5.8))
gs  = GridSpec(1, 3, figure=fig, wspace=0.35,
               left=0.06, right=0.97, top=0.88, bottom=0.15)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2])

# ── (a) 单因素R²柱状对比 ─────────────────────────────────────────
cats   = ["朝代\n(单因素)", "体裁\n(单因素)", "联合R²\n(双因素)"]
r2s    = [r2_single_dyn, r2_single_genre, r2_joint]
colors_a = ["#4ECDC4", "#FF6B6B", "#95A5A6"]
bars_a   = ax0.bar(cats, r2s, color=colors_a, alpha=0.88, edgecolor='black', lw=1.2)

for bar, val in zip(bars_a, r2s):
    ax0.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.012,
             f"{val:.3f}", ha='center', fontsize=11, fontweight='bold')

ax0.set_ylabel("$R^2$（方差解释比例）", fontsize=11)
ax0.set_title("(a) 单因素PERMANOVA：朝代 vs 体裁", fontsize=12, fontweight='bold')
ax0.set_ylim(0, 1.08)
ax0.grid(True, alpha=0.25, axis='y')
ax0.axhline(0.1, color='red', lw=1, ls='--', label='$R^2=0.1$ 弱效应基线')
ax0.legend(fontsize=9, loc='upper right')

# 标注：朝代R²高≠独立效应
ax0.annotate("⚠ 生态谬误\n（高度共线）",
             xy=(0, r2_single_dyn), xytext=(0.5, 0.72),
             arrowprops=dict(arrowstyle='->', color='gray'),
             fontsize=9, color='gray',
             bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.85))

# ── (b) 双因素条件效应分解 ──────────────────────────────────────
# 堆叠条形：联合R² = 重叠(可归于任一) + 纯朝代 + 纯体裁
overlap = r2_joint - r2_cond_dyn - r2_cond_genre  # ≈ 0.158

y_base = 0.35
bar_h  = 0.28

# 底层：重叠
b1 = ax1.barh(y_base, overlap, color="#D2B48C", alpha=0.85,
              edgecolor='black', lw=0.8, height=bar_h,
              label=f'共享方差 {overlap:.3f}')
# 中层：纯朝代（n.s.）
b2 = ax1.barh(y_base, r2_cond_dyn, color="#4ECDC4", alpha=0.85,
              edgecolor='black', lw=0.8, height=bar_h,
              left=overlap, label=f'纯朝代 {r2_cond_dyn:.3f} (p=0.283 n.s.)')
# 顶层：纯体裁（***）
b3 = ax1.barh(y_base, r2_cond_genre, color="#FF6B6B", alpha=0.90,
              edgecolor='black', lw=0.8, height=bar_h,
              left=overlap+r2_cond_dyn, label=f'纯体裁 {r2_cond_genre:.3f} (p<0.001***)')

# 标注文字
mid1 = overlap/2
ax1.text(mid1, y_base, f"共享\n{overlap:.3f}",
         ha='center', va='center', fontsize=8, fontweight='bold', color='black')
mid2 = overlap + r2_cond_dyn/2
ax1.text(mid2, y_base, f"纯朝代\n{r2_cond_dyn:.3f}",
         ha='center', va='center', fontsize=8, fontweight='bold', color='white')
mid3 = overlap + r2_cond_dyn + r2_cond_genre/2
ax1.text(mid3, y_base, f"纯\n{r2_cond_genre:.3f}",
         ha='center', va='center', fontsize=8, fontweight='bold', color='white')

# Bootstrap CI 标注
ci_text = f"纯体裁 Bootstrap CI\n[{boot_ci['ci_low']:.3f}, {boot_ci['ci_high']:.3f}]"
ax1.text(overlap+r2_cond_dyn+0.01, y_base-0.22, ci_text,
         fontsize=8, color='#CC0000',
         bbox=dict(boxstyle='round,pad=0.25', fc='#FFF0F0', alpha=0.9))

ax1.set_yticks([y_base])
ax1.set_yticklabels(["$R^2$ 条件效应分解\n（控制另一因素）"], fontsize=10)
ax1.set_xlabel("$R^2$（方差解释比例）", fontsize=11)
ax1.set_title("(b) 双因素PERMANOVA：条件效应分解", fontsize=12, fontweight='bold')
ax1.set_xlim(0, 1.08)
ax1.set_ylim(-0.05, 0.72)
ax1.grid(True, alpha=0.25, axis='x')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(fontsize=9, loc='upper right', framealpha=0.9)

# ── (c) 朝代间语义距离热力图 ─────────────────────────────────────
dyn_avg_dist = np.zeros((len(dyn_list), len(dyn_list)))
for di, da in enumerate(dyn_list):
    for dj, db in enumerate(dyn_list):
        ma = np.array([p.get("dynasty","其他")==da for p in poets])
        mb = np.array([p.get("dynasty","其他")==db for p in poets])
        if ma.sum()>0 and mb.sum()>0:
            dyn_avg_dist[di,dj] = D[np.ix_(ma, mb)].mean()

vmax = max(0.14, dyn_avg_dist.max())
vmin = 0.0
im = ax2.imshow(dyn_avg_dist, cmap='YlOrRd', aspect='auto', vmin=vmin, vmax=vmax)
ax2.set_xticks(range(len(dyn_list))); ax2.set_xticklabels(dyn_list, fontsize=10)
ax2.set_yticks(range(len(dyn_list))); ax2.set_yticklabels(dyn_list, fontsize=10)
ax2.set_title("(c) 朝代间平均语义距离热力图", fontsize=12, fontweight='bold')
for di in range(len(dyn_list)):
    for dj in range(len(dyn_list)):
        c = 'white' if dyn_avg_dist[di,dj] > (vmax+vmin)/2 else 'black'
        ax2.text(dj, di, f"{dyn_avg_dist[di,dj]:.3f}",
                 ha='center', va='center', fontsize=8.5, color=c, fontweight='bold')
plt.colorbar(im, ax=ax2, shrink=0.85, label='平均余弦距离')

fig.suptitle("图3  PERMANOVA效应量分解：体裁是语义空间的第一组织力量",
             fontsize=14, fontweight='bold', y=0.99)

out = f"{BASE}/data/figures/fig3_permanova.png"
fig.savefig(out, dpi=150, facecolor='white')
plt.close()
print(f"✓ fig2_permanova_v2 saved: {out}")
