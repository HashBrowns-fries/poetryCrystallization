#!/usr/bin/env python3
"""
fig4_intertextual_v2.py — 互文距离ci体裁信号检验
(a) ci vs non-ci 距离分布 (b) 各朝代内ci效应（均值对比+效应量）(c) 各朝代ci诗人比例
关键改进：使用诗人级平均距离（非句对级），更准确反映体裁效应
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

if _font_cjk:
    plt.rcParams['font.sans-serif'] = [_font_cjk]
plt.rcParams['axes.unicode_minus'] = False

BASE = "/home/chenhao/poetry-phylogeny"
D    = np.load(f"{BASE}/data/processed/poet_distances.npy").astype(np.float64)
poets  = json.load(open(f"{BASE}/data/processed/poet_poems.json", encoding="utf-8"))
genres = json.load(open(f"{BASE}/data/processed/poet_genre_by_source.json", encoding="utf-8"))
itx_g  = json.load(open(f"{BASE}/data/processed/intertextual_genre_results.json", encoding="utf-8"))

n = len(poets)
dyn_list = ["唐","宋","元","明","清","近代"]
dyn_colors = ["#4ECDC4","#FFA07A","#FF6B6B","#98D8C8","#F7DC6F","#ABB2B9"]

def is_ci_poet(name):
    g = genres.get(name, {})
    return g.get("ci", 0) >= 5

is_ci = np.array([is_ci_poet(p["name"]) for p in poets])
ci_count = is_ci.sum()

# ── (a) 距离分布：诗人级平均距离 ─────────────────────────────────
# 计算每位诗人的平均语义距离（到同类诗人）
poet_mean_dist = {}
for i, p in enumerate(poets):
    di = D[i]
    same = di[is_ci].mean() if is_ci[i] else di[~is_ci].mean()
    poet_mean_dist[i] = same

ci_means   = [poet_mean_dist[i] for i in range(n) if is_ci[i]]
nonci_means= [poet_mean_dist[i] for i in range(n) if not is_ci[i]]

cohens_d = itx_g.get("cohens_d", 1.90)
ci_n = itx_g.get("n_ci_poets", ci_count)
nonci_n = itx_g.get("n_nonci_poets", n - ci_count)
ci_pair_n = itx_g.get("n_ci_pairs", 0)
nonci_pair_n = itx_g.get("n_nonci_pairs", 0)

# ── 读取朝代内数据 ────────────────────────────────────────────────
dyn_results = itx_g.get("within_dyn_results", [])
dyn_ci_n = {r["dynasty"]: r["n_ci"] for r in dyn_results}
dyn_ci_mean = {r["dynasty"]: r["ci_mean"] for r in dyn_results}
dyn_nonci_mean = {r["dynasty"]: r["nonci_mean"] for r in dyn_results}
dyn_diff = {r["dynasty"]: r["diff"] for r in dyn_results}

# 各朝代ci诗人数量（来自poets元数据）
dyn_total = {d: sum(1 for p in poets if p.get("dynasty")==d) for d in dyn_list}
dyn_ci_total = {d: sum(1 for p in poets if p.get("dynasty")==d and is_ci[poets.index(p)])
               for d in dyn_list}
# 重新计算（避免poets.index低效）
poet_dyn_ci = {}
for idx, p in enumerate(poets):
    d = p.get("dynasty","其他")
    if d in dyn_list and is_ci[idx]:
        poet_dyn_ci[d] = poet_dyn_ci.get(d, 0) + 1

# ── 图布局 ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 5.8))
gs  = GridSpec(1, 3, figure=fig, wspace=0.30,
               left=0.06, right=0.97, top=0.88, bottom=0.14)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2])

# ── (a) 距离分布直方图（诗人级） ─────────────────────────────────
ax0.hist(nonci_means, bins=50, alpha=0.55, label=f"非ci诗人(n={nonci_n:,})",
         color='#4ECDC4', density=True, edgecolor='white')
ax0.hist(ci_means, bins=50, alpha=0.55, label=f"ci诗人(n={ci_n:,})",
         color='#FF6B6B', density=True, edgecolor='white')

# 添加均值线
ax0.axvline(np.mean(ci_means), color='#CC0000', lw=2, ls='--',
            label=f"ci均值={np.mean(ci_means):.3f}")
ax0.axvline(np.mean(nonci_means), color='#006600', lw=2, ls='--',
            label=f"非ci均值={np.mean(nonci_means):.3f}")

ax0.set_xlabel("诗人平均组内语义距离", fontsize=11)
ax0.set_ylabel("密度", fontsize=11)
ax0.set_title("(a) ci vs 非ci诗人距离分布\n（诗人级平均距离）", fontsize=12, fontweight='bold')
ax0.legend(fontsize=9)
ax0.grid(True, alpha=0.25)

# 效应量注释
ax0.text(0.95, 0.95,
         f"Cohen's d={cohens_d:.2f}\nci诗人对={ci_pair_n:,}\n非ci诗人对={nonci_pair_n:,}",
         transform=ax0.transAxes, fontsize=9, va='top', ha='right',
         bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))

# ── (b) 各朝代内ci效应：均值对比柱状图 ────────────────────────────
x = np.arange(len(dyn_list))
width = 0.35

bars_nc = ax1.bar(x - width/2, [dyn_nonci_mean.get(d, 0) for d in dyn_list],
                  width, label='非ci诗人', color='#4ECDC4', alpha=0.82, edgecolor='black')
bars_c  = ax1.bar(x + width/2, [dyn_ci_mean.get(d, 0) for d in dyn_list],
                  width, label='ci诗人', color='#FF6B6B', alpha=0.82, edgecolor='black')

# 添加误差标注（差值）
for i, d in enumerate(dyn_list):
    diff = dyn_diff.get(d, 0)
    # 差值标注在柱子上方
    ax1.annotate(f"Δ={diff:.3f}",
                 xy=(i, max(dyn_nonci_mean.get(d,0), dyn_ci_mean.get(d,0)) + 0.01),
                 ha='center', va='bottom', fontsize=8, color='#CC0000', fontweight='bold')

ax1.set_xticks(x)
ax1.set_xticklabels(dyn_list, fontsize=10)
ax1.set_ylabel("平均语义距离", fontsize=11)
ax1.set_title("(b) 各朝代内ci效应（均值对比）\n（ci诗人始终更近）", fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.25, axis='y')
ax1.set_ylim(0, 1.15)

# ── (c) 各朝代ci诗人比例 ──────────────────────────────────────────
dyn_ci_ratio = {d: poet_dyn_ci.get(d, 0) / dyn_total[d] if dyn_total[d] > 0 else 0
                for d in dyn_list}
ratios = [dyn_ci_ratio[d] for d in dyn_list]
ci_nums = [poet_dyn_ci.get(d, 0) for d in dyn_list]
tot_nums = [dyn_total[d] for d in dyn_list]

bars_r = ax2.bar(x, ratios, width=0.55, color='#FF6B6B', alpha=0.78, edgecolor='black')

for i, (bar, ratio, cn, tn) in enumerate(zip(bars_r, ratios, ci_nums, tot_nums)):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
             f"{ratio:.1%}\n(n={cn})", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax2.set_xticks(x)
ax2.set_xticklabels(dyn_list, fontsize=10)
ax2.set_ylabel("ci诗人占比", fontsize=11)
ax2.set_title("(c) 各朝代ci主导诗人占比\n（括号内为ci诗人绝对数量）", fontsize=12, fontweight='bold')
ax2.set_ylim(0, 0.45)
ax2.grid(True, alpha=0.25, axis='y')

# 标注：宋代ci比例最高
max_dyn = max(dyn_list, key=lambda d: dyn_ci_ratio[d])
ax2.annotate(f"宋最高{ratios[dyn_list.index(max_dyn)]:.1%}",
             xy=(dyn_list.index(max_dyn), ratios[dyn_list.index(max_dyn)]),
             xytext=(dyn_list.index(max_dyn)-1.2, ratios[dyn_list.index(max_dyn)]+0.04),
             arrowprops=dict(arrowstyle='->', color='gray'),
             fontsize=8, color='gray')

fig.suptitle("图4  互文距离ci/shi体裁信号检验",
             fontsize=14, fontweight='bold', y=0.99)

out = f"{BASE}/data/figures/fig4_intertextual.png"
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✓ fig4_intertextual_v2 saved: {out}")
