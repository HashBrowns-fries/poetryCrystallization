#!/usr/bin/env python3
"""
fig3_community_v2.py — Louvain社区检测：体裁纯度 vs 朝代纯度
(a) 社区规模分布  (b) 体裁纯度 vs 朝代纯度散点  (c) 主要社区环形布局
重点：正确呈现体裁纯度>>朝代纯度，并标注随机期望基线
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
from collections import defaultdict
import community as community_louvain
import networkx as nx

if _font_cjk:
    plt.rcParams['font.sans-serif'] = [_font_cjk]
plt.rcParams['axes.unicode_minus'] = False

BASE = "/home/chenhao/poetry-phylogeny"
D    = np.load(f"{BASE}/data/processed/poet_distances.npy").astype(np.float64)
poets  = json.load(open(f"{BASE}/data/processed/poet_poems.json", encoding="utf-8"))
genres = json.load(open(f"{BASE}/data/processed/poet_genre_by_source.json", encoding="utf-8"))
lv    = json.load(open(f"{BASE}/data/processed/louvain_genre_cross.json", encoding="utf-8"))
lpn   = json.load(open(f"{BASE}/data/processed/louvain_purity_null.json", encoding="utf-8"))

n = len(poets)

def dominant_genre(name):
    g = genres.get(name, {})
    ci = g.get("ci",0); shi = g.get("shi",0); qu = g.get("qu",0)
    t = ci+shi+qu
    if t==0: return "other"
    if ci/t > 0.5: return "ci"
    if qu/t > 0.5: return "qu"
    return "shi"

gen_lb = [dominant_genre(p["name"]) for p in poets]

# ── 重建 Louvain ────────────────────────────────────────────────
K = 20
D_sorted = np.argsort(D, axis=1)
knn = D_sorted[:, 1:K+1]
G = nx.Graph()
for i in range(n):
    for j in knn[i]:
        if i != j:
            w = max(0.0, 1.0 - D[i,j])
            if w > 0.15:
                G.add_edge(i, j, weight=w)

partition  = community_louvain.best_partition(G, weight='weight', resolution=1.0, random_state=42)
comm_ids   = set(partition.values())
n_comm     = len(comm_ids)
modularity = lv.get("modularity", 0.648)

comm_size   = defaultdict(int)
comm_dyn    = defaultdict(lambda: defaultdict(int))
comm_genre  = defaultdict(lambda: defaultdict(int))
for i, p in enumerate(poets):
    c = partition[i]
    comm_size[c]  += 1
    comm_dyn[c][p.get("dynasty","其他")]  += 1
    comm_genre[c][gen_lb[i]] += 1

def purity(dist):
    t = sum(dist.values())
    return max(dist.values())/t if t>0 else 0

# ── 图布局 ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 5.8))
gs  = GridSpec(1, 3, figure=fig, wspace=0.30,
               left=0.06, right=0.97, top=0.88, bottom=0.14)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2])

# ── (a) 社区规模分布 ─────────────────────────────────────────────
comm_sizes_list = sorted([comm_size[c] for c in comm_ids if comm_size[c]>=5], reverse=True)
ax0.hist(comm_sizes_list, bins=25, color='steelblue', alpha=0.72, edgecolor='black')
ax0.set_xlabel("社区规模（诗人数量）", fontsize=11)
ax0.set_ylabel("频数（社区个数）", fontsize=11)
ax0.set_title(f"(a) Louvain社区规模分布\n(共{n_comm}个社区, Q={modularity:.3f})",
              fontsize=12, fontweight='bold')
ax0.grid(True, alpha=0.25, axis='y')

# ── (b) 体裁纯度 vs 朝代纯度 ─────────────────────────────────────
purity_genre_v, purity_dyn_v, sizes_v = [], [], []
for c in comm_ids:
    if comm_size[c] < 10: continue
    purity_dyn_v.append(purity(comm_dyn[c]))
    purity_genre_v.append(purity(comm_genre[c]))
    sizes_v.append(comm_size[c])

ax1.scatter(purity_dyn_v, purity_genre_v, s=[s*3 for s in sizes_v],
            c='#FF6B6B', alpha=0.65, edgecolors='black', linewidth=0.6, zorder=3)
ax1.plot([0,1],[0,1], 'k--', lw=1.2, label='y=x（等纯度线）', zorder=2)

# 随机期望水平线
null_mean = lpn.get("null_mean", 0.935)
ax1.axhline(null_mean, color='green', lw=1.5, ls=':',
            label=f'随机期望={null_mean:.3f}', zorder=2)
ax1.axhline(1.0, color='gray', lw=0.8, ls='-', alpha=0.4, zorder=1)

# 观察均值水平线
obs_mean = lpn.get("observed_genre_purity", 0.956)
ax1.hlines(obs_mean, 0, 1, colors='#CC0000', lw=2,
           label=f'观察纯度={obs_mean:.3f}')

ax1.set_xlabel("朝代纯度", fontsize=11)
ax1.set_ylabel("体裁纯度", fontsize=11)
ax1.set_title("(b) 社区纯度：体裁 vs 朝代\n（圆圈大小∝社区规模）", fontsize=12, fontweight='bold')
ax1.set_xlim(0, 1); ax1.set_ylim(0.3, 1.04)
ax1.legend(fontsize=9, loc='lower right')
ax1.grid(True, alpha=0.25)

# 标注净增量
increment = obs_mean - null_mean
ax1.annotate("", xy=(0.52, null_mean), xytext=(0.52, obs_mean),
             arrowprops=dict(arrowstyle='<->', color='#CC0000', lw=1.5))
ax1.text(0.54, (null_mean+obs_mean)/2, f"+{increment:.3f}净增量",
         fontsize=9, color='#CC0000', fontweight='bold', va='center')

# 文本框
txt = (f"体裁纯度均值={np.mean(purity_genre_v):.3f}\n"
       f"朝代纯度均值={np.mean(purity_dyn_v):.3f}\n"
       f"随机期望={null_mean:.3f}±{lpn.get('null_std',0.004):.3f}")
ax1.text(0.03, 0.97, txt, transform=ax1.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))

# ── (c) 主要社区环形布局 ──────────────────────────────────────────
top_comms = sorted(comm_ids, key=lambda c: -comm_size[c])[:8]
n_top = len(top_comms)
cmap = plt.cm.Set3(np.linspace(0, 1, n_top))
comm_centers = {}
for ci, c in enumerate(top_comms):
    theta = 2*np.pi*ci/n_top - np.pi/2
    comm_centers[c] = (np.cos(theta)*2.4, np.sin(theta)*2.4)

for ci, c in enumerate(top_comms):
    cx, cy = comm_centers[c]
    r = 0.55
    circ = plt.Circle((cx, cy), r, color=cmap[ci], alpha=0.78, zorder=3,
                       lw=1.5, edgecolor='white')
    ax2.add_patch(circ)
    dom_dyn  = max(comm_dyn[c], key=comm_dyn[c].get)
    dom_genr = max(comm_genre[c], key=comm_genre[c].get)
    g_lb = {"ci":"词","shi":"诗","qu":"曲","other":"其他"}.get(dom_genr, dom_genr)
    ax2.text(cx, cy+0.08, f"C{c}", ha='center', va='center',
             fontsize=8, fontweight='bold', zorder=4)
    ax2.text(cx, cy-0.15, f"{dom_dyn}/{g_lb}", ha='center', va='center',
             fontsize=7.5, color='white', fontweight='bold', zorder=4)
    ax2.text(cx, cy-0.38, f"n={comm_size[c]}", ha='center', va='center',
             fontsize=7, color='white', zorder=4)

legend_patches = [mpatches.Patch(color=cmap[i], label=f"C{top_comms[i]}({comm_size[top_comms[i]]}人)")
                 for i in range(n_top)]
ax2.legend(handles=legend_patches, fontsize=8, loc='upper right',
           title='主要社区', framealpha=0.9)
ax2.set_xlim(-3.8, 3.8); ax2.set_ylim(-3.8, 3.8)
ax2.set_aspect('equal'); ax2.axis('off')
ax2.set_title("(c) 主要Louvain社区环形布局", fontsize=12, fontweight='bold')

fig.suptitle("图7  Louvain社区检测：体裁纯度远高于朝代纯度",
             fontsize=14, fontweight='bold', y=0.99)

out = f"{BASE}/data/figures/fig7_community.png"
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✓ fig3_community_v2 saved: {out}")
