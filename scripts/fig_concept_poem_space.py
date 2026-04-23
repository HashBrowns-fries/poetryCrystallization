#!/usr/bin/env python3
"""
语义空间概念示意图
展示 shi/ci/qu 三类在语义空间的几何关系（体裁分离、语义引力、ci独立子空间）
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
for _p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"]:
    try: fm.fontManager.addfont(_p)
    except: pass
_cjk = [f.name for f in fm.fontManager.ttflist if "CJK" in f.name]
FONT_CJK = _cjk[0] if _cjk else None
import matplotlib; matplotlib.use("Agg")
from matplotlib.patches import Circle, FancyArrowPatch, Arc, FancyBboxPatch, Wedge
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import warnings; warnings.filterwarnings("ignore")

if FONT_CJK:
    plt.rcParams["font.sans-serif"] = [FONT_CJK]
plt.rcParams["axes.unicode_minus"] = False

# ── 模拟语义空间数据 ─────────────────────────────────────────────────
np.random.seed(42)

# Tang semantic origin (唐语义原点 = 语义引力中心)
tang = np.array([0.0, 0.0])

# shi (诗) poets: 围绕唐语义原点，分布较散，呈椭圆形（唐→宋→清历史扩散）
n_shi = 200
# 主分布在唐原点附近，但随朝代推进有轻微漂移
shi_pc1 = np.random.randn(n_shi) * 0.35
shi_pc2 = np.random.randn(n_shi) * 0.20
# 添加朝代效应：宋→清略有偏移
dynasty_shift = np.random.choice([0.0, 0.05, 0.10, 0.15, 0.20], n_shi)
shi_pc1 += dynasty_shift

# ci (词) poets: 形成紧凑内聚簇，独立于唐原点（挣脱引力场）
n_ci = 120
# ci簇在PC1上与shi有部分重叠，但在PC2上有明显偏移
ci_center = np.array([0.08, -0.25])   # ci独立子空间中心
ci_pc1 = np.random.randn(n_ci) * 0.12 + ci_center[0]   # 更紧凑
ci_pc2 = np.random.randn(n_ci) * 0.10 + ci_center[1]

# qu (曲) poets: 小而独立的聚簇
n_qu = 60
qu_center = np.array([-0.15, 0.12])
qu_pc1 = np.random.randn(n_qu) * 0.10 + qu_center[0]
qu_pc2 = np.random.randn(n_qu) * 0.08 + qu_center[1]

# ── 绘图 ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 10))

ax.set_xlim(-0.85, 0.85)
ax.set_ylim(-0.55, 0.55)
ax.set_aspect("equal")
ax.grid(True, alpha=0.2, linestyle="--")

# 轴标签
ax.set_xlabel("PC1（22.1%）— 体裁-历史复合轴（ci/shi分化主导）", fontsize=13)
ax.set_ylabel("PC2（13.6%）— 元曲戏剧轴", fontsize=13)

# 语义引力方向箭头（从ci/qu指向唐原点）
# ci → 唐引力
ax.annotate("",
    xy=tang, xytext=ci_center + np.array([0.08, 0.05]),
    arrowprops=dict(arrowstyle="->", color="#888888", lw=2.0,
                   connectionstyle="arc3,rad=0.15", linestyle="dashed"))
# qu → 唐引力
ax.annotate("",
    xy=tang, xytext=qu_center + np.array([-0.05, -0.04]),
    arrowprops=dict(arrowstyle="->", color="#888888", lw=2.0,
                   connectionstyle="arc3,rad=-0.1", linestyle="dashed"))
# 唐引力标注
ax.text(0.12, -0.08, "语义引力\n（指向唐原点）",
        fontsize=10, color="#888888", style="italic",
        ha="left", va="center")

# 语义引力范围圆（虚线，表示引力场）
circle_tang = Circle(tang, 0.30, fill=False, linestyle="--",
                      color="#aaaaaa", linewidth=1.5, alpha=0.6, zorder=1)
ax.add_patch(circle_tang)
ax.text(0.32, 0.32, "语义引力场", fontsize=9, color="#aaaaaa", style="italic")

# ── ci 独立子空间（半透明椭圆区域）──────────────────────────────────
from matplotlib.patches import Ellipse
ci_ellipse = Ellipse(ci_center, width=0.55, height=0.40,
                      fill=True, facecolor="#FF6B6B", alpha=0.08,
                      edgecolor="#FF6B6B", linewidth=2,
                      linestyle="--", zorder=2)
ax.add_patch(ci_ellipse)
ax.text(ci_center[0]-0.02, ci_center[1]-0.30,
        "ci独立语义子空间\n（挣脱唐引力场）",
        fontsize=10, color="#C0392B", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

# ── 散点 ─────────────────────────────────────────────────────────────
# shi：蓝色，围绕原点
ax.scatter(shi_pc1, shi_pc2, c="#4ECDC4", s=30, alpha=0.65,
           edgecolors="white", linewidth=0.5, zorder=3, label="shi（诗）")

# ci：红色，紧凑内聚
ax.scatter(ci_pc1, ci_pc2, c="#FF6B6B", s=40, alpha=0.80,
           edgecolors="white", linewidth=0.5, zorder=4, label="ci（词）")

# qu：紫色，小聚簇
ax.scatter(qu_pc1, qu_pc2, c="#9B59B6", s=35, alpha=0.80,
           edgecolors="white", linewidth=0.5, zorder=4, label="qu（元曲）")

# ── 特殊标注：词牌名密集区 ───────────────────────────────────────────
# 在ci簇中心附近标注词牌名
cipa示例 = [
    (-0.02, -0.22), (0.05, -0.28), (0.12, -0.20),
    (0.08, -0.30), (0.15, -0.24), (-0.08, -0.18),
]
for xp, yp in cipa示例:
    ax.text(xp, yp, "■", fontsize=8, color="#C0392B", alpha=0.7,
            ha="center", va="center", zorder=6)
ax.text(0.18, -0.16, "词牌名\n密集区", fontsize=9, color="#C0392B",
        ha="center", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", alpha=0.9))

# ── 唐语义原点 ───────────────────────────────────────────────────────
ax.scatter([0], [0], c="black", s=200, marker="*", zorder=10, label="唐语义原点")
ax.text(0.04, 0.04, "唐\n原点", fontsize=10, fontweight="bold", ha="left", va="bottom")

# ── 朝代扩散箭头（shi诗人历史扩散方向）─────────────────────────────
ax.annotate("",
    xy=(0.50, 0.0), xytext=(0.0, 0.0),
    arrowprops=dict(arrowstyle="->", color="#4ECDC4", lw=2.5,
                   connectionstyle="arc3,rad=-0.1"))
ax.text(0.25, -0.06, "历史时间→\n（诗人体裁扩散）",
        fontsize=9, color="#4ECDC4", ha="center")

# ── 体裁分离带（PC1上的ci/shi分离）──────────────────────────────────
ax.annotate("",
    xy=(0.18, 0.15), xytext=(-0.12, -0.18),
    arrowprops=dict(arrowstyle="<->", color="#888888", lw=1.5,
                   connectionstyle="arc3,rad=0"))
ax.text(0.03, -0.04, "体裁分离\n(PC1)", fontsize=9, color="#888888",
        ha="center", style="italic")

# ── 图例 ─────────────────────────────────────────────────────────────
legend_elems = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#4ECDC4",
           markersize=12, label="shi（诗）诗人"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF6B6B",
           markersize=12, label="ci（词）诗人（内聚）"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#9B59B6",
           markersize=12, label="qu（元曲）诗人"),
    Line2D([0], [0], marker="*", color="black", markersize=16,
           linestyle="none", label="唐语义原点（引力中心）"),
    mpatches.Patch(color="#FF6B6B", alpha=0.2, label="ci独立语义子空间"),
    mpatches.Patch(facecolor="none", edgecolor="#aaaaaa",
                   linestyle="--", label="语义引力场范围"),
]
ax.legend(handles=legend_elems, loc="upper left", fontsize=11,
          framealpha=0.92, edgecolor="gray")

# ── 标题 ─────────────────────────────────────────────────────────────
ax.set_title("图2  语义空间概念图：体裁分离与语义引力\n"
             "（基于PCA前两主成分的示意性呈现）",
             fontsize=14, fontweight="bold", pad=10)

plt.tight_layout()
out = "/home/chenhao/poetry-phylogeny/data/figures/fig2_concept_semantic_space.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"已保存: {out}")
