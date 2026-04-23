#!/usr/bin/env python3
"""
35_fig5_bert_classification.py
Fig5: BERT-CCPoem 体裁分类实验综合结果（仅3个子图）
  (a) 消融实验对比表
  (b) 混淆矩阵（3×3）
  (c) Macro-F1 与 ci-F1 柱状对比
"""
import json, os
import numpy as np
import matplotlib.font_manager as fm
for _p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"]:
    try: fm.fontManager.addfont(_p)
    except: pass
_cjk = [f.name for f in fm.fontManager.ttflist if "CJK" in f.name]
FONT_CJK = _cjk[0] if _cjk else None
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings; warnings.filterwarnings("ignore")

BASE    = "/home/chenhao/poetry-phylogeny"
OUT_DIR = f"{BASE}/data/figures"
os.makedirs(OUT_DIR, exist_ok=True)

if FONT_CJK:
    plt.rcParams["font.sans-serif"] = [FONT_CJK]
plt.rcParams["axes.unicode_minus"] = False

res_file = f"{BASE}/data/processed/expC_v2_results.json"
exp_data = json.load(open(res_file, encoding="utf-8")) if os.path.exists(res_file) else None

# ── 图布局：2行 ─────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 12))
gs  = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.25,
               left=0.06, right=0.97, top=0.92, bottom=0.10)

ax_a = fig.add_subplot(gs[0, :2])   # (a) 消融表（跨2列）
ax_b = fig.add_subplot(gs[0, 2])    # (b) 混淆矩阵（1列）
ax_c = fig.add_subplot(gs[1, :])    # (c) 柱状图（跨全宽）

# ═════════════════════════════════════════════════════════════════════════════
# (a) 消融实验对比表
# ═════════════════════════════════════════════════════════════════════════════
ax_a.axis("off")

headers = ["配置", "池化", "损失函数", "学习率",
           "Accuracy", "Macro-F1", "ci-F1", "qu-F1", "ROC-AUC"]

if exp_data and "ablation" in exp_data:
    tbl_rows = []
    for k in ["A1", "A2", "A3", "A4"]:
        r = exp_data["ablation"].get(k, {})
        label = {"A1":"A1(CE基准)","A2":"A2(Focal★最优)","A3":"A3(Focal)","A4":"A4(最终)"}[k]
        tbl_rows.append([
            label,
            "Mean Pooling",
            {"A1":"CE","A2":"Focal(γ=2)","A3":"Focal(γ=2)","A4":"Focal(γ=2)"}[k],
            {"A1":"2e-5","A2":"2e-5","A3":"1e-5","A4":"1e-5"}[k],
            f"{r.get('acc',0):.4f}",
            f"{r.get('f1_macro',0):.4f}",
            f"{r.get('f1_ci',0):.4f}",
            f"{r.get('f1_qu',0):.4f}",
            f"{r.get('auc',0):.4f}",
        ])
else:
    tbl_rows = [
        ["A1(CE基准)","Mean","CE","2e-5","—","—","—","—","—"],
        ["A2(Focal★)","Mean","Focal","2e-5","—","—","—","—","—"],
        ["A3(Focal)","Mean","Focal","1e-5","—","—","—","—","—"],
        ["A4(最终)","Mean","Focal","1e-5","—","—","—","—","—"],
    ]

all_rows = [headers] + tbl_rows
cell_colors = [["#4ECDC4"]*len(headers)]
for row in tbl_rows:
    clr = "#FFE4B5" if "★" in row[0] else "#FFFFFF"
    cell_colors.append([clr]*len(headers))

table = ax_a.table(
    cellText=all_rows,
    cellColours=cell_colors,
    colLabels=None,
    loc="center",
    cellLoc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.2)

for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_text_props(fontweight="bold", color="white")
        cell.set_facecolor("#2C3E50")
    elif c == 0:
        cell.set_text_props(fontweight="bold")
    if r > 0 and "★" in all_rows[r][0]:
        cell.set_edgecolor("#FF8C00")
        cell.set_linewidth(2)

ax_a.set_title("(a) BERT-CCPoem 三分类消融实验结果（shi/ci/qu，Poem-level）",
               fontsize=12, fontweight="bold", pad=6)

# ═════════════════════════════════════════════════════════════════════════════
# (b) 混淆矩阵（3×3）
# ═════════════════════════════════════════════════════════════════════════════
if exp_data and "best_result" in exp_data and "cm_mean" in exp_data["best_result"]:
    cm = np.array(exp_data["best_result"]["cm_mean"], dtype=float)
    cm_pct = cm / cm.sum(axis=1, keepdims=True)
else:
    cm_pct = np.zeros((3, 3))
    cm = np.zeros((3, 3))

im = ax_b.imshow(cm_pct, cmap="Blues", vmin=0, vmax=1)
ax_b.set_xticks([0,1,2]); ax_b.set_yticks([0,1,2])
ax_b.set_xticklabels(["shi","ci","qu"], fontsize=13)
ax_b.set_yticklabels(["shi","ci","qu"], fontsize=13)
ax_b.set_xlabel("预测", fontsize=13)
ax_b.set_ylabel("实际", fontsize=13)
for i in range(3):
    for j in range(3):
        color = "white" if cm_pct[i,j] > 0.5 else "black"
        ax_b.text(j, i, f"{cm_pct[i,j]:.1%}\n({int(cm[i,j])})", ha="center", va="center",
                  fontsize=13, fontweight="bold", color=color)
ax_b.set_title("(b) 混淆矩阵（A2 Focal 最优）", fontsize=12, fontweight="bold")
plt.colorbar(im, ax=ax_b, shrink=0.75, label="正确率")

# ═════════════════════════════════════════════════════════════════════════════
# (c) 柱状图对比
# ═════════════════════════════════════════════════════════════════════════════
names = ["A1(CE)", "A2(Focal★)", "A3(Focal)", "A4(最终)"]
ci_f1s, f1s = [], []
for k in ["A1", "A2", "A3", "A4"]:
    r = exp_data["ablation"].get(k, {}) if exp_data else {}
    ci_f1s.append(r.get("f1_ci", 0))
    f1s.append(r.get("f1_macro", 0))

x = np.arange(len(names))
w = 0.3
bars1 = ax_c.bar(x - w/2, f1s, w, label="Macro-F1", color="#4ECDC4", alpha=0.85, edgecolor="black", lw=0.8)
bars2 = ax_c.bar(x + w/2, ci_f1s, w, label="ci-F1", color="#FF6B6B", alpha=0.85, edgecolor="black", lw=0.8)

for bar in bars1:
    h = bar.get_height()
    if h > 0:
        ax_c.text(bar.get_x()+bar.get_width()/2, h+0.003, f"{h:.3f}",
                  ha="center", fontsize=10, fontweight="bold")
for bar in bars2:
    h = bar.get_height()
    if h > 0:
        ax_c.text(bar.get_x()+bar.get_width()/2, h+0.003, f"{h:.3f}",
                  ha="center", fontsize=10, fontweight="bold")

ax_c.set_xticks(x)
ax_c.set_xticklabels(names, fontsize=13)
ax_c.set_ylabel("F1 分数", fontsize=13)
ax_c.set_ylim(0, 1.08)
ax_c.set_title("(c) 消融实验 F1 分数对比", fontsize=12, fontweight="bold")
ax_c.legend(fontsize=12, loc="upper right", framealpha=0.9)
ax_c.grid(True, alpha=0.25, axis="y")
ax_c.axhline(0.5, color="red", lw=1, ls="--", alpha=0.5)

best_idx = int(np.argmax(f1s))
ax_c.annotate(f"Best\nA{best_idx+1}: {f1s[best_idx]:.3f}",
              xy=(best_idx, f1s[best_idx]),
              xytext=(best_idx + 0.65, f1s[best_idx] + 0.05),
              arrowprops=dict(arrowstyle="->", color="green"),
              fontsize=11, color="darkgreen",
              bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.8))

fig.suptitle(
    "图8  BERT-CCPoem 体裁分类：消融实验结果",
    fontsize=15, fontweight="bold", y=0.995)

out = f"{OUT_DIR}/fig8_bert_classification.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"已保存: {out}")
