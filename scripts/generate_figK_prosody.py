#!/usr/bin/env python3
"""
generate_figK_prosody.py
为实验 K（格律 baseline）生成可视化图
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
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

# 加载实验K结果
with open(DATA_DIR / "processed/expK_prosody_baseline.json") as f:
    results = json.load(f)

prosody = results["prosody_features"]
bert = results["bert_embedding"]

# ═══════════════════════════════════════════════════════════
# 图1: 三指标对比柱状图 + 混淆矩阵
# ═══════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# (a) 三指标对比柱状图
ax = axes[0]
metrics = ['Accuracy', 'Macro F1', 'Balanced Acc']
prosody_vals = [prosody["test_accuracy"], prosody["test_f1_macro"], prosody["test_balanced_acc"]]
bert_vals = [bert["test_accuracy"], bert["test_f1_macro"], bert["test_balanced_acc"]]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, prosody_vals, width, label='格律特征 (10D)',
               color='#E74C3C', alpha=0.85, edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, bert_vals, width, label='BERT 语义 (512D)',
               color='#3498DB', alpha=0.85, edgecolor='black', linewidth=1.2)

for bar, val in zip(bars1, prosody_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f"{val:.3f}", ha='center', fontsize=10, fontweight='bold')
for bar, val in zip(bars2, bert_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f"{val:.3f}", ha='center', fontsize=10, fontweight='bold')

ax.set_ylabel("分类性能", fontsize=12)
ax.set_title("(a) 格律特征 vs BERT语义嵌入", fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 1.1)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(0.333, color='gray', linestyle='--', linewidth=1, alpha=0.6)
ax.text(2.4, 0.345, '随机基线\n(三分类)', fontsize=8, color='gray', ha='right')

# (b) 格律特征混淆矩阵
ax = axes[1]
cm_prosody = np.array(prosody["confusion_matrix"])
labels = ['ci', 'shi', 'qu']
im = ax.imshow(cm_prosody, cmap='Reds', aspect='auto')
for i in range(3):
    for j in range(3):
        text_color = 'white' if cm_prosody[i, j] > cm_prosody.max() / 2 else 'black'
        ax.text(j, i, str(cm_prosody[i, j]), ha='center', va='center',
                color=text_color, fontsize=14, fontweight='bold')
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(labels, fontsize=11)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel("预测", fontsize=11)
ax.set_ylabel("真实", fontsize=11)
ax.set_title("(b) 格律特征混淆矩阵\n(全部判为shi)", fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# (c) BERT 混淆矩阵
ax = axes[2]
cm_bert = np.array(bert["confusion_matrix"])
im = ax.imshow(cm_bert, cmap='Blues', aspect='auto')
for i in range(3):
    for j in range(3):
        text_color = 'white' if cm_bert[i, j] > cm_bert.max() / 2 else 'black'
        ax.text(j, i, str(cm_bert[i, j]), ha='center', va='center',
                color=text_color, fontsize=14, fontweight='bold')
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(labels, fontsize=11)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel("预测", fontsize=11)
ax.set_ylabel("真实", fontsize=11)
ax.set_title("(c) BERT语义嵌入混淆矩阵\n(成功识别 ci/qu)", fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
fig_path = BASE / "data/figures/fig_prosody_baseline.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_path.with_suffix('.pdf'), bbox_inches='tight')
print(f"✅ 已保存: {fig_path}")
plt.close()

# ═══════════════════════════════════════════════════════════
# 图2: 双层逻辑链条概念图（理论可视化）
# ═══════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# 顶层：Experiment K
rect_topL = plt.Rectangle((0.5, 6), 5, 1.5, linewidth=2, edgecolor='#E74C3C',
                           facecolor='#FADBD8', alpha=0.8)
ax.add_patch(rect_topL)
ax.text(3, 6.75, "Experiment K\n显式格律特征 → 失败 (F1=0.32)\n[平仄、韵部、字数无法分类]",
        ha='center', va='center', fontsize=11, fontweight='bold')

rect_topR = plt.Rectangle((6.5, 6), 5, 1.5, linewidth=2, edgecolor='#E67E22',
                           facecolor='#FAE5D3', alpha=0.8)
ax.add_patch(rect_topR)
ax.text(9, 6.75, "§5.12 词牌阴性结果\n同词牌 vs 跨词牌 → 无差异 (p=0.50)\n[微观格律不强制语义]",
        ha='center', va='center', fontsize=11, fontweight='bold')

# 箭头：双层证据 → 中间结论
ax.annotate('', xy=(4.5, 4.5), xytext=(3, 6),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#34495E'))
ax.annotate('', xy=(7.5, 4.5), xytext=(9, 6),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#34495E'))

# 中间：核心修正立论
rect_mid = plt.Rectangle((1.5, 3), 9, 1.5, linewidth=2.5, edgecolor='#2C3E50',
                          facecolor='#D5DBDB', alpha=0.9)
ax.add_patch(rect_mid)
ax.text(6, 3.75, "【核心修正】 体裁约束力 ≠ 微观格律模板 (cipai)\n体裁约束力 = 宏观文化记忆与集体美学范式 (genre)",
        ha='center', va='center', fontsize=12, fontweight='bold')

# 箭头：理论 → 三大支撑
ax.annotate('', xy=(2, 1.5), xytext=(4, 3),
            arrowprops=dict(arrowstyle='->', lw=2, color='#2C3E50'))
ax.annotate('', xy=(6, 1.5), xytext=(6, 3),
            arrowprops=dict(arrowstyle='->', lw=2, color='#2C3E50'))
ax.annotate('', xy=(10, 1.5), xytext=(8, 3),
            arrowprops=dict(arrowstyle='->', lw=2, color='#2C3E50'))

# 底层：三大支撑实证
rect_bot1 = plt.Rectangle((0.2, 0), 3.6, 1.5, linewidth=1.5, edgecolor='#27AE60',
                           facecolor='#D5F5E3', alpha=0.8)
ax.add_patch(rect_bot1)
ax.text(2, 0.75, "BERT 语义嵌入\n体裁分类 F1=0.76\n(2.35× 格律 baseline)",
        ha='center', va='center', fontsize=10)

rect_bot2 = plt.Rectangle((4.2, 0), 3.6, 1.5, linewidth=1.5, edgecolor='#27AE60',
                           facecolor='#D5F5E3', alpha=0.8)
ax.add_patch(rect_bot2)
ax.text(6, 0.75, "时间窗口稳健性\n唐宋 R²=0.115\n明清 R²=0.109",
        ha='center', va='center', fontsize=10)

rect_bot3 = plt.Rectangle((8.2, 0), 3.6, 1.5, linewidth=1.5, edgecolor='#27AE60',
                           facecolor='#D5F5E3', alpha=0.8)
ax.add_patch(rect_bot3)
ax.text(10, 0.75, "ci 体裁聚类\nCohen's d = 1.90\n词派跨七朝",
        ha='center', va='center', fontsize=10)

ax.set_title("形式约束的层级突变性：从微观格律失效到宏观体裁主导",
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
fig_path = BASE / "data/figures/fig_hierarchical_argument.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_path.with_suffix('.pdf'), bbox_inches='tight')
print(f"✅ 已保存: {fig_path}")
plt.close()

print("\n=== 实验 K 可视化完成 ===")
