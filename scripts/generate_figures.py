"""
generate_figures.py
为四项稳健性实验生成图表并保存到 data/figures/
"""
import json, random, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import Counter

import matplotlib.font_manager as fm

# Register Noto Serif CJK for Chinese characters
cjk_font_path = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'
fm.fontManager.addfont(cjk_font_path)
cjk_prop = fm.FontProperties(fname=cjk_font_path)
cjk_family = fm.FontProperties(fname=cjk_font_path).get_name()

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': [cjk_family, 'DejaVu Sans', 'sans-serif'],
    'font.size': 10,
    'axes.unicode_minus': False,
    'figure.dpi': 150,
})

random.seed(42); np.random.seed(42)
BASE = "/home/chenhao/poetryCrystallization"
FIGS = f"{BASE}/data/figures"

# ═══════════════════════════════════════════════════════════════════════════
# Fig A: TF-IDF baseline bar chart — Genre vs Dynasty F1
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Fig A: TF-IDF baseline comparison...")
expF = json.load(open(f"{BASE}/data/processed/expF_tfidf_baseline.json"))

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# Left: main comparison
tasks = ['Genre\nbinary\n(shi/ci)', 'Dynasty\n6-class']
f1s = [expF['genre_binary']['macro_f1_mean'], expF['dynasty_6class']['macro_f1_mean']]
accs = [expF['genre_binary']['accuracy_mean'], expF['dynasty_6class']['accuracy_mean']]

x = np.arange(len(tasks))
w = 0.35
bars1 = axes[0].bar(x - w/2, f1s, w, label='Macro-F1', color='#4472C4', edgecolor='white')
bars2 = axes[0].bar(x + w/2, accs, w, label='Accuracy', color='#ED7D31', edgecolor='white')
for bar in bars1:
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in bars2:
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
axes[0].set_xticks(x)
axes[0].set_xticklabels(tasks)
axes[0].set_ylabel('Score')
axes[0].set_title('TF-IDF Baseline: Genre vs Dynasty', fontweight='bold')
axes[0].legend(loc='lower right')
axes[0].set_ylim(0, 1.05)
# Arrow showing reverse pattern
axes[0].annotate('Dynasty > Genre\n(reverse of BERT)', xy=(1, max(f1s[1], accs[1])),
                xytext=(0.5, 0.95), fontsize=8, color='red',
                arrowprops=dict(arrowstyle='->', color='red'), ha='center')

# Right: within-dynasty control
within_data = {
    'Genre-in-Song': expF['genre_within_song_f1'],
    'Genre-in-Qing': expF['genre_within_qing_f1'],
    'Dynasty-in-Shi': expF['dynasty_within_shi_f1'],
    'Dynasty-in-Ci': expF['dynasty_within_ci_f1'],
}
colors2 = ['#4472C4', '#4472C4', '#ED7D31', '#ED7D31']
bars3 = axes[1].bar(within_data.keys(), within_data.values(), color=colors2, edgecolor='white')
for bar in bars3:
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
axes[1].axhline(y=0.50, color='gray', linestyle='--', linewidth=0.8, label='Random baseline')
axes[1].set_ylabel('Macro-F1')
axes[1].set_title('Within-Dynasty / Within-Genre Control', fontweight='bold')
axes[1].legend(fontsize=8)
axes[1].set_ylim(0, 0.85)
plt.setp(axes[1].get_xticklabels(), rotation=15, ha='right', fontsize=8)

plt.tight_layout()
fig.savefig(f"{FIGS}/fig_tfidf_baseline.pdf", bbox_inches='tight')
fig.savefig(f"{FIGS}/fig_tfidf_baseline.png", bbox_inches='tight', dpi=200)
plt.close()
print("  → fig_tfidf_baseline.pdf/png")

# ═══════════════════════════════════════════════════════════════════════════
# Fig B: Louvain k×τ heatmap
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Fig B: Louvain parameter sensitivity heatmap...")
expG = json.load(open(f"{BASE}/data/processed/expG_sensitivity.json"))
louvain = expG['louvain_scan']
k_list = louvain['k_list']
tau_list = louvain['tau_list']
data = np.array(louvain['net_increment'])

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(data, cmap='RdBu_r', aspect='auto', vmin=-0.004, vmax=0.006,
               interpolation='nearest')
ax.set_xticks(range(len(tau_list)))
ax.set_xticklabels([f'{t:.2f}' for t in tau_list])
ax.set_yticks(range(len(k_list)))
ax.set_yticklabels([str(k) for k in k_list])
ax.set_xlabel(r'$\tau$ (similarity threshold)', fontsize=11)
ax.set_ylabel(r'$k$ (k-NN neighbors)', fontsize=11)
ax.set_title('Louvain Genre Purity Net Increment\n(Observed − Null Expectation)', fontweight='bold')

# Annotate each cell
for i in range(len(k_list)):
    for j in range(len(tau_list)):
        val = data[i, j]
        color = 'white' if abs(val) > 0.002 and val < 0 else 'black'
        ax.text(j, i, f'{val:+.4f}', ha='center', va='center', fontsize=9,
                fontweight='bold', color=color)

cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label('Net Purity Increment', fontsize=10)

# Highlight positive vs negative
for i in range(len(k_list)):
    for j in range(len(tau_list)):
        if data[i, j] < 0:
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                        edgecolor='black', linewidth=2, linestyle='--'))

plt.tight_layout()
fig.savefig(f"{FIGS}/fig_louvain_heatmap.pdf", bbox_inches='tight')
fig.savefig(f"{FIGS}/fig_louvain_heatmap.png", bbox_inches='tight', dpi=200)
plt.close()
print("  → fig_louvain_heatmap.pdf/png")

# ═══════════════════════════════════════════════════════════════════════════
# Fig C: Distance metric comparison — grouped bar chart
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Fig C: Distance metric comparison...")
dm = expG['distance_metric_comparison']
metrics = list(dm.keys())
ci_ci = [dm[m]['ci_ci_mean'] for m in metrics]
ci_shi = [dm[m]['ci_shi_mean'] for m in metrics]
gap_pct = [dm[m]['genre_gap_pct'] for m in metrics]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# Left: ci-ci vs ci-shi
x = np.arange(len(metrics))
w = 0.35
ax1.bar(x - w/2, ci_ci, w, label='ci-ci (within)', color='#4472C4', edgecolor='white')
ax1.bar(x + w/2, ci_shi, w, label='ci-shi (between)', color='#ED7D31', edgecolor='white')
for i in range(len(metrics)):
    ax1.text(i - w/2, ci_ci[i] + 0.3, f'{ci_ci[i]:.3f}', ha='center', fontsize=8, rotation=90)
    ax1.text(i + w/2, ci_shi[i] + 0.3, f'{ci_shi[i]:.3f}', ha='center', fontsize=8, rotation=90)
ax1.set_xticks(x)
ax1.set_xticklabels([m.capitalize() for m in metrics])
ax1.set_ylabel('Distance')
ax1.set_title('Within-Genre vs Between-Genre Distance', fontweight='bold')
ax1.legend(fontsize=9)

# Right: gap %
bars = ax2.bar(metrics, gap_pct, color=['#4472C4', '#ED7D31', '#A5A5A5'], edgecolor='white')
for bar, pct in zip(bars, gap_pct):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
             f'{pct:.1f}%', ha='center', fontweight='bold', fontsize=11)
ax2.set_ylabel('Gap (%)')
ax2.set_title('ci-ci / ci-shi Gap', fontweight='bold')
ax2.set_xticklabels([m.capitalize() for m in metrics])

plt.tight_layout()
fig.savefig(f"{FIGS}/fig_distance_metrics.pdf", bbox_inches='tight')
fig.savefig(f"{FIGS}/fig_distance_metrics.png", bbox_inches='tight', dpi=200)
plt.close()
print("  → fig_distance_metrics.pdf/png")

# ═══════════════════════════════════════════════════════════════════════════
# Fig D: Cross-dynasty generalization — scatter/bar
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Fig D: Cross-dynasty generalization...")
expH = json.load(open(f"{BASE}/data/processed/expH_cross_dynasty.json"))
cross = expH['cross_dynasty_genre']

# Group by train dynasty
train_dyns = sorted(set(k.split('→')[0] for k in cross.keys()))
fig, ax = plt.subplots(figsize=(10, 5.5))

# Create a matrix: train_dyn × test_dyn
test_dyns = sorted(set(k.split('→')[1] for k in cross.keys()))
f1_matrix = np.full((len(train_dyns), len(test_dyns)), np.nan)
acc_matrix = np.full((len(train_dyns), len(test_dyns)), np.nan)
for key, val in cross.items():
    train, test = key.split('→')
    i = train_dyns.index(train)
    j = test_dyns.index(test)
    f1_matrix[i, j] = val['macro_f1']
    acc_matrix[i, j] = val['accuracy']

im = ax.imshow(f1_matrix, cmap='YlOrRd', aspect='auto', vmin=0.3, vmax=0.75)
ax.set_xticks(range(len(test_dyns)))
ax.set_xticklabels(test_dyns, fontsize=11)
ax.set_yticks(range(len(train_dyns)))
ax.set_yticklabels(train_dyns, fontsize=11)
ax.set_xlabel('Test Dynasty', fontsize=12)
ax.set_ylabel('Train Dynasty', fontsize=12)
ax.set_title('Cross-Dynasty Genre Classification F1\n(Trained on row, tested on column)',
             fontweight='bold')

# Annotate
for i in range(len(train_dyns)):
    for j in range(len(test_dyns)):
        if not np.isnan(f1_matrix[i, j]):
            text_color = 'white' if f1_matrix[i, j] > 0.55 else 'black'
            ax.text(j, i, f'{f1_matrix[i,j]:.3f}', ha='center', va='center',
                    fontsize=8, fontweight='bold', color=text_color)

cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label('Macro-F1', fontsize=10)

plt.tight_layout()
fig.savefig(f"{FIGS}/fig_cross_dynasty.pdf", bbox_inches='tight')
fig.savefig(f"{FIGS}/fig_cross_dynasty.png", bbox_inches='tight', dpi=200)
plt.close()
print("  → fig_cross_dynasty.pdf/png")

# ═══════════════════════════════════════════════════════════════════════════
# Fig E: Cipai comparison — boxplot + per-cipai bar
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Fig E: Cipai within vs across comparison...")
expI = json.load(open(f"{BASE}/data/processed/expI_cipai_compare.json"))
per_cipai = expI['per_cipai']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: per-cipai cohesion bar chart
cipai_names = [c for c, m, s, n in per_cipai]
cipai_means = [m for c, m, s, n in per_cipai]
cipai_ns = [n for c, m, s, n in per_cipai]

colors_bar = plt.cm.viridis(np.linspace(0.2, 0.8, len(cipai_names)))
bars = ax1.barh(range(len(cipai_names)), cipai_means, color=colors_bar, edgecolor='white')
ax1.set_yticks(range(len(cipai_names)))
ax1.set_yticklabels(cipai_names, fontsize=9)
ax1.set_xlabel('Mean Cosine Distance')
ax1.set_title('Per-Cipai Internal Cohesion', fontweight='bold')
ax1.axvline(x=expI['poet_level']['within_cipai_mean'], color='red', linestyle='--',
            linewidth=1.5, label=f"Overall ci within: {expI['poet_level']['within_cipai_mean']:.4f}")
ax1.legend(fontsize=8)
ax1.invert_yaxis()
for i, (bar, n) in enumerate(zip(bars, cipai_ns)):
    ax1.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2.,
             f'n={n}', va='center', fontsize=8)

# Right: within vs across + shi control comparison
comparison_data = {
    'Ci within-\ncipai': expI['poet_level']['within_cipai_mean'],
    'Ci across-\ncipai': expI['poet_level']['across_cipai_mean'],
    'Shi random\npairs': 0.0943,
    'Ci overall\n(within genre)': expI['poet_level']['within_cipai_mean'],
}
comp_colors = ['#4472C4', '#4472C4', '#ED7D31', '#70AD47']
bars2 = ax2.bar(comparison_data.keys(), comparison_data.values(),
                color=comp_colors, edgecolor='white')
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
             f'{bar.get_height():.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
ax2.set_ylabel('Mean Cosine Distance')
ax2.set_title('Genre-level vs Cipai-level Cohesion', fontweight='bold')
ax2.axhline(y=expI['poet_level']['within_cipai_mean'], color='gray',
            linestyle=':', linewidth=0.8)
# Add annotation
ax2.annotate(f"Within ≈ Across\n(p={expI['poet_level']['mann_whitney_p']:.2f})",
            xy=(1, expI['poet_level']['across_cipai_mean']),
            xytext=(1.5, 0.06), fontsize=9, color='red',
            arrowprops=dict(arrowstyle='->', color='red'))
plt.setp(ax2.get_xticklabels(), rotation=15, ha='right', fontsize=8)

plt.tight_layout()
fig.savefig(f"{FIGS}/fig_cipai_comparison.pdf", bbox_inches='tight')
fig.savefig(f"{FIGS}/fig_cipai_comparison.png", bbox_inches='tight', dpi=200)
plt.close()
print("  → fig_cipai_comparison.pdf/png")

print("\nAll figures generated in data/figures/")
