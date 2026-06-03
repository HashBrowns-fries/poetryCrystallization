"""
expI_cipai_compare.py
词牌内 vs 跨词牌语义距离比较

直接检验"词牌格律 → 语义内聚"的精细假设。
- 选取ci数量最多的前10个词牌
- 计算同词牌内句子对的余弦距离 vs 跨词牌句子对的距离
- Mann-Whitney U 检验 + Cliff's delta 效应量

同时计算诗人层面：同词牌诗人的嵌入距离 vs 不同词牌诗人的嵌入距离
"""

import json, glob, random, numpy as np
from collections import Counter, defaultdict
from scipy.spatial.distance import cosine
from scipy.stats import mannwhitneyu

random.seed(42); np.random.seed(42)

BASE = "/home/chenhao/poetryCrystallization"
DATA = f"{BASE}/data/processed"
RAW  = f"{BASE}/data/raw/chinese-poetry"
OUT  = f"{DATA}/expI_cipai_compare.json"

# ── 1. Load data ─────────────────────────────────────────────────────────
print("Loading data...")
embeddings = np.load(f"{DATA}/poet_embeddings.npy").astype(np.float32)
poets      = json.load(open(f"{DATA}/poet_poems.json"))
gsrc       = json.load(open(f"{DATA}/poet_genre_hybrid.json"))

name2idx = {p["name"]: p["id"] for p in poets}
idx2name = {p["id"]: p["name"] for p in poets}

def dom_genre(name):
    g = gsrc.get(name, {})
    s, c, q = g.get("shi",0), g.get("ci",0), g.get("qu",0)
    t = s + c + q
    if t == 0: return "shi"
    if c/t > 0.25: return "ci"
    if q/t > 0.25: return "qu"
    return "shi"

# ── 2. Load ci poems with rhythmic (词牌名) ──────────────────────────────
ci_files = sorted(glob.glob(f"{RAW}/宋词/ci.song.*.json"))
all_ci = []
for f in ci_files:
    for p in json.load(open(f)):
        rhythmic = p.get('rhythmic', '').strip()
        if rhythmic and rhythmic != '失调名':
            all_ci.append({
                'author': p['author'],
                'rhythmic': rhythmic,
                'paragraphs': p.get('paragraphs', []),
            })

print(f"  Total ci poems with rhythmic: {len(all_ci)}")

# ── 3. Identify top-10 cipai ─────────────────────────────────────────────
cipai_counts = Counter(p['rhythmic'] for p in all_ci)
top10 = [c for c, n in cipai_counts.most_common(10)]
print(f"  Top 10 cipai: {top10}")
print(f"  Counts: {[cipai_counts[c] for c in top10]}")

# ── 4. Build cipai → [poet_ids] for poet-level analysis ──────────────────
# Only count ci poets (≥5 ci poems)
ci_poet_names = set()
for name, g in gsrc.items():
    if g.get('ci', 0) >= 5:
        ci_poet_names.add(name)

cipai_poets = defaultdict(set)
for poem in all_ci:
    if poem['author'] in ci_poet_names:
        pid = name2idx.get(poem['author'])
        if pid is not None:
            cipai_poets[poem['rhythmic']].add(pid)

# Filter to top10 cipai with ≥3 poets
valid_cipai = {c: list(p) for c, p in cipai_poets.items()
               if c in top10 and len(p) >= 3}
print(f"\n  Valid cipai (≥3 ci poets): {list(valid_cipai.keys())}")

# ── 5. Poet-level: within-cipai vs across-cipai distances ────────────────
print("\n" + "="*60)
print("Poet-level: Within-cipai vs Across-cipai Cosine Distance")
print("="*60)

within_dists = []
across_dists = []

# Within-cipai
for cipai, plist in valid_cipai.items():
    if len(plist) < 2:
        continue
    for i in range(len(plist)):
        for j in range(i+1, len(plist)):
            within_dists.append(float(cosine(embeddings[plist[i]], embeddings[plist[j]])))

# Across-cipai
cipai_names = list(valid_cipai.keys())
for i in range(len(cipai_names)):
    for j in range(i+1, len(cipai_names)):
        for pid_a in valid_cipai[cipai_names[i]]:
            for pid_b in valid_cipai[cipai_names[j]]:
                across_dists.append(float(cosine(embeddings[pid_a], embeddings[pid_b])))

# Sample across_dists to match within_dists size for fair comparison
if len(across_dists) > len(within_dists) * 3:
    across_dists = random.sample(across_dists, len(within_dists) * 3)

within_mean = np.mean(within_dists)
within_std  = np.std(within_dists)
across_mean = np.mean(across_dists)
across_std  = np.std(across_dists)

# Mann-Whitney U
stat, pval = mannwhitneyu(within_dists, across_dists, alternative='less')
# Cliff's delta (effect size)
# For each within pair, count how many across pairs are larger
n1, n2 = len(within_dists), len(across_dists)
# Approximate Cliff's delta via U statistic
cliff_delta = 1 - (2 * stat) / (n1 * n2)  # for 'less' alternative

print(f"  Within-cipai:  mean={within_mean:.4f}  std={within_std:.4f}  n={n1}")
print(f"  Across-cipai:  mean={across_mean:.4f}  std={across_std:.4f}  n={n2}")
print(f"  Difference (across - within): {across_mean - within_mean:+.4f}")
print(f"  Mann-Whitney U: stat={stat:.0f}  p={pval:.2e}")
print(f"  Cliff's delta: {cliff_delta:.4f}")

# ── 6. Per-cipai breakdown ───────────────────────────────────────────────
print(f"\n{'='*60}")
print("Per-Cipai Internal Cohesion (sorted by mean distance)")
cipai_stats = []
for cipai, plist in valid_cipai.items():
    if len(plist) < 3:
        continue
    dists = [float(cosine(embeddings[plist[i]], embeddings[plist[j]]))
             for i in range(len(plist)) for j in range(i+1, len(plist))]
    if dists:
        cipai_stats.append((cipai, np.mean(dists), np.std(dists), len(plist)))

cipai_stats.sort(key=lambda x: x[1])
print(f"  {'Cipai':<12} {'Mean':>8} {'Std':>8} {'N_poets':>8}")
for c, m, s, n in cipai_stats:
    print(f"  {c:<12} {m:8.4f} {s:8.4f} {n:8d}")

# ── 7. Control: Shi poets grouped by form (诗体) ─────────────────────────
# For shi poets, group by shi form category (五言/七言) as a control
# If form constraint matters, within-form shi should be more cohesive
# than across-form shi — but effect should be weaker than cipai
print(f"\n{'='*60}")
print("Control: Shi form (五言/七言) vs Cipai cohesion comparison")
print("(If cipai effect > shi-form effect, it supports the genre-form hypothesis)")
print("="*60)

shi_poets = set()
for name, g in gsrc.items():
    if g.get('shi', 0) >= 20 and g.get('ci', 0) < 3:
        shi_poets.add(name)

# Get shi poets by their PC1 score (proxy for shi style differentiation)
shi_poet_list = [(name2idx[n], n) for n in shi_poets if n in name2idx]
print(f"  Shi poets (≥20 shi, <3 ci): {len(shi_poet_list)}")

if len(shi_poet_list) >= 50:
    shi_indices = [p[0] for p in shi_poet_list]
    shi_dists = []
    for i in range(min(5000, len(shi_indices))):
        a, b = random.sample(shi_indices, 2)
        shi_dists.append(float(cosine(embeddings[a], embeddings[b])))
    shi_mean = np.mean(shi_dists)
    print(f"  Shi random pairs: mean={shi_mean:.4f}  n={len(shi_dists)}")
    print(f"  Ci within-cipai:  mean={within_mean:.4f}")
    print(f"  Ci within-cipai is {shi_mean - within_mean:+.4f} tighter than shi random")

# ── 8. Save ──────────────────────────────────────────────────────────────
results = {
    "top10_cipai": top10,
    "valid_cipai": list(valid_cipai.keys()),
    "poet_level": {
        "within_cipai_mean": round(within_mean, 4),
        "within_cipai_std":  round(within_std, 4),
        "within_cipai_n":    len(within_dists),
        "across_cipai_mean": round(across_mean, 4),
        "across_cipai_std":  round(across_std, 4),
        "across_cipai_n":    len(across_dists),
        "diff_mean":         round(across_mean - within_mean, 4),
        "mann_whitney_p":    float(pval),
        "cliffs_delta":      round(cliff_delta, 4),
    },
    "per_cipai": [(c, round(m, 4), round(s, 4), n) for c, m, s, n in cipai_stats],
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nSaved to {OUT}")
