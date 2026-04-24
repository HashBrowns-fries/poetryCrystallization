"""
expE_cipai_counterfactual.py
反事实实验E：词牌格律是ci语义内聚的机制吗？

实验逻辑：
  - H0: ci诗人语义相近是因为词牌格律（形式约束）
    → 同一词牌内诗人应比同ci体裁但不同词牌的诗人更近
  - 若同词牌距离 ≈ 同ci不同词牌距离 → 词牌格律不是机制

数据：
  - poet_embeddings.npy: 4634 × 512 诗人嵌入
  - poet_poems.json: 4634诗人，id=0..4633（与嵌入索引一一对应）
  - poet_genre_hybrid.json: {poet_name → {ci:N, shi:N, ...}}
  - data/raw/chinese-poetry/宋词/: 21053首宋词，带词牌名(rhythmic)
"""

import json, glob, numpy as np, random
from collections import defaultdict
from scipy.spatial.distance import cosine
from scipy.stats import mannwhitneyu

BASE = '/home/chenhao/poetryCrystallization'
DATA  = f'{BASE}/data/processed'
RAW   = f'{BASE}/data/raw/chinese-poetry'

# ── 1. Load ──────────────────────────────────────────────────────────────────
embeddings = np.load(f'{DATA}/poet_embeddings.npy')
poets      = json.load(open(f'{DATA}/poet_poems.json'))
genre_h    = json.load(open(f'{DATA}/poet_genre_hybrid.json'))
print(f'Poets: {len(poets)}, Embeddings: {embeddings.shape}')

id2name = {p['id']: p['name'] for p in poets}
name2id = {p['name']: p['id'] for p in poets}

# ── 2. Identify ci poets (≥5 ci poems) ─────────────────────────────────────
ci_poet_ids = set()
for name, g in genre_h.items():
    if g.get('ci', 0) >= 5:
        pid = name2id.get(name)
        if pid is not None:
            ci_poet_ids.add(pid)
print(f'Ci poets (≥5 ci poems): {len(ci_poet_ids)}')

# ── 3. Load ci poems with (author, rhythmic) from raw ───────────────────────
ci_files = sorted(glob.glob(f'{RAW}/宋词/ci.song.*.json'))
all_ci = []
for f in ci_files:
    for p in json.load(open(f)):
        all_ci.append({'author': p['author'], 'rhythmic': p.get('rhythmic', '失调名')})
print(f'Total ci poems from raw: {len(all_ci)}')

# ── 4. Build: cipa → [poet_ids] ────────────────────────────────────────────
cipa_poets = defaultdict(list)
for poem in all_ci:
    pid = name2id.get(poem['author'])
    if pid is not None and pid in ci_poet_ids:
        cipa_poets[poem['rhythmic']].append(pid)
for cipa in cipa_poets:
    cipa_poets[cipa] = list(set(cipa_poets[cipa]))

valid_cipa = {c: pl for c, pl in cipa_poets.items() if len(pl) >= 2}
print(f'Valid cipai (≥2 ci poets): {len(valid_cipa)}')

# ── 5. Compute distances ────────────────────────────────────────────────────
same_cipa_dists = []
diff_cipa_dists = []

poet_main_cipa = {}
for cipa, pl in valid_cipa.items():
    for pid in pl:
        if pid not in poet_main_cipa:
            poet_main_cipa[pid] = cipa

for cipa, pl in valid_cipa.items():
    if len(pl) < 2:
        continue
    for i in range(len(pl)):
        for j in range(i + 1, len(pl)):
            same_cipa_dists.append(cosine(embeddings[pl[i]], embeddings[pl[j]]))

random.seed(42)
ci_poet_list = list(ci_poet_ids)
n_ci = len(ci_poet_list)
sampled = 0
attempts = 0
while sampled < 15000 and attempts < 200000:
    attempts += 1
    i, j = random.sample(range(n_ci), 2)
    pid_i, pid_j = ci_poet_list[i], ci_poet_list[j]
    cipa_i = poet_main_cipa.get(pid_i, None)
    cipa_j = poet_main_cipa.get(pid_j, None)
    if cipa_i and cipa_j and cipa_i != cipa_j:
        diff_cipa_dists.append(cosine(embeddings[pid_i], embeddings[pid_j]))
        sampled += 1

print(f'\nSame-cipa pairs: {len(same_cipa_dists)}')
print(f'Diff-cipa pairs: {len(diff_cipa_dists)}')

# ── 6. Statistics ───────────────────────────────────────────────────────────
same_mean = np.mean(same_cipa_dists)
same_std  = np.std(same_cipa_dists)
diff_mean = np.mean(diff_cipa_dists)
diff_std  = np.std(diff_cipa_dists)

print(f'\n{"="*50}')
print(f'同词牌诗人对距离:  mean={same_mean:.4f}  std={same_std:.4f}  n={len(same_cipa_dists)}')
print(f'异词牌诗人对距离:  mean={diff_mean:.4f}  std={diff_std:.4f}  n={len(diff_cipa_dists)}')
print(f'差值 (异-同):      {diff_mean - same_mean:+.4f}')

stat, pval = mannwhitneyu(same_cipa_dists, diff_cipa_dists, alternative='less')
pooled_std = np.sqrt((np.var(same_cipa_dists) + np.var(diff_cipa_dists)) / 2)
cohens_d   = (diff_mean - same_mean) / pooled_std

print(f'\nMann-Whitney U (H1: same-cipa < diff-cipa):')
print(f'  stat={stat:.0f},  p={pval:.2e}')
print(f"Cohen's d (diff - same): {cohens_d:.4f}")

# ── 7. Per-cipa breakdown ───────────────────────────────────────────────────
print(f'\n{"="*50}')
print('各词牌内部距离（按均值升序）:')
cipa_stats = []
for cipa, pl in valid_cipa.items():
    if len(pl) < 3:
        continue
    dists = [cosine(embeddings[pl[i]], embeddings[pl[j]])
             for i in range(len(pl)) for j in range(i+1, len(pl))]
    if dists:
        cipa_stats.append((cipa, np.mean(dists), np.std(dists), len(pl), len(dists)))

cipa_stats.sort(key=lambda x: x[1])
print(f'{"词牌":<12} {"均值":>8} {"标准差":>8} {"诗人数":>8} {"诗人对数":>8}')
for cipa, m, s, n, npairs in cipa_stats[:20]:
    print(f'{cipa:<12} {m:8.4f} {s:8.4f} {n:8d} {npairs:8d}')

# ── 8. Save ──────────────────────────────────────────────────────────────────
results = {
    'same_cipa_mean': float(same_mean),
    'same_cipa_std':  float(same_std),
    'same_cipa_n':    len(same_cipa_dists),
    'diff_cipa_mean': float(diff_mean),
    'diff_cipa_std':  float(diff_std),
    'diff_cipa_n':    len(diff_cipa_dists),
    'mann_whitney_p': float(pval),
    'cohens_d':       float(cohens_d),
    'cipa_stats':     [(c, float(m), float(s), int(n), int(npairs))
                       for c, m, s, n, npairs in cipa_stats],
}
out = f'{DATA}/expE_cipai_counterfactual.json'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f'\nResults → {out}')
