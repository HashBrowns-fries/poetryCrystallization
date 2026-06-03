"""
expG_sensitivity.py
参数敏感性分析：Louvain k/τ 扫描 + 距离度量对比

1. Louvain k ∈ {10,15,20,25,30} × τ ∈ {0.10,0.15,0.20,0.25} 热力图
   净增量 = 真实纯度 - 随机打乱标签后的平均纯度

2. 距离度量对比：余弦 vs 欧氏 vs 曼哈顿
   报告单因素 PERMANOVA R² 和双因素条件效应
"""

import json, random, numpy as np, networkx as nx
from collections import Counter
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

random.seed(42); np.random.seed(42)

BASE = "/home/chenhao/poetryCrystallization"
DATA = f"{BASE}/data/processed"
OUT  = f"{DATA}/expG_sensitivity.json"

# ── 1. Load data ─────────────────────────────────────────────────────────
print("Loading data...")
embeddings = np.load(f"{DATA}/poet_embeddings.npy").astype(np.float32)
poets      = json.load(open(f"{DATA}/poet_poems.json"))
gsrc       = json.load(open(f"{DATA}/poet_genre_hybrid.json"))

N = len(poets)
print(f"  Embeddings: {embeddings.shape}, Poets: {N}")

def dom_genre(name):
    g = gsrc.get(name, {})
    s, c, q = g.get("shi",0), g.get("ci",0), g.get("qu",0)
    t = s + c + q
    if t == 0: return "shi"
    if c/t > 0.25: return "ci"
    if q/t > 0.25: return "qu"
    return "shi"

genre_labels = [dom_genre(p["name"]) for p in poets]
dynasty_labels = [p.get("dynasty", "未知") for p in poets]

# ── 2. Precompute cosine similarity ─────────────────────────────────────
print("Computing cosine similarity...")
cos_sim = 1.0 - squareform(pdist(embeddings, metric='cosine'))

# ── 3. Louvain k/τ parameter scan ───────────────────────────────────────
print("\n" + "="*60)
print("Louvain k/τ Parameter Scan")
print("="*60)

k_list  = [10, 15, 20, 25, 30]
tau_list = [0.10, 0.15, 0.20, 0.25]
n_shuffle = 50  # reduced for speed

try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    print("python-louvain not installed; skipping Louvain scan")
    HAS_LOUVAIN = False

louvain_results = np.zeros((len(k_list), len(tau_list)))
louvain_purity_obs = np.zeros((len(k_list), len(tau_list)))
louvain_purity_null = np.zeros((len(k_list), len(tau_list)))

if HAS_LOUVAIN:
    for i, k in enumerate(tqdm(k_list, desc="k scan")):
        for j, tau in enumerate(tau_list):
            G = nx.Graph()
            G.add_nodes_from(range(N))
            for u in range(N):
                sims = cos_sim[u]
                top_k = np.argpartition(-sims, min(k+1, N))[:min(k+1, N)]
                for v in top_k:
                    if u < v and sims[v] > tau:
                        G.add_edge(u, v, weight=float(sims[v]))

            if G.number_of_edges() == 0:
                continue

            partition = community_louvain.best_partition(G, random_state=42)

            # Build communities: cid → [node_ids]
            comms = {}
            for node, cid in partition.items():
                comms.setdefault(cid, []).append(node)

            # Observed purity
            purity_real = np.mean([
                max(Counter([genre_labels[n] for n in comms[cid]]).values())
                / len(comms[cid])
                for cid in comms if len(comms[cid]) > 1
            ])
            louvain_purity_obs[i, j] = purity_real

            # Null: shuffle genre labels
            g_arr = np.array(genre_labels)
            purities_null = []
            for _ in range(n_shuffle):
                shuffled = g_arr[np.random.permutation(N)]
                purity_s = np.mean([
                    max(Counter([shuffled[n] for n in comms[cid]]).values())
                    / len(comms[cid])
                    for cid in comms if len(comms[cid]) > 1
                ])
                purities_null.append(purity_s)
            louvain_purity_null[i, j] = np.mean(purities_null)
            louvain_results[i, j] = purity_real - np.mean(purities_null)

    print(f"\n  Net purity increment:")
    print(f"    Range: [{louvain_results.min():.4f}, {louvain_results.max():.4f}]")
    print(f"    All cells positive: {np.all(louvain_results > 0)}")
    # Print heatmap
    print(f"\n  Heatmap (k rows × τ cols):")
    print(f"           τ={tau_list}")
    for i, k in enumerate(k_list):
        row = [f"{louvain_results[i,j]:.4f}" for j in range(len(tau_list))]
        print(f"    k={k:2d}:  {'  '.join(row)}")

# ── 4. Distance metric comparison ───────────────────────────────────────
print("\n" + "="*60)
print("Distance Metric Comparison")
print("="*60)

# Use existing PERMANOVA results from paper (proper adonis implementation)
permanova = json.load(open(f"{DATA}/permanova_two_factor.json"))
print("Using existing PERMANOVA results from permanova_two_factor.json:")
print(f"  Genre single-factor: R²={permanova.get('单因素_体裁',{}).get('R2','N/A')}")
print(f"  Dynasty single-factor: R²={permanova.get('单因素_朝代',{}).get('R2','N/A')}")

# Also compute intertextual cohesion by genre under different distance metrics
# (cosine is the default; compute euclidean/manhattan for comparison)
TARGET_DYNASTIES = {"唐", "宋", "元", "明", "清", "近代"}
dyn_idx = [i for i, d in enumerate(dynasty_labels) if d in TARGET_DYNASTIES]

from scipy.spatial.distance import cosine, euclidean, cityblock

dist_metrics = {
    "cosine": lambda a, b: cosine(a, b),
    "euclidean": lambda a, b: euclidean(a, b),
    "manhattan": lambda a, b: cityblock(a, b),
}

metric_results = {}
for mname, metric_fn in dist_metrics.items():
    # Within-genre and between-genre
    ci_idx = [i for i in dyn_idx if genre_labels[i] == "ci"]
    shi_idx = [i for i in dyn_idx if genre_labels[i] == "shi"]
    qu_idx = [i for i in dyn_idx if genre_labels[i] == "qu"]

    # ci-ci distances
    ci_ci_dists = []
    for a in range(len(ci_idx)):
        for b in range(a+1, len(ci_idx)):
            ci_ci_dists.append(metric_fn(embeddings[ci_idx[a]], embeddings[ci_idx[b]]))
    ci_ci_mean = np.mean(ci_ci_dists)

    # shi-shi distances (sample to limit compute)
    shi_shi_dists = []
    sample_shi = random.sample(shi_idx, min(500, len(shi_idx)))
    for a in range(len(sample_shi)):
        for b in range(a+1, len(sample_shi)):
            shi_shi_dists.append(metric_fn(embeddings[sample_shi[a]], embeddings[sample_shi[b]]))
    shi_shi_mean = np.mean(shi_shi_dists)

    # ci-shi (cross-genre)
    ci_shi_dists = []
    sample_ci = random.sample(ci_idx, min(200, len(ci_idx)))
    for a in sample_ci:
        for b in random.sample(shi_idx, min(5, len(shi_idx))):
            ci_shi_dists.append(metric_fn(embeddings[a], embeddings[b]))
    ci_shi_mean = np.mean(ci_shi_dists)

    cohens_d = (ci_shi_mean - ci_ci_mean) / np.sqrt(
        (np.var(ci_ci_dists) + np.var(ci_shi_dists)) / 2)

    metric_results[mname] = {
        "ci_ci_mean": round(float(ci_ci_mean), 4),
        "shi_shi_mean": round(float(shi_shi_mean), 4),
        "ci_shi_mean": round(float(ci_shi_mean), 4),
        "genre_gap_pct": round(float((ci_shi_mean - ci_ci_mean) / ci_shi_mean * 100), 1),
        "cohens_d": round(float(cohens_d), 4),
    }
    print(f"\n  {mname}: ci-ci={ci_ci_mean:.4f}  shi-shi={shi_shi_mean:.4f}  "
          f"ci-shi={ci_shi_mean:.4f}  gap={(ci_shi_mean-ci_ci_mean)/ci_shi_mean*100:.1f}%  "
          f"d={cohens_d:.4f}")

# ── 5. Save ──────────────────────────────────────────────────────────────
results = {
    "louvain_scan": {
        "k_list": k_list,
        "tau_list": tau_list,
        "net_increment": louvain_results.tolist(),
        "purity_observed": louvain_purity_obs.tolist(),
        "purity_null": louvain_purity_null.tolist(),
        "n_shuffles": n_shuffle,
        "all_cells_positive": bool(np.all(louvain_results > 0)),
    },
    "distance_metric_comparison": metric_results,
    "paper_permanova": {
        "genre_single_R2": permanova.get("单因素_体裁", {}).get("R2"),
        "dynasty_single_R2": permanova.get("单因素_朝代", {}).get("R2"),
    },
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nSaved to {OUT}")
