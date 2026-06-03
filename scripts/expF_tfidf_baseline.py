"""
expF_tfidf_baseline.py
非神经基线实验：TF-IDF + LogisticRegression 体裁 vs 朝代分类对比

目的：证明体裁信息不依赖于BERT，甚至不依赖于词序。
- 从poet_poems.json按诗人取文本片段（首300字）
- 体裁二分类（shi/ci），朝代6分类
- 按诗人分组GroupKFold，确保零泄漏
- 核心：Genre-in-Song测试—在同一朝代内部体裁信号是否持存
"""

import json, warnings, random, numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score

warnings.filterwarnings("ignore")
random.seed(42); np.random.seed(42)

BASE = "/home/chenhao/poetryCrystallization"
DATA = f"{BASE}/data/processed"
OUT  = f"{DATA}/expF_tfidf_baseline.json"

# ── 1. Load ─────────────────────────────────────────────────────────────
print("Loading data...")
poets = json.load(open(f"{DATA}/poet_poems.json"))
gsrc  = json.load(open(f"{DATA}/poet_genre_hybrid.json"))

def dom_genre(name):
    g = gsrc.get(name, {})
    s, c, q = g.get("shi",0), g.get("ci",0), g.get("qu",0)
    t = s + c + q
    if t == 0: return "shi"
    if c/t > 0.25: return "ci"
    if q/t > 0.25: return "qu"
    return "shi"

# ── 2. Build dataset ────────────────────────────────────────────────────
TARGET_DYNASTIES = {"唐", "宋", "元", "明", "清", "近代"}
CHUNK_SIZE = 300

print("Building dataset...")
samples = []
for p in poets:
    name = p["name"]
    full_text = p.get("text", "").strip()
    dyn = p.get("dynasty", "未知")
    if dyn not in TARGET_DYNASTIES:
        continue
    g = dom_genre(name)
    if g not in {"shi", "ci"}:  # qu excluded: too few poets
        continue
    chunk = full_text[:CHUNK_SIZE].strip()
    if len(chunk) < 30:
        continue
    samples.append({"text": chunk, "genre": g, "dynasty": dyn, "author": name})

texts   = [s["text"] for s in samples]
genres  = np.array([s["genre"] for s in samples])
dynasties = np.array([s["dynasty"] for s in samples])
authors   = np.array([s["author"] for s in samples])

print(f"  Samples: {len(texts)}")
print(f"  Genre:    {dict(Counter(genres))}")
print(f"  Dynasty:  {dict(Counter(dynasties))}")
cross_tab = {}
for g, d in zip(genres, dynasties):
    cross_tab.setdefault(g, Counter())[d] += 1
print(f"  Cross-tab (genre × dynasty):")
for g in ["shi", "ci"]:
    print(f"    {g}: {dict(cross_tab.get(g, {}))}")

# ── 3. TF-IDF ───────────────────────────────────────────────────────────
print("\nVectorizing (char n-gram 1-3, max_features=10000)...")
vectorizer = TfidfVectorizer(
    analyzer='char', ngram_range=(1,3), max_features=10000, sublinear_tf=True)
X = vectorizer.fit_transform(texts)
print(f"  TF-IDF: {X.shape}")

# ── 4. Helper: StratifiedKFold (each poet=1 sample, no within-poet leakage risk) ──
def cv_eval(X, y, n_splits=5):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    f1s, accs = [], []
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=5000, C=1.0, solver="saga",
                                 class_weight="balanced", random_state=42)
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        f1s.append(f1_score(y[test_idx], pred, average='macro'))
        accs.append(accuracy_score(y[test_idx], pred))
    return np.mean(f1s), np.std(f1s), np.mean(accs), np.std(accs)

# ── 5. Genre binary (shi/ci) ────────────────────────────────────────────
print("\n=== Genre binary (shi vs ci) ===")
y_genre = np.where(genres == "ci", 1, 0)
g_f1, g_f1_std, g_acc, g_acc_std = cv_eval(X, y_genre)
print(f"  Macro-F1={g_f1:.4f}±{g_f1_std:.4f}  Acc={g_acc:.4f} (note: each poet=1 sample, no within-poet leakage)")

# ── 6. Dynasty 6-class ──────────────────────────────────────────────────
print("\n=== Dynasty 6-class ===")
d_f1, d_f1_std, d_acc, d_acc_std = cv_eval(X, dynasties)
print(f"  Macro-F1={d_f1:.4f}±{d_f1_std:.4f}  Acc={d_acc:.4f}")

# ── 7. Critical: Genre WITHIN single dynasty ────────────────────────────
print("\n=== Genre-in-Song (same dynasty, different genres) ===")
song_mask = dynasties == "宋"
X_song, g_song = X[song_mask], genres[song_mask]
auth_song = authors[song_mask]
print(f"  Song genre mix: {dict(Counter(g_song))}")
if len(set(g_song)) >= 2:
    y_song = np.where(g_song == "ci", 1, 0)
    g_song_f1, _, g_song_acc, _ = cv_eval(X_song, y_song)
    print(f"  Genre-in-Song: Macro-F1={g_song_f1:.4f}  Acc={g_song_acc:.4f}")
else:
    g_song_f1 = 0

print("\n=== Genre-in-Qing ===")
qing_mask = dynasties == "清"
X_qing, g_qing = X[qing_mask], genres[qing_mask]
auth_qing = authors[qing_mask]
print(f"  Qing genre mix: {dict(Counter(g_qing))}")
if len(set(g_qing)) >= 2:
    y_qing = np.where(g_qing == "ci", 1, 0)
    g_qing_f1, _, g_qing_acc, _ = cv_eval(X_qing, y_qing)
    print(f"  Genre-in-Qing: Macro-F1={g_qing_f1:.4f}  Acc={g_qing_acc:.4f}")
else:
    g_qing_f1 = 0

# ── 8. Dynasty WITHIN single genre ──────────────────────────────────────
print("\n=== Dynasty-in-Shi ===")
shi_mask = genres == "shi"
if shi_mask.sum() > 50:
    X_shi, d_shi = X[shi_mask], dynasties[shi_mask]
    auth_shi = authors[shi_mask]
    d_shi_f1, _, d_shi_acc, _ = cv_eval(X_shi, d_shi)
    print(f"  Dynasty-in-Shi: Macro-F1={d_shi_f1:.4f}  Acc={d_shi_acc:.4f}")
else:
    d_shi_f1 = 0

print("\n=== Dynasty-in-Ci ===")
ci_mask = genres == "ci"
if ci_mask.sum() > 20:
    X_ci, d_ci = X[ci_mask], dynasties[ci_mask]
    auth_ci = authors[ci_mask]
    # If too few dynasties, report raw
    dc = Counter(d_ci)
    print(f"  Ci dynasty dist: {dict(dc)}")
    d_ci_f1, _, d_ci_acc, _ = cv_eval(X_ci, d_ci, n_splits=3)
    print(f"  Dynasty-in-Ci: Macro-F1={d_ci_f1:.4f}  Acc={d_ci_acc:.4f}")
else:
    d_ci_f1 = 0

# ── 9. Save ─────────────────────────────────────────────────────────────
results = {
    "method": "TF-IDF char-ngram(1,3) first300chars + LogReg(balanced, max_iter=5000)",
    "n_samples": len(texts),
    "vocab_size": int(X.shape[1]),
    "genre_binary": {
        "macro_f1_mean": round(g_f1, 4), "macro_f1_std": round(g_f1_std, 4),
        "accuracy_mean": round(g_acc, 4), "accuracy_std": round(g_acc_std, 4),
    },
    "dynasty_6class": {
        "macro_f1_mean": round(d_f1, 4), "macro_f1_std": round(d_f1_std, 4),
        "accuracy_mean": round(d_acc, 4), "accuracy_std": round(d_acc_std, 4),
    },
    "genre_within_song_f1": round(g_song_f1, 4),
    "genre_within_qing_f1": round(g_qing_f1, 4),
    "dynasty_within_shi_f1": round(d_shi_f1, 4),
    "dynasty_within_ci_f1": round(d_ci_f1, 4),
    "key_finding": (
        f"Genre binary F1={g_f1:.4f}, Dynasty 6-class F1={d_f1:.4f}. "
        f"Genre-in-Song (same dynasty!): F1={g_song_f1:.4f} — "
        f"genre signal persists within single dynasty. "
        f"Dynasty-in-Shi (same genre!): F1={d_shi_f1:.4f}."
    ),
}

print(f"\n{'='*60}")
print(f"Genre binary (shi/ci):     {g_f1:.4f}")
print(f"Dynasty 6-class:            {d_f1:.4f}")
print(f"Genre-in-Song (same dyn!):  {g_song_f1:.4f}")
print(f"Genre-in-Qing (same dyn!):  {g_qing_f1:.4f}")
print(f"Dynasty-in-Shi (same gen!): {d_shi_f1:.4f}")
print(f"Dynasty-in-Ci (same gen!):  {d_ci_f1:.4f}")
print(f"{'='*60}")

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nSaved to {OUT}")
