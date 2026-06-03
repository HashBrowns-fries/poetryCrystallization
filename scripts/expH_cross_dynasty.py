"""
expH_cross_dynasty.py
跨朝代泛化测试：体裁分类器 vs 朝代分类器的跨时间迁移能力

设计：
- 体裁三分类（shi/ci/qu）：Tang训练 → Song测试（可跨朝代泛化）
- 朝代二分类（朝代A vs 朝代B）：训练朝代A+B → 测试朝代C（不可泛化）
  用一对多设置：Tang+Song训练 (Tang vs Song)二元分类器 → 测试Qing
  Qing全预测为Tang或Song → 准确率接近随机

关键：体裁判别器可以跨代泛化，朝代判别器高度时间特定。
"""

import json, random, numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report

random.seed(42); np.random.seed(42)

BASE = "/home/chenhao/poetryCrystallization"
DATA = f"{BASE}/data/processed"
OUT  = f"{DATA}/expH_cross_dynasty.json"

# ── 1. Load data ─────────────────────────────────────────────────────────
print("Loading data...")
embeddings = np.load(f"{DATA}/poet_embeddings.npy").astype(np.float32)
poets      = json.load(open(f"{DATA}/poet_poems.json"))
gsrc       = json.load(open(f"{DATA}/poet_genre_hybrid.json"))

def dom_genre(name):
    g = gsrc.get(name, {})
    s, c, q = g.get("shi",0), g.get("ci",0), g.get("qu",0)
    t = s + c + q
    if t == 0: return "shi"
    if c/t > 0.25: return "ci"
    if q/t > 0.25: return "qu"
    return "shi"

genre_labels = np.array([dom_genre(p["name"]) for p in poets])
dynasty_labels = np.array([p.get("dynasty", "未知") for p in poets])

MAJOR_DYN = ["唐", "宋", "元", "明", "清", "近代"]
# Map dynasty to group index for cross-dynasty
dyn_map = {d: i for i, d in enumerate(MAJOR_DYN)}
print(f"  Total poets: {len(poets)}")

# ── 2. Cross-dynasty genre classification ────────────────────────────────
print("\n" + "="*60)
print("Cross-Dynasty Genre Classification")
print("  Train on dynasty A, test on dynasty B")
print("="*60)

cross_results = {}

for train_dyn in MAJOR_DYN:
    train_mask = dynasty_labels == train_dyn
    train_idx = np.where(train_mask)[0]
    if len(train_idx) < 30:
        continue

    X_train = embeddings[train_idx]
    y_train = genre_labels[train_idx]

    # Only classes present in training set
    valid_classes = set(y_train)

    for test_dyn in MAJOR_DYN:
        if test_dyn == train_dyn:
            continue
        test_mask = dynasty_labels == test_dyn
        test_idx = np.where(test_mask)[0]
        if len(test_idx) < 20:
            continue

        # Filter test set to classes seen in training
        y_test_raw = genre_labels[test_idx]
        test_valid_mask = np.isin(y_test_raw, list(valid_classes))
        if test_valid_mask.sum() < 10:
            continue

        X_test = embeddings[test_idx][test_valid_mask]
        y_test = y_test_raw[test_valid_mask]

        # Train classifier
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="saga",
                                 class_weight="balanced", random_state=42)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        acc = accuracy_score(y_test, pred)
        f1  = f1_score(y_test, pred, average='macro')

        key = f"{train_dyn}→{test_dyn}"
        cross_results[key] = {
            "train_n": len(train_idx), "test_n": len(X_test),
            "accuracy": round(float(acc), 4), "macro_f1": round(float(f1), 4),
            "train_classes": list(valid_classes),
        }
        print(f"  {key}: acc={acc:.4f}  F1={f1:.4f}  (n_train={len(train_idx)}, n_test={len(X_test)})")

# ── 3. Control: Dynasty binary classifier (cannot generalize) ────────────
print("\n" + "="*60)
print("Control: Dynasty Binary Classifier Cross-Dynasty Test")
print("  Train on (Dyn A vs Dyn B), test on Dyn C")
print("  Expectation: performance ≈ majority-class baseline")
print("="*60)

dynasty_control_results = {}

# Tang vs Song classifier, test on Qing
for (dyn_a, dyn_b), test_dyn in [
    (("唐", "宋"), "清"),
    (("唐", "宋"), "明"),
    (("明", "清"), "宋"),
    (("元", "明"), "清"),
]:
    train_mask = np.isin(dynasty_labels, [dyn_a, dyn_b])
    test_mask  = dynasty_labels == test_dyn

    train_idx = np.where(train_mask)[0]
    test_idx  = np.where(test_mask)[0]

    if len(train_idx) < 30 or len(test_idx) < 20:
        continue

    X_train = embeddings[train_idx]
    y_train = np.where(dynasty_labels[train_idx] == dyn_a, 0, 1)

    X_test = embeddings[test_idx]
    # There's no "dyn_a or dyn_b" in test — the task is impossible
    # Any prediction is arbitrary → accuracy should be near chance
    y_test = np.zeros(len(test_idx))  # dummy

    clf = LogisticRegression(max_iter=2000, C=1.0, solver="saga",
                             class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    # What fraction did it label as dyn_a (class 0)?
    pct_dyn_a = np.mean(pred == 0)

    key = f"{dyn_a}vs{dyn_b}→{test_dyn}"
    dynasty_control_results[key] = {
        "train_n": len(train_idx), "test_n": len(test_idx),
        "pct_predicted_as": {dyn_a: round(float(pct_dyn_a), 4),
                             dyn_b: round(float(1 - pct_dyn_a), 4)},
        "interpretation": (
            f"Trained to distinguish {dyn_a} vs {dyn_b}, "
            f"tested on {test_dyn} (neither class seen in training). "
            f"Classifier arbitrarily assigns {pct_dyn_a*100:.1f}% to {dyn_a}. "
            f"No meaningful generalization possible."
        ),
    }
    print(f"  {key}: {pct_dyn_a*100:.1f}% → {dyn_a}, "
          f"{(1-pct_dyn_a)*100:.1f}% → {dyn_b} (arbitrary)")

# ── 4. Summary ───────────────────────────────────────────────────────────
genre_f1s = [v["macro_f1"] for v in cross_results.values()]
print(f"\n{'='*60}")
print(f"Cross-Dynasty Genre Classification:")
print(f"  Mean F1: {np.mean(genre_f1s):.4f} ± {np.std(genre_f1s):.4f}")
print(f"  Range: [{min(genre_f1s):.4f}, {max(genre_f1s):.4f}]")
print(f"  N pairs: {len(genre_f1s)}")
print(f"\nDynasty Binary Classifier:")
print(f"  CANNOT generalize cross-dynasty (task is ill-posed)")
print(f"{'='*60}")

results = {
    "cross_dynasty_genre": cross_results,
    "dynasty_control": dynasty_control_results,
    "summary": {
        "genre_cross_f1_mean": round(float(np.mean(genre_f1s)), 4),
        "genre_cross_f1_std":  round(float(np.std(genre_f1s)), 4),
        "genre_cross_n_pairs": len(genre_f1s),
        "dynasty_cannot_generalize": True,
    },
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nSaved to {OUT}")
