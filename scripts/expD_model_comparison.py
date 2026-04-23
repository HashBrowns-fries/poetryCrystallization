#!/usr/bin/env python3
"""
expD_model_comparison.py
Exp D: BERT-CCPoem vs GuwenBERT 诗人-体裁分类对比（80/20划分）

课程对应：词向量模型对比实验
"""

import json, os, time, random
import numpy as np
from numpy.linalg import norm

BASE = "/home/chenhao/poetryCrystallization"
OUT  = f"{BASE}/data/processed/expD_model_comparison.json"

random.seed(42); np.random.seed(42)

# ── 数据加载 ────────────────────────────────────────────────────────────────
print("加载数据...")
poets = json.load(open(f"{BASE}/data/processed/poet_poems.json", encoding="utf-8"))
gsrc  = json.load(open(f"{BASE}/data/processed/poet_genre_by_source.json", encoding="utf-8"))
X_cc  = np.load(f"{BASE}/data/processed/poet_embeddings.npy").astype(np.float32)
X_cc_n = X_cc / (norm(X_cc, axis=1, keepdims=True) + 1e-9)

def dom_genre(name):
    g = gsrc.get(name, {})
    s,c,q,f = g.get("shi",0),g.get("ci",0),g.get("qu",0),g.get("fu",0)
    t=s+c+q+f
    if t==0: return "shi"
    if c/t>0.25: return "ci"
    if q/t>0.25: return "qu"
    return "shi"

poet_labels = [1 if dom_genre(p["name"])=="ci" else 0 for p in poets]
poet_names  = [p["name"] for p in poets]
ci_names    = [n for n in poet_names if dom_genre(n)=="ci"]
shi_names   = [n for n in poet_names if dom_genre(n)=="shi"]
all_names   = ci_names + shi_names
all_labels  = [1]*len(ci_names) + [0]*len(shi_names)
all_idx     = [i for i,p in enumerate(poets) if p["name"] in set(all_names)]
X_cc_feat   = X_cc_n[all_idx]

print(f"ci诗人数: {len(ci_names)}, shi诗人数: {len(shi_names)}")
print(f"BERT-CCPoem嵌入: {X_cc_feat.shape}")

# ── sklearn + 交叉验证 ──────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 80/20 划分，一次性评估
X_train, X_test, y_train, y_test = train_test_split(
    X_cc_feat, np.array(all_labels), test_size=0.2,
    stratify=np.array(all_labels), random_state=42)

def eval_model(X_feat_train, X_feat_test, y_train, y_test, name):
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", class_weight="balanced")
    clf.fit(X_feat_train, y_train)
    y_pred = clf.predict(X_feat_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="binary")
    print(f"  {name}: acc={acc:.4f}  F1={f1:.4f}")
    return {"acc": round(float(acc),4), "f1": round(float(f1),4)}

print("\n80/20 划分评估（LogisticRegression）：")
r_cc = eval_model(X_train, X_test, y_train, y_test, "BERT-CCPoem (512D)")

# ── GuwenBERT 嵌入 ────────────────────────────────────────────────────────
print("\n尝试加载 ethanyt/guwenbert-base ...")
try:
    from transformers import AutoTokenizer, AutoModel
    tok  = AutoTokenizer.from_pretrained("ethanyt/guwenbert-base")
    gwen = AutoModel.from_pretrained("ethanyt/guwenbert-base")
    gwen.eval()
    print("GuwenBERT 加载成功！")
    HAS_GUWEN = True
except Exception as e:
    print(f"GuwenBERT 加载失败: {e}")
    HAS_GUWEN = False

if HAS_GUWEN:
    print("获取 GuwenBERT 诗人嵌入（均值池化，每诗人最多30首）...")
    import torch
    poems_by_name = {p["name"]: p.get("text","") for p in poets}

    gwen_feats = []
    with torch.no_grad():
        for i, name in enumerate(all_names):
            text = poems_by_name.get(name, "")
            if not text:
                gwen_feats.append(np.zeros(768))
                continue
            # 取前500字（避免过长）
            enc = tok([text[:500]], truncation=True, max_length=512,
                       padding=True, return_tensors="pt")
            out = gwen(**enc)
            emb = out.last_hidden_state.mean(dim=1).cpu().numpy().mean(axis=0)
            gwen_feats.append(emb)
            if (i+1) % 500 == 0:
                print(f"  已处理 {i+1}/{len(all_names)} 位诗人...")

    X_gwen = np.array(gwen_feats)
    X_gwen_n = X_gwen / (norm(X_gwen, axis=1, keepdims=True) + 1e-9)
    print(f"GuwenBERT嵌入: {X_gwen_n.shape}")
    X_gw_train, X_gw_test, y_gw_train, y_gw_test = train_test_split(
        X_gwen_n, np.array(all_labels), test_size=0.2,
        stratify=np.array(all_labels), random_state=42)
    r_gw = eval_model(X_gw_train, X_gw_test, y_gw_train, y_gw_test, "GuwenBERT (768D)")
    acc_diff = r_cc["acc"] - r_gw["acc"]
    print(f"\nBERT-CCPoem vs GuwenBERT acc差值: {acc_diff:+.4f}")
    interp = (f"BERT-CCPoem {'优于' if acc_diff>0 else '劣于'}GuwenBERT "
              f"{abs(acc_diff)*100:.2f}%")
else:
    r_gw = None
    acc_diff = None
    interp = "GuwenBERT加载失败，仅报告BERT-CCPoem结果"

print(f"\n解读: {interp}")

# ── 保存 ─────────────────────────────────────────────────────────────────
results = {
    "n_ci": len(ci_names), "n_shi": len(shi_names),
    "split": "80/20 stratified",
    "classifier": "LogisticRegression(C=1.0)",
    "BERT-CCPoem": r_cc,
    "GuwenBERT": r_gw,
    "acc_diff_ccpoem_minus_gwen": round(float(acc_diff),4) if acc_diff is not None else None,
    "interpretation": interp
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n已保存: {OUT}")
