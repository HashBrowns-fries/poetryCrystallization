#!/usr/bin/env python3
"""
expM_label_leakage.py
§4 BERT 分类 label-leakage 三档对照实验

回应审稿意见：F1=0.98 的体裁分类是否来自语义，还是来自元数据/格式/正字法泄漏？

三档输入清洗（同一模型、同一超参、同一诗人级分割，仅改变输入文本）：
  A = raw + metadata   : 词牌名(rhythmic)/曲牌名/标题(title)/作者(author) + 正文，
                         保留标点、换行、原始繁/简正字法
  B = body + format    : 仅正文，保留标点/换行/原始正字法，去掉所有元数据
  C = normalized body  : 仅正文，繁→简统一(opencc t2s) + 去标点 + 去换行
                         （消除正字法 + 标点 + 行结构三类格式泄漏）

两个分类任务：
  T1  3-class shi/ci/qu          —— 体裁↔朝代高度共线（唐=shi, 宋词=ci, 元=qu）
  T2  within-Song  shi vs ci     —— 朝代受控：宋诗(全唐诗/poet.song) vs 宋词
                                    若 C 档仍显著>50%，则体裁信号独立于朝代+正字法

附带诊断：每档每体裁的 tokenizer [UNK] 率（量化正字法泄漏）

决策规则（用户既定）：
  - 若 C 档 ci/qu 仍显著高于随机基线 → §4 保留为「语义证据」
  - 若 C 档性能大幅坍塌            → §4 降级为「格式/元数据可识别性实验」
"""

import json, os, time, random, re, glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings; warnings.filterwarnings("ignore")
from opencc import OpenCC

SEED = 42
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
set_seed(SEED)

BASE   = "/home/chenhao/poetryCrystallization"
MODEL  = f"{BASE}/data/models/BERT_CCPoem_v1"
RAW    = f"{BASE}/data/raw/chinese-poetry/chinese-poetry-master"
OUT    = f"{BASE}/data/processed/expM_label_leakage.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device {device}] SEED={SEED}")

cc = OpenCC('t2s')
CJK = re.compile(r'[一-鿿]')

def strip_to_cjk(s):
    return ''.join(CJK.findall(s))

def make_text(rec, tier):
    body = ''.join(rec.get('paragraphs', []))
    if tier == 'A':
        meta = ' '.join(x for x in [rec.get('rhythmic',''), rec.get('title',''),
                                    rec.get('author','')] if x)
        return (meta + ' ' + body).strip()
    if tier == 'B':
        return body
    if tier == 'C':
        return strip_to_cjk(cc.convert(body))
    raise ValueError(tier)

# ════════════════════════════════════════════════════════════════════════════
# 1. 加载原始数据（保留 title/rhythmic/author/paragraphs 分离字段）
# ════════════════════════════════════════════════════════════════════════════
def load_files(patterns, dynasty, genre):
    recs = []
    for pat in patterns:
        for fp in sorted(glob.glob(os.path.join(RAW, pat))):
            try:
                data = json.load(open(fp, encoding='utf-8'))
            except Exception:
                continue
            for r in data:
                paras = r.get('paragraphs', [])
                if not paras:
                    continue
                if len(''.join(paras)) < 8:
                    continue
                recs.append({
                    "author": r.get('author', '?'),
                    "title": r.get('title', '') or '',
                    "rhythmic": r.get('rhythmic', '') or '',
                    "paragraphs": paras,
                    "dynasty": dynasty,
                    "genre": genre,
                })
    return recs

print("\n[1] 加载原始数据 ...")
tang_shi = load_files(["全唐诗/poet.tang.*.json"], "Tang", "shi")
song_shi = load_files(["全唐诗/poet.song.*.json"], "Song", "shi")
song_ci  = load_files(["宋词/ci.song.*.json"],     "Song", "ci")
yuan_qu  = load_files(["元曲/yuanqu.json"],         "Yuan", "qu")
for nm, r in [("Tang shi",tang_shi),("Song shi",song_shi),
              ("Song ci",song_ci),("Yuan qu",yuan_qu)]:
    print(f"  {nm}: {len(r)} poems, {len({x['author'] for x in r})} authors")

# ════════════════════════════════════════════════════════════════════════════
# 2. 诗人级分割 + 平衡采样工具
# ════════════════════════════════════════════════════════════════════════════
def poet_split(recs, test_frac=0.2, seed=SEED):
    """同一作者所有诗同在 train 或 val，杜绝诗人级泄漏"""
    rng = random.Random(seed)
    authors = sorted({r['author'] for r in recs})
    rng.shuffle(authors)
    n_val = int(len(authors) * test_frac)
    val_a = set(authors[:n_val])
    tr = [r for r in recs if r['author'] not in val_a]
    va = [r for r in recs if r['author'] in val_a]
    return tr, va

def balance(recs_by_label, max_per, seed=SEED):
    rng = random.Random(seed)
    out = []
    for lab, recs in recs_by_label.items():
        rr = recs[:]
        rng.shuffle(rr)
        for r in rr[:max_per]:
            out.append((r, lab))
    rng.shuffle(out)
    return out

# ════════════════════════════════════════════════════════════════════════════
# 3. Dataset / 模型 (mean-pooling, CE) —— 与 expC_v2 同构
# ════════════════════════════════════════════════════════════════════════════
tok = BertTokenizer.from_pretrained(MODEL)

class DS(Dataset):
    def __init__(self, pairs, tier, max_len=256):
        self.pairs = pairs; self.tier = tier; self.max = max_len
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        rec, lab = self.pairs[i]
        enc = tok(make_text(rec, self.tier), truncation=True, max_length=self.max,
                  padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}, lab

class Clf(nn.Module):
    def __init__(self, n_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(512, n_labels)
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        emb = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.fc(self.drop(emb))

def unk_rate(pairs, tier, n_sample=2000):
    rng = random.Random(SEED)
    sample = rng.sample(pairs, min(n_sample, len(pairs)))
    tot = unk = 0
    for rec, _ in sample:
        ids = tok(make_text(rec, tier), truncation=True, max_length=256)['input_ids']
        tot += len(ids)
        unk += sum(1 for x in ids if x == tok.unk_token_id)
    return unk / max(tot, 1)

def run(train_pairs, val_pairs, tier, n_labels, label_names,
        epochs=3, bs=32, lr=2e-5, tag=""):
    set_seed(SEED)
    trd = DataLoader(DS(train_pairs, tier), batch_size=bs, shuffle=True,
                     num_workers=4, pin_memory=True)
    vad = DataLoader(DS(val_pairs, tier), batch_size=bs*2, shuffle=False,
                     num_workers=4, pin_memory=True)
    model = Clf(n_labels).to(device)
    crit = nn.CrossEntropyLoss()
    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.1*len(trd)*epochs), len(trd)*epochs)

    best = None; best_f1 = -1
    for ep in range(epochs):
        model.train()
        for batch, y in trd:
            opt.zero_grad()
            logits = model(**{k: v.to(device) for k, v in batch.items()})
            loss = crit(logits, y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
        st = _eval(model, vad)
        print(f"    [{tag}|{tier}] ep{ep+1} acc={st['acc']:.4f} F1={st['f1_macro']:.4f} "
              + " ".join(f"{n}={f:.3f}" for n, f in zip(label_names, st['f1_per'])))
        if st['f1_macro'] > best_f1:
            best_f1 = st['f1_macro']
            best = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict({k: v.to(device) for k, v in best.items()})
    st = _eval(model, vad)
    st['unk_rate'] = unk_rate(val_pairs, tier)
    del model; torch.cuda.empty_cache()
    return st

@torch.no_grad()
def _eval(model, dl):
    model.eval()
    P, T = [], []
    for batch, y in dl:
        logits = model(**{k: v.to(device) for k, v in batch.items()})
        P.extend(logits.argmax(1).cpu().tolist()); T.extend(y.tolist())
    P, T = np.array(P), np.array(T)
    return {"acc": accuracy_score(T, P),
            "f1_macro": f1_score(T, P, average="macro"),
            "f1_per": f1_score(T, P, average=None).tolist(),
            "cm": confusion_matrix(T, P).tolist(),
            "n_val": int(len(T))}

# ════════════════════════════════════════════════════════════════════════════
# 4. T1: 3-class shi/ci/qu （体裁↔朝代共线）
# ════════════════════════════════════════════════════════════════════════════
MAX_TRAIN = 8000     # 每类训练上限
MAX_VAL   = 2000     # 每类验证上限
LABELS3   = ["shi", "ci", "qu"]

print("\n[2] T1 数据准备（诗人级分割, shi=Tang+Song诗, ci=宋词, qu=元曲）...")
# shi 合并 Tang+Song 诗；分割在各自来源内做（保证诗人级）
shi_all = tang_shi + song_shi
shi_tr, shi_va = poet_split(shi_all)
ci_tr,  ci_va  = poet_split(song_ci)
qu_tr,  qu_va  = poet_split(yuan_qu)

t1_train = balance({0: shi_tr, 1: ci_tr, 2: qu_tr}, MAX_TRAIN)
t1_val   = balance({0: shi_va, 1: ci_va, 2: qu_va}, MAX_VAL)
print(f"  T1 train={len(t1_train)} val={len(t1_val)}")

# ════════════════════════════════════════════════════════════════════════════
# 5. T2: within-Song shi vs ci （朝代受控）
# ════════════════════════════════════════════════════════════════════════════
LABELS2 = ["song_shi", "song_ci"]
print("\n[3] T2 数据准备（朝代受控：宋诗 vs 宋词）...")
sshi_tr, sshi_va = poet_split(song_shi)
sci_tr,  sci_va  = poet_split(song_ci)
t2_train = balance({0: sshi_tr, 1: sci_tr}, MAX_TRAIN)
t2_val   = balance({0: sshi_va, 1: sci_va}, MAX_VAL)
print(f"  T2 train={len(t2_train)} val={len(t2_val)}")

# ════════════════════════════════════════════════════════════════════════════
# 6. 跑全部 (T1×ABC, T2×ABC)
# ════════════════════════════════════════════════════════════════════════════
results = {"meta": {
    "model": "BERT-CCPoem (mean-pool, CE)", "seed": SEED,
    "max_train_per_class": MAX_TRAIN, "max_val_per_class": MAX_VAL,
    "split": "poet-level (author disjoint train/val)",
    "tiers": {"A": "metadata(rhythmic+title+author)+body, raw orthography/punct",
              "B": "body only, raw orthography/punct",
              "C": "body only, t2s-normalized + CJK-only (no punct/linebreak)"},
    "tasks": {"T1": "3-class shi/ci/qu (genre~dynasty collinear)",
              "T2": "within-Song shi vs ci (dynasty controlled)"},
    "n_tang_shi": len(tang_shi), "n_song_shi": len(song_shi),
    "n_song_ci": len(song_ci), "n_yuan_qu": len(yuan_qu)},
    "T1": {}, "T2": {}}

for tier in ["A", "B", "C"]:
    print(f"\n=== T1 tier {tier} ===")
    t0 = time.time()
    results["T1"][tier] = run(t1_train, t1_val, tier, 3, LABELS3, tag="T1")
    results["T1"][tier]["minutes"] = (time.time()-t0)/60
    print(f"  T1/{tier}: acc={results['T1'][tier]['acc']:.4f} "
          f"F1={results['T1'][tier]['f1_macro']:.4f} "
          f"UNK={results['T1'][tier]['unk_rate']:.3f}")

for tier in ["A", "B", "C"]:
    print(f"\n=== T2 tier {tier} ===")
    t0 = time.time()
    results["T2"][tier] = run(t2_train, t2_val, tier, 2, LABELS2, tag="T2")
    results["T2"][tier]["minutes"] = (time.time()-t0)/60
    print(f"  T2/{tier}: acc={results['T2'][tier]['acc']:.4f} "
          f"F1={results['T2'][tier]['f1_macro']:.4f} "
          f"UNK={results['T2'][tier]['unk_rate']:.3f}")

# 随机基线
results["meta"]["baseline_T1_acc"] = 1/3
results["meta"]["baseline_T2_acc"] = 0.5

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n" + "="*70)
print("SUMMARY  (acc / macro-F1 / UNK-rate)")
print("="*70)
for T in ["T1", "T2"]:
    print(f"\n{T}: {results['meta']['tasks'][T]}")
    for tier in ["A", "B", "C"]:
        r = results[T][tier]
        print(f"  {tier}: acc={r['acc']:.4f}  F1={r['f1_macro']:.4f}  "
              f"UNK={r['unk_rate']:.3f}  per-class-F1={[round(x,3) for x in r['f1_per']]}")
print(f"\n  随机基线: T1={1/3:.3f}, T2=0.500")
print(f"\n已保存: {OUT}")


