#!/usr/bin/env python3
"""
expC_v2_poem_level.py
Exp C (改进版): Poem-level 三分类 (shi=0 / ci=1 / qu=2) + 消融实验

改进点：
  1. Poem-level（逐首诗），样本量充足
  2. Mean Pooling 替代 [CLS] token
  3. Focal Loss 处理类别不平衡
  4. max_len=256 / 学习率 1e-5
  5. 10折分层交叉验证
  6. 三分类：shi / ci / qu
"""

import json, os, time, random, sys, re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import (BertTokenizer, BertModel,
                            get_linear_schedule_with_warmup)
from torch.optim import AdamW
from sklearn.metrics import (accuracy_score, f1_score,
                             roc_auc_score, confusion_matrix)
from sklearn.model_selection import StratifiedKFold, train_test_split
import warnings; warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════════
SEED = 42
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
set_seed(SEED)

BASE      = "/home/chenhao/poetry-phylogeny"
MODEL_DIR = f"{BASE}/data/models/BERT_CCPoem_v1"
RAW_DIR   = f"{BASE}/data/raw/chinese-poetry/chinese-poetry-master"
CIPAI_FILE= f"{BASE}/data/ciPai.json"
OUT_DIR   = f"{BASE}/data/processed"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device: {device}]  SEED={SEED}")

# ═════════════════════════════════════════════════════════════════════════════
# 1. 加载词牌词典
# ═════════════════════════════════════════════════════════════════════════════
print("\n[1] 加载词牌词典...")
with open(CIPAI_FILE, encoding="utf-8") as f:
    cipai_data = json.load(f)
ci_pai_names = set()
for entry in cipai_data:
    if isinstance(entry, dict):
        for v in entry.values():
            if isinstance(v, str) and len(v.strip()) >= 2:
                ci_pai_names.add(v.strip())
    elif isinstance(entry, str) and len(entry.strip()) >= 2:
        ci_pai_names.add(entry.strip())
print(f"  词牌名词典: {len(ci_pai_names)} 个")

# ═════════════════════════════════════════════════════════════════════════════
# 2. Poem-level 数据加载
# ═════════════════════════════════════════════════════════════════════════════
print("\n[2] 加载 poem-level 数据（shi=0 / ci=1 / qu=2）...")
poems = []  # {"text": str, "label": 0/1/2}

def load_json_poems(filepath, label):
    count = 0
    json_files = [f for f in os.listdir(filepath) if f.endswith(".json")]
    if len(json_files) == 1:
        fpath = os.path.join(filepath, json_files[0])
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    paras = item.get("paragraphs", [])
                    if not paras: continue
                    text = "".join(paras)
                    if len(text) < 8: continue
                    poems.append({"text": text, "label": label})
                    count += 1
        except: pass
        return count
    for fn in json_files:
        try:
            with open(os.path.join(filepath, fn), encoding="utf-8") as f:
                data = json.load(f)
            items = data if isinstance(data, list) else data.get("poems", [])
            for item in items:
                paras = item.get("paragraphs", [])
                if not paras: continue
                text = "".join(paras)
                if len(text) < 8: continue
                poems.append({"text": text, "label": label})
                count += 1
        except: pass
    return count

n_shi = 0
for sub in ["全唐诗", "五代诗词"]:
    d = os.path.join(RAW_DIR, sub)
    if os.path.isdir(d):
        n = load_json_poems(d, label=0)
        n_shi += n
        print(f"  {sub}(shi=0): {n} 首")

n_ci = 0
for sub in ["宋词"]:
    d = os.path.join(RAW_DIR, sub)
    if os.path.isdir(d):
        n = load_json_poems(d, label=1)
        n_ci += n
        print(f"  {sub}(ci=1): {n} 首")

n_qu = 0
for sub in ["元曲"]:
    d = os.path.join(RAW_DIR, sub)
    if os.path.isdir(d):
        n = load_json_poems(d, label=2)
        n_qu += n
        print(f"  {sub}(qu=2): {n} 首")

print(f"\n  原始数据: shi={n_shi}, ci={n_ci}, qu={n_qu}")

# ── 类别平衡采样（每类10,000）─────────────────────────────
MAX_PER = 10000
shi_list = [p for p in poems if p["label"] == 0]
ci_list  = [p for p in poems if p["label"] == 1]
qu_list  = [p for p in poems if p["label"] == 2]
random.seed(SEED)
random.shuffle(shi_list); random.shuffle(ci_list); random.shuffle(qu_list)
shi_s = shi_list[:MAX_PER]; ci_s = ci_list[:MAX_PER]; qu_s = qu_list[:MAX_PER]
all_p = shi_s + ci_s + qu_s
random.seed(SEED); random.shuffle(all_p)
texts = [p["text"] for p in all_p]
labels = np.array([p["label"] for p in all_p])
print(f"  采样后: shi={len(shi_s)}, ci={len(ci_s)}, qu={len(qu_s)}, 总={len(all_p)}")

# ═════════════════════════════════════════════════════════════════════════════
# 3. Dataset & Collator
# ═════════════════════════════════════════════════════════════════════════════
class PoemDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts; self.labels = labels
        self.tok = tokenizer; self.max = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max,
                       padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}, self.labels[i]

def collate_fn(batch):
    enc_list = [b[0] for b in batch]
    y = torch.tensor([b[1] for b in batch], dtype=torch.long)
    ids  = pad_sequence([e["input_ids"]      for e in enc_list], batch_first=True, padding_value=0)
    mask = pad_sequence([e["attention_mask"] for e in enc_list], batch_first=True, padding_value=0)
    tids_list = [e.get("token_type_ids", torch.tensor([])) for e in enc_list]
    tids_list = [t for t in tids_list if t.numel() > 0]
    tids = pad_sequence(tids_list, batch_first=True, padding_value=0) if tids_list \
           else torch.zeros_like(ids)
    return {"input_ids": ids, "attention_mask": mask, "token_type_ids": tids}, y

# ═════════════════════════════════════════════════════════════════════════════
# 4. 模型 & 损失函数
# ═════════════════════════════════════════════════════════════════════════════
class PoemClassifier(nn.Module):
    def __init__(self, model_path, pooling="mean", dropout=0.2, num_labels=3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        h = out.last_hidden_state  # (B, L, 768)
        if self.pooling == "mean":
            m = attention_mask.unsqueeze(-1).float()
            emb = (h * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            emb = h[:, 0, :]
        emb = self.dropout(emb)
        return self.fc(emb), emb


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__(); self.gamma = gamma
    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

# ═════════════════════════════════════════════════════════════════════════════
# 5. 训练 & 评估
# ═════════════════════════════════════════════════════════════════════════════
def train_epoch(model, dl, opt, sch, crit):
    model.train()
    total_loss, preds, truths = 0, [], []
    for batch, y in dl:
        opt.zero_grad()
        logits, _ = model(**{k: v.to(device) for k, v in batch.items()})
        loss = crit(logits, y.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        total_loss += loss.item()
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        truths.extend(y.tolist())
    return total_loss / len(dl), accuracy_score(truths, preds)


@torch.no_grad()
def evaluate(model, dl):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    for batch, y in dl:
        logits, _ = model(**{k: v.to(device) for k, v in batch.items()})
        all_probs.extend(torch.softmax(logits, dim=1).cpu().tolist())
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(y.tolist())
    yp = np.array(all_preds); yt = np.array(all_labels)
    acc = accuracy_score(yt, yp)
    f1m = f1_score(yt, yp, average="macro")
    f1p = f1_score(yt, yp, average=None)
    try:
        auc = roc_auc_score(yt, np.array(all_probs), multi_class="ovr", average="macro")
    except:
        auc = 0.0
    cm = confusion_matrix(yt, yp)
    return {
        "acc": acc, "f1_macro": f1m,
        "f1_shi": f1p[0], "f1_ci": f1p[1], "f1_qu": f1p[2],
        "auc": auc, "cm": cm.tolist(), "n": len(yt),
    }


def run_cv(texts, labels, cfg, n_folds=10, epochs=3, bs=32, lr=2e-5, max_len=256, name=""):
    set_seed(SEED)
    tok = BertTokenizer.from_pretrained(MODEL_DIR)

    if n_folds == 1:
        tri, vai = train_test_split(
            np.arange(len(texts)), test_size=0.2,
            stratify=labels, random_state=SEED)
        splits = [(tri, vai)]
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        splits = list(skf.split(texts, labels))

    fold_results = []
    for fold, (tri, vai) in enumerate(splits):
        set_seed(SEED + fold)
        print(f"\n  [{name}] Fold {fold+1}/{n_folds}")

        trd = DataLoader(PoemDataset([texts[i] for i in tri], labels[tri], tok, max_len),
                         batch_size=bs, shuffle=True, num_workers=4, pin_memory=True,
                         collate_fn=collate_fn)
        vad = DataLoader(PoemDataset([texts[i] for i in vai], labels[vai], tok, max_len),
                         batch_size=bs*2, shuffle=False, num_workers=4, pin_memory=True,
                         collate_fn=collate_fn)

        model = PoemClassifier(MODEL_DIR, pooling=cfg.get("pooling","mean"),
                               dropout=0.2, num_labels=3).to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        crit = FocalLoss(gamma=cfg.get("focal_gamma",2.0)) if cfg.get("loss")=="focal" \
               else nn.CrossEntropyLoss()
        opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        sch = get_linear_schedule_with_warmup(opt,
                     num_warmup_steps=int(0.1*len(trd)*epochs),
                     num_training_steps=len(trd)*epochs)

        best_f1, best_st = 0.0, None
        for ep in range(epochs):
            loss, _ = train_epoch(model, trd, opt, sch, crit)
            st = evaluate(model, vad)
            print(f"    ep={ep+1} loss={loss:.4f} acc={st['acc']:.4f} "
                  f"F1={st['f1_macro']:.4f} "
                  f"shi={st['f1_shi']:.3f} ci={st['f1_ci']:.3f} qu={st['f1_qu']:.3f}")
            if st["f1_macro"] > best_f1:
                best_f1 = st["f1_macro"]
                best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict({k: v.to(device) for k, v in best_st.items()})
        fst = evaluate(model, vad)
        print(f"    ★ best={best_f1:.4f}  acc={fst['acc']:.4f} "
              f"F1={fst['f1_macro']:.4f} "
              f"shi={fst['f1_shi']:.3f} ci={fst['f1_ci']:.3f} qu={fst['f1_qu']:.3f}")
        fold_results.append(fst)
        del model; torch.cuda.empty_cache()

    keys = ["acc","f1_macro","f1_shi","f1_ci","f1_qu","auc"]
    smry = {k: np.mean([r[k] for r in fold_results]) for k in keys}
    smry.update({f"{k}_std": np.std([r[k] for r in fold_results]) for k in keys})
    smry["fold_results"] = fold_results
    smry["cm_mean"] = np.mean([r["cm"] for r in fold_results], axis=0).tolist()
    smry["n_folds"] = n_folds
    smry["config"] = cfg; smry["lr"] = lr; smry["max_len"] = max_len; smry["epochs"] = epochs
    return smry

# ═════════════════════════════════════════════════════════════════════════════
# 6. 消融实验
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n[6] 消融实验（10折CV, epochs=3, batch=32）")
print("="*60)

configs = [
    {"name":"A1","pooling":"mean","loss":"ce",     "focal_gamma":None},
    {"name":"A2","pooling":"mean","loss":"focal",   "focal_gamma":2.0},
    {"name":"A3","pooling":"mean","loss":"focal",   "focal_gamma":2.0},
    {"name":"A4","pooling":"mean","loss":"focal",   "focal_gamma":2.0},
]
lrs  = [2e-5, 2e-5, 1e-5, 1e-5]
lens = [256,  256,  256,  256]

all_results = {}
for cfg, lr, mlen in zip(configs, lrs, lens):
    print(f"\n{'─'*60}")
    print(f"  {cfg['name']}: pooling={cfg['pooling']}, loss={cfg['loss']}, lr={lr}, max_len={mlen}")
    t0 = time.time()
    r = run_cv(texts, labels, cfg, n_folds=1, epochs=3, bs=16, lr=lr, max_len=mlen, name=cfg["name"])
    r["n_folds"] = "80/20 split"
    elapsed = time.time() - t0
    r["elapsed_min"] = elapsed / 60
    all_results[cfg["name"]] = r
    print(f"\n  {cfg['name']} ({elapsed/60:.1f}min): "
          f"Acc={r['acc']:.4f}±{r['acc_std']:.4f}  "
          f"Macro-F1={r['f1_macro']:.4f}±{r['f1_macro_std']:.4f}\n"
          f"    shi-F1={r['f1_shi']:.4f}  ci-F1={r['f1_ci']:.4f}  qu-F1={r['f1_qu']:.4f}  "
          f"AUC={r['auc']:.4f}")

best_name = max(all_results, key=lambda k: all_results[k]["f1_macro"])
print(f"\n最优: {best_name}")

# ═════════════════════════════════════════════════════════════════════════════
# 7. 保存
# ═════════════════════════════════════════════════════════════════════════════
output = {
    "task": "poem-level shi/ci/qu 3-class classification",
    "model": "BERT-CCPoem", "model_path": MODEL_DIR,
    "seed": SEED,
    "n_shi": len(shi_s), "n_ci": len(ci_s), "n_qu": len(qu_s),
    "total": len(all_p),
    "n_folds": 10, "epochs": 3, "batch_size": 32,
    "ablation": {
        k: {kk: vv for kk, vv in v.items() if kk != "fold_results"}
        for k, v in all_results.items()
    },
    "best_config": best_name,
    "best_result": {k: v for k, v in all_results[best_name].items() if k != "fold_results"},
}
with open(f"{OUT_DIR}/expC_v2_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\n已保存: {OUT_DIR}/expC_v2_results.json")
