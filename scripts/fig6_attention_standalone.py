#!/usr/bin/env python3
"""
fig6_attention_standalone.py
Figure 6: BERT-CCPoem 分类 Attention 权重可视化（独立大图）
"""
import json, os, warnings
import numpy as np
import matplotlib.font_manager as fm
for _p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"]:
    try: fm.fontManager.addfont(_p)
    except: pass
_cjk = [f.name for f in fm.fontManager.ttflist if "CJK" in f.name]
FONT_CJK = _cjk[0] if _cjk else None
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings; warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE      = "/home/chenhao/poetry-phylogeny"
MODEL_DIR = f"{BASE}/data/models/BERT_CCPoem_v1"
RAW_DIR   = f"{BASE}/data/raw/chinese-poetry/chinese-poetry-master"
CIPAI_FILE= f"{BASE}/data/ciPai.json"
OUT_DIR   = f"{BASE}/data/figures"
device = "cuda"

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

def load_poems(dir_path, label, n_max=200):
    poems = []
    if not os.path.isdir(dir_path): return poems
    json_files = [f for f in os.listdir(dir_path) if f.endswith(".json")]
    if len(json_files) == 1:
        fpath = os.path.join(dir_path, json_files[0])
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    paras = item.get("paragraphs", [])
                    if not paras: continue
                    text = "".join(paras)
                    if len(text) >= 8:
                        poems.append({"text": text, "label": label})
                        if len(poems) >= n_max: return poems
        except: pass
        return poems
    for fn in json_files:
        try:
            with open(os.path.join(dir_path, fn), encoding="utf-8") as f:
                data = json.load(f)
            items = data if isinstance(data, list) else data.get("poems", [])
            for item in items:
                paras = item.get("paragraphs", [])
                if not paras: continue
                text = "".join(paras)
                if len(text) >= 8:
                    poems.append({"text": text, "label": label})
                    if len(poems) >= n_max: return poems
        except: pass
    return poems

ci_samples  = load_poems(os.path.join(RAW_DIR, "宋词"), label=1, n_max=200)
shi_samples = load_poems(os.path.join(RAW_DIR, "全唐诗"), label=0, n_max=200)

def find_cipai(text):
    for name in ci_pai_names:
        if name in text: return name
    return None

ci_with_name = [(p, find_cipai(p["text"])) for p in ci_samples]
ci_with_name = [(p, n) for p, n in ci_with_name if n]

from transformers import BertTokenizer, BertModel

tok = BertTokenizer.from_pretrained(MODEL_DIR)
bert_model = BertModel.from_pretrained(MODEL_DIR, attn_implementation="eager")
bert_model.eval()
bert_model.to(device)

def get_attention(text, max_len=256):
    enc = tok(text, truncation=True, max_length=max_len, padding=False, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = bert_model(**enc, output_attentions=True)
    last_attn = out.attentions[-1][0].cpu().numpy()
    tokens    = tok.convert_ids_to_tokens(enc["input_ids"][0])
    cls_attn  = last_attn[:, 0, :].mean(axis=0)
    return tokens, cls_attn

def clean_tokens(tokens, attn):
    valid = [t not in ("[CLS]","[SEP]","[PAD]","[UNK]") for t in tokens]
    return [t for t, v in zip(tokens, valid) if v], \
           np.array([a for a, v in zip(attn, valid) if v])

if FONT_CJK:
    plt.rcParams["font.sans-serif"] = [FONT_CJK]
plt.rcParams["axes.unicode_minus"] = False

# ── 独立大图：Attention 热力图 ─────────────────────────────
# 两行：ci样本 + shi样本，每行4个样本
import torch
vis_ci  = (ci_with_name[:8] if len(ci_with_name) >= 8
           else [(p, find_cipai(p["text"]) or "") for p in ci_samples[:8]])
vis_shi = shi_samples[:8]

n_rows = 4
fig, axes = plt.subplots(n_rows, 2, figsize=(20, n_rows * 5.5))

def plot_attn_bars(ax, text, cipai_name, row_label):
    tokens, attn = get_attention(text)
    tokens, attn  = clean_tokens(tokens, attn)
    attn_norm     = (attn - attn.min() + 1e-9) / (attn.max() - attn.min() + 1e-9)
    n_show        = min(len(tokens), 60)
    tks, ats      = tokens[:n_show], attn_norm[:n_show]

    cmap = plt.cm.YlOrRd
    colors = [cmap(a) for a in ats]
    for i, tk in enumerate(tks):
        if cipai_name and tk in cipai_name:
            colors[i] = (0.8, 0.1, 0.1, 0.95)
        elif tk in ci_pai_names:
            colors[i] = (1.0, 0.55, 0.0, 0.95)

    ax.bar(range(len(tks)), ats, color=colors, alpha=0.88,
           edgecolor="black", lw=0.5, width=0.8)
    ax.set_xticks(range(len(tks)))
    ax.set_xticklabels(tks, rotation=90, fontsize=8)
    ax.set_ylabel(row_label, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.grid(True, alpha=0.2, axis="y")

    if cipai_name:
        pos = [i for i, tk in enumerate(tks) if tk in cipai_name]
        for p in pos:
            ax.add_patch(plt.Rectangle((p-0.5, 0), 1, 1.06,
                       fill=False, edgecolor="red", lw=2, linestyle="--"))
            ax.text(p, 1.04, f"★{tks[p]}", ha="center", fontsize=9,
                    color="darkred", rotation=45, fontweight="bold")

    title = text[:30] + "…" if len(text) > 30 else text
    ax.set_title(title, fontsize=9, loc="left", color="#333")

for row in range(n_rows):
    if row < len(vis_ci):
        p, cname = vis_ci[row]
        plot_attn_bars(axes[row, 0], p["text"], cname, f"ci #{row+1}")
    if row < len(vis_shi):
        plot_attn_bars(axes[row, 1], vis_shi[row]["text"], "", f"shi #{row+1}")

axes[0, 0].set_title("词（ci）样本 — 词牌名红色标注，★标记", fontsize=11,
                       fontweight="bold", color="darkred")
axes[0, 1].set_title("诗（shi）样本 — 无词牌名", fontsize=11,
                       fontweight="bold", color="#333")

legend_els = [
    plt.Rectangle((0,0),1,1,fc=(0.8,0.1,0.1,0.85), label="词牌名"),
    plt.Rectangle((0,0),1,1,fc=(0.95,0.55,0.0,0.85), label="格律标记词"),
    plt.Rectangle((0,0),1,1,fc=(1.0,0.85,0.4,0.7), label="高Attention"),
    plt.Rectangle((0,0),1,1,fc=(1.0,1.0,0.7,0.5), label="低Attention"),
]
fig.legend(handles=legend_els, loc="upper center", ncol=4,
           fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, 1.01))
fig.suptitle("图9  BERT-CCPoem 分类 Token 级 Attention 权重（词牌名红色标注）",
             fontsize=15, fontweight="bold", y=1.025)
plt.tight_layout(rect=[0, 0, 1, 0.99])
out = f"{OUT_DIR}/fig9_attention_standalone.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"已保存: {out}")
