#!/usr/bin/env python3
"""
expC_attention_viz.py
Exp C 可视化: Attention Weight 可视化 + 形式约束词验证

功能：
  1. 从最优BERT模型提取 attention 权重
  2. 对 ci/shi 各8个典型样本，绘制 token-level attention 热力图
  3. 验证词牌名获得最高注意力（形式约束词验证）
  4. 形式约束词注意力权重统计（ci vs shi 箱线图）
"""

import json, os, random, math
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from scipy import stats
import matplotlib.font_manager as fm
for _p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"]:
    try: fm.fontManager.addfont(_p)
    except: pass
_cjk = [f.name for f in fm.fontManager.ttflist if "CJK" in f.name]
FONT_CJK = _cjk[0] if _cjk else None
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════════
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE       = "/home/chenhao/poetry-phylogeny"
MODEL_DIR  = f"{BASE}/data/models/BERT_CCPoem_v1"
RAW_DIR    = f"{BASE}/data/raw/chinese-poetry/chinese-poetry-master"
CIPAI_FILE = f"{BASE}/data/ciPai.json"
OUT_DIR    = f"{BASE}/data/figures"
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── 词牌词典 ───────────────────────────────────────────────
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
print(f"词牌名词典: {len(ci_pai_names)} 个")

# ── 加载数据 ────────────────────────────────────────────────
def load_poems(dir_path, label, n_max=200):
    poems = []
    if not os.path.isdir(dir_path): return poems

    # 情况1：目录下有单个大json文件（宋词ci.song.0.json）
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

    # 情况2：分诗人json文件（全唐诗/*.json）
    for fn in os.listdir(dir_path):
        if not fn.endswith(".json"): continue
        try:
            with open(os.path.join(dir_path, fn), encoding="utf-8") as f:
                data = json.load(f)
            # data 可能是 dict（诗人信息+poems列表）或 list（多首）
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

print("加载样本...")
ci_samples  = load_poems(os.path.join(RAW_DIR, "宋词"), label=1, n_max=200)
qu_samples  = load_poems(os.path.join(RAW_DIR, "元曲"), label=2, n_max=200)
shi_samples = load_poems(os.path.join(RAW_DIR, "全唐诗"), label=0, n_max=200)

def find_cipai(text):
    for name in ci_pai_names:
        if name in text: return name
    return None

ci_has_cipai = [(p, find_cipai(p["text"])) for p in ci_samples]
ci_with_name = [(p, n) for p, n in ci_has_cipai if n]
print(f"ci样本: {len(ci_samples)}, 含词牌名: {len(ci_with_name)}")
print(f"qu样本: {len(qu_samples)}, shi样本: {len(shi_samples)}")

# ── 加载模型 ────────────────────────────────────────────────
print("加载BERT模型（提取attention）...")
tok = BertTokenizer.from_pretrained(MODEL_DIR)
bert_model = BertModel.from_pretrained(MODEL_DIR, attn_implementation="eager")
bert_model.eval()
bert_model.to(device)


def get_attention(text, max_len=256):
    """提取最后一层 [CLS]→all tokens 的平均 attention"""
    enc = tok(text, truncation=True, max_length=max_len, padding=False,
               return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = bert_model(**enc, output_attentions=True)
    # (n_layers, B, n_heads, L, L)
    last_attn = out.attentions[-1][0].cpu().numpy()   # (n_heads, L, L)
    tokens    = tok.convert_ids_to_tokens(enc["input_ids"][0])
    cls_attn  = last_attn[:, 0, :].mean(axis=0)         # (L,) 所有head平均
    return tokens, cls_attn


def clean_tokens(tokens, attn):
    """去掉[CLS][SEP][PAD][UNK]"""
    valid = [t not in ("[CLS]","[SEP]","[PAD]","[UNK]") for t in tokens]
    return [t for t, v in zip(tokens, valid) if v], \
           np.array([a for a, v in zip(attn, valid) if v])


# ═════════════════════════════════════════════════════════════════════════════
# 图1: Attention 热力图（ci vs shi，各8个样本）
# ═════════════════════════════════════════════════════════════════════════════
print("\n生成 attention 热力图...")

if FONT_CJK:
    plt.rcParams["font.sans-serif"] = [FONT_CJK]
plt.rcParams["axes.unicode_minus"] = False

n_rows = 4
fig, axes = plt.subplots(n_rows, 2, figsize=(16, n_rows * 4.2))

def plot_attn_bars(ax, text, cipai_name, row_label):
    tokens, attn = get_attention(text)
    tokens, attn  = clean_tokens(tokens, attn)
    attn_norm     = (attn - attn.min() + 1e-9) / (attn.max() - attn.min() + 1e-9)
    n_show        = min(len(tokens), 60)
    tks, ats      = tokens[:n_show], attn_norm[:n_show]

    # 颜色：词牌名红色，其他按权重渐变
    from matplotlib.cm import ScalarMappable
    cmap = plt.cm.YlOrRd
    colors = [cmap(a) for a in ats]
    for i, tk in enumerate(tks):
        if cipai_name and tk in cipai_name:
            colors[i] = (0.8, 0.1, 0.1, 0.9)
        elif tk in ci_pai_names:
            colors[i] = (1.0, 0.55, 0.0, 0.9)

    ax.bar(range(len(tks)), ats, color=colors, alpha=0.85,
           edgecolor="black", lw=0.4, width=0.8)
    ax.set_xticks(range(len(tks)))
    ax.set_xticklabels(tks, rotation=90, fontsize=6)
    ax.set_ylabel(row_label, fontsize=8.5)
    ax.set_ylim(0, 1.08)
    ax.grid(True, alpha=0.2, axis="y")

    # 高亮词牌名
    if cipai_name:
        pos = [i for i, tk in enumerate(tks) if tk in cipai_name]
        for p in pos:
            ax.add_patch(plt.Rectangle((p-0.5, 0), 1, 1.06,
                       fill=False, edgecolor="red", lw=1.5, linestyle="--"))
            ax.text(p, 1.04, f"★{tks[p]}", ha="center", fontsize=7,
                    color="darkred", rotation=45)

    title = text[:28] + "…" if len(text) > 28 else text
    ax.set_title(title, fontsize=7.5, loc="left", color="#333")

vis_ci  = (ci_with_name[:8] if len(ci_with_name) >= 8
           else [(p, find_cipai(p["text"]) or "") for p in ci_samples[:8]])
vis_shi = shi_samples[:8]

for row in range(n_rows):
    if row < len(vis_ci):
        p, cname = vis_ci[row]
        plot_attn_bars(axes[row, 0], p["text"], cname, f"ci #{row+1}")
    if row < len(vis_shi):
        plot_attn_bars(axes[row, 1], vis_shi[row]["text"], "", f"shi #{row+1}")

axes[0, 0].set_title("词（ci）样本 — 词牌名红色标注", fontsize=9,
                       fontweight="bold", color="darkred")
axes[0, 1].set_title("诗（shi）样本 — 无词牌名", fontsize=9,
                       fontweight="bold", color="#333")

legend_els = [
    plt.Rectangle((0,0),1,1,fc=(0.8,0.1,0.1,0.85), label="词牌名"),
    plt.Rectangle((0,0),1,1,fc=(0.95,0.55,0.0,0.85), label="格律标记"),
    plt.Rectangle((0,0),1,1,fc=(1.0,0.85,0.4,0.7), label="普通词（高attn）"),
    plt.Rectangle((0,0),1,1,fc=(1.0,1.0,0.7,0.5), label="普通词（低attn）"),
]
fig.legend(handles=legend_els, loc="upper center", ncol=4,
           fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, 1.01))
fig.suptitle("BERT-CCPoem 分类 ci/shi 时 Token 级 Attention 权重",
             fontsize=13, fontweight="bold", y=1.03)
plt.tight_layout(rect=[0, 0, 1, 0.98])
out1 = f"{OUT_DIR}/figC1_attention_heatmap.png"
fig.savefig(out1, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  已保存: {out1}")


# ═════════════════════════════════════════════════════════════════════════════
# 图2: 形式约束词 Attention 权重统计（ci vs shi 箱线图）
# ═════════════════════════════════════════════════════════════════════════════
print("生成形式约束词注意力统计图...")

N_STAT = 300
ci_s  = random.sample(ci_samples,  min(N_STAT, len(ci_samples)))
shi_s = random.sample(shi_samples, min(N_STAT, len(shi_samples)))

def extract_weights(poem_text):
    tokens, attn = get_attention(poem_text)
    tokens, attn = clean_tokens(tokens, attn)
    attn_n = (attn - attn.min() + 1e-9) / (attn.max() - attn.min() + 1e-9)
    cipai_w = [attn_n[i] for i, t in enumerate(tokens) if t in ci_pai_names]
    formal_w = [attn_n[i] for i, t in enumerate(tokens)
                if t in ("平","仄","韵","律","调","曲","词","韵","律")]
    other_w  = [attn_n[i] for i, t in enumerate(tokens)
                if t not in ci_pai_names and
                t not in ("平","仄","韵","律","调","曲","词","韵","律")]
    return cipai_w, formal_w, other_w

ci_cipai_all, ci_formal_all, ci_other_all = [], [], []
shi_cipai_all, shi_formal_all, shi_other_all = [], [], []

for p in ci_s:
    a, b, c = extract_weights(p["text"])
    ci_cipai_all.extend(a); ci_formal_all.extend(b); ci_other_all.extend(c)
for p in shi_s:
    a, b, c = extract_weights(p["text"])
    if a: shi_cipai_all.extend(a)   # shi样本一般无词牌名
    shi_formal_all.extend(b); shi_other_all.extend(c)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
if FONT_CJK:
    plt.rcParams["font.sans-serif"] = [FONT_CJK]
plt.rcParams["axes.unicode_minus"] = False

def boxplt(ax, data_dict, title, ylabel):
    names = list(data_dict.keys())
    vals  = list(data_dict.values())
    colors = ["#FF6B6B","#4ECDC4","#FFD700","#95A5A6","#98D8C8"]
    bp = ax.boxplot(vals, patch_artist=True, notch=False,
                    medianprops=dict(color="black", lw=2),
                    whiskerprops=dict(lw=1.2),
                    capprops=dict(lw=1.2),
                    flierprops=dict(markersize=3, alpha=0.5))
    for patch, color in zip(bp["boxes"], colors[:len(vals)]):
        patch.set_facecolor(color); patch.set_alpha(0.72)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")
    for i, v in enumerate(vals):
        if len(v) > 0:
            ax.text(i+1, max(v)*0.93, f"n={len(v)}",
                    ha="center", fontsize=8.5, color="#555")

boxplt(axes[0],
    {"ci: 词牌名attn": ci_cipai_all,
     "ci: 形式词attn": ci_formal_all,
     "ci: 普通词attn": ci_other_all},
    "ci 样本各类 Token 平均 Attention",
    "归一化 Attention 权重")

boxplt(axes[1],
    {"shi: 形式词attn": shi_formal_all,
     "shi: 普通词attn": shi_other_all,
     "ci: 词牌名attn": ci_cipai_all},
    "形式约束词 Attention: ci vs shi",
    "归一化 Attention 权重")

# 统计检验
if ci_cipai_all and shi_formal_all:
    u, p = stats.mannwhitneyu(ci_cipai_all, shi_formal_all, alternative="greater")
    print(f"  词牌名attn(ci) vs 形式词attn(shi): U={u:.0f}, p={p:.2e}")
    axes[1].text(0.5, 0.97,
                 f"Mann-Whitney U={u:.0f}, p{'<0.001***' if p<0.001 else f'={p:.3f}'}",
                 transform=axes[1].transAxes, ha="center", fontsize=9,
                 color="darkred",
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.85))

fig.suptitle("形式约束词 Attention 权重分布验证", fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
out2 = f"{OUT_DIR}/figC2_formal_word_attention.png"
fig.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  已保存: {out2}")
print(f"\n可视化完成:")
print(f"  {out1}")
print(f"  {out2}")
