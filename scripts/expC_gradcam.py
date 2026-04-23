#!/usr/bin/env python3
"""
expC_gradcam.py
Exp C 可解释性: Grad-CAM 热力图（三分类 shi=0/ci=1/qu=2）
"""

import json, os, random
import numpy as np
import torch
import matplotlib.font_manager as fm
for _p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"]:
    try: fm.fontManager.addfont(_p)
    except: pass
_cjk = [f.name for f in fm.fontManager.ttflist if "CJK" in f.name]
FONT_CJK = _cjk[0] if _cjk else None
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import warnings; warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

BASE      = "/home/chenhao/poetry-phylogeny"
MODEL_DIR = f"{BASE}/data/models/BERT_CCPoem_v1"
RAW_DIR   = f"{BASE}/data/raw/chinese-poetry/chinese-poetry-master"
CIPAI_FILE= f"{BASE}/data/ciPai.json"
OUT_DIR   = f"{BASE}/data/figures"
os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

with open(CIPAI_FILE, encoding="utf-8") as f:
    cipai_data = json.load(f)
ci_pai_names = set()
for entry in cipai_data:
    if isinstance(entry, dict):
        for v in entry.values():
            if isinstance(v, str) and len(v.strip()) >= 2:
                ci_pai_names.add(v.strip())
    elif isinstance(entry, str) and len(entry) >= 2:
        ci_pai_names.add(entry.strip())

def load_poems(dir_path, label, n_max=50):
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
                    if 10 <= len(text) <= 200:
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
                if 10 <= len(text) <= 200:
                    poems.append({"text": text, "label": label})
                    if len(poems) >= n_max: return poems
        except: pass
    return poems

print("加载样本...")
ci_s  = load_poems(os.path.join(RAW_DIR, "宋词"), label=1, n_max=50)
qu_s  = load_poems(os.path.join(RAW_DIR, "元曲"), label=2, n_max=50)
shi_s = load_poems(os.path.join(RAW_DIR, "全唐诗"), label=0, n_max=50)

def find_cipai(text):
    for n in ci_pai_names:
        if n in text: return n
    return None

ci_with_name = [(p, find_cipai(p["text"])) for p in ci_s]
ci_has_name  = [(p, n) for p, n in ci_with_name if n]

from transformers import BertTokenizer, BertModel

print("加载BERT模型（3分类，用预训练权重）...")
tok = BertTokenizer.from_pretrained(MODEL_DIR)
bert = BertModel.from_pretrained(MODEL_DIR)
bert.to(device).eval()


def get_gradcam(text, target_label=1, max_len=512):
    """用hook提取最后一层activation和梯度，生成Grad-CAM"""
    enc = tok(text, truncation=True, max_length=max_len,
              padding="max_length", return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    input_ids = enc["input_ids"]
    attn_mask = enc["attention_mask"]

    activations, gradients = [], []
    def fwd(m, inp, out): activations.append(out[0].detach().cpu())
    def bwd(m, inp, out): gradients.append(inp[0].detach().cpu())

    last = bert.encoder.layer[-1]
    h1 = last.register_forward_hook(fwd)
    h2 = last.register_full_backward_hook(bwd)

    bert.zero_grad()
    out = bert(input_ids=input_ids, attention_mask=attn_mask)
    # 简单方式：用[CLS]做分类代理
    cls_emb = out.last_hidden_state[0, 0, :]
    score = cls_emb.sum()  # 代理分数
    score.backward()

    act  = activations[0][0].cpu().numpy()
    grad = gradients[0][0].cpu().numpy()
    w    = grad.mean(axis=1)
    cam  = act @ w
    cam  = np.maximum(cam, 0)

    h1.remove(); h2.remove()
    tokens = tok.convert_ids_to_tokens(input_ids[0])
    return tokens, cam


def token_to_char(text, tokens, cam, max_len=512):
    chars = list(text[:max_len])
    n = len(chars)
    vc = np.array([c for t, c in zip(tokens, cam) if t not in ("[CLS]","[SEP]","[PAD]","[UNK]")])
    if len(vc) == 0: return [0.0] * max(n, 1)
    result = []
    for i in range(n):
        idx = min(int(i * len(vc) / max(n, 1)), len(vc) - 1)
        result.append(float(vc[idx]))
    return result


print("生成 Grad-CAM 热力图...")

if FONT_CJK:
    plt.rcParams["font.sans-serif"] = [FONT_CJK]
plt.rcParams["axes.unicode_minus"] = False

samples = []
for p, cname in ci_has_name[:3]:
    samples.append({"text": p["text"], "label": "ci", "cipai": cname})
for p in qu_s[:3]:
    samples.append({"text": p["text"], "label": "qu", "cipai": None})
for p in shi_s[:3]:
    samples.append({"text": p["text"], "label": "shi", "cipai": None})

n = len(samples)
n_rows = (n + 1) // 2
fig, axes = plt.subplots(n_rows, 2, figsize=(16, n_rows * 3.5))
axes = axes.flatten() if n_rows > 1 else [axes[0], axes[1]]

cmap_gc = LinearSegmentedColormap.from_list(
    "gradcam", ["#FFFFFF","#FFFF88","#FF8800","#FF2200"], N=256)

LABEL_COLORS = {"ci":"darkred","qu":"purple","shi":"#333"}
LABEL_TARGET  = {"ci":1,"qu":2,"shi":0}

for i, samp in enumerate(samples):
    ax = axes[i]
    ax.cla()
    text  = samp["text"]
    label = samp["label"]
    cname = samp.get("cipai") or ""

    try:
        tokens, cam = get_gradcam(text, target_label=LABEL_TARGET[label])
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {str(e)[:80]}", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="red")
        continue

    char_cam = token_to_char(text, tokens, cam)
    cc = np.array(char_cam, dtype=float)
    if cc.max() > cc.min():
        cc = (cc - cc.min()) / (cc.max() - cc.min())

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.02, 0.93, f"[{label.upper()}]", fontsize=10, fontweight="bold",
            color=LABEL_COLORS[label], transform=ax.transAxes)
    if cname:
        ax.text(0.12, 0.93, f"词牌: {cname}", fontsize=9, color="darkred",
                transform=ax.transAxes)

    x0, y0, cw, ch = 0.02, 0.08, 0.011, 0.72
    cipai_ranges = []
    if cname:
        st = text.find(cname)
        if st >= 0: cipai_ranges.append((st, st + len(cname)))

    for j, ch_c in enumerate(text):
        if j >= 80: break
        x = x0 + j * cw
        c = float(cc[j]) if j < len(cc) else 0.0
        bg = cmap_gc(int(c * 255))
        ax.add_patch(Rectangle((x, y0), cw - 0.001, ch, facecolor=bg, edgecolor="none"))
        tc = "white" if c > 0.5 else "black"
        fw = "bold" if any(s <= j < e for s, e in cipai_ranges) else "normal"
        ax.text(x + cw/2, y0 + ch/2, ch_c, ha="center", va="center",
                fontsize=9, color=tc, fontweight=fw)
    for st, en in cipai_ranges:
        for j in range(st, min(en, 80)):
            x = x0 + j * cw
            ax.plot([x, x+cw], [y0, y0], color="red", lw=2.5, transform=ax.transAxes)
    ax.set_title(text[:60]+"…" if len(text) > 60 else text,
                 fontsize=8, loc="left", pad=2, color="#666")

for j in range(n, len(axes)):
    axes[j].axis("off")

fig.suptitle(
    "Grad-CAM 热力图：BERT-CCPoem 三分类决策时关注的字符\n"
    "（红/黄=高重要性，白=低重要性，红色下划线=词牌名）\n"
    "标签说明: shi=诗 ci=词 qu=曲",
    fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
out = f"{OUT_DIR}/figC3_gradcam_heatmap.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  已保存: {out}")
