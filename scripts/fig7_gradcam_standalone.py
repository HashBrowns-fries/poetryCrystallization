#!/usr/bin/env python3
"""
fig7_gradcam_standalone.py
Figure 7: BERT-CCPoem 分类 Grad-CAM 热力图（独立大图）
"""
import json, os, random, warnings
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

tok = BertTokenizer.from_pretrained(MODEL_DIR)
bert = BertModel.from_pretrained(MODEL_DIR)
bert.to(device).eval()

def get_gradcam(text, max_len=512):
    """提取Grad-CAM，返回 (tokens列表, cam数组，长度=seq_len)"""
    enc = tok(text, truncation=True, max_length=max_len,
              padding="max_length", return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    input_ids = enc["input_ids"]
    attn_mask = enc["attention_mask"]
    L = input_ids.size(1)

    activations, gradients = [], []
    def fwd(m, inp, out): activations.append(out[0].detach())
    def bwd(m, inp, out): gradients.append(inp[0].detach())

    last = bert.encoder.layer[-1]
    h1 = last.register_forward_hook(fwd)
    h2 = last.register_full_backward_hook(bwd)

    bert.zero_grad()
    out = bert(input_ids=input_ids, attention_mask=attn_mask)
    cls_emb = out.last_hidden_state[0, 0, :]
    score = cls_emb.sum()
    score.backward()

    act  = activations[0].cpu()             # (L, 512)  hook返回out[0]
    grad = gradients[0][0].cpu()              # (1, L, H) → squeeze batch
    grad = grad.squeeze(0)                  # (L, 512)
    w    = grad.mean(axis=0)                  # (512,) 沿sequence维平均
    cam  = np.maximum(act.numpy() @ w.numpy(), 0)  # (L,) ReLU

    h1.remove(); h2.remove()
    tokens = tok.convert_ids_to_tokens(input_ids[0])
    return tokens, cam

def token_to_char(text, tokens, cam):
    """将token级CAM映射到字符级"""
    skip = {"[CLS]", "[SEP]", "[PAD]", "[UNK]"}
    valid_idx = [i for i, t in enumerate(tokens) if t not in skip]
    valid_cam = cam[valid_idx]
    if valid_cam.size == 0:
        return [0.0] * min(len(text), 80)

    chars = list(text)
    result = np.zeros(len(chars), dtype=float)
    for i, vc_idx in enumerate(valid_idx):
        char_pos = int(i * len(chars) / max(len(valid_idx), 1))
        if char_pos < len(chars):
            result[char_pos] = valid_cam[i]

    if result.max() > result.min():
        result = (result - result.min()) / (result.max() - result.min() + 1e-9)
    return result.tolist()

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
fig, axes = plt.subplots(n_rows, 2, figsize=(22, n_rows * 7))
axes = axes.flatten() if n_rows > 1 else [axes[0], axes[1]]

cmap_gc = LinearSegmentedColormap.from_list(
    "gradcam", ["#FFFFFF","#FFFF88","#FF8800","#FF2200"], N=256)

LABEL_COLORS = {"ci":"darkred","qu":"purple","shi":"#333"}

for i, samp in enumerate(samples):
    ax = axes[i]
    ax.cla()
    text  = samp["text"]
    label = samp["label"]
    cname = samp.get("cipai") or ""

    try:
        tokens, cam = get_gradcam(text)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {str(e)[:100]}", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="red")
        continue

    char_cam = token_to_char(text, tokens, cam)
    cc = np.array(char_cam, dtype=float)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.01, 0.94, f"[{label.upper()}]", fontsize=13, fontweight="bold",
            color=LABEL_COLORS[label], transform=ax.transAxes)
    if cname:
        ax.text(0.07, 0.94, f"词牌: {cname}", fontsize=12, color="darkred",
                transform=ax.transAxes, fontweight="bold")

    n_show = min(len(text), 80)
    cw = 0.012
    x0, y0, ch = 0.01, 0.06, 0.80

    cipai_ranges = []
    if cname:
        st = text.find(cname)
        if st >= 0:
            cipai_ranges.append((st, st + len(cname)))

    for j in range(n_show):
        x = x0 + j * cw
        c = float(cc[j]) if j < len(cc) else 0.0
        bg = cmap_gc(int(c * 255))
        ax.add_patch(Rectangle((x, y0), cw - 0.001, ch, facecolor=bg, edgecolor="none"))
        tc = "white" if c > 0.5 else "black"
        is_cipai = any(s <= j < e for s, e in cipai_ranges)
        fw = "bold" if is_cipai else "normal"
        ax.text(x + cw/2, y0 + ch/2, text[j], ha="center", va="center",
                fontsize=12, color=tc, fontweight=fw)

    for st, en in cipai_ranges:
        for j in range(st, min(en, n_show)):
            x = x0 + j * cw
            ax.plot([x, x+cw], [y0, y0], color="red", lw=2.5, transform=ax.transAxes)

    title = text[:55] + "…" if len(text) > 55 else text
    ax.set_title(title, fontsize=10, loc="left", pad=2, color="#555")

for j in range(n, len(axes)):
    axes[j].axis("off")

fig.suptitle(
    "图10  Grad-CAM 热力图：BERT-CCPoem 三分类决策时关注的字符\n"
    "红色=高重要性，白=低重要性，红色下划线=词牌名。标签: shi=诗 ci=词 qu=曲",
    fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
out = f"{OUT_DIR}/fig10_gradcam_standalone.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"已保存: {out}")
