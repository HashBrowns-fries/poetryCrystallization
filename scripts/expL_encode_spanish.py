#!/usr/bin/env python3
"""
expL_encode_spanish.py
用 XLM-RoBERTa (multilingual BERT) 编码西班牙十四行诗

XLM-RoBERTa: 在 100 种语言上预训练，与 BERT-CCPoem 架构相同
优势：跨语言语义对齐，适合跨文化比较
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data"

# ═══════════════════════════════════════════════════════════════════════════
# 1. 加载西班牙诗人数据
# ═══════════════════════════════════════════════════════════════════════════

print("=== 加载西班牙诗人数据 ===")
with open(DATA_DIR / "processed/spanish_poets_disco.json") as f:
    spanish_data = json.load(f)

poets = spanish_data['poets']
print(f"诗人数: {len(poets)}")
print(f"总诗歌数: {spanish_data['metadata']['total_poems']}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. 加载 XLM-RoBERTa
# ═══════════════════════════════════════════════════════════════════════════

print("\n=== 加载 XLM-RoBERTa ===")
model_name = "xlm-roberta-base"  # 或 "xlm-roberta-large"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"设备: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

print(f"模型: {model_name}")
print(f"隐层维度: {model.config.hidden_size}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. 编码函数
# ═══════════════════════════════════════════════════════════════════════════

def encode_poem(text, max_length=512):
    """编码单首诗，返回 [CLS] token 嵌入"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # 取 [CLS] token (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    return cls_embedding.squeeze()

# ═══════════════════════════════════════════════════════════════════════════
# 4. 诗人级编码（平均池化）
# ═══════════════════════════════════════════════════════════════════════════

print("\n=== 编码西班牙诗人（诗人级聚合）===")

poet_embeddings = []
poet_names = []

for poet in tqdm(poets, desc="编码诗人"):
    name = poet['name']
    poems = poet['poems']

    # 编码所有诗歌
    poem_embeds = []
    for poem_text in poems:
        try:
            embed = encode_poem(poem_text)
            poem_embeds.append(embed)
        except Exception as e:
            print(f"警告: 编码失败 {name} - {e}")
            continue

    if poem_embeds:
        # 诗人级聚合：平均池化
        poet_embed = np.mean(poem_embeds, axis=0)
        poet_embeddings.append(poet_embed)
        poet_names.append(name)

poet_embeddings = np.array(poet_embeddings)
print(f"\n✅ 编码完成")
print(f"诗人数: {len(poet_names)}")
print(f"嵌入维度: {poet_embeddings.shape}")

# ═══════════════════════════════════════════════════════════════════════════
# 5. 保存嵌入
# ═══════════════════════════════════════════════════════════════════════════

# 保存嵌入矩阵
np.save(DATA_DIR / "processed/spanish_poet_embeddings.npy", poet_embeddings)

# 保存诗人名称列表
with open(DATA_DIR / "processed/spanish_poet_names.json", 'w', encoding='utf-8') as f:
    json.dump(poet_names, f, ensure_ascii=False, indent=2)

print(f"\n✅ 已保存:")
print(f"  - {DATA_DIR / 'processed/spanish_poet_embeddings.npy'}")
print(f"  - {DATA_DIR / 'processed/spanish_poet_names.json'}")

# ═══════════════════════════════════════════════════════════════════════════
# 6. 下一步说明
# ═══════════════════════════════════════════════════════════════════════════

print("\n=== 下一步：跨文化分析 ===")
print("运行: python scripts/expL_cross_cultural_analysis.py")
print("  - 联合 PCA：中国 + 西班牙诗人")
print("  - PERMANOVA：文化效应 vs 体裁效应")
print("  - 跨文化距离矩阵")
