#!/usr/bin/env python3
"""
expL_cross_cultural.py
实验 L: 跨文化对照（DISCO 西班牙十四行诗）

研究问题：
1. 形式约束（十四行诗 sonnet）是否在跨文化语境中也产生语义内聚？
2. 西班牙 sonnet 与中国 shi/ci/qu 的语义距离如何？
3. 形式约束的跨文化普遍性 vs 文化特异性

方法：
1. 从 DISCO 提取西班牙诗人（≥10 首十四行诗）
2. 用 Claude API / multilingual BERT 编码西班牙语诗歌
3. 联合 PCA：中国诗人（ci/shi/qu）+ 西班牙诗人（sonnet）
4. PERMANOVA：文化效应 vs 体裁效应（within-culture）
5. 跨文化距离分析：sonnet vs ci, sonnet vs shi, sonnet vs qu

预期：
- H1: sonnet 诗人在西班牙语境中形成内聚子空间
- H2: sonnet 与 ci 距离 < sonnet 与 shi 距离（形式相似性：长短句 vs 整齐句）
- H3: 文化效应 > 体裁效应（语言/文化差异主导）
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import re

BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data"
DISCO_DIR = DATA_DIR / "raw/disco/disco-main"

# ═══════════════════════════════════════════════════════════════════════════
# 1. 加载 DISCO 元数据
# ═══════════════════════════════════════════════════════════════════════════

print("=== 加载 DISCO 元数据 ===")

# 作者元数据
authors_df = pd.read_csv(DISCO_DIR / "author_metadata.tsv", sep='\t')
print(f"总作者数: {len(authors_df)}")

# 诗歌元数据
poems_df = pd.read_csv(DISCO_DIR / "poem_metadata.tsv", sep='\t')
print(f"总诗歌数: {len(poems_df)}")

# 统计每位作者的诗歌数
author_poem_counts = poems_df['author_id'].value_counts()
print(f"\n诗歌数分布:")
print(author_poem_counts.describe())

# 筛选：≥10 首诗的作者
min_poems = 10
qualified_authors = author_poem_counts[author_poem_counts >= min_poems].index.tolist()
print(f"\n≥{min_poems} 首诗的作者数: {len(qualified_authors)}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. 提取诗歌文本
# ═══════════════════════════════════════════════════════════════════════════

print("\n=== 提取诗歌文本 ===")

spanish_poets = {}

for author_id in qualified_authors:
    # 获取作者信息
    author_row = authors_df[authors_df['aid'] == author_id]
    if author_row.empty:
        continue

    author_name = author_row.iloc[0]['author']

    # 获取该作者的所有诗歌
    author_poems = poems_df[poems_df['author_id'] == author_id]

    poems_text = []
    for _, poem_row in author_poems.iterrows():
        poem_id = poem_row['poem_id']

        # 构建文件路径（添加 disco 前缀）
        txt_file = None
        for period in ['15th-17th', '18th', '19th', '20th']:
            candidate = DISCO_DIR / f"txt/{period}/per-sonnet/disco{poem_id}.txt"
            if candidate.exists():
                txt_file = candidate
                break

        if txt_file and txt_file.exists():
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                if text:
                    poems_text.append(text)

    if len(poems_text) >= min_poems:
        spanish_poets[author_name] = {
            'author_id': author_id,
            'poems': poems_text[:50],  # 限制最多50首，与中国诗人一致
            'n_poems': len(poems_text)
        }

print(f"成功提取 {len(spanish_poets)} 位西班牙诗人")
print(f"诗歌总数: {sum(p['n_poems'] for p in spanish_poets.values())}")

# 示例
sample_poet = list(spanish_poets.keys())[0]
print(f"\n示例诗人: {sample_poet}")
print(f"诗歌数: {spanish_poets[sample_poet]['n_poems']}")
print(f"首诗前200字:\n{spanish_poets[sample_poet]['poems'][0][:200]}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. 保存西班牙诗人数据
# ═══════════════════════════════════════════════════════════════════════════

output_data = {
    'metadata': {
        'corpus': 'DISCO v4.0',
        'n_poets': len(spanish_poets),
        'min_poems': min_poems,
        'total_poems': sum(p['n_poems'] for p in spanish_poets.values())
    },
    'poets': [
        {
            'name': name,
            'author_id': data['author_id'],
            'n_poems': data['n_poems'],
            'poems': data['poems']
        }
        for name, data in spanish_poets.items()
    ]
}

output_path = DATA_DIR / "processed/spanish_poets_disco.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"\n✅ 西班牙诗人数据已保存至: {output_path}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. 下一步说明
# ═══════════════════════════════════════════════════════════════════════════

print("\n=== 下一步：诗歌编码 ===")
print("需要选择编码方法：")
print("  选项 A: multilingual BERT (mBERT / XLM-RoBERTa)")
print("  选项 B: Claude API（语义嵌入）")
print("  选项 C: OpenAI text-embedding-3-large")
print("")
print("推荐：选项 A (multilingual BERT)，与中国诗歌 BERT 嵌入最可比")
print("")
print("运行: python scripts/expL_encode_spanish.py")
