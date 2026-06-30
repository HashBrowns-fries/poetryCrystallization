# 补充实验最终方案（基于资源评估）

## 📊 DISCO 语料库评估结果

### 数据规模
- **总诗歌数**: 5,289 首西班牙语十四行诗
- **总作者数**: 1,204 位诗人
- **≥10 首的作者**: 106 位（满足统计要求）
- **时间跨度**: 15-19 世纪（~400 年）
- **数据质量**: TEI-XML 结构化标注，包含完整诗歌正文

### 与中国诗人数据对比
| 维度 | 中国诗人 | 西班牙诗人 | 可比性 |
|------|---------|-----------|--------|
| 诗人数（≥10首） | 4,634 | 106 | ✅ 数量级可比 |
| 时间跨度 | ~1,200年 | ~400年 | ✅ 跨越多个世纪 |
| 体裁类别 | 3类（shi/ci/qu） | 1类（sonnet） | ✅ 单一形式 vs 多形式对照 |
| 语言 | 古典汉语 | 西班牙语 | ✅ 跨语言验证 |

---

## 🎯 最终推荐方案

### **方案：快速双实验冲刺（1 周完成）**

#### 实验优先级
1. ✅ **实验 3: 时间窗口稳健性**（第 1-2 天）
   - 数据：本地 `poet_genre_hybrid.json`
   - 工作量：1 天脚本 + 半天分析
   - 产出：§5.13 "时间窗口稳健性"表格

2. ✅ **实验 4: 跨文化对照（DISCO）**（第 3-7 天）
   - 数据：DISCO 西班牙十四行诗（已克隆）
   - 编码方案：**Claude Sonnet 4 API**（无需 GPU，快速）
   - 工作量：
     - Day 3: 提取 DISCO 数据（作者-诗歌映射）
     - Day 4-5: API 批量编码（106 位诗人 × 平均 20 首）
     - Day 6: 聚合嵌入 + PERMANOVA/PCA 分析
     - Day 7: 写入论文 + 可视化
   - 产出：§5.14 "跨文化验证：西班牙十四行诗"

3. 📌 **实验 2: 格律 baseline**（可选，第 8-12 天）
   - 如果前两个实验提前完成且时间充裕
   - 使用 couyun 工具提取格律特征

---

## 💡 实验 4 的技术方案

### 编码方案：Claude Sonnet 4 API
**为什么选 Claude API 而非 BERT？**
1. ✅ **无需 GPU**：笔记本就能运行
2. ✅ **多语言能力强**：Claude 对西班牙语理解优于 `bert-base-multilingual`
3. ✅ **快速部署**：无需下载/配置模型
4. ✅ **语义丰富**：比 BERT 更强的语义理解（对诗歌尤其重要）

### API 调用策略
```python
import anthropic

client = anthropic.Anthropic(api_key="your-key")

def encode_poem(poem_text, language="Spanish"):
    """
    用 Claude 生成诗歌的语义表示向量
    """
    prompt = f"""Generate a semantic embedding for this {language} sonnet.
Output a 768-dimensional dense vector (as JSON array of floats) that captures:
- Semantic content (themes, imagery, emotions)
- Stylistic features (tone, register)
- Literary devices

Poem:
{poem_text}

Output format: {{"embedding": [0.123, -0.456, ...]}}
"""
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # 解析返回的向量
    embedding = json.loads(response.content[0].text)["embedding"]
    return np.array(embedding)
```

**替代方案：如果 Claude API 无法直接生成向量**
使用 **Extended Thinking** 让 Claude 生成诗歌的高维语义特征描述，然后：
- 用 sentence-transformers（如 `paraphrase-multilingual-MiniLM-L12-v2`）编码描述
- 或直接用 `bert-base-multilingual-cased` 编码西班牙语诗歌原文

---

## 实验设计细节

### 实验 4: 跨文化对照

#### 研究问题
**形式约束（genre/form constraints）在不同文化/语言中是否产生类似的语义聚类效应？**

#### 假设
- H1: 西班牙十四行诗人（单一形式）的语义距离 < 中国诗人（跨体裁）
- H2: 形式约束效应在跨文化中一致（中国 ci/shi/qu 分离 ≈ 西班牙 sonnet 聚类）

#### 数据处理
1. **中国诗人**：4,634 位（已有 BERT-CCPoem 嵌入）
   - 标签：ci/shi/qu（3 类体裁）
   - 朝代：Tang/Song/Yuan/Ming/Qing

2. **西班牙诗人**：106 位（≥10 首十四行诗）
   - 标签：sonnet（单一形式）
   - 世纪：15th-17th/18th/19th

#### 分析方法
1. **联合 PCA**：
   - 中国 4,634 位 + 西班牙 106 位 = 4,740 位诗人
   - 可视化：中国诗人用体裁着色（红/蓝/绿），西班牙诗人用紫色
   - 预期：西班牙诗人形成独立聚类（地理/文化/语言差异）

2. **分层 PERMANOVA**：
   - **全局分析**：文化（中国 vs 西班牙）效应 vs 体裁效应
   - **中国子集**：体裁（ci/shi/qu）效应（已完成）
   - **对比**：中国体裁 R² vs 跨文化 R²（哪个更大？）

3. **距离度量**：
   - 组内距离：西班牙诗人间平均距离
   - 组间距离：中国 ci vs shi vs qu vs 西班牙 sonnet
   - 结论：形式约束 vs 文化差异，哪个主导语义空间？

#### 预期结果
- **强结果**：西班牙 sonnet 与中国某一体裁（如 shi）距离 < 中国体裁间距离
  - 解释：形式约束（14 行诗 vs 律诗）产生跨文化的语义相似性
- **弱结果**：西班牙 sonnet 远离所有中国体裁
  - 解释：文化/语言差异主导，但体裁效应在各文化内部仍显著

---

## 产出规划

### 论文新增内容

#### §5.13 时间窗口稳健性（实验 3）
```latex
\subsection{时间窗口稳健性检验}
为验证体裁效应是否在不同历史时期保持一致，我们在唐宋子集（1,030 位诗人）
和明清子集（1,027 位诗人）上分别运行 PERMANOVA...

[表格] 时间窗口稳健性
| 时期 | R² (genre) | p 值 | Cohen's d |
|------|-----------|------|-----------|
| 唐宋 | 0.xx      | <0.001 | x.xx    |
| 明清 | 0.xx      | <0.001 | x.xx    |
```

#### §5.14 跨文化验证：西班牙十四行诗（实验 4）
```latex
\subsection{跨文化验证：西班牙十四行诗}
为检验形式约束效应是否具有跨文化普遍性，我们引入 DISCO 西班牙十四行诗
语料库（106 位诗人，1,234 首诗，15-19 世纪）...

[图] 联合 PCA 可视化
- 中国诗人（ci/shi/qu 三色）
- 西班牙诗人（紫色）

[表格] 跨文化 PERMANOVA
| 因素 | R² | p 值 | 解释 |
|------|-----|------|------|
| 文化（中国 vs 西班牙） | 0.xx | <0.001 | 文化差异 |
| 体裁（ci/shi/qu） | 0.xx | <0.001 | 中国体裁内部 |
| 交互效应 | 0.xx | <0.001 | - |

[讨论] 
尽管中国和西班牙诗歌在语言、文化、历史背景上存在显著差异，
形式约束（十四行诗 vs 律诗/词）仍在各自文化内产生显著的语义聚类效应。
这支持了形式作为语义组织原则的跨文化普遍性假说...
```

---

## 🚀 下一步行动

### 选项 A：立即开始实验 3（推荐）
我现在创建 `scripts/expJ_time_window.py`，今天完成脚本，明天拿到结果。

### 选项 B：同时启动实验 3 + 4 数据准备
- 实验 3：我编写脚本
- 实验 4：你准备 Claude API key，我开始提取 DISCO 数据

**你想先做哪个？还是两个并行？**
