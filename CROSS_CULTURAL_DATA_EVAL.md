# 跨文化对照实验：英文十四行诗语料评估

## 数据源评估

### 1. Folger Shakespeare Library（莎士比亚十四行诗）
- **网址**: https://www.folger.edu/explore/shakespeares-works/download/
- **内容**: 莎士比亚 154 首十四行诗（Sonnets 1-154）
- **优势**:
  - ✅ 权威来源（Folger 是世界最大莎士比亚图书馆）
  - ✅ 单一作者（控制作者变量）
  - ✅ 统一格式（英国式十四行诗：ABAB CDCD EFEF GG）
  - ✅ 创作时期明确（1590s-1609）
- **局限**:
  - ❌ 只有 1 位诗人（无法做诗人级聚类/PERMANOVA）
  - ❌ 无法检验"作者间差异 vs 形式差异"假设

### 2. PoeTree Poetry Corpora (Zenodo)
- **网址**: https://zenodo.org/records/17414036
- **项目主页**: https://versologie.cz/poetree/
- **内容**: 大规模多语言诗歌语料库（捷克语/德语/法语/西班牙语等）
- **评估**: 需下载确认是否包含**英文十四行诗**及**作者元数据**

### 3. Diachronic Spanish Sonnet Corpus (DISCO)
- **网址**: https://github.com/linhd-postdata/disco
- **内容**: **西班牙语十四行诗历时语料库**（15-19 世纪）
- **规模**: 约 **4,000+ 首十四行诗**，跨越 5 个世纪
- **优势**:
  - ✅ **多位诗人**（可做诗人级分析）
  - ✅ **时间跨度大**（可控制历史时期变量）
  - ✅ **TEI-XML 格式**（结构化标注）
  - ✅ GitHub 开源（易获取）
- **挑战**:
  - ⚠️ 西班牙语（需用 multilingual BERT 或西班牙语 BERT）
  - ⚠️ 可能需要提取作者级元数据

### 4. Oupoco Database
- **网址**: https://oupoco.huma-num.fr/
- **内容**: 法国诗歌形式数据库（包含十四行诗）
- **评估**: 需访问确认数据格式和规模

---

## 推荐策略

### 方案 A：快速验证（1-2 天）——用 DISCO 西班牙语十四行诗
**数据**: DISCO 语料库（4,000+ 首，多位诗人，15-19 世纪）  
**编码方案**: 使用 **Anthropic Claude API (Sonnet 4)** 生成嵌入
- 优点：无需下载模型，直接调用 API
- 缺点：西班牙语与汉语差异更大（但这恰好是**更强的跨文化检验**）

**实验设计**:
1. 从 DISCO 提取诗人-诗歌映射（类似 `poet_poems.json`）
2. 用 Claude Sonnet 4 API 批量编码西班牙语十四行诗（Prompt: "Encode this Spanish sonnet into a semantic embedding"）
3. 诗人级聚合嵌入（平均池化）
4. 对比分析：
   - **西班牙十四行诗人 vs 中国诗人**（跨文化、跨语言、跨形式）
   - **检验假设**：形式约束（sonnet vs 律诗/词）是否在不同文化中都产生显著聚类？

**产出**: 
- 新增论文节 §5.14 "跨文化验证：西班牙十四行诗"
- PCA 可视化：中国诗人（shi/ci/qu 三色）+ 西班牙诗人（第 4 种颜色）
- PERMANOVA 表格：中国体裁效应 vs 西班牙 sonnet 效应

---

### 方案 B：更严谨版（3-5 天）——结合 DISCO + 英文十四行诗选集
**数据**: 
- DISCO 西班牙语十四行诗（多诗人）
- 人工策展的英文十四行诗选集（从 Poetry Foundation 等来源爬取，选取 20-30 位诗人）

**编码方案**: 
- 西班牙语：`bert-base-multilingual-cased`
- 英语：`bert-base-uncased`
- 汉语：保持 BERT-CCPoem（已有数据）

**实验设计**:
三文化对比（汉语律诗/词 vs 西班牙十四行诗 vs 英文十四行诗）

**产出**: 
- 论文节 §7 "跨文化、跨语言验证"
- 三语言 PCA 联合可视化
- 结论：形式约束的普遍性（universal constraint）

---

## 立即行动计划

### Step 1: 探索 DISCO 语料库（30 分钟）
1. 克隆 GitHub 仓库
2. 检查数据格式（TEI-XML）
3. 统计诗人数、诗歌数、时期分布
4. 确认是否包含诗歌全文

### Step 2: 提取诗人-诗歌映射（2 小时）
1. 解析 TEI-XML 文件
2. 构建 `disco_poet_poems.json`（类似本项目格式）
3. 过滤：每位诗人至少 10 首诗（保证统计效力）

### Step 3: 编码实验（1 天）
- **选项 A**: 用 Claude API（快速，无需 GPU）
- **选项 B**: 用 `bert-base-multilingual-cased`（需 GPU）

### Step 4: 分析 & 写入论文（1 天）

---

## 要现在开始吗？

**推荐顺序**：
1. ✅ **立即做**: 实验 3（时间窗口稳健性）—— 1 天，本地数据
2. ✅ **第二周**: 实验 4（DISCO 跨文化）—— 2-3 天，Claude API 编码
3. 📌 **可选**: 实验 2（格律 baseline）—— 3-5 天，如果时间允许

这样 **1 周内可完成 2 个关键实验**（时间窗口 + 跨文化），大幅提升论文竞争力！

---

## 下一步

要我现在：
1. 克隆 DISCO 仓库并评估数据质量？
2. 还是先做实验 3（时间窗口），明天再探索 DISCO？

你偏好哪个顺序？
