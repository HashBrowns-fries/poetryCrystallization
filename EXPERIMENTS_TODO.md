# 补充实验计划

## 外部资源

### 1. Hanpoetry 数据集
- **仓库**: https://github.com/TASYU78/Hanpoetry
- **内容**: 
  - `tang_db`: 唐诗数据库
  - `heian_db`: 平安朝汉诗数据库（日本诗人用古典汉语创作）
  - `tools`: 可视化和数据处理工具
- **论文**: "Toward a Semantic Framework for Han Poetry: Multilingual and Decentralized Integration of East Asian Literary Heritage"
- **价值**: 
  - ✅ **语言相同**（古典汉语）→ 可直接用 BERT-CCPoem 编码
  - ✅ **文化/地理不同**（日本平安朝 vs 中国）→ 完美的跨文化对照
  - ✅ **时期重叠**（平安朝 794-1185 约等于唐末-宋初）→ 可控制时间变量

### 2. Couyun 格律检测工具
- **仓库**: https://github.com/hulbji/couyun
- **功能**: 中文诗歌格律检测（平仄、韵部、格律模式）
- **价值**: 可用于实验 2（格律特征 baseline）

---

## 实验优先级（修订版）

### 🚀 Phase 1: 立即可做（本周完成）

#### ✅ 实验 3: 时间窗口稳健性
- **时间**: 1 天
- **数据**: 本地已有（poet_genre_hybrid.json）
- **步骤**: 唐宋子集、明清子集分别运行 PERMANOVA + PCA
- **产出**: §5.13 "时间窗口稳健性"表格

#### ✅ 实验 4: 跨文化对照（Heian 平安朝）
- **时间**: 2-3 天（依赖 Hanpoetry 数据质量）
- **数据**: Hanpoetry `heian_db`（需克隆）
- **步骤**:
  1. 解析 heian_db 诗歌文本和作者元数据
  2. 用 BERT-CCPoem 编码平安朝汉诗（~500-1000 首？）
  3. 对比中国唐宋诗人 vs 日本平安朝诗人的语义距离
  4. 检验假设：地理/文化差异是否大于体裁差异？
- **产出**: §5.14 "跨文化验证：平安朝汉诗"

---

### 🔄 Phase 2: 中等成本（下周完成）

#### 实验 2: 格律特征 baseline
- **时间**: 3-5 天（有 couyun 工具加速）
- **数据**: 本地 poet_poems.json
- **步骤**:
  1. 集成 couyun 格律检测器
  2. 对每首诗提取平仄序列、韵部、字数模式
  3. 诗人级聚合（格律特征向量）
  4. 训练 Logistic Regression 三分类 ci/shi/qu
  5. 与 BERT 语义嵌入准确率对比
- **产出**: §5.15 "格律 vs 语义 baseline"

---

### 🌐 Phase 3: 可选增强（时间允许）

#### 实验 1: GuwenBERT 交叉验证
- **时间**: 3-4 天（需 GPU 重新编码 111 万首）
- **数据**: 本地 poet_poems.json + sentence_embeddings_by_poet.npz
- **步骤**: 下载 GuwenBERT → 重新编码 → PERMANOVA + PCA
- **产出**: §5.16 "跨模型验证"
- **优先级**: 降低（因为有 Heian 跨文化数据更有新意）

---

## 修订后的投稿策略

### 方案 A: 快速顶刊版（1-2 周）
- ✅ 实验 3（时间窗口）
- ✅ 实验 4（Heian 跨文化）
- **亮点**: 平安朝汉诗是极强的创新点——同语言跨文化验证，国际期刊会非常感兴趣
- **投稿目标**: *Journal of Cultural Analytics*, *Digital Scholarship in the Humanities*

### 方案 B: 完整顶刊版（3-4 周）
- 实验 3 + 4 + 2（格律 baseline）
- 可选：实验 1（GuwenBERT）
- **投稿目标**: *JCA* 特刊（跨文化数字人文）

---

## 下一步

1. ✅ 启动 Clash 代理
2. 克隆 Hanpoetry 和 couyun 仓库
3. 探索 heian_db 数据结构和质量
4. 决定：先做实验 3（时间窗口）还是实验 4（Heian）？

**建议顺序**: 实验 3 → 实验 4 → 实验 2（如果时间允许）
