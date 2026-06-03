# 体裁先于朝代：古典诗歌语义空间的层级组织

**Genre Precedes Dynasty: Hierarchical Organization of Semantic Space in Classical Chinese Poetry**

> 本项目为课程论文配套研究代码与数据。

## 核心结论

体裁形式是中国古典诗歌语义空间的**第一组织力量**，朝代效应是体裁效应的副产品。

## 数据

实验结果 JSON 位于 `data/processed/`。完整语料库（需另行获取）包含 111 万余首中国古典诗歌，覆盖 4,634 位诗人。

## 环境

```bash
uv sync
```

## 实验脚本

### 核心实验（A-I）

| 脚本 | 实验内容 |
|------|---------|
| `scripts/expA_tsne_kmeans.py` | t-SNE + K-means 聚类质量对比 |
| `scripts/expB_ttest.py` | Welch t检验 ci vs shi 诗人嵌入距离 |
| `scripts/expC_bert_finetune.py` | BERT-CCPoem 微调 ci/shi 二分类器 |
| `scripts/expD_model_comparison.py` | BERT-CCPoem vs GuwenBERT 对比 |
| `scripts/expE_cipai_counterfactual.py` | 词牌级别反事实实验（形式约束机制验证） |
| `scripts/expF_tfidf_baseline.py` | TF-IDF 非神经基线（体裁 vs 朝代分类对比） |
| `scripts/expG_sensitivity.py` | 参数敏感性分析（Louvain k/τ 扫描 + 距离度量对比） |
| `scripts/expH_cross_dynasty.py` | 跨朝代泛化测试（体裁分类器的跨时间迁移能力） |
| `scripts/expI_cipai_compare.py` | 词牌内 vs 跨词牌语义距离比较 |

### 方差分解与统计检验

| 脚本 | 实验内容 |
|------|---------|
| `scripts/40_genre_dominance_fast.py` | PERMANOVA 体裁 vs 朝代方差分解 |
| `scripts/balance_check.py` | 类别平衡 PERMANOVA |
| `scripts/permdisp_test.py` | 散布效应 Levene 检验 |
| `scripts/louvain_purity_null.py` | Louvain 纯度零假设检验 |

## 可视化

| 脚本 | 输出 |
|------|------|
| `scripts/30_fig1_pca_semantic_gravity_v2.py` | 图1 PCA语义空间 |
| `scripts/31_fig2_permanova_v2.py` | 图2 PERMANOVA分解 |
| `scripts/31_fig3_community_v2.py` | 图3 Louvain社区 |
| `scripts/32_fig4_intertextual_v2.py` | 图4 互文距离 |
| `scripts/34_fig6_geographic_gravity_v2.py` | 图6 语义引力 |
| `scripts/35_fig5_bert_classification.py` | 图5/8 BERT分类 |
| `scripts/generate_figures.py` | 批量生成实验图表（expF/G/H/I 对应图） |
| `scripts/generate_review_figures.py` | 生成审稿补充图表 |

## 论文编译

使用 XeLaTeX 编译（需要 CJK 字体支持）：

```bash
xelatex paper.tex
```

生成的 PDF：`paper.pdf`（49 页）
