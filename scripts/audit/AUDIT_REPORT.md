# PERMANOVA 审计报告：数字内部矛盾诊断

**日期**: 2026-06-04  
**审计人**: Claude Code  
**问题来源**: 审稿人质疑论文中 "R²=0.733, p=0.283" 的矛盾

---

## 执行摘要

审稿人的质疑**完全正确**。论文存在三个层面的数学/统计错误：

1. **置换检验逻辑错误**: 代码计算的是交互效应的 p 值，不是纯效应的 p 值
2. **Bootstrap 数字混乱**: 论文中引用了至少三个不同的 R² 值，互相矛盾
3. **表述与计算不匹配**: 论文文本的解释与实际计算结果不对应

这不是小的报告错误，而是**结构性的统计实现问题**。必须重写双因素 PERMANOVA 部分。

---

## 1. 核心矛盾：R²=0.733 vs p=0.283

### 1.1 论文声称

> 纯朝代效应（朝代条件于体裁）：R² = 0.733, p = 0.283 (n.s.)

### 1.2 实际计算

从 `poet_distances.npy` (4634×4634) 重新计算：

```
SS_Total = 246,146.92
SS_genre (边际) = 42,277.92  → R²_genre = 0.1718
SS_dynasty (边际) = 219,254.08  → R²_dynasty = 0.8907
SS_joint = 222,693.24  → R²_joint = 0.9047

纯效应（条件效应）:
SS_genre|dynasty = SS_joint - SS_dynasty = 3,439.16  → R²_genre|dynasty = 0.0140
SS_dynasty|genre = SS_joint - SS_genre = 180,415.32  → R²_dynasty|genre = 0.7330
```

**F 统计量**:
```
df_genre = 2, df_dynasty = 6, df_residual = 4620
MS_dynasty|genre = 180415.32 / 6 = 30,069.22
MS_residual = (SS_Total - SS_joint) / 4620 = 5.08
F_dynasty = 30,069.22 / 5.08 = 5,923.16
```

### 1.3 矛盾

若 F = 5,923.16，这是一个**极端显著**的统计量。即使用保守的 Bonferroni 校正，p 值也应 << 0.001。

但论文报告 p = 0.283。这在数学上不可能，除非置换检验的实现有误。

---

## 2. 代码逻辑错误：置换的是交互效应，不是纯效应

### 2.1 原代码逻辑 (`_archive/40_genre_dominance.py`, line 135-147)

```python
def perm_test(cond_ss_func, n_p=N_PERM):
    cnt = 0
    for _ in range(n_p):
        perm_fB = fB_lbl.copy()
        np.random.shuffle(perm_fB)
        jp = np.array([f"{a}_{b}" for a, b in zip(fA_lbl, perm_fB)])
        ss_jp  = ss_between(jp)
        ss_Bp  = ss_between(perm_fB)
        cp     = max(0, ss_jp - ss_b_A - ss_Bp)  # ← 这是交互效应!
        if cp >= cond_ss_func: cnt += 1          # ← 比较错了!
    return round(cnt / n_p, 4)

p_A = perm_test(ss_A_cond)
p_B = perm_test(ss_B_cond)
```

### 2.2 错误诊断

**问题**: 函数固定 `fA_lbl` (体裁), 置换 `fB_lbl` (朝代), 然后计算：
```
cp = ss_jp - ss_b_A - ss_Bp
```
这个 `cp` 是**交互效应的置换值**，不是纯朝代效应的置换值。

然后用 `cp >= ss_B_cond` (纯朝代效应) 判断显著性。这在统计上没有意义：
- **交互效应 SS** 通常远小于 **纯效应 SS**
- 用小的统计量去比较大的观测值，必然得到大的 p 值

### 2.3 正确的置换逻辑

要检验 "纯朝代效应 (朝代|体裁)" 的显著性，应该：

1. **固定体裁标签**，**置换朝代标签**
2. 计算置换后的 `ss_dynasty_perm|genre = ss_joint_perm - ss_genre`
3. 比较 `ss_dynasty_perm|genre >= ss_dynasty_obs|genre`

伪代码：
```python
def perm_test_dynasty_cond(n_perm=999):
    cnt = 0
    ss_genre_obs = ss_between(g_id)  # 固定
    ss_dynasty_cond_obs = ss_joint_obs - ss_genre_obs
    
    for _ in range(n_perm):
        dyn_perm = dyn_id.copy()
        np.random.shuffle(dyn_perm)
        joint_perm = [f"{g}_{d}" for g, d in zip(g_id, dyn_perm)]
        ss_joint_perm = ss_between(joint_perm)
        ss_dynasty_cond_perm = ss_joint_perm - ss_genre_obs
        
        if ss_dynasty_cond_perm >= ss_dynasty_cond_obs:
            cnt += 1
    
    return cnt / n_perm
```

**预测**: 若用正确逻辑，p_dynasty 应该 < 0.001，不是 0.283。

---

## 3. Bootstrap 数字混乱

### 3.1 三个不一致的数字

| 来源 | 数值 | 含义 |
|------|------|------|
| `genre_dominance.json: point_R2` | 0.1243 | ❓未知 |
| `genre_dominance.json: exp1_genre_permanova.R2` | 0.1718 | 单因素体裁 R² (正确) |
| `genre_dominance.json: exp2_two_factor.R2_genre_cond` | 0.0140 | 双因素纯体裁 R² (正确) |
| 论文 §5.1.3: "体裁 R² 的点估计" | 0.1243 | 引用了 point_R2 |
| 论文 §5.1.3: "Bootstrap CI" | [0.1057, 0.1431] | 与 0.1243 对应 |
| 论文结论处 | R²=0.014, CI=[0.106, 0.143] | 混淆了条件 R² 与 Bootstrap CI |

### 3.2 问题根源

Bootstrap 代码 (line 300) 写的是：
```python
"point_R2": r2_genre["R2"],  # ← 这是单因素 R² = 0.1718
```

但实际保存的 `point_R2 = 0.1243` 不等于 0.1718。怀疑：
- 运行 `40_genre_dominance.py` 时用了不同的数据或参数
- 或者 `genre_dominance.json` 是旧版本代码生成的

### 3.3 论文引用错误

论文结论处写：
> R² = 0.014, Bootstrap CI = [0.106, 0.143]

这句话混淆了：
- **0.014** 是**条件体裁 R²** (双因素)
- **[0.106, 0.143]** 是**单因素体裁 R²** 的 Bootstrap CI

它们不是同一个估计量的点估计与区间估计！

---

## 4. 其他发现

### 4.1 公式标注错误

论文中写的公式 (§5.1.1):
```latex
R²_genre|dy = R²_joint - R²_dynasty
R²_dynasty|gen = R²_joint - R²_genre
```

这个公式在 **SS 层面** 成立，但在 **R² 层面** 不完全成立，因为：
```
R²_A = SS_A / SS_Total
R²_joint - R²_genre ≠ R²_genre|dynasty (在分母相同时可以直接减)
```

更严格的表述应该：
```
SS_genre|dynasty = SS_joint - SS_dynasty
R²_genre|dynasty = SS_genre|dynasty / SS_Total
```

### 4.2 df 未报告

论文表格只报告了 R²、p 值，没有报告 df、F、SS。审稿人无法验证计算。

---

## 5. 修复建议

### 5.1 立即修复（blocking）

1. **重写双因素 PERMANOVA 的置换检验**
   - 分别为 `genre|dynasty` 和 `dynasty|genre` 写独立的置换函数
   - 固定条件变量，置换被检验变量
   - 计算正确的纯效应 SS

2. **重跑所有 PERMANOVA 分析**
   - 用新代码重新生成 `genre_dominance.json`
   - 验证所有 R²、F、p 值的一致性

3. **统一 Bootstrap 报告**
   - 明确 Bootstrap 的是**单因素 R²** 还是**条件 R²**
   - 论文中只报告一种，不要混淆

4. **补充完整统计表**
   - 报告 SS、df、MS、F、p 的完整分解
   - 让审稿人可以验证计算

### 5.2 表述修正

论文中 "高 R² 与不显著 p 值并存" 的解释 (PERMDISP) 是**错误的**。正确的原因是：
- **实现错误**: 置换检验用错了统计量
- 不是 "散布差异导致 F 效率低"

修改后应该诚实报告：
- 若 p_dynasty < 0.001: "纯朝代效应显著，但 R² 远小于边际 R²"
- 若 p_dynasty > 0.05 (用正确置换后): "纯朝代效应不显著，支持体裁是更紧致的聚类因子"

### 5.3 降级推断

即使修复后，也要避免过强表述：
- ❌ "体裁压倒朝代"
- ✓ "在控制朝代后，体裁仍呈现可检测的局部聚类信号"

---

## 6. 待验证（等置换检验完成）

我已经启动 `audit_permanova.py`，正在运行 999 次置换检验（预计 20-30 分钟）。完成后将确认：

1. **正确的 p_genre|dynasty** 是否仍 < 0.001
2. **正确的 p_dynasty|genre** 到底是多少
3. **原代码的 p=0.283** 是否真的来自交互效应置换

初步预测：
- p_genre ≈ 0.001 (仍显著)
- p_dynasty ≈ 0.001 (也会显著，因为 F=5923)
- 原代码的 p=0.283 应该等于 "交互效应显著性"

---

## 7. 结论

审稿人的质疑击中要害。这不是表述问题，是**统计实现错误**。

**拒稿风险**: 极高。顶刊审稿人看到这种矛盾会直接拒稿。

**修复成本**: 中等。需要：
1. 重写置换函数（1 天）
2. 重跑所有分析（半天）
3. 重写论文 §5.1（1 天）
4. 降级所有过强表述（半天）

**修复后论文是否还有贡献**: 取决于正确的 p 值。若两个条件效应都显著，论文就变成 "体裁与朝代都有独立贡献"，而不是 "体裁压倒朝代"。

---

**审计状态**: 数学关系已确认，置换检验实证验证中（预计 00:50 完成）

**下一步**: 等置换完成后，编写修正代码 `permanova_corrected.py`
