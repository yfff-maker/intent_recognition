# 多臂老虎机记忆库使用说明

## 📁 文件说明

### 原有文件（不变）
- `main.py` - 使用简单记忆库（FIFO淘汰策略）
- `memory_bank.py` - 简单记忆库实现
- 输出文件：
  - `output/intent_inference_results.csv`
  - `output/intent_inference_results.xlsx`

### 新增文件
- `main_bandit.py` - 使用Bandit记忆库（智能淘汰策略）
- `memory_bank_bandit.py` - Bandit记忆库实现
- 输出文件：
  - `output/intent_inference_results_bandit.csv`
  - `output/intent_inference_results_bandit.xlsx`
  - `output/memory_bank_statistics.xlsx` ⭐ **新增统计信息**

---

## 🚀 运行方式

### 方式1：运行原有版本（简单记忆库）
```bash
cd tool_src
python main.py
```
**输出：** `output/intent_inference_results.xlsx`

---

### 方式2：运行Bandit版本（多臂老虎机记忆库）
```bash
cd tool_src
python main_bandit.py
```
**输出：**
1. `output/intent_inference_results_bandit.xlsx` - 意图识别结果
2. `output/memory_bank_statistics.xlsx` - 记忆库统计信息

---

## 📊 输出文件对比

### 1. `intent_inference_results.xlsx` vs `intent_inference_results_bandit.xlsx`

| 列名 | 说明 | 两者是否相同？ |
|------|------|--------------|
| Participant | 参与者ID | ✅ 相同 |
| AnchorTimestamp | 异常点时间戳 | ✅ 相同 |
| AnomalyType | 异常类型 | ✅ 相同 |
| Strategy | 窗口策略(A/B/C) | ✅ 相同 |
| Intent | 推断的意图 | ⚠️ **可能不同** |
| Confidence | 置信度 | ⚠️ **可能不同** |
| Evidence | 证据引用 | ⚠️ **可能不同** |

**关键区别：**
- 简单版本：静态LTM，FIFO淘汰
- Bandit版本：动态LTM，智能淘汰 + STM提升机制
- **预期：** Bandit版本在长任务中表现更好

---

### 2. `memory_bank_statistics.xlsx` - 新增统计文件

这个文件**仅在运行main_bandit.py时生成**，记录每个chunk的详细信息：

| 列名 | 说明 | 示例 |
|------|------|------|
| Participant | 参与者ID | P1 |
| ChunkID | Chunk唯一标识 | P1_0, P1_promoted_50000 |
| TimeStart | Chunk时间范围起点(ms) | 0 |
| TimeEnd | Chunk时间范围终点(ms) | 3000 |
| EventIdxRange | 事件索引范围 | 0-30 |
| AccessCount | 被检索次数 | 5 |
| UsefulCount | 被采用次数 | 3 |
| EstimatedValue | 估计价值 | 0.8542 |
| ConfidenceBound | UCB置信上界 | 0.3215 |
| LastAccessTime | 最后访问时间(ms) | 45000 |
| CreationTime | 创建时间(ms) | 0 |
| UsageRate | 采用率 (UsefulCount/AccessCount) | 0.6000 |

**用途：**
- 分析哪些chunk最有价值（EstimatedValue高）
- 验证时间衰减机制（EstimatedValue随时间变化）
- 识别"被遗忘"的chunk（AccessCount低）
- 发现"明星chunk"（UsageRate高）

---

## 🔬 对比实验设计

### 步骤1：运行两个版本
```bash
# 运行简单版本
python main.py

# 运行Bandit版本
python main_bandit.py
```

### 步骤2：对比结果
使用Excel或Python分析：
```python
import pandas as pd

# 读取两个结果文件
df_baseline = pd.read_excel("output/intent_inference_results.xlsx")
df_bandit = pd.read_excel("output/intent_inference_results_bandit.xlsx")

# 对比准确率（假设有ground truth）
# ...

# 对比置信度
print("简单版本平均置信度:", df_baseline['Confidence'].mean())
print("Bandit版本平均置信度:", df_bandit['Confidence'].mean())
```

### 步骤3：分析Bandit统计
```python
df_stats = pd.read_excel("output/memory_bank_statistics.xlsx")

# 找出最有价值的chunk
top_chunks = df_stats.nlargest(10, 'EstimatedValue')
print("最有价值的10个chunk:")
print(top_chunks[['ChunkID', 'AccessCount', 'UsefulCount', 'EstimatedValue']])

# 分析采用率分布
import matplotlib.pyplot as plt
df_stats['UsageRate'].hist(bins=20)
plt.xlabel('Usage Rate')
plt.ylabel('Frequency')
plt.title('Chunk Usage Rate Distribution')
plt.show()
```

---

## ⚙️ 关键配置参数

### config.py中的参数（两个版本共用）
```python
KEY_EVENT_TARGET_K = 600      # 选择600个关键事件
MEMORY_CHUNK_SIZE = 30         # 每个chunk 30个事件
MEMORY_MAX_ITEMS = 50          # 最多存储50个chunk
MEMORY_RETRIEVE_TOP_K = 5      # 检索返回5个chunk
```

### main_bandit.py中的Bandit特有参数
```python
mb = MemoryBankWithBandit(
    max_items=MEMORY_MAX_ITEMS,
    exploration_factor=1.5  # UCB探索因子（越大越倾向探索）
)

# STM→LTM提升条件
if parsed.get("confidence", 0) > 0.8:  # 置信度阈值
    mb.promote_stm_to_ltm(...)
```

---

## 📈 预期实验结果

### 假设验证

**假设1：** Bandit版本在长任务中表现更好
- 验证方法：比较任务时长>15分钟的参与者的准确率
- 预期：Bandit版本准确率更高

**假设2：** 智能淘汰比FIFO更有效
- 验证方法：查看`memory_bank_statistics.xlsx`，分析被删除的chunk是否确实价值低
- 预期：被删除的chunk的UsageRate < 平均值

**假设3：** STM→LTM提升机制有效
- 验证方法：统计有多少chunk是从STM提升的（ChunkID包含"promoted"）
- 预期：提升的chunk的EstimatedValue > 平均值

---

## 🐛 故障排查

### 问题1：运行main_bandit.py报错 "No module named 'memory_bank_bandit'"
**解决：** 确保在tool_src目录下运行

### 问题2：Excel文件被占用，无法保存
**解决：** 关闭已打开的Excel文件，或者只使用CSV版本

### 问题3：Bandit统计文件为空
**解决：** 确保至少处理了一个参与者，且检测到了异常点

---

## 📞 技术支持

如有问题，请检查：
1. Python版本 >= 3.8
2. 已安装依赖：`pandas`, `openpyxl`
3. 数据集路径正确（config.py中的DATASET_ROOT）

---

## 🎯 下一步计划

1. ✅ 运行两个版本，生成结果文件
2. ⬜ 对比分析两个版本的准确率差异
3. ⬜ 可视化Bandit统计信息（价值分布、时间衰减曲线）
4. ⬜ 消融实验（不同exploration_factor的影响）
5. ⬜ 向导师汇报实验结果

---

生成日期: 2026-02-04
