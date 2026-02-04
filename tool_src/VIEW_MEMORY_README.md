# 📖 LTM & STM 内容查看器使用说明

## 🎯 功能概述

这个工具可以帮你查看：
1. **LTM记忆库统计信息** - 所有chunk的统计概览
2. **特定LTM chunk详情** - 查看某个chunk的完整内容
3. **STM+LTM在推理中的使用** - 查看LLM推理时实际使用的短期和长期记忆

---

## 🚀 使用方法

### 方法1：交互式模式（推荐）

```bash
python view_memory_contents.py --interactive
```

然后按照菜单提示选择操作。

---

### 方法2：命令行模式

#### 查看LTM统计概览
```bash
python view_memory_contents.py
```

#### 查看特定参与者的推理场景
```bash
python view_memory_contents.py --participant P1 --strategy C
```

#### 查看特定chunk详情
```bash
python view_memory_contents.py --chunk P1_0
```

#### 查看特定异常点的推理
```bash
python view_memory_contents.py --participant P1 --anomaly 0 --strategy A
```

---

## 📊 查看内容示例

### 1. LTM记忆库统计
显示：
- 总chunk数量
- 每个参与者的chunk分布
- chunk价值统计（最高/最低/平均）
- 使用率统计
- 最有价值的Top-5 chunk
- 从STM提升的chunk（如果有）

### 2. 特定LTM Chunk详情
显示：
- ChunkID、参与者
- 事件索引范围、时间范围
- Bandit统计（访问次数、有用次数、使用率、估计价值）
- 内容摘要（高频页面/控件/操作）
- Signature特征

### 3. STM+LTM在推理中的内容
显示：
- 推理场景信息（参与者、异常点、策略）
- LLM推理结果（意图、置信度、证据）
- 完整Prompt，包括：
  - **STM部分**：异常点附近的事件序列
  - **LTM部分**：检索到的Top-5相关chunk摘要

---

## 📁 数据文件位置

查看器从以下文件读取数据：

1. **LTM统计**：`output/memory_bank_statistics.xlsx`
   - 包含所有chunk的详细统计信息
   
2. **推理结果**：`output/intent_inference_results_bandit.xlsx`
   - 包含每次推理的Prompt和结果

> **注意**：运行此工具前，请确保已运行 `main_bandit.py` 生成这些文件。

---

## 💡 使用场景示例

### 场景1：检查LTM是否正常构建
```bash
python view_memory_contents.py --interactive
# 选择 1 - 查看LTM统计概览
```

预期看到：每个参与者约20个chunk（600个关键事件 ÷ 30 = 20）

---

### 场景2：分析某个参与者的记忆质量
```bash
python view_memory_contents.py --interactive
# 选择 2 - 查看特定chunk详情
# 输入参与者ID: P1
```

查看各chunk的：
- 访问频率（AccessCount）
- 有用程度（UsefulCount）
- 综合价值（EstimatedValue）

---

### 场景3：理解LLM推理时看到了什么
```bash
python view_memory_contents.py --participant P1 --strategy C --anomaly 0
```

可以看到：
- **STM窗口**：异常点附近的200个左侧+50个右侧事件（策略C）
- **LTM检索**：根据异常点特征检索到的Top-5相关chunk
- **LLM输出**：基于这些上下文推理出的意图和证据

---

## 🔍 关键字段解释

### LTM Chunk字段

| 字段 | 含义 |
|------|------|
| ChunkID | chunk标识符（如 P1_0, P1_promoted_123456） |
| AccessCount | 被检索的总次数 |
| UsefulCount | 被LLM实际使用的次数（相似度>阈值） |
| UsageRate | 使用率 = UsefulCount / AccessCount |
| EstimatedValue | 综合价值（考虑历史效用+时间衰减+信息稀有度） |
| Summary | 内容摘要（Top-5 pages/widgets/ops） |
| Signature | 特征集合（用于相似度计算） |

### Prompt字段
- **Short-Term Memory (STM)**：窗口内的事件序列（压缩后）
- **Long-Term Memory (LTM)**：检索到的chunk摘要（Top-5）
- **Output Schema**：要求LLM输出的JSON格式

---

## ❓ 常见问题

### Q1: 为什么有些chunk的AccessCount是0？
A: 如果某个chunk与任何异常点都不相似，就不会被检索到。

### Q2: 为什么有些chunk是 "promoted"？
A: 这些是从STM提升到LTM的chunk（LLM推理置信度>0.8且策略为C）。

### Q3: 如何判断LTM质量好不好？
A: 观察：
- UsageRate较高（说明检索准确）
- EstimatedValue分布合理（有高有低，说明区分度好）
- 有chunk被提升（说明STM→LTM机制生效）

---

## 🛠️ 调试技巧

1. **如果看不到LTM内容**：检查 `output/memory_bank_statistics.xlsx` 是否存在
2. **如果看不到STM内容**：检查 `output/intent_inference_results_bandit.xlsx` 的Prompt列
3. **如果想深入分析**：用Excel打开这两个文件，进行自定义筛选和统计

---

## 📞 需要帮助？

如果遇到问题，请检查：
1. 是否已运行 `main_bandit.py`
2. output目录下是否有相应的Excel文件
3. Python环境是否已安装 pandas 和 openpyxl
