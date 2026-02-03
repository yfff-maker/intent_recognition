"""
生成人工标注模板：从 LLM 输出结果里抽样，生成 Excel 供标注
"""
import pandas as pd
import os
import random

# 输入：LLM 输出的 CSV
csv_path = "./output/intent_inference_results.csv"
output_annotation_path = "./output/annotation_template.xlsx"

if not os.path.exists(csv_path):
    print(f"错误：找不到 {csv_path}，请先运行 main.py 生成结果。")
    exit(1)

df = pd.read_csv(csv_path, encoding='utf-8-sig')

# 只保留 Strategy=A 的（因为同一 anomaly 的 A/B/C 会一起标注）
df_a = df[df['Strategy'] == 'A'].copy()

# 抽样策略：每个参与者抽 2-3 个，优先选不同异常类型
random.seed(42)  # 可复现
samples = []

for p_id in df_a['Participant'].unique():
    p_data = df_a[df_a['Participant'] == p_id]
    
    # 按异常类型分组
    rep = p_data[p_data['AnomalyType'].str.contains('Repetitive', na=False)]
    long_dur = p_data[p_data['AnomalyType'].str.contains('Long', na=False)]
    
    # 每类抽 1 个（如果有的话）
    if len(rep) > 0:
        samples.append(rep.sample(1, random_state=42).iloc[0])
    if len(long_dur) > 0:
        samples.append(long_dur.sample(1, random_state=42).iloc[0])

# 转成 DataFrame
df_samples = pd.DataFrame(samples)

# 构造标注模板（加上 A/B/C 的 LLM 输出）
annotation_rows = []

for idx, row in df_samples.iterrows():
    p_id = row['Participant']
    ts = row['AnchorTimestamp']
    
    # 获取该 anomaly 的 A/B/C 输出
    abc_data = df[(df['Participant'] == p_id) & (df['AnchorTimestamp'] == ts)]
    
    intent_a = abc_data[abc_data['Strategy'] == 'A']['Intent'].values[0] if len(abc_data[abc_data['Strategy'] == 'A']) > 0 else ''
    intent_b = abc_data[abc_data['Strategy'] == 'B']['Intent'].values[0] if len(abc_data[abc_data['Strategy'] == 'B']) > 0 else ''
    intent_c = abc_data[abc_data['Strategy'] == 'C']['Intent'].values[0] if len(abc_data[abc_data['Strategy'] == 'C']) > 0 else ''
    
    conf_a = abc_data[abc_data['Strategy'] == 'A']['Confidence'].values[0] if len(abc_data[abc_data['Strategy'] == 'A']) > 0 else ''
    conf_b = abc_data[abc_data['Strategy'] == 'B']['Confidence'].values[0] if len(abc_data[abc_data['Strategy'] == 'B']) > 0 else ''
    conf_c = abc_data[abc_data['Strategy'] == 'C']['Confidence'].values[0] if len(abc_data[abc_data['Strategy'] == 'C']) > 0 else ''
    
    annotation_rows.append({
        'SampleID': len(annotation_rows) + 1,
        'Participant': p_id,
        'AnchorTimestamp': ts,
        'AnomalyType': row['AnomalyType'],
        'LLM_Intent_A': intent_a,
        'LLM_Confidence_A': conf_a,
        'LLM_Intent_B': intent_b,
        'LLM_Confidence_B': conf_b,
        'LLM_Intent_C': intent_c,
        'LLM_Confidence_C': conf_c,
        # 人工标注列（留空）
        'GroundTruth_Intent': '',
        'Quality_A (1-5)': '',
        'Quality_B (1-5)': '',
        'Quality_C (1-5)': '',
        'Notes': '',
    })

df_annotation = pd.DataFrame(annotation_rows)

# 保存为 Excel
df_annotation.to_excel(output_annotation_path, index=False, sheet_name='Annotation')

print(f"✓ 已生成标注模板：{output_annotation_path}")
print(f"  - 共抽样 {len(df_annotation)} 个 anomaly")
print(f"  - 请在 Excel 里填写以下列：")
print(f"    · GroundTruth_Intent: 真实意图（从 INTENT_LABELS 里选）")
print(f"    · Quality_A/B/C: 每个策略的输出质量（1-5 分）")
print(f"    · Notes: 可选备注")
print(f"\n📖 标注指南见: ./output/annotation_guide.txt")

# 同时生成标注指南
guide_path = "./output/annotation_guide.txt"
with open(guide_path, 'w', encoding='utf-8') as f:
    f.write("""
==================== 人工标注指南 ====================

一、如何判断"真实意图"（GroundTruth_Intent）

1. 查看原始数据（可选但推荐）：
   - 打开 anonymous_data/{Participant}/behavior_sequences.json
   - 找到 AnchorTimestamp 附近的事件（±5 秒内）
   - 看用户在做什么：page, widget, operationId

2. 根据上下文判断意图，从以下标签选一个：
   ["Login", "Navigate", "Search/Explore", "FillForm", 
    "Upload/Download", "Submit/Confirm", "ErrorRecovery", 
    "Waiting/NoFeedback", "Hesitation/Uncertainty", "Other"]

3. 判断规则：
   - Repetitive Interaction（重复点击）→ 通常是 Waiting/NoFeedback 或 ErrorRecovery
   - Long Duration（长时停留）→ 通常是 Hesitation/Uncertainty 或 Search/Explore
   - 如果在登录/注册页面 → Login
   - 如果在表单页面填写 → FillForm
   - 如果点击上传/下载按钮 → Upload/Download
   - 如果点击提交/确认 → Submit/Confirm
   - 如果在导航/切换页面 → Navigate
   - 不确定时选 Other

二、如何评分 Quality_A/B/C（1-5 分）

对每个策略（A/B/C）的 LLM 输出，从以下维度综合打分：

评分标准：
  5分 - 完美：意图准确 + 证据充分 + 有洞察
  4分 - 良好：意图合理 + 证据基本充分
  3分 - 中等：意图大致对但不够精准，或证据薄弱
  2分 - 较差：意图偏离或证据不足/矛盾
  1分 - 很差：意图完全错误或明显瞎编

三、标注流程（推荐顺序）

1. 先标 GroundTruth_Intent（不看 LLM 输出，独立判断）
2. 再对比 A/B/C 的输出，分别打分
3. 在 Notes 里记录任何值得注意的点（如 C 明显比 A 好的原因）

四、如何查看原始行为数据（可选）

如果你想更准确判断，可以打开对应的 behavior_sequences.json：
  文件位置：anonymous_data/{Participant}/behavior_sequences.json
  搜索："startTimeTick": {AnchorTimestamp} （或附近值±1000）
  看前后 5-10 个事件，理解用户在做什么

五、标注示例

假设某行：
  - AnomalyType: "Repetitive Interaction"
  - LLM_Intent_A: "Waiting/NoFeedback"
  - LLM_Intent_B: "Waiting/NoFeedback"
  - LLM_Intent_C: "ErrorRecovery"

你的标注可能是：
  - GroundTruth_Intent: "Waiting/NoFeedback"  （你判断是在等反馈）
  - Quality_A: 4  （意图对，但可能证据不够充分）
  - Quality_B: 5  （意图对，证据更充分）
  - Quality_C: 3  （意图偏了，虽然证据多但推断错了）
  - Notes: "C 可能被长序列中的错误恢复操作误导"

==================== 开始标注吧！ ====================
""")
print(f"✓ 标注指南已生成：{guide_path}")
