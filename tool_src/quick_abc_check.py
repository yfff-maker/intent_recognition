"""快速检查ABC一致性"""
import pandas as pd

# 读取结果
df = pd.read_excel("output/intent_inference_results_bandit.xlsx")

# 按照异常点分组
grouped = df.groupby(['Participant', 'AnchorTimestamp'])

all_same = 0
two_same = 0
all_different = 0
total = 0

for (p, t), group in grouped:
    if len(group) != 3:
        continue
    
    intents = group.set_index('Strategy')['Intent'].to_dict()
    if len(intents) != 3:
        continue
    
    total += 1
    a, b, c = intents.get('A'), intents.get('B'), intents.get('C')
    
    if a == b == c:
        all_same += 1
    elif a == b or b == c or a == c:
        two_same += 1
    else:
        all_different += 1

print(f"总异常点数: {total}")
print(f"三个策略完全一致: {all_same} ({all_same/total*100:.1f}%)")
print(f"两个策略一致: {two_same} ({two_same/total*100:.1f}%)")
print(f"三个策略完全不同: {all_different} ({all_different/total*100:.1f}%)")

if all_same / total > 0.7:
    print(f"\n🚨 警告：{all_same/total*100:.1f}%的案例ABC完全一致！大窗口可能没什么用！")
elif all_same / total < 0.3:
    print(f"\n✅ 良好：只有{all_same/total*100:.1f}%完全一致，说明窗口大小有影响")
