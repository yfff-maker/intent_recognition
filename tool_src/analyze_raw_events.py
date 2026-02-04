"""
分析原始behavior_sequences.json中的事件数量和结构

输出：每个参与者的原始事件数、事件类型统计等
"""

import os
import json
import pandas as pd
from collections import Counter

DATASET_ROOT = "../anonymous_data"


def analyze_participant(p_id):
    """分析单个参与者的原始事件"""
    json_path = os.path.join(DATASET_ROOT, p_id, "behavior_sequences.json")
    
    if not os.path.exists(json_path):
        return None
    
    with open(json_path, 'r', encoding='utf-8') as f:
        events = json.load(f)
    
    if not events:
        return None
    
    # 统计信息
    total_events = len(events)
    
    # 时间跨度
    time_start = events[0].get('startTimeTick', 0)
    time_end = events[-1].get('startTimeTick', 0)
    duration_seconds = (time_end - time_start) / 1000.0  # 转换为秒
    duration_minutes = duration_seconds / 60.0
    
    # 页面统计
    pages = [e.get('page', 'None') for e in events if e.get('page') != 'None']
    unique_pages = len(set(pages))
    page_counts = Counter(pages)
    top_pages = page_counts.most_common(3)
    
    # 控件统计
    widgets = [e.get('widget', 'None') for e in events if e.get('widget') != 'None']
    unique_widgets = len(set(widgets))
    
    # 操作统计
    operations = [e.get('operationId', 'None') for e in events]
    unique_operations = len(set(operations))
    
    return {
        'Participant': p_id,
        'TotalEvents': total_events,
        'DurationSeconds': round(duration_seconds, 2),
        'DurationMinutes': round(duration_minutes, 2),
        'UniquePages': unique_pages,
        'UniqueWidgets': unique_widgets,
        'UniqueOperations': unique_operations,
        'TopPage1': top_pages[0][0] if len(top_pages) > 0 else 'None',
        'TopPage1Count': top_pages[0][1] if len(top_pages) > 0 else 0,
        'TopPage2': top_pages[1][0] if len(top_pages) > 1 else 'None',
        'TopPage2Count': top_pages[1][1] if len(top_pages) > 1 else 0,
        'AvgEventsPerMinute': round(total_events / duration_minutes, 2) if duration_minutes > 0 else 0,
    }


def main():
    print("="*80)
    print("📊 原始事件数量分析")
    print("="*80)
    
    # 获取所有参与者
    participants = [
        d for d in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, d)) and d.startswith("P")
    ]
    participants = sorted(participants, key=lambda x: int(x[1:]))  # 按数字排序
    
    print(f"\n发现 {len(participants)} 个参与者: {', '.join(participants)}\n")
    
    # 分析每个参与者
    results = []
    for p_id in participants:
        print(f"分析 {p_id}...", end=' ')
        result = analyze_participant(p_id)
        if result:
            results.append(result)
            print(f"✓ {result['TotalEvents']} 个事件, {result['DurationMinutes']:.1f} 分钟")
        else:
            print("✗ 无数据")
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 显示详细统计
    print("\n" + "="*80)
    print("📈 统计摘要")
    print("="*80)
    
    print("\n【总体统计】")
    print(f"  总参与者数: {len(df)}")
    print(f"  总事件数: {df['TotalEvents'].sum():,}")
    print(f"  平均每人事件数: {df['TotalEvents'].mean():.0f}")
    print(f"  最多事件数: {df['TotalEvents'].max():,} ({df.loc[df['TotalEvents'].idxmax(), 'Participant']})")
    print(f"  最少事件数: {df['TotalEvents'].min():,} ({df.loc[df['TotalEvents'].idxmin(), 'Participant']})")
    
    print("\n【时间统计】")
    print(f"  平均任务时长: {df['DurationMinutes'].mean():.1f} 分钟")
    print(f"  最长任务: {df['DurationMinutes'].max():.1f} 分钟 ({df.loc[df['DurationMinutes'].idxmax(), 'Participant']})")
    print(f"  最短任务: {df['DurationMinutes'].min():.1f} 分钟 ({df.loc[df['DurationMinutes'].idxmin(), 'Participant']})")
    
    print("\n【交互统计】")
    print(f"  平均独特页面数: {df['UniquePages'].mean():.1f}")
    print(f"  平均独特控件数: {df['UniqueWidgets'].mean():.1f}")
    print(f"  平均事件速率: {df['AvgEventsPerMinute'].mean():.1f} 事件/分钟")
    
    # 显示详细表格
    print("\n" + "="*80)
    print("📋 详细信息（按事件数排序）")
    print("="*80)
    
    df_sorted = df.sort_values('TotalEvents', ascending=False)
    display_cols = ['Participant', 'TotalEvents', 'DurationMinutes', 'UniquePages', 
                    'UniqueWidgets', 'AvgEventsPerMinute', 'TopPage1']
    print(df_sorted[display_cols].to_string(index=False))
    
    # 保存到Excel
    output_path = "./output/raw_events_analysis.xlsx"
    os.makedirs("./output", exist_ok=True)
    df_sorted.to_excel(output_path, index=False)
    print(f"\n✓ 详细统计已保存到: {output_path}")
    
    # 事件划分说明
    print("\n" + "="*80)
    print("📖 事件结构说明")
    print("="*80)
    print("""
每个事件 (Event) 包含以下字段：

1. operationId: 操作唯一标识符
   - 格式: "{状态}-{时间戳}-{序号}"
   - 示例: "NotLogin-1728361522885-00000001"

2. page: 当前页面名称
   - 示例: "Home", "Log in", "Course List"
   - 用于追踪用户在哪个页面操作

3. module: 功能模块名称
   - 示例: "Login", "NLogin", "Course"
   - 表示页面内的功能区域

4. widget: 具体控件名称
   - 示例: "L-Username", "L-Password", "N-Login"
   - 表示用户交互的具体UI元素
   - 也可能是坐标 "Blank(974, 370)" 表示空白区域点击

5. startTimeTick: 事件开始时间戳（毫秒）
   - 相对于任务开始的时间偏移

6. duration: 事件持续时间（毫秒）
   - 例如停留时间、输入时间等

【事件划分逻辑】
- 每个用户操作（点击、输入、导航等）= 1个事件
- 按时间戳顺序排列
- 从任务开始(t=0)到任务结束
- 包含显式操作和隐式状态（如页面停留）
    """)
    
    print("\n" + "="*80)
    print("🔄 压缩流程说明")
    print("="*80)
    print(f"""
原始事件 → 关键事件选择 → STM/LTM

以 {df.loc[df['TotalEvents'].idxmax(), 'Participant']} 为例（事件最多）：
1. 原始事件: {df['TotalEvents'].max():,} 个
2. 关键事件选择 (KEY_EVENT_TARGET_K=600): 600 个
   压缩率: {(1 - 600/df['TotalEvents'].max())*100:.1f}%
   
3. LTM分块 (MEMORY_CHUNK_SIZE=30): 600 ÷ 30 = 20 个chunk
   每个chunk代表约 {df.loc[df['TotalEvents'].idxmax(), 'DurationMinutes']/20:.1f} 分钟的活动
   
4. LTM检索 (MEMORY_RETRIEVE_TOP_K=5): 返回5个最相关chunk
   最终LTM token: ~20行文本
   
5. STM窗口 (策略C: k_left=200, k_right=50): 最多251个事件
   压缩后: ~60行文本（合并连续重复）
   
总计输入LLM: STM(60行) + LTM(20行) = ~80行 ≈ 6k tokens
原始: {df['TotalEvents'].max():,} 事件 ≈ {df['TotalEvents'].max()*80//1000}k tokens
最终压缩率: {(1 - 6/(df['TotalEvents'].max()*80//1000))*100:.1f}%
    """)


if __name__ == "__main__":
    main()
