"""
分析ABC策略的意图一致性

用途：检查ABC三种策略识别出的意图是否大多数都一样
如果大多数一样，说明大窗口/长期记忆可能没什么用
"""

import pandas as pd
import os


def analyze_abc_consistency(excel_path: str):
    """
    分析ABC策略的意图一致性
    
    输出：
    1. 三个策略完全一致的比例
    2. 两个策略一致的比例
    3. 三个策略完全不同的比例
    4. 具体案例分析
    """
    print("=" * 80)
    print("🔍 ABC策略意图一致性分析")
    print("=" * 80)
    
    # 读取数据
    df = pd.read_excel(excel_path)
    
    print(f"\n📊 数据概览:")
    print(f"  总记录数: {len(df)}")
    print(f"  参与者数: {df['Participant'].nunique()}")
    print(f"  异常点数: {len(df) // 3}")  # 每个异常点3条记录（A/B/C）
    
    # 按照(Participant, AnchorTimestamp)分组，获取ABC三种策略的意图
    grouped = df.groupby(['Participant', 'AnchorTimestamp'])
    
    consistency_results = []
    
    for (participant, timestamp), group in grouped:
        if len(group) != 3:
            # 如果不是恰好3条记录（A/B/C），跳过
            continue
        
        # 提取ABC三种策略的意图
        strategies = {}
        for _, row in group.iterrows():
            strategy = row['Strategy']
            intent = row['Intent']
            confidence = row.get('Confidence', 0)
            strategies[strategy] = {
                'intent': intent,
                'confidence': confidence
            }
        
        if len(strategies) != 3:
            continue
        
        # 检查一致性
        intent_a = strategies.get('A', {}).get('intent')
        intent_b = strategies.get('B', {}).get('intent')
        intent_c = strategies.get('C', {}).get('intent')
        
        # 统计一致性
        if intent_a == intent_b == intent_c:
            consistency = "all_same"
        elif intent_a == intent_b or intent_b == intent_c or intent_a == intent_c:
            consistency = "two_same"
        else:
            consistency = "all_different"
        
        consistency_results.append({
            'Participant': participant,
            'AnchorTimestamp': timestamp,
            'Intent_A': intent_a,
            'Intent_B': intent_b,
            'Intent_C': intent_c,
            'Confidence_A': strategies.get('A', {}).get('confidence'),
            'Confidence_B': strategies.get('B', {}).get('confidence'),
            'Confidence_C': strategies.get('C', {}).get('confidence'),
            'Consistency': consistency
        })
    
    # 转换为DataFrame
    df_consistency = pd.DataFrame(consistency_results)
    
    # 统计结果
    print(f"\n" + "=" * 80)
    print(f"📈 一致性统计:")
    print("=" * 80)
    
    total = len(df_consistency)
    all_same = len(df_consistency[df_consistency['Consistency'] == 'all_same'])
    two_same = len(df_consistency[df_consistency['Consistency'] == 'two_same'])
    all_different = len(df_consistency[df_consistency['Consistency'] == 'all_different'])
    
    print(f"\n总异常点数: {total}")
    print(f"\n✅ 三个策略完全一致: {all_same} ({all_same/total*100:.1f}%)")
    print(f"⚠️  两个策略一致:     {two_same} ({two_same/total*100:.1f}%)")
    print(f"❌ 三个策略完全不同: {all_different} ({all_different/total*100:.1f}%)")
    
    # 如果大多数一致，说明问题
    if all_same / total > 0.7:
        print(f"\n🚨 警告：{all_same/total*100:.1f}% 的异常点ABC策略识别出的意图完全一致！")
        print(f"   这可能说明：")
        print(f"   1. 大窗口/长期记忆对推理没什么影响")
        print(f"   2. 意图太简单，小窗口就足够了")
        print(f"   3. LTM检索没有提供新信息")
    elif all_same / total < 0.3:
        print(f"\n✅ 良好：只有{all_same/total*100:.1f}% 的异常点ABC完全一致")
        print(f"   说明窗口大小/长期记忆确实影响了推理结果")
    
    # 详细分析不同意图的分布
    print(f"\n" + "=" * 80)
    print(f"📊 意图分布对比:")
    print("=" * 80)
    
    print(f"\n策略A（小窗口）的意图分布:")
    intent_a_counts = df_consistency['Intent_A'].value_counts()
    for intent, count in intent_a_counts.items():
        print(f"  {intent}: {count} ({count/total*100:.1f}%)")
    
    print(f"\n策略B（中窗口）的意图分布:")
    intent_b_counts = df_consistency['Intent_B'].value_counts()
    for intent, count in intent_b_counts.items():
        print(f"  {intent}: {count} ({count/total*100:.1f}%)")
    
    print(f"\n策略C（大窗口）的意图分布:")
    intent_c_counts = df_consistency['Intent_C'].value_counts()
    for intent, count in intent_c_counts.items():
        print(f"  {intent}: {count} ({count/total*100:.1f}%)")
    
    # 展示一些不一致的案例
    print(f"\n" + "=" * 80)
    print(f"📋 案例分析 - 策略不一致的情况:")
    print("=" * 80)
    
    # 找出完全不同的案例
    different_cases = df_consistency[df_consistency['Consistency'] == 'all_different']
    if len(different_cases) > 0:
        print(f"\n🔍 三个策略完全不同的案例 (展示前5个):")
        for idx, row in different_cases.head(5).iterrows():
            print(f"\n  案例 {idx+1}:")
            print(f"    参与者: {row['Participant']}, 时间: {row['AnchorTimestamp']}")
            print(f"    策略A (小窗口): {row['Intent_A']} (置信度: {row['Confidence_A']:.2f})")
            print(f"    策略B (中窗口): {row['Intent_B']} (置信度: {row['Confidence_B']:.2f})")
            print(f"    策略C (大窗口): {row['Intent_C']} (置信度: {row['Confidence_C']:.2f})")
    
    # 找出两个一致的案例
    two_same_cases = df_consistency[df_consistency['Consistency'] == 'two_same']
    if len(two_same_cases) > 0:
        print(f"\n🔍 两个策略一致的案例 (展示前3个):")
        for idx, row in two_same_cases.head(3).iterrows():
            print(f"\n  案例 {idx+1}:")
            print(f"    参与者: {row['Participant']}, 时间: {row['AnchorTimestamp']}")
            print(f"    策略A: {row['Intent_A']} (置信度: {row['Confidence_A']:.2f})")
            print(f"    策略B: {row['Intent_B']} (置信度: {row['Confidence_B']:.2f})")
            print(f"    策略C: {row['Intent_C']} (置信度: {row['Confidence_C']:.2f})")
    
    # 分析置信度差异
    print(f"\n" + "=" * 80)
    print(f"📊 置信度分析:")
    print("=" * 80)
    
    print(f"\n平均置信度:")
    print(f"  策略A: {df_consistency['Confidence_A'].mean():.3f}")
    print(f"  策略B: {df_consistency['Confidence_B'].mean():.3f}")
    print(f"  策略C: {df_consistency['Confidence_C'].mean():.3f}")
    
    # 对于完全一致的案例，检查置信度是否有差异
    all_same_df = df_consistency[df_consistency['Consistency'] == 'all_same']
    if len(all_same_df) > 0:
        print(f"\n对于三个策略意图完全一致的{len(all_same_df)}个案例:")
        print(f"  策略A平均置信度: {all_same_df['Confidence_A'].mean():.3f}")
        print(f"  策略B平均置信度: {all_same_df['Confidence_B'].mean():.3f}")
        print(f"  策略C平均置信度: {all_same_df['Confidence_C'].mean():.3f}")
        
        if all_same_df['Confidence_C'].mean() > all_same_df['Confidence_A'].mean() + 0.05:
            print(f"\n  ✅ 策略C的置信度明显更高，说明大窗口虽然得出相同意图，但更自信")
        elif all_same_df['Confidence_C'].mean() < all_same_df['Confidence_A'].mean() - 0.05:
            print(f"\n  ⚠️  策略C的置信度反而更低，可能长上下文导致混淆")
        else:
            print(f"\n  ➡️  三种策略的置信度相近")
    
    # 保存详细结果
    output_path = os.path.join(os.path.dirname(excel_path), "abc_consistency_analysis.xlsx")
    df_consistency.to_excel(output_path, index=False)
    print(f"\n✅ 详细结果已保存到: {output_path}")
    
    print(f"\n" + "=" * 80)
    
    # 返回统计结果
    return {
        'total': total,
        'all_same': all_same,
        'two_same': two_same,
        'all_different': all_different,
        'all_same_ratio': all_same / total if total > 0 else 0
    }


if __name__ == "__main__":
    import sys
    
    # 默认分析最新的结果文件
    default_file = "output/intent_inference_results.xlsx"
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = default_file
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        print(f"\n请先运行 main.py 或 main_bandit.py 生成结果文件")
        print(f"\n可用的结果文件:")
        output_dir = "output"
        if os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                if f.endswith(".xlsx") and "intent" in f:
                    print(f"  - {os.path.join(output_dir, f)}")
    else:
        results = analyze_abc_consistency(file_path)
        
        # 总结
        print(f"\n" + "=" * 80)
        print(f"📌 总结:")
        print("=" * 80)
        
        if results['all_same_ratio'] > 0.7:
            print(f"\n🚨 问题严重：{results['all_same_ratio']*100:.1f}% 的案例ABC完全一致")
            print(f"\n建议:")
            print(f"  1. 检查LTM是否真的被使用（查看prompt）")
            print(f"  2. 检查ABC窗口大小是否真的不同")
            print(f"  3. 可能需要更复杂的任务/意图标签")
        elif results['all_same_ratio'] > 0.5:
            print(f"\n⚠️  一致性较高：{results['all_same_ratio']*100:.1f}% 的案例ABC完全一致")
            print(f"\n可能原因:")
            print(f"  1. 大部分意图确实只需要局部信息就能判断")
            print(f"  2. LTM提供的信息与STM重复")
            print(f"  3. 意图标签过于粗粒度")
        else:
            print(f"\n✅ 一致性合理：{results['all_same_ratio']*100:.1f}% 的案例ABC完全一致")
            print(f"\n说明:")
            print(f"  - 窗口大小/长期记忆确实影响推理结果")
            print(f"  - 方法设计有效，ABC策略有区分度")
