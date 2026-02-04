"""快速查看LTM和STM内容的简化版本"""
import os
import pandas as pd

OUTPUT_DIR = "../output"

print("=" * 80)
print("📊 LTM & STM 快速查看".center(80))
print("=" * 80)

# 1. 查看LTM统计
stats_file = os.path.join(OUTPUT_DIR, "memory_bank_statistics.xlsx")
if os.path.exists(stats_file):
    print("\n✅ 找到LTM统计文件")
    df_stats = pd.read_excel(stats_file)
    
    print(f"\n📦 LTM记忆库概览:")
    print(f"  - 总chunk数: {len(df_stats)}")
    print(f"  - 参与者数: {df_stats['Participant'].nunique()}")
    
    print(f"\n各参与者的chunk数量:")
    for pid in sorted(df_stats['Participant'].unique()):
        count = len(df_stats[df_stats['Participant'] == pid])
        print(f"  {pid}: {count} 个chunk")
    
    print(f"\n⭐ 最有价值的5个chunk:")
    top5 = df_stats.nlargest(5, 'EstimatedValue')[
        ['Participant', 'ChunkID', 'AccessCount', 'UsefulCount', 'EstimatedValue']
    ]
    print(top5.to_string(index=False))
    
    # 查看第一个chunk的详细内容
    print(f"\n📝 示例chunk详情 (第一个chunk):")
    first_chunk = df_stats.iloc[0]
    print(f"  ChunkID: {first_chunk['ChunkID']}")
    print(f"  参与者: {first_chunk['Participant']}")
    print(f"  事件范围: {first_chunk['EventIdxRange']}")
    print(f"  访问次数: {first_chunk['AccessCount']}")
    print(f"  有用次数: {first_chunk['UsefulCount']}")
    print(f"  估计价值: {first_chunk['EstimatedValue']:.4f}")
    if 'Summary' in first_chunk.index and pd.notna(first_chunk['Summary']):
        print(f"\n  摘要内容:")
        for line in str(first_chunk['Summary']).split('\n')[:5]:
            print(f"    {line}")
else:
    print("\n❌ 未找到LTM统计文件:", stats_file)
    print("   请先运行 main_bandit.py")

# 2. 查看推理结果中的STM+LTM
results_file = os.path.join(OUTPUT_DIR, "intent_inference_results_bandit.xlsx")
if os.path.exists(results_file):
    print("\n" + "=" * 80)
    print("\n✅ 找到推理结果文件")
    df_results = pd.read_excel(results_file)
    
    print(f"\n🔍 推理记录概览:")
    print(f"  - 总推理次数: {len(df_results)}")
    print(f"  - 参与者数: {df_results['Participant'].nunique()}")
    print(f"  - 策略分布: A={len(df_results[df_results['Strategy']=='A'])}, "
          f"B={len(df_results[df_results['Strategy']=='B'])}, "
          f"C={len(df_results[df_results['Strategy']=='C'])}")
    
    # 显示第一条推理记录的STM+LTM
    print(f"\n📄 示例推理场景 (第一条记录):")
    first_row = df_results.iloc[0]
    print(f"  参与者: {first_row['Participant']}")
    print(f"  异常点时间: {first_row['AnchorTimestamp']}")
    print(f"  异常类型: {first_row['AnomalyType']}")
    print(f"  策略: {first_row['Strategy']}")
    print(f"  推理结果: {first_row['Intent']} (置信度: {first_row['Confidence']})")
    
    if 'Prompt' in first_row.index and pd.notna(first_row['Prompt']):
        prompt = str(first_row['Prompt'])
        
        # 提取STM部分
        if "### Short-Term Memory (STM)" in prompt:
            print(f"\n  🔸 STM内容 (前10行):")
            stm_start = prompt.find("### Short-Term Memory (STM)")
            ltm_start = prompt.find("### Long-Term Memory (LTM)")
            if ltm_start > stm_start:
                stm_content = prompt[stm_start:ltm_start]
                stm_lines = stm_content.split('\n')[1:11]  # 跳过标题，取10行
                for line in stm_lines:
                    if line.strip():
                        print(f"    {line[:80]}")  # 限制每行80字符
        
        # 提取LTM部分
        if "### Long-Term Memory (LTM)" in prompt:
            print(f"\n  🔹 LTM内容:")
            ltm_start = prompt.find("### Long-Term Memory (LTM)")
            ltm_end = prompt.find("### Output Schema", ltm_start)
            if ltm_end == -1:
                ltm_end = len(prompt)
            ltm_content = prompt[ltm_start:ltm_end]
            ltm_lines = ltm_content.split('\n')[1:20]  # 取前20行
            for line in ltm_lines:
                if line.strip():
                    print(f"    {line[:100]}")  # 限制每行100字符
    
    print(f"\n💡 提示: 要查看完整Prompt，请打开Excel文件查看 'Prompt' 列")

else:
    print("\n❌ 未找到推理结果文件:", results_file)
    print("   请先运行 main_bandit.py")

print("\n" + "=" * 80)
print("\n📖 详细使用说明请查看: VIEW_MEMORY_README.md")
print("🔧 交互式查看请运行: python view_memory_contents.py --interactive")
print("=" * 80)
