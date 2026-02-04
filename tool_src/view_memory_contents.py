"""
查看LTM和STM的详细内容

使用方法：
python view_memory_contents.py

可选参数：
--participant P1  # 查看特定参与者
--anomaly 0       # 查看特定异常点（按索引）
--strategy A      # 查看特定策略（A/B/C）
"""

import os
import sys
import pandas as pd
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = "../output"


def print_separator(title="", width=80):
    """打印分隔线"""
    if title:
        print(f"\n{'=' * width}")
        print(f"{title:^{width}}")
        print(f"{'=' * width}\n")
    else:
        print(f"{'=' * width}\n")


def view_ltm_statistics():
    """查看LTM记忆库统计信息"""
    stats_file = os.path.join(OUTPUT_DIR, "memory_bank_statistics.xlsx")
    
    if not os.path.exists(stats_file):
        print(f"❌ 未找到统计文件: {stats_file}")
        print("   请先运行 main_bandit.py 生成结果")
        return None
    
    df_stats = pd.read_excel(stats_file)
    
    print_separator("📊 LTM记忆库统计概览")
    
    print(f"总chunk数: {len(df_stats)}")
    print(f"参与者数: {df_stats['Participant'].nunique()}")
    
    # 按参与者分组统计
    print("\n各参与者的chunk数量:")
    participant_counts = df_stats['Participant'].value_counts().sort_index()
    for pid, count in participant_counts.items():
        print(f"  {pid}: {count} 个chunk")
    
    # 价值统计
    print(f"\nchunk价值统计:")
    print(f"  平均EstimatedValue: {df_stats['EstimatedValue'].mean():.4f}")
    print(f"  最高EstimatedValue: {df_stats['EstimatedValue'].max():.4f}")
    print(f"  最低EstimatedValue: {df_stats['EstimatedValue'].min():.4f}")
    
    # 使用率统计
    if 'UsageRate' in df_stats.columns:
        print(f"\nchunk使用率统计:")
        print(f"  平均UsageRate: {df_stats['UsageRate'].mean():.2%}")
        print(f"  被使用过的chunk: {(df_stats['UsefulCount'] > 0).sum()} / {len(df_stats)}")
    
    # 最有价值的chunk
    print(f"\n⭐ 最有价值的5个chunk:")
    top_chunks = df_stats.nlargest(5, 'EstimatedValue')[
        ['Participant', 'ChunkID', 'AccessCount', 'UsefulCount', 'EstimatedValue']
    ]
    print(top_chunks.to_string(index=False))
    
    # 检查提升的chunk
    promoted_chunks = df_stats[df_stats['ChunkID'].str.contains('promoted', na=False)]
    if len(promoted_chunks) > 0:
        print(f"\n🚀 从STM提升的chunk:")
        print(f"  数量: {len(promoted_chunks)}")
        print(f"  平均价值: {promoted_chunks['EstimatedValue'].mean():.4f}")
        print(promoted_chunks[['Participant', 'ChunkID', 'EstimatedValue']].to_string(index=False))
    
    return df_stats


def view_specific_ltm_chunk(df_stats, chunk_id=None, participant=None, chunk_index=0):
    """查看特定LTM chunk的详细内容"""
    if chunk_id:
        chunk = df_stats[df_stats['ChunkID'] == chunk_id]
    elif participant:
        participant_chunks = df_stats[df_stats['Participant'] == participant]
        if chunk_index >= len(participant_chunks):
            print(f"❌ {participant} 只有 {len(participant_chunks)} 个chunk，索引 {chunk_index} 超出范围")
            return
        chunk = participant_chunks.iloc[[chunk_index]]
    else:
        chunk = df_stats.iloc[[chunk_index]]
    
    if len(chunk) == 0:
        print(f"❌ 未找到chunk: {chunk_id}")
        return
    
    chunk = chunk.iloc[0]
    
    print_separator(f"📦 LTM Chunk详情: {chunk['ChunkID']}")
    
    print(f"参与者: {chunk['Participant']}")
    print(f"ChunkID: {chunk['ChunkID']}")
    print(f"事件索引范围: {chunk['EventIdxRange']}")
    print(f"时间范围: {chunk['TimeStart']} -> {chunk['TimeEnd']}")
    
    print(f"\n📊 Bandit统计:")
    print(f"  访问次数 (AccessCount): {chunk['AccessCount']}")
    print(f"  有用次数 (UsefulCount): {chunk['UsefulCount']}")
    if 'UsageRate' in chunk.index and pd.notna(chunk['UsageRate']):
        print(f"  使用率 (UsageRate): {chunk['UsageRate']:.2%}")
    print(f"  估计价值 (EstimatedValue): {chunk['EstimatedValue']:.4f}")
    print(f"  最后访问时间: {chunk['LastAccessTime']}")
    
    print(f"\n📝 内容摘要:")
    if 'Summary' in chunk.index:
        print(chunk['Summary'])
    
    print(f"\n🔑 Signature特征:")
    print(f"  Pages: {chunk.get('SignaturePages', 'N/A')}")
    print(f"  Widgets: {chunk.get('SignatureWidgets', 'N/A')}")
    print(f"  Ops: {chunk.get('SignatureOps', 'N/A')}")


def view_stm_and_ltm_in_prompt(participant=None, anomaly_idx=None, strategy=None):
    """查看特定推理中使用的STM和LTM内容"""
    results_file = os.path.join(OUTPUT_DIR, "intent_inference_results_bandit.xlsx")
    
    if not os.path.exists(results_file):
        print(f"❌ 未找到结果文件: {results_file}")
        print("   请先运行 main_bandit.py 生成结果")
        return
    
    df_results = pd.read_excel(results_file)
    
    # 筛选
    filtered = df_results
    if participant:
        filtered = filtered[filtered['Participant'] == participant]
    if strategy:
        filtered = filtered[filtered['Strategy'] == strategy]
    
    if anomaly_idx is not None and anomaly_idx < len(filtered):
        filtered = filtered.iloc[[anomaly_idx]]
    elif len(filtered) > 0:
        filtered = filtered.iloc[[0]]  # 默认显示第一个
    
    if len(filtered) == 0:
        print("❌ 没有找到匹配的推理记录")
        return
    
    row = filtered.iloc[0]
    
    print_separator(f"🔍 推理场景详情")
    
    print(f"参与者: {row['Participant']}")
    print(f"异常点时间戳: {row['AnchorTimestamp']}")
    print(f"异常类型: {row['AnomalyType']}")
    print(f"策略: {row['Strategy']}")
    
    print(f"\n💡 LLM推理结果:")
    print(f"  意图: {row['Intent']}")
    print(f"  置信度: {row['Confidence']}")
    print(f"  证据: {row['Evidence']}")
    
    print_separator("📄 完整Prompt（包含STM + LTM）")
    
    if 'Prompt' in row.index and pd.notna(row['Prompt']):
        prompt = row['Prompt']
        
        # 尝试提取STM和LTM部分
        if "### Short-Term Memory (STM)" in prompt:
            stm_start = prompt.find("### Short-Term Memory (STM)")
            ltm_start = prompt.find("### Long-Term Memory (LTM)")
            
            if ltm_start > stm_start:
                print("\n🔸 STM部分 (Short-Term Memory):")
                print("-" * 80)
                stm_content = prompt[stm_start:ltm_start].strip()
                # 只显示前30行，避免太长
                stm_lines = stm_content.split('\n')
                for line in stm_lines[:30]:
                    print(line)
                if len(stm_lines) > 30:
                    print(f"... (省略 {len(stm_lines) - 30} 行)")
                
                print("\n\n🔹 LTM部分 (Long-Term Memory):")
                print("-" * 80)
                ltm_end = prompt.find("### Output Schema", ltm_start)
                if ltm_end == -1:
                    ltm_end = len(prompt)
                ltm_content = prompt[ltm_start:ltm_end].strip()
                print(ltm_content)
            else:
                print(prompt[:2000])  # 显示前2000字符
                if len(prompt) > 2000:
                    print(f"\n... (总长度: {len(prompt)} 字符)")
        else:
            print(prompt[:2000])  # 显示前2000字符
            if len(prompt) > 2000:
                print(f"\n... (总长度: {len(prompt)} 字符)")
    else:
        print("❌ Prompt内容不可用")


def interactive_menu():
    """交互式菜单"""
    print_separator("🧠 LTM & STM 内容查看器")
    
    while True:
        print("\n请选择操作:")
        print("  1. 查看LTM记忆库统计概览")
        print("  2. 查看特定LTM chunk详情")
        print("  3. 查看特定推理中的STM+LTM内容")
        print("  4. 查看所有参与者的LTM分布")
        print("  5. 退出")
        
        choice = input("\n输入选项 (1-5): ").strip()
        
        if choice == "1":
            view_ltm_statistics()
        
        elif choice == "2":
            participant = input("输入参与者ID (如 P1，留空查看所有): ").strip() or None
            
            if participant:
                stats_file = os.path.join(OUTPUT_DIR, "memory_bank_statistics.xlsx")
                if os.path.exists(stats_file):
                    df_stats = pd.read_excel(stats_file)
                    participant_chunks = df_stats[df_stats['Participant'] == participant]
                    print(f"\n{participant} 的chunk列表:")
                    for idx, row in participant_chunks.iterrows():
                        print(f"  [{idx}] {row['ChunkID']} (价值: {row['EstimatedValue']:.4f})")
                    
                    chunk_idx = input(f"\n输入chunk索引 (0-{len(participant_chunks)-1}): ").strip()
                    if chunk_idx.isdigit():
                        view_specific_ltm_chunk(df_stats, participant=participant, chunk_index=int(chunk_idx))
            else:
                chunk_id = input("输入ChunkID (如 P1_0): ").strip()
                stats_file = os.path.join(OUTPUT_DIR, "memory_bank_statistics.xlsx")
                if os.path.exists(stats_file):
                    df_stats = pd.read_excel(stats_file)
                    view_specific_ltm_chunk(df_stats, chunk_id=chunk_id)
        
        elif choice == "3":
            participant = input("输入参与者ID (如 P1，留空查看第一个): ").strip() or None
            strategy = input("输入策略 (A/B/C，留空查看所有): ").strip() or None
            anomaly_idx = input("输入异常点索引 (留空显示第一个): ").strip()
            anomaly_idx = int(anomaly_idx) if anomaly_idx.isdigit() else None
            
            view_stm_and_ltm_in_prompt(participant, anomaly_idx, strategy)
        
        elif choice == "4":
            df_stats = view_ltm_statistics()
            if df_stats is not None:
                print("\n各参与者的chunk详情:")
                for pid in sorted(df_stats['Participant'].unique()):
                    participant_chunks = df_stats[df_stats['Participant'] == pid]
                    avg_value = participant_chunks['EstimatedValue'].mean()
                    print(f"\n{pid}: {len(participant_chunks)} 个chunk, 平均价值: {avg_value:.4f}")
                    print(participant_chunks[['ChunkID', 'AccessCount', 'UsefulCount', 'EstimatedValue']].to_string(index=False))
        
        elif choice == "5":
            print("\n👋 再见!")
            break
        
        else:
            print("❌ 无效选项，请重新输入")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="查看LTM和STM的详细内容")
    parser.add_argument("--participant", type=str, help="参与者ID (如 P1)")
    parser.add_argument("--anomaly", type=int, help="异常点索引")
    parser.add_argument("--strategy", type=str, choices=["A", "B", "C"], help="策略")
    parser.add_argument("--chunk", type=str, help="ChunkID")
    parser.add_argument("--interactive", action="store_true", help="交互式模式")
    
    args = parser.parse_args()
    
    if args.interactive or (not any([args.participant, args.chunk])):
        interactive_menu()
    else:
        if args.chunk:
            stats_file = os.path.join(OUTPUT_DIR, "memory_bank_statistics.xlsx")
            if os.path.exists(stats_file):
                df_stats = pd.read_excel(stats_file)
                view_specific_ltm_chunk(df_stats, chunk_id=args.chunk)
        else:
            view_stm_and_ltm_in_prompt(args.participant, args.anomaly, args.strategy)
