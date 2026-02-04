"""
对比分析脚本：比较简单记忆库 vs Bandit记忆库的结果

使用方法：
    python compare_results.py

输出：
    1. 控制台打印对比统计
    2. 生成对比报告Excel文件
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = "./output"


def load_results():
    """加载两个版本的结果文件"""
    baseline_path = os.path.join(OUTPUT_DIR, "intent_inference_results.xlsx")
    bandit_path = os.path.join(OUTPUT_DIR, "intent_inference_results_bandit.xlsx")
    stats_path = os.path.join(OUTPUT_DIR, "memory_bank_statistics.xlsx")
    
    if not os.path.exists(baseline_path):
        print(f"❌ 未找到简单版本结果文件: {baseline_path}")
        print("   请先运行: python main.py")
        return None, None, None
    
    if not os.path.exists(bandit_path):
        print(f"❌ 未找到Bandit版本结果文件: {bandit_path}")
        print("   请先运行: python main_bandit.py")
        return None, None, None
    
    df_baseline = pd.read_excel(baseline_path)
    df_bandit = pd.read_excel(bandit_path)
    df_stats = pd.read_excel(stats_path) if os.path.exists(stats_path) else None
    
    print(f"✓ 已加载简单版本结果: {len(df_baseline)} 条")
    print(f"✓ 已加载Bandit版本结果: {len(df_bandit)} 条")
    if df_stats is not None:
        print(f"✓ 已加载Bandit统计信息: {len(df_stats)} 个chunk")
    
    return df_baseline, df_bandit, df_stats


def compare_confidence(df_baseline, df_bandit):
    """对比置信度"""
    print("\n" + "="*60)
    print("📊 置信度对比")
    print("="*60)
    
    # 确保Confidence列是数值类型
    df_baseline['Confidence'] = pd.to_numeric(df_baseline['Confidence'], errors='coerce')
    df_bandit['Confidence'] = pd.to_numeric(df_bandit['Confidence'], errors='coerce')
    
    baseline_conf = df_baseline['Confidence'].dropna()
    bandit_conf = df_bandit['Confidence'].dropna()
    
    print(f"简单版本平均置信度: {baseline_conf.mean():.4f}")
    print(f"Bandit版本平均置信度: {bandit_conf.mean():.4f}")
    print(f"差异: {(bandit_conf.mean() - baseline_conf.mean()):.4f}")
    
    print(f"\n简单版本置信度>0.8的比例: {(baseline_conf > 0.8).sum() / len(baseline_conf):.2%}")
    print(f"Bandit版本置信度>0.8的比例: {(bandit_conf > 0.8).sum() / len(bandit_conf):.2%}")
    
    return {
        "baseline_mean": baseline_conf.mean(),
        "bandit_mean": bandit_conf.mean(),
        "baseline_high_conf_ratio": (baseline_conf > 0.8).sum() / len(baseline_conf),
        "bandit_high_conf_ratio": (bandit_conf > 0.8).sum() / len(bandit_conf),
    }


def compare_by_strategy(df_baseline, df_bandit):
    """按策略A/B/C对比"""
    print("\n" + "="*60)
    print("📊 分策略对比")
    print("="*60)
    
    results = []
    for strategy in ['A', 'B', 'C']:
        baseline_strategy = df_baseline[df_baseline['Strategy'] == strategy]['Confidence'].dropna()
        bandit_strategy = df_bandit[df_bandit['Strategy'] == strategy]['Confidence'].dropna()
        
        baseline_mean = baseline_strategy.mean() if len(baseline_strategy) > 0 else 0
        bandit_mean = bandit_strategy.mean() if len(bandit_strategy) > 0 else 0
        
        print(f"\n策略 {strategy}:")
        print(f"  简单版本: {baseline_mean:.4f}")
        print(f"  Bandit版本: {bandit_mean:.4f}")
        print(f"  提升: {(bandit_mean - baseline_mean):.4f}")
        
        results.append({
            "Strategy": strategy,
            "Baseline_Mean": baseline_mean,
            "Bandit_Mean": bandit_mean,
            "Improvement": bandit_mean - baseline_mean
        })
    
    return pd.DataFrame(results)


def compare_by_participant(df_baseline, df_bandit):
    """按参与者对比"""
    print("\n" + "="*60)
    print("📊 分参与者对比")
    print("="*60)
    
    participants = df_baseline['Participant'].unique()
    
    results = []
    for p_id in participants:
        baseline_p = df_baseline[df_baseline['Participant'] == p_id]['Confidence'].dropna()
        bandit_p = df_bandit[df_bandit['Participant'] == p_id]['Confidence'].dropna()
        
        if len(baseline_p) == 0 or len(bandit_p) == 0:
            continue
        
        baseline_mean = baseline_p.mean()
        bandit_mean = bandit_p.mean()
        improvement = bandit_mean - baseline_mean
        
        results.append({
            "Participant": p_id,
            "Baseline_Mean": baseline_mean,
            "Bandit_Mean": bandit_mean,
            "Improvement": improvement,
            "Sample_Count": len(baseline_p)
        })
    
    df_results = pd.DataFrame(results)
    
    # 显示提升最大的5个参与者
    top_5 = df_results.nlargest(5, 'Improvement')
    print("\n提升最大的5个参与者:")
    print(top_5.to_string(index=False))
    
    # 显示下降最大的5个参与者
    bottom_5 = df_results.nsmallest(5, 'Improvement')
    print("\n下降最大的5个参与者:")
    print(bottom_5.to_string(index=False))
    
    return df_results


def analyze_bandit_stats(df_stats):
    """分析Bandit统计信息"""
    if df_stats is None:
        print("\n⚠️  未找到Bandit统计文件，跳过分析")
        return None
    
    print("\n" + "="*60)
    print("📊 Bandit记忆库统计分析")
    print("="*60)
    
    print(f"\n总chunk数: {len(df_stats)}")
    print(f"平均访问次数: {df_stats['AccessCount'].mean():.2f}")
    print(f"平均采用次数: {df_stats['UsefulCount'].mean():.2f}")
    print(f"平均采用率: {df_stats['UsageRate'].mean():.2%}")
    print(f"平均估计价值: {df_stats['EstimatedValue'].mean():.4f}")
    
    # 找出明星chunk
    print("\n⭐ 最有价值的10个chunk:")
    top_chunks = df_stats.nlargest(10, 'EstimatedValue')
    print(top_chunks[['ChunkID', 'AccessCount', 'UsefulCount', 'EstimatedValue', 'UsageRate']].to_string(index=False))
    
    # 找出被遗忘的chunk
    print("\n❌ 最少被使用的10个chunk:")
    bottom_chunks = df_stats.nsmallest(10, 'AccessCount')
    print(bottom_chunks[['ChunkID', 'AccessCount', 'UsefulCount', 'EstimatedValue']].to_string(index=False))
    
    # 统计提升的chunk
    promoted_chunks = df_stats[df_stats['ChunkID'].str.contains('promoted', na=False)]
    if len(promoted_chunks) > 0:
        print(f"\n🚀 从STM提升的chunk数: {len(promoted_chunks)}")
        print(f"   平均价值: {promoted_chunks['EstimatedValue'].mean():.4f}")
        print(f"   平均采用率: {promoted_chunks['UsageRate'].mean():.2%}")
    
    return {
        "total_chunks": len(df_stats),
        "avg_access": df_stats['AccessCount'].mean(),
        "avg_useful": df_stats['UsefulCount'].mean(),
        "avg_usage_rate": df_stats['UsageRate'].mean(),
        "avg_value": df_stats['EstimatedValue'].mean(),
        "promoted_count": len(promoted_chunks) if len(promoted_chunks) > 0 else 0,
    }


def save_comparison_report(conf_stats, strategy_comp, participant_comp, bandit_stats):
    """保存对比报告到Excel"""
    output_path = os.path.join(OUTPUT_DIR, "comparison_report.xlsx")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: 总体对比
        df_overall = pd.DataFrame([conf_stats])
        df_overall.to_excel(writer, sheet_name='总体对比', index=False)
        
        # Sheet 2: 策略对比
        strategy_comp.to_excel(writer, sheet_name='策略对比', index=False)
        
        # Sheet 3: 参与者对比
        participant_comp.to_excel(writer, sheet_name='参与者对比', index=False)
        
        # Sheet 4: Bandit统计
        if bandit_stats:
            df_bandit_summary = pd.DataFrame([bandit_stats])
            df_bandit_summary.to_excel(writer, sheet_name='Bandit统计', index=False)
    
    print(f"\n✓ 对比报告已保存到: {output_path}")


def plot_comparison(df_baseline, df_bandit, df_stats):
    """生成对比可视化图表"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('简单记忆库 vs Bandit记忆库 对比分析', fontsize=16, fontweight='bold')
    
    # 图1: 置信度分布对比
    ax1 = axes[0, 0]
    baseline_conf = df_baseline['Confidence'].dropna()
    bandit_conf = df_bandit['Confidence'].dropna()
    ax1.hist([baseline_conf, bandit_conf], bins=20, label=['简单版本', 'Bandit版本'], alpha=0.7)
    ax1.set_xlabel('置信度')
    ax1.set_ylabel('频次')
    ax1.set_title('置信度分布对比')
    ax1.legend()
    
    # 图2: 策略对比
    ax2 = axes[0, 1]
    strategy_data = []
    for strategy in ['A', 'B', 'C']:
        baseline_mean = df_baseline[df_baseline['Strategy'] == strategy]['Confidence'].mean()
        bandit_mean = df_bandit[df_bandit['Strategy'] == strategy]['Confidence'].mean()
        strategy_data.append([baseline_mean, bandit_mean])
    
    x = range(len(['A', 'B', 'C']))
    width = 0.35
    ax2.bar([i - width/2 for i in x], [d[0] for d in strategy_data], width, label='简单版本', alpha=0.8)
    ax2.bar([i + width/2 for i in x], [d[1] for d in strategy_data], width, label='Bandit版本', alpha=0.8)
    ax2.set_xlabel('策略')
    ax2.set_ylabel('平均置信度')
    ax2.set_title('不同策略的置信度对比')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['A', 'B', 'C'])
    ax2.legend()
    
    # 图3: Chunk价值分布
    if df_stats is not None:
        ax3 = axes[1, 0]
        ax3.hist(df_stats['EstimatedValue'], bins=20, alpha=0.7, color='green')
        ax3.set_xlabel('估计价值')
        ax3.set_ylabel('Chunk数量')
        ax3.set_title('Chunk价值分布')
        ax3.axvline(df_stats['EstimatedValue'].mean(), color='red', linestyle='--', label='平均值')
        ax3.legend()
    else:
        ax3 = axes[1, 0]
        ax3.text(0.5, 0.5, '无Bandit统计数据', ha='center', va='center', fontsize=14)
        ax3.axis('off')
    
    # 图4: 采用率分布
    if df_stats is not None:
        ax4 = axes[1, 1]
        ax4.hist(df_stats['UsageRate'], bins=20, alpha=0.7, color='orange')
        ax4.set_xlabel('采用率 (UsefulCount / AccessCount)')
        ax4.set_ylabel('Chunk数量')
        ax4.set_title('Chunk采用率分布')
        ax4.axvline(df_stats['UsageRate'].mean(), color='red', linestyle='--', label='平均值')
        ax4.legend()
    else:
        ax4 = axes[1, 1]
        ax4.text(0.5, 0.5, '无Bandit统计数据', ha='center', va='center', fontsize=14)
        ax4.axis('off')
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(OUTPUT_DIR, "comparison_plots.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ 对比图表已保存到: {plot_path}")
    
    # 显示图表（可选）
    # plt.show()


def main():
    print("="*60)
    print("📊 开始对比分析：简单记忆库 vs Bandit记忆库")
    print("="*60)
    
    # 加载数据
    df_baseline, df_bandit, df_stats = load_results()
    if df_baseline is None or df_bandit is None:
        return
    
    # 对比分析
    conf_stats = compare_confidence(df_baseline, df_bandit)
    strategy_comp = compare_by_strategy(df_baseline, df_bandit)
    participant_comp = compare_by_participant(df_baseline, df_bandit)
    bandit_stats = analyze_bandit_stats(df_stats)
    
    # 保存报告
    save_comparison_report(conf_stats, strategy_comp, participant_comp, bandit_stats)
    
    # 生成图表
    try:
        plot_comparison(df_baseline, df_bandit, df_stats)
    except Exception as e:
        print(f"⚠️  生成图表时出错: {e}")
        print("   跳过图表生成，但对比报告已保存")
    
    print("\n" + "="*60)
    print("✅ 对比分析完成！")
    print("="*60)
    print("\n生成的文件:")
    print(f"  1. {os.path.join(OUTPUT_DIR, 'comparison_report.xlsx')}")
    print(f"  2. {os.path.join(OUTPUT_DIR, 'comparison_plots.png')}")


if __name__ == "__main__":
    main()
