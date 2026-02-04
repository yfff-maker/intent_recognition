"""
ABC窗口策略对比分析 - Bandit架构专用

分析不同异常点附近，使用A/B/C三种窗口策略的效果差异

输出：
1. 详细统计报告Excel
2. 可视化对比图表
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

# 设置中文字体和样式
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

OUTPUT_DIR = "./output"


def load_bandit_results():
    """加载Bandit版本的结果"""
    bandit_path = os.path.join(OUTPUT_DIR, "intent_inference_results_bandit.xlsx")
    
    if not os.path.exists(bandit_path):
        print(f"❌ 未找到Bandit结果文件: {bandit_path}")
        print("   请先运行: python main_bandit.py")
        return None
    
    df = pd.read_excel(bandit_path)
    
    # 确保Confidence是数值类型
    df['Confidence'] = pd.to_numeric(df['Confidence'], errors='coerce')
    
    print(f"✓ 已加载Bandit结果: {len(df)} 条记录")
    print(f"  - 参与者数: {df['Participant'].nunique()}")
    print(f"  - 异常点数: {len(df[df['Strategy']=='A'])}")  # 每个异常点有ABC三条记录
    print(f"  - 策略分布: A={len(df[df['Strategy']=='A'])}, B={len(df[df['Strategy']=='B'])}, C={len(df[df['Strategy']=='C'])}")
    
    return df


def analyze_strategy_performance(df):
    """总体策略性能对比"""
    print("\n" + "="*80)
    print("📊 ABC策略总体性能对比")
    print("="*80)
    
    results = []
    
    for strategy in ['A', 'B', 'C']:
        df_strategy = df[df['Strategy'] == strategy]
        conf = df_strategy['Confidence'].dropna()
        
        if len(conf) == 0:
            continue
        
        stats = {
            'Strategy': strategy,
            'Count': len(conf),
            'MeanConfidence': conf.mean(),
            'MedianConfidence': conf.median(),
            'StdConfidence': conf.std(),
            'MinConfidence': conf.min(),
            'MaxConfidence': conf.max(),
            'HighConfRate': (conf > 0.8).sum() / len(conf),  # 高置信度比例
            'LowConfRate': (conf < 0.5).sum() / len(conf),   # 低置信度比例
        }
        
        results.append(stats)
        
        print(f"\n策略 {strategy}:")
        print(f"  样本数: {stats['Count']}")
        print(f"  平均置信度: {stats['MeanConfidence']:.4f}")
        print(f"  中位数置信度: {stats['MedianConfidence']:.4f}")
        print(f"  标准差: {stats['StdConfidence']:.4f}")
        print(f"  高置信度(>0.8)比例: {stats['HighConfRate']:.2%}")
        print(f"  低置信度(<0.5)比例: {stats['LowConfRate']:.2%}")
    
    df_results = pd.DataFrame(results)
    
    # 计算相对于策略A的提升
    if len(df_results) == 3:
        baseline = df_results[df_results['Strategy']=='A']['MeanConfidence'].values[0]
        df_results['ImprovementVsA'] = df_results['MeanConfidence'] - baseline
        df_results['ImprovementVsA_Pct'] = (df_results['MeanConfidence'] / baseline - 1) * 100
        
        print("\n📈 相对于策略A的提升:")
        for _, row in df_results.iterrows():
            if row['Strategy'] != 'A':
                print(f"  {row['Strategy']} vs A: {row['ImprovementVsA']:+.4f} ({row['ImprovementVsA_Pct']:+.2f}%)")
    
    return df_results


def analyze_by_anomaly_type(df):
    """按异常类型分析ABC策略差异"""
    print("\n" + "="*80)
    print("📊 按异常类型的ABC策略对比")
    print("="*80)
    
    anomaly_types = df['AnomalyType'].unique()
    
    results = []
    
    for atype in anomaly_types:
        df_type = df[df['AnomalyType'] == atype]
        
        print(f"\n【{atype}】")
        
        for strategy in ['A', 'B', 'C']:
            df_s = df_type[df_type['Strategy'] == strategy]
            conf = df_s['Confidence'].dropna()
            
            if len(conf) == 0:
                continue
            
            results.append({
                'AnomalyType': atype,
                'Strategy': strategy,
                'Count': len(conf),
                'MeanConfidence': conf.mean(),
                'StdConfidence': conf.std(),
            })
            
            print(f"  策略{strategy}: 平均={conf.mean():.4f}, 样本数={len(conf)}")
    
    return pd.DataFrame(results)


def analyze_by_participant(df):
    """按参与者分析ABC策略差异"""
    print("\n" + "="*80)
    print("📊 按参与者的ABC策略对比")
    print("="*80)
    
    participants = sorted(df['Participant'].unique())
    
    results = []
    
    for p_id in participants:
        df_p = df[df['Participant'] == p_id]
        
        row = {'Participant': p_id}
        
        for strategy in ['A', 'B', 'C']:
            df_s = df_p[df_p['Strategy'] == strategy]
            conf = df_s['Confidence'].dropna()
            row[f'Strategy_{strategy}_Mean'] = conf.mean() if len(conf) > 0 else np.nan
            row[f'Strategy_{strategy}_Count'] = len(conf)
        
        # 计算B相对A、C相对A的提升
        if not np.isnan(row['Strategy_A_Mean']):
            row['B_vs_A'] = row['Strategy_B_Mean'] - row['Strategy_A_Mean']
            row['C_vs_A'] = row['Strategy_C_Mean'] - row['Strategy_A_Mean']
            row['Best_Strategy'] = max(['A', 'B', 'C'], 
                                      key=lambda s: row[f'Strategy_{s}_Mean'])
        
        results.append(row)
    
    df_results = pd.DataFrame(results)
    
    # 显示提升最大的5个参与者
    print("\n提升最大的参与者 (C vs A):")
    top_5 = df_results.nlargest(5, 'C_vs_A')
    for _, row in top_5.iterrows():
        print(f"  {row['Participant']}: C比A提升 {row['C_vs_A']:+.4f}")
    
    # 显示提升最小（或下降）的参与者
    print("\n提升最小的参与者 (C vs A):")
    bottom_5 = df_results.nsmallest(5, 'C_vs_A')
    for _, row in bottom_5.iterrows():
        print(f"  {row['Participant']}: C比A变化 {row['C_vs_A']:+.4f}")
    
    return df_results


def analyze_by_anomaly_point(df):
    """按每个异常点分析ABC策略差异"""
    print("\n" + "="*80)
    print("📊 按异常点的ABC策略对比")
    print("="*80)
    
    # 为每个异常点创建唯一ID
    df['AnomalyID'] = df['Participant'] + '_' + df['AnchorTimestamp'].astype(str)
    
    anomaly_ids = df['AnomalyID'].unique()
    
    results = []
    
    for aid in anomaly_ids:
        df_a = df[df['AnomalyID'] == aid]
        
        if len(df_a) != 3:  # 应该有ABC三条记录
            continue
        
        row = {
            'AnomalyID': aid,
            'Participant': df_a['Participant'].iloc[0],
            'Timestamp': df_a['AnchorTimestamp'].iloc[0],
            'AnomalyType': df_a['AnomalyType'].iloc[0],
        }
        
        for strategy in ['A', 'B', 'C']:
            df_s = df_a[df_a['Strategy'] == strategy]
            if len(df_s) > 0:
                row[f'Conf_{strategy}'] = df_s['Confidence'].iloc[0]
                row[f'Intent_{strategy}'] = df_s['Intent'].iloc[0]
        
        # 计算策略间的置信度差异
        row['B_minus_A'] = row['Conf_B'] - row['Conf_A']
        row['C_minus_A'] = row['Conf_C'] - row['Conf_A']
        row['C_minus_B'] = row['Conf_C'] - row['Conf_B']
        
        # 判断最佳策略
        row['Best_Strategy'] = max(['A', 'B', 'C'], 
                                   key=lambda s: row[f'Conf_{s}'])
        
        # 判断ABC是否给出相同意图
        intents = [row[f'Intent_{s}'] for s in ['A', 'B', 'C']]
        row['Intent_Agreement'] = len(set(intents)) == 1  # 三个策略意图一致
        
        results.append(row)
    
    df_results = pd.DataFrame(results)
    
    print(f"\n总异常点数: {len(df_results)}")
    print(f"意图完全一致的异常点: {df_results['Intent_Agreement'].sum()} ({df_results['Intent_Agreement'].sum()/len(df_results):.1%})")
    print(f"\n最佳策略分布:")
    print(df_results['Best_Strategy'].value_counts())
    
    # 显示C策略提升最大的异常点
    print("\nC策略提升最大的10个异常点:")
    top_10 = df_results.nlargest(10, 'C_minus_A')
    for _, row in top_10.iterrows():
        print(f"  {row['Participant']} @{row['Timestamp']}: C比A提升 {row['C_minus_A']:.4f} "
              f"(A={row['Conf_A']:.3f} → C={row['Conf_C']:.3f})")
    
    return df_results


def plot_abc_comparison(df, df_overall, df_by_type, df_by_participant):
    """生成ABC策略对比图表"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 图1: 总体置信度对比（柱状图）
    ax1 = fig.add_subplot(gs[0, 0])
    strategies = df_overall['Strategy'].values
    means = df_overall['MeanConfidence'].values
    stds = df_overall['StdConfidence'].values
    
    bars = ax1.bar(strategies, means, yerr=stds, capsize=5, alpha=0.7, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_ylabel('平均置信度', fontsize=11)
    ax1.set_title('ABC策略总体置信度对比', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    
    # 在柱子上标注数值
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 图2: 置信度分布（箱线图）
    ax2 = fig.add_subplot(gs[0, 1])
    data_for_box = [df[df['Strategy']==s]['Confidence'].dropna() for s in ['A', 'B', 'C']]
    bp = ax2.boxplot(data_for_box, labels=['A', 'B', 'C'], patch_artist=True)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel('置信度', fontsize=11)
    ax2.set_title('ABC策略置信度分布', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 图3: 高置信度比例对比
    ax3 = fig.add_subplot(gs[0, 2])
    high_conf_rates = df_overall['HighConfRate'].values
    bars = ax3.bar(strategies, high_conf_rates, alpha=0.7, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax3.set_ylabel('高置信度(>0.8)比例', fontsize=11)
    ax3.set_title('高置信度比例对比', fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 1])
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, rate in zip(bars, high_conf_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1%}', ha='center', va='bottom', fontsize=10)
    
    # 图4: 按异常类型的对比
    ax4 = fig.add_subplot(gs[1, :2])
    if len(df_by_type) > 0:
        pivot = df_by_type.pivot(index='AnomalyType', columns='Strategy', values='MeanConfidence')
        pivot.plot(kind='bar', ax=ax4, alpha=0.7, width=0.7)
        ax4.set_xlabel('异常类型', fontsize=11)
        ax4.set_ylabel('平均置信度', fontsize=11)
        ax4.set_title('不同异常类型下的ABC策略对比', fontsize=12, fontweight='bold')
        ax4.legend(title='策略', loc='upper right')
        ax4.grid(axis='y', alpha=0.3)
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
    
    # 图5: 参与者提升分布（散点图）
    ax5 = fig.add_subplot(gs[1, 2])
    if 'C_vs_A' in df_by_participant.columns:
        improvements = df_by_participant['C_vs_A'].dropna()
        ax5.scatter(range(len(improvements)), sorted(improvements), 
                   alpha=0.6, s=50, c=improvements, cmap='RdYlGn')
        ax5.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='无提升线')
        ax5.set_xlabel('参与者排名', fontsize=11)
        ax5.set_ylabel('C相对A的提升', fontsize=11)
        ax5.set_title('C策略提升分布（按参与者）', fontsize=12, fontweight='bold')
        ax5.grid(alpha=0.3)
        ax5.legend()
    
    # 图6: 策略选择频次（饼图）
    ax6 = fig.add_subplot(gs[2, 0])
    # 统计每个异常点的最佳策略
    df['AnomalyID'] = df['Participant'] + '_' + df['AnchorTimestamp'].astype(str)
    best_strategies = []
    for aid in df['AnomalyID'].unique():
        df_a = df[df['AnomalyID'] == aid]
        if len(df_a) == 3:
            best = df_a.loc[df_a['Confidence'].idxmax(), 'Strategy']
            best_strategies.append(best)
    
    strategy_counts = pd.Series(best_strategies).value_counts()
    colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    ax6.pie(strategy_counts.values, labels=strategy_counts.index, autopct='%1.1f%%',
           colors=colors_pie, startangle=90)
    ax6.set_title('最佳策略分布', fontsize=12, fontweight='bold')
    
    # 图7: 置信度改善趋势（折线图）
    ax7 = fig.add_subplot(gs[2, 1:])
    # 按时间戳排序，看策略效果的时序变化
    df_sorted = df.sort_values('AnchorTimestamp')
    for strategy in ['A', 'B', 'C']:
        df_s = df_sorted[df_sorted['Strategy'] == strategy]
        # 使用滚动平均平滑曲线
        window_size = min(10, len(df_s)//3)
        if window_size > 0:
            rolling_mean = df_s['Confidence'].rolling(window=window_size, min_periods=1).mean()
            ax7.plot(range(len(rolling_mean)), rolling_mean, 
                    label=f'策略{strategy}', linewidth=2, alpha=0.8)
    
    ax7.set_xlabel('异常点序号（按时间排序）', fontsize=11)
    ax7.set_ylabel('置信度（滚动平均）', fontsize=11)
    ax7.set_title('ABC策略置信度时序趋势对比', fontsize=12, fontweight='bold')
    ax7.legend(loc='best')
    ax7.grid(alpha=0.3)
    
    plt.suptitle('ABC窗口策略全面对比分析 - Bandit架构', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # 保存图表
    plot_path = os.path.join(OUTPUT_DIR, "abc_strategy_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 对比图表已保存到: {plot_path}")
    
    plt.close()


def plot_detailed_anomaly_comparison(df_by_anomaly):
    """生成详细的异常点对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('异常点级别的ABC策略详细对比', fontsize=16, fontweight='bold')
    
    # 图1: C-A差异分布
    ax1 = axes[0, 0]
    improvements = df_by_anomaly['C_minus_A'].dropna()
    ax1.hist(improvements, bins=30, alpha=0.7, color='#45B7D1', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='无差异线')
    ax1.axvline(x=improvements.mean(), color='green', linestyle='--', 
               linewidth=2, label=f'平均差异={improvements.mean():.3f}')
    ax1.set_xlabel('C策略 - A策略 (置信度差异)', fontsize=11)
    ax1.set_ylabel('异常点数量', fontsize=11)
    ax1.set_title('C相对A的置信度差异分布', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 图2: B-A vs C-A散点图
    ax2 = axes[0, 1]
    ax2.scatter(df_by_anomaly['B_minus_A'], df_by_anomaly['C_minus_A'], 
               alpha=0.5, s=50, c=df_by_anomaly['C_minus_A'], cmap='RdYlGn')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.plot([-0.5, 0.5], [-0.5, 0.5], 'r--', alpha=0.3, label='B=C线')
    ax2.set_xlabel('B - A (置信度差异)', fontsize=11)
    ax2.set_ylabel('C - A (置信度差异)', fontsize=11)
    ax2.set_title('B策略 vs C策略 相对A的提升', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 图3: 意图一致性分析
    ax3 = axes[1, 0]
    agreement_counts = df_by_anomaly['Intent_Agreement'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4']
    labels = ['意图不一致', '意图一致']
    ax3.bar(labels, [agreement_counts.get(False, 0), agreement_counts.get(True, 0)],
           color=colors, alpha=0.7)
    ax3.set_ylabel('异常点数量', fontsize=11)
    ax3.set_title('ABC策略意图推断一致性', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 在柱子上标注百分比
    total = len(df_by_anomaly)
    for i, (label, count) in enumerate([(False, agreement_counts.get(False, 0)), 
                                         (True, agreement_counts.get(True, 0))]):
        ax3.text(i, count, f'{count}\n({count/total:.1%})', 
                ha='center', va='bottom', fontsize=10)
    
    # 图4: 按异常类型的C-A提升
    ax4 = axes[1, 1]
    type_improvement = df_by_anomaly.groupby('AnomalyType')['C_minus_A'].mean().sort_values()
    type_improvement.plot(kind='barh', ax=ax4, alpha=0.7, color='#45B7D1')
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax4.set_xlabel('平均置信度提升 (C - A)', fontsize=11)
    ax4.set_title('不同异常类型下C策略的平均提升', fontsize=12, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(OUTPUT_DIR, "abc_anomaly_detail_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ 详细异常点对比图已保存到: {plot_path}")
    
    plt.close()


def save_analysis_report(df_overall, df_by_type, df_by_participant, df_by_anomaly):
    """保存分析报告到Excel"""
    output_path = os.path.join(OUTPUT_DIR, "abc_strategy_analysis_report.xlsx")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: 总体对比
        df_overall.to_excel(writer, sheet_name='总体对比', index=False)
        
        # Sheet 2: 按异常类型对比
        if len(df_by_type) > 0:
            df_by_type.to_excel(writer, sheet_name='按异常类型', index=False)
        
        # Sheet 3: 按参与者对比
        df_by_participant.to_excel(writer, sheet_name='按参与者', index=False)
        
        # Sheet 4: 按异常点详细对比
        df_by_anomaly.to_excel(writer, sheet_name='按异常点', index=False)
    
    print(f"\n✓ 分析报告已保存到: {output_path}")


def main():
    print("="*80)
    print("📊 ABC窗口策略对比分析 - Bandit架构")
    print("="*80)
    
    # 加载数据
    df = load_bandit_results()
    if df is None:
        return
    
    # 总体性能分析
    df_overall = analyze_strategy_performance(df)
    
    # 按异常类型分析
    df_by_type = analyze_by_anomaly_type(df)
    
    # 按参与者分析
    df_by_participant = analyze_by_participant(df)
    
    # 按异常点分析
    df_by_anomaly = analyze_by_anomaly_point(df)
    
    # 生成对比图表
    print("\n" + "="*80)
    print("📈 生成可视化图表...")
    print("="*80)
    
    plot_abc_comparison(df, df_overall, df_by_type, df_by_participant)
    plot_detailed_anomaly_comparison(df_by_anomaly)
    
    # 保存报告
    save_analysis_report(df_overall, df_by_type, df_by_participant, df_by_anomaly)
    
    print("\n" + "="*80)
    print("✅ ABC策略对比分析完成！")
    print("="*80)
    print("\n生成的文件:")
    print(f"  1. {os.path.join(OUTPUT_DIR, 'abc_strategy_analysis_report.xlsx')}")
    print(f"  2. {os.path.join(OUTPUT_DIR, 'abc_strategy_comparison.png')}")
    print(f"  3. {os.path.join(OUTPUT_DIR, 'abc_anomaly_detail_comparison.png')}")
    
    # 输出关键发现
    print("\n" + "="*80)
    print("🔍 关键发现总结")
    print("="*80)
    
    if len(df_overall) == 3:
        baseline = df_overall[df_overall['Strategy']=='A']['MeanConfidence'].values[0]
        b_mean = df_overall[df_overall['Strategy']=='B']['MeanConfidence'].values[0]
        c_mean = df_overall[df_overall['Strategy']=='C']['MeanConfidence'].values[0]
        
        print(f"\n1. 总体置信度提升:")
        print(f"   - 策略A（短窗口）: {baseline:.4f}")
        print(f"   - 策略B（中窗口）: {b_mean:.4f} ({(b_mean/baseline-1)*100:+.2f}%)")
        print(f"   - 策略C（长窗口）: {c_mean:.4f} ({(c_mean/baseline-1)*100:+.2f}%)")
        
        if c_mean > b_mean > baseline:
            print("\n   ✅ 结论: 更长的上下文窗口带来更高的置信度！")
        elif c_mean < baseline:
            print("\n   ⚠️ 注意: C策略反而降低了置信度，可能是噪音过多！")
    
    print(f"\n2. 意图推断一致性:")
    agreement_rate = df_by_anomaly['Intent_Agreement'].mean()
    print(f"   - ABC三策略推断相同意图的比例: {agreement_rate:.1%}")
    if agreement_rate > 0.7:
        print("   ✅ 策略间一致性高，结果可靠")
    else:
        print("   ⚠️ 策略间一致性较低，需要进一步分析")
    
    print(f"\n3. 最佳策略分布:")
    best_dist = df_by_anomaly['Best_Strategy'].value_counts()
    for strategy, count in best_dist.items():
        print(f"   - 策略{strategy}: {count} 次 ({count/len(df_by_anomaly):.1%})")


if __name__ == "__main__":
    main()
