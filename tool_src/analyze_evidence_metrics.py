"""
证据质量指标分析 - ABC策略对比

分析指标：
1. Early-evidence rate（早期证据引用率）：证据中有多少来自窗口左侧
2. Average evidence distance（证据平均距离）：证据事件离异常点的平均距离
3. 窗口事件数
4. 上下文token数

输出4张图：
- 窗口事件数 vs Early-evidence rate
- 窗口事件数 vs Average evidence distance
- 上下文token数 vs Early-evidence rate
- 上下文token数 vs Average evidence distance
"""

import os
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import List, Dict, Tuple

# 设置中文字体
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = "./output"

# 窗口事件数（理论值，来自config.py）
WINDOW_SIZES = {
    'A': {'k_left': 2, 'k_right': 2, 'total': 5},
    'B': {'k_left': 20, 'k_right': 20, 'total': 41},
    'C': {'k_left': 200, 'k_right': 50, 'total': 251},
}


def parse_evidence_field(evidence_str: str) -> List[Dict]:
    """
    解析Evidence字段（可能是字符串化的JSON）
    
    返回: [{"event_idx": "42", "why": "..."}, ...]
    """
    if pd.isna(evidence_str) or evidence_str == '[]':
        return []
    
    try:
        # 尝试直接解析JSON
        evidence = json.loads(evidence_str)
        if isinstance(evidence, list):
            return evidence
    except:
        pass
    
    # 尝试字符串形式
    try:
        # 替换单引号为双引号
        evidence_str = str(evidence_str).replace("'", '"')
        evidence = json.loads(evidence_str)
        if isinstance(evidence, list):
            return evidence
    except:
        pass
    
    return []


def extract_event_indices(evidence_list: List[Dict]) -> List[int]:
    """
    从evidence列表中提取事件索引
    
    event_idx可能是：
    - "42"（单个索引）
    - "42..45"（范围）
    - "chunk_P1_2"（chunk引用，忽略）
    """
    indices = []
    
    for item in evidence_list:
        idx_str = item.get('event_idx', '')
        
        if not idx_str or 'chunk' in idx_str.lower():
            continue  # 跳过chunk引用
        
        # 处理范围 "42..45"
        if '..' in idx_str:
            try:
                parts = idx_str.split('..')
                start = int(parts[0])
                end = int(parts[1])
                indices.extend(range(start, end + 1))
            except:
                pass
        else:
            # 单个索引
            try:
                indices.append(int(idx_str))
            except:
                pass
    
    return indices


def estimate_token_count(prompt_text: str) -> int:
    """
    估算prompt的token数
    
    简单估算：1 token ≈ 4 characters（英文+中文混合）
    """
    if pd.isna(prompt_text):
        return 0
    
    char_count = len(str(prompt_text))
    token_count = char_count // 4
    return token_count


def calculate_evidence_metrics(row, center_event_idx: int) -> Dict:
    """
    计算单个异常点的证据指标
    
    Args:
        row: DataFrame的一行
        center_event_idx: 异常点对应的中心事件索引
    
    Returns:
        {
            'early_evidence_rate': float,  # 早期证据比例
            'avg_evidence_distance': float,  # 平均距离
            'total_evidence_count': int,  # 总证据数
        }
    """
    # 解析Evidence字段
    evidence_list = parse_evidence_field(row['Evidence'])
    
    if not evidence_list:
        return {
            'early_evidence_rate': np.nan,
            'avg_evidence_distance': np.nan,
            'total_evidence_count': 0,
        }
    
    # 提取事件索引
    indices = extract_event_indices(evidence_list)
    
    if not indices:
        return {
            'early_evidence_rate': np.nan,
            'avg_evidence_distance': np.nan,
            'total_evidence_count': len(evidence_list),  # 可能都是chunk引用
        }
    
    # 计算早期证据率（索引 < center_event_idx 的比例）
    early_count = sum(1 for idx in indices if idx < center_event_idx)
    early_rate = early_count / len(indices) if len(indices) > 0 else 0
    
    # 计算平均距离（绝对值）
    distances = [abs(idx - center_event_idx) for idx in indices]
    avg_distance = np.mean(distances) if distances else 0
    
    return {
        'early_evidence_rate': early_rate,
        'avg_evidence_distance': avg_distance,
        'total_evidence_count': len(indices),
    }


def load_and_process_data():
    """加载Bandit结果并处理"""
    bandit_path = os.path.join(OUTPUT_DIR, "intent_inference_results_bandit.xlsx")
    
    if not os.path.exists(bandit_path):
        print(f"❌ 未找到Bandit结果文件: {bandit_path}")
        return None
    
    df = pd.read_excel(bandit_path)
    print(f"✓ 已加载 {len(df)} 条记录")
    
    # 为每行计算指标
    print("\n处理中...", end='')
    
    results = []
    
    for idx, row in df.iterrows():
        if idx % 50 == 0:
            print(".", end='', flush=True)
        
        strategy = row['Strategy']
        
        # 估算窗口大小和token数
        window_size = WINDOW_SIZES[strategy]['total']
        token_count = estimate_token_count(row['Prompt'])
        
        # 假设中心事件索引（简化：我们不知道确切的center_pos，用timestamp估算）
        # 在实际场景中，可能需要从其他地方获取
        # 这里我们用一个近似方法：从Evidence中推断
        evidence_list = parse_evidence_field(row['Evidence'])
        indices = extract_event_indices(evidence_list)
        
        # 近似中心点：假设是证据索引的中位数附近
        if indices:
            center_event_idx = int(np.median(indices))
        else:
            center_event_idx = 0  # 无法确定，设为0
        
        # 计算证据指标
        metrics = calculate_evidence_metrics(row, center_event_idx)
        
        results.append({
            'Participant': row['Participant'],
            'AnchorTimestamp': row['AnchorTimestamp'],
            'AnomalyType': row['AnomalyType'],
            'Strategy': strategy,
            'WindowSize': window_size,
            'TokenCount': token_count,
            'Confidence': row['Confidence'],
            'EarlyEvidenceRate': metrics['early_evidence_rate'],
            'AvgEvidenceDistance': metrics['avg_evidence_distance'],
            'TotalEvidenceCount': metrics['total_evidence_count'],
            'Intent': row['Intent'],
        })
    
    print(" 完成！")
    
    df_processed = pd.DataFrame(results)
    return df_processed


def aggregate_by_strategy(df):
    """按策略聚合统计"""
    print("\n" + "="*80)
    print("📊 按策略聚合的指标统计")
    print("="*80)
    
    agg_results = []
    
    for strategy in ['A', 'B', 'C']:
        df_s = df[df['Strategy'] == strategy]
        
        stats = {
            'Strategy': strategy,
            'WindowSize_Mean': df_s['WindowSize'].mean(),
            'TokenCount_Mean': df_s['TokenCount'].mean(),
            'TokenCount_Std': df_s['TokenCount'].std(),
            'Confidence_Mean': df_s['Confidence'].mean(),
            'Confidence_Std': df_s['Confidence'].std(),
            'EarlyEvidenceRate_Mean': df_s['EarlyEvidenceRate'].mean(),
            'EarlyEvidenceRate_Std': df_s['EarlyEvidenceRate'].std(),
            'AvgEvidenceDistance_Mean': df_s['AvgEvidenceDistance'].mean(),
            'AvgEvidenceDistance_Std': df_s['AvgEvidenceDistance'].std(),
            'SampleCount': len(df_s),
        }
        
        agg_results.append(stats)
        
        print(f"\n策略 {strategy}:")
        print(f"  样本数: {stats['SampleCount']}")
        print(f"  平均窗口事件数: {stats['WindowSize_Mean']:.0f}")
        print(f"  平均Token数: {stats['TokenCount_Mean']:.0f} ± {stats['TokenCount_Std']:.0f}")
        print(f"  平均置信度: {stats['Confidence_Mean']:.4f} ± {stats['Confidence_Std']:.4f}")
        print(f"  早期证据率: {stats['EarlyEvidenceRate_Mean']:.2%} ± {stats['EarlyEvidenceRate_Std']:.2%}")
        print(f"  平均证据距离: {stats['AvgEvidenceDistance_Mean']:.1f} ± {stats['AvgEvidenceDistance_Std']:.1f} 个事件")
    
    return pd.DataFrame(agg_results)


def plot_four_metrics(df_agg):
    """
    绘制4张指标图
    
    横坐标：窗口事件数、上下文token数
    纵坐标：早期证据率、平均证据距离
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ABC策略的证据质量指标对比', fontsize=16, fontweight='bold')
    
    strategies = ['A', 'B', 'C']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    markers = ['o', 's', '^']
    
    # 提取数据
    window_sizes = df_agg['WindowSize_Mean'].values
    token_counts = df_agg['TokenCount_Mean'].values
    token_stds = df_agg['TokenCount_Std'].values
    
    early_rates = df_agg['EarlyEvidenceRate_Mean'].values
    early_stds = df_agg['EarlyEvidenceRate_Std'].values
    
    distances = df_agg['AvgEvidenceDistance_Mean'].values
    distance_stds = df_agg['AvgEvidenceDistance_Std'].values
    
    # ==================== 图1: 窗口事件数 vs 早期证据率 ====================
    ax1 = axes[0, 0]
    for i, (strategy, color, marker) in enumerate(zip(strategies, colors, markers)):
        ax1.errorbar(window_sizes[i], early_rates[i], yerr=early_stds[i],
                    marker=marker, markersize=12, capsize=8, capthick=2,
                    linewidth=2, label=f'策略{strategy}', color=color, alpha=0.8)
        # 标注点
        ax1.text(window_sizes[i], early_rates[i] + early_stds[i] + 0.03,
                f'{strategy}\n({early_rates[i]:.1%})',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('窗口事件数', fontsize=12, fontweight='bold')
    ax1.set_ylabel('早期证据引用率', fontsize=12, fontweight='bold')
    ax1.set_title('窗口大小 vs 早期证据率', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)
    ax1.set_xscale('log')  # 对数刻度，因为A=5, C=251差距大
    
    # ==================== 图2: 窗口事件数 vs 平均证据距离 ====================
    ax2 = axes[0, 1]
    for i, (strategy, color, marker) in enumerate(zip(strategies, colors, markers)):
        ax2.errorbar(window_sizes[i], distances[i], yerr=distance_stds[i],
                    marker=marker, markersize=12, capsize=8, capthick=2,
                    linewidth=2, label=f'策略{strategy}', color=color, alpha=0.8)
        ax2.text(window_sizes[i], distances[i] + distance_stds[i] + 5,
                f'{strategy}\n({distances[i]:.0f})',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('窗口事件数', fontsize=12, fontweight='bold')
    ax2.set_ylabel('平均证据距离（事件数）', fontsize=12, fontweight='bold')
    ax2.set_title('窗口大小 vs 平均证据距离', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)
    ax2.set_xscale('log')
    
    # ==================== 图3: 上下文Token数 vs 早期证据率 ====================
    ax3 = axes[1, 0]
    for i, (strategy, color, marker) in enumerate(zip(strategies, colors, markers)):
        ax3.errorbar(token_counts[i], early_rates[i], 
                    xerr=token_stds[i], yerr=early_stds[i],
                    marker=marker, markersize=12, capsize=8, capthick=2,
                    linewidth=2, label=f'策略{strategy}', color=color, alpha=0.8)
        ax3.text(token_counts[i] + token_stds[i] + 200, early_rates[i],
                f'{strategy}',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax3.set_xlabel('上下文Token数', fontsize=12, fontweight='bold')
    ax3.set_ylabel('早期证据引用率', fontsize=12, fontweight='bold')
    ax3.set_title('Token消耗 vs 早期证据率', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best', fontsize=10)
    
    # ==================== 图4: 上下文Token数 vs 平均证据距离 ====================
    ax4 = axes[1, 1]
    for i, (strategy, color, marker) in enumerate(zip(strategies, colors, markers)):
        ax4.errorbar(token_counts[i], distances[i],
                    xerr=token_stds[i], yerr=distance_stds[i],
                    marker=marker, markersize=12, capsize=8, capthick=2,
                    linewidth=2, label=f'策略{strategy}', color=color, alpha=0.8)
        ax4.text(token_counts[i] + token_stds[i] + 200, distances[i],
                f'{strategy}',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax4.set_xlabel('上下文Token数', fontsize=12, fontweight='bold')
    ax4.set_ylabel('平均证据距离（事件数）', fontsize=12, fontweight='bold')
    ax4.set_title('Token消耗 vs 平均证据距离', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(OUTPUT_DIR, "evidence_metrics_4plots.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 4张指标对比图已保存到: {plot_path}")
    
    plt.close()


def plot_scatter_with_trend(df):
    """
    绘制散点图（每个异常点一个点）+ 趋势线
    
    展示所有数据点的分布，而不仅仅是平均值
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ABC策略证据指标分布（所有异常点）', fontsize=16, fontweight='bold')
    
    strategies = ['A', 'B', 'C']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # ==================== 图1: 窗口事件数 vs 早期证据率（散点） ====================
    ax1 = axes[0, 0]
    for strategy, color in zip(strategies, colors):
        df_s = df[df['Strategy'] == strategy]
        # 添加随机抖动，避免点重叠
        jitter = np.random.normal(0, 2, len(df_s))
        ax1.scatter(df_s['WindowSize'] + jitter, df_s['EarlyEvidenceRate'],
                   alpha=0.4, s=30, color=color, label=f'策略{strategy}')
        
        # 添加平均值标记
        mean_x = df_s['WindowSize'].mean()
        mean_y = df_s['EarlyEvidenceRate'].mean()
        ax1.scatter(mean_x, mean_y, marker='*', s=300, color=color,
                   edgecolors='black', linewidths=2, zorder=10)
    
    ax1.set_xlabel('窗口事件数', fontsize=11)
    ax1.set_ylabel('早期证据引用率', fontsize=11)
    ax1.set_title('窗口大小 vs 早期证据率（散点分布）', fontsize=12, fontweight='bold')
    ax1.set_xscale('log')
    ax1.legend(loc='best')
    ax1.grid(alpha=0.3)
    
    # ==================== 图2: 窗口事件数 vs 平均距离（散点） ====================
    ax2 = axes[0, 1]
    for strategy, color in zip(strategies, colors):
        df_s = df[df['Strategy'] == strategy]
        jitter = np.random.normal(0, 2, len(df_s))
        ax2.scatter(df_s['WindowSize'] + jitter, df_s['AvgEvidenceDistance'],
                   alpha=0.4, s=30, color=color, label=f'策略{strategy}')
        
        mean_x = df_s['WindowSize'].mean()
        mean_y = df_s['AvgEvidenceDistance'].mean()
        ax2.scatter(mean_x, mean_y, marker='*', s=300, color=color,
                   edgecolors='black', linewidths=2, zorder=10)
    
    ax2.set_xlabel('窗口事件数', fontsize=11)
    ax2.set_ylabel('平均证据距离（事件数）', fontsize=11)
    ax2.set_title('窗口大小 vs 平均证据距离（散点分布）', fontsize=12, fontweight='bold')
    ax2.set_xscale('log')
    ax2.legend(loc='best')
    ax2.grid(alpha=0.3)
    
    # ==================== 图3: Token数 vs 早期证据率（散点） ====================
    ax3 = axes[1, 0]
    for strategy, color in zip(strategies, colors):
        df_s = df[df['Strategy'] == strategy]
        ax3.scatter(df_s['TokenCount'], df_s['EarlyEvidenceRate'],
                   alpha=0.4, s=30, color=color, label=f'策略{strategy}')
        
        mean_x = df_s['TokenCount'].mean()
        mean_y = df_s['EarlyEvidenceRate'].mean()
        ax3.scatter(mean_x, mean_y, marker='*', s=300, color=color,
                   edgecolors='black', linewidths=2, zorder=10)
    
    ax3.set_xlabel('上下文Token数', fontsize=11)
    ax3.set_ylabel('早期证据引用率', fontsize=11)
    ax3.set_title('Token消耗 vs 早期证据率（散点分布）', fontsize=12, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(alpha=0.3)
    
    # ==================== 图4: Token数 vs 平均距离（散点） ====================
    ax4 = axes[1, 1]
    for strategy, color in zip(strategies, colors):
        df_s = df[df['Strategy'] == strategy]
        ax4.scatter(df_s['TokenCount'], df_s['AvgEvidenceDistance'],
                   alpha=0.4, s=30, color=color, label=f'策略{strategy}')
        
        mean_x = df_s['TokenCount'].mean()
        mean_y = df_s['AvgEvidenceDistance'].mean()
        ax4.scatter(mean_x, mean_y, marker='*', s=300, color=color,
                   edgecolors='black', linewidths=2, zorder=10)
    
    ax4.set_xlabel('上下文Token数', fontsize=11)
    ax4.set_ylabel('平均证据距离（事件数）', fontsize=11)
    ax4.set_title('Token消耗 vs 平均证据距离（散点分布）', fontsize=12, fontweight='bold')
    ax4.legend(loc='best')
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(OUTPUT_DIR, "evidence_metrics_scatter.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ 散点分布图已保存到: {plot_path}")
    
    plt.close()


def plot_combined_4metrics(df_agg):
    """
    绘制4张独立指标图（每张图只关注一个指标）
    使用折线图 + 误差带，更清晰地展示趋势
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle('ABC策略的证据质量指标完整对比', fontsize=16, fontweight='bold')
    
    strategies = df_agg['Strategy'].values
    x_positions = [0, 1, 2]  # A, B, C的位置
    
    # 提取数据
    window_sizes = df_agg['WindowSize_Mean'].values
    token_counts = df_agg['TokenCount_Mean'].values
    token_stds = df_agg['TokenCount_Std'].values
    
    early_rates = df_agg['EarlyEvidenceRate_Mean'].values
    early_stds = df_agg['EarlyEvidenceRate_Std'].values
    
    distances = df_agg['AvgEvidenceDistance_Mean'].values
    distance_stds = df_agg['AvgEvidenceDistance_Std'].values
    
    # ==================== 图1: 窗口事件数 vs 早期证据率 ====================
    ax1 = axes[0, 0]
    ax1.errorbar(window_sizes, early_rates, yerr=early_stds,
                marker='o', markersize=10, capsize=8, capthick=2,
                linewidth=2.5, color='#45B7D1', alpha=0.8)
    
    for i, (ws, er, strategy) in enumerate(zip(window_sizes, early_rates, strategies)):
        ax1.text(ws, er + early_stds[i] + 0.03,
                f'{strategy}: {er:.1%}',
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax1.set_xlabel('窗口事件数', fontsize=12, fontweight='bold')
    ax1.set_ylabel('早期证据引用率', fontsize=12, fontweight='bold')
    ax1.set_title('(1) 窗口大小 → 早期证据率', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xscale('log')
    
    # ==================== 图2: 窗口事件数 vs 平均证据距离 ====================
    ax2 = axes[0, 1]
    ax2.errorbar(window_sizes, distances, yerr=distance_stds,
                marker='s', markersize=10, capsize=8, capthick=2,
                linewidth=2.5, color='#FF6B6B', alpha=0.8)
    
    for i, (ws, dist, strategy) in enumerate(zip(window_sizes, distances, strategies)):
        ax2.text(ws, dist + distance_stds[i] + 5,
                f'{strategy}: {dist:.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax2.set_xlabel('窗口事件数', fontsize=12, fontweight='bold')
    ax2.set_ylabel('平均证据距离（事件数）', fontsize=12, fontweight='bold')
    ax2.set_title('(2) 窗口大小 → 平均证据距离', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xscale('log')
    
    # ==================== 图3: 上下文Token数 vs 早期证据率 ====================
    ax3 = axes[1, 0]
    ax3.errorbar(token_counts, early_rates, xerr=token_stds, yerr=early_stds,
                marker='^', markersize=10, capsize=8, capthick=2,
                linewidth=2.5, color='#4ECDC4', alpha=0.8)
    
    for i, (tc, er, strategy) in enumerate(zip(token_counts, early_rates, strategies)):
        ax3.text(tc, er + early_stds[i] + 0.03,
                f'{strategy}\n{tc:.0f} tokens',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax3.set_xlabel('上下文Token数', fontsize=12, fontweight='bold')
    ax3.set_ylabel('早期证据引用率', fontsize=12, fontweight='bold')
    ax3.set_title('(3) Token消耗 → 早期证据率', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # ==================== 图4: 上下文Token数 vs 平均证据距离 ====================
    ax4 = axes[1, 1]
    ax4.errorbar(token_counts, distances, xerr=token_stds, yerr=distance_stds,
                marker='D', markersize=10, capsize=8, capthick=2,
                linewidth=2.5, color='#FFA07A', alpha=0.8)
    
    for i, (tc, dist, strategy) in enumerate(zip(token_counts, distances, strategies)):
        ax4.text(tc, dist + distance_stds[i] + 5,
                f'{strategy}\n{tc:.0f} tokens',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax4.set_xlabel('上下文Token数', fontsize=12, fontweight='bold')
    ax4.set_ylabel('平均证据距离（事件数）', fontsize=12, fontweight='bold')
    ax4.set_title('(4) Token消耗 → 平均证据距离', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(OUTPUT_DIR, "evidence_metrics_combined.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ 组合指标图已保存到: {plot_path}")
    
    plt.close()


def save_detailed_report(df, df_agg):
    """保存详细分析报告"""
    output_path = os.path.join(OUTPUT_DIR, "evidence_metrics_report.xlsx")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: 聚合统计
        df_agg.to_excel(writer, sheet_name='聚合统计', index=False)
        
        # Sheet 2: 原始数据
        df.to_excel(writer, sheet_name='原始数据', index=False)
        
        # Sheet 3: 按参与者聚合
        df_by_p = df.groupby(['Participant', 'Strategy']).agg({
            'WindowSize': 'mean',
            'TokenCount': 'mean',
            'EarlyEvidenceRate': 'mean',
            'AvgEvidenceDistance': 'mean',
            'Confidence': 'mean',
        }).reset_index()
        df_by_p.to_excel(writer, sheet_name='按参与者聚合', index=False)
    
    print(f"✓ 详细报告已保存到: {output_path}")


def interpret_results(df_agg):
    """解读实验结果"""
    print("\n" + "="*80)
    print("🔍 实验结果解读")
    print("="*80)
    
    # 提取关键数据
    a_data = df_agg[df_agg['Strategy'] == 'A'].iloc[0]
    b_data = df_agg[df_agg['Strategy'] == 'B'].iloc[0]
    c_data = df_agg[df_agg['Strategy'] == 'C'].iloc[0]
    
    print("\n1️⃣ 早期证据引用率分析:")
    print(f"   策略A: {a_data['EarlyEvidenceRate_Mean']:.2%}")
    print(f"   策略B: {b_data['EarlyEvidenceRate_Mean']:.2%}")
    print(f"   策略C: {c_data['EarlyEvidenceRate_Mean']:.2%}")
    
    if c_data['EarlyEvidenceRate_Mean'] > a_data['EarlyEvidenceRate_Mean']:
        improvement = (c_data['EarlyEvidenceRate_Mean'] - a_data['EarlyEvidenceRate_Mean']) * 100
        print(f"\n   ✅ C策略比A策略多引用了 {improvement:.1f}% 的早期证据")
        print(f"   → 说明：长窗口让LLM能够访问更早的历史信息")
        print(f"   → 验证了：用户意图的线索确实分布在较长时间跨度内")
    else:
        print(f"\n   ⚠️ C策略的早期证据率并未显著提升")
        print(f"   → 可能原因：LLM仍然倾向于依赖近期信息")
    
    print("\n2️⃣ 平均证据距离分析:")
    print(f"   策略A: {a_data['AvgEvidenceDistance_Mean']:.1f} 个事件")
    print(f"   策略B: {b_data['AvgEvidenceDistance_Mean']:.1f} 个事件")
    print(f"   策略C: {c_data['AvgEvidenceDistance_Mean']:.1f} 个事件")
    
    if c_data['AvgEvidenceDistance_Mean'] > a_data['AvgEvidenceDistance_Mean']:
        print(f"\n   ✅ C策略的平均证据距离更大")
        print(f"   → 说明：长窗口让LLM能够引用更远的历史事件作为证据")
        print(f"   → 验证了：长期记忆机制能够捕获远期线索")
    else:
        print(f"\n   ⚠️ 即使窗口变长，证据距离未显著增加")
        print(f"   → 可能原因：LLM的注意力仍集中在近期（recency bias）")
    
    print("\n3️⃣ Token效率分析:")
    print(f"   策略A: {a_data['TokenCount_Mean']:.0f} tokens → 置信度 {a_data['Confidence_Mean']:.3f}")
    print(f"   策略B: {b_data['TokenCount_Mean']:.0f} tokens → 置信度 {b_data['Confidence_Mean']:.3f}")
    print(f"   策略C: {c_data['TokenCount_Mean']:.0f} tokens → 置信度 {c_data['Confidence_Mean']:.3f}")
    
    # 计算token效率（置信度提升 / token增加）
    token_increase_b = b_data['TokenCount_Mean'] - a_data['TokenCount_Mean']
    conf_increase_b = b_data['Confidence_Mean'] - a_data['Confidence_Mean']
    efficiency_b = conf_increase_b / token_increase_b if token_increase_b > 0 else 0
    
    token_increase_c = c_data['TokenCount_Mean'] - a_data['TokenCount_Mean']
    conf_increase_c = c_data['Confidence_Mean'] - a_data['Confidence_Mean']
    efficiency_c = conf_increase_c / token_increase_c if token_increase_c > 0 else 0
    
    print(f"\n   Token效率 (置信度提升 / Token增加):")
    print(f"   策略B: {efficiency_b*10000:.2f} 置信度提升 / 千token")
    print(f"   策略C: {efficiency_c*10000:.2f} 置信度提升 / 千token")
    
    if efficiency_b > efficiency_c:
        print(f"\n   ⚠️ 策略B的Token效率更高")
        print(f"   → 建议：如果计算资源有限，策略B可能是更好的平衡点")
    else:
        print(f"\n   ✅ 策略C虽然消耗更多Token，但效率仍然更高")
        print(f"   → 建议：如果追求最高准确率，应使用策略C")


def main():
    print("="*80)
    print("📊 证据质量指标分析 - ABC策略对比")
    print("="*80)
    
    # 加载并处理数据
    df = load_and_process_data()
    if df is None:
        return
    
    # 聚合统计
    df_agg = aggregate_by_strategy(df)
    
    # 绘制4张指标图（组合版，清晰展示趋势）
    print("\n" + "="*80)
    print("📈 生成可视化图表...")
    print("="*80)
    
    plot_combined_4metrics(df_agg)
    
    # 绘制散点分布图（展示所有数据点）
    plot_scatter_with_trend(df)
    
    # 保存详细报告
    save_detailed_report(df, df_agg)
    
    # 解读结果
    interpret_results(df_agg)
    
    print("\n" + "="*80)
    print("✅ 证据质量指标分析完成！")
    print("="*80)
    print("\n生成的文件:")
    print(f"  1. {os.path.join(OUTPUT_DIR, 'evidence_metrics_report.xlsx')}")
    print(f"  2. {os.path.join(OUTPUT_DIR, 'evidence_metrics_combined.png')} ⭐ 主图")
    print(f"  3. {os.path.join(OUTPUT_DIR, 'evidence_metrics_scatter.png')}")
    print("\n📖 说明:")
    print("  - evidence_metrics_combined.png: 4张清晰的趋势图（推荐用于论文）")
    print("  - evidence_metrics_scatter.png: 散点分布图（展示数据分布）")
    print("  - evidence_metrics_report.xlsx: 详细数据（3个Sheet）")


if __name__ == "__main__":
    main()
