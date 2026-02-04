"""
改进的LTM检索策略：结合相似度和时间距离

解决问题：
1. 纯相似度检索忽视了时间连续性
2. 时间上相邻的chunk可能更有因果关系
3. 太久远的相似chunk可能已经不相关
"""

from __future__ import annotations

import math
from typing import List, Set, Tuple
from memory_bank_bandit import MemoryItemWithBandit, _jaccard


def retrieve_with_temporal_awareness(
    items: List[MemoryItemWithBandit],
    query_pages: Set[str],
    query_widgets: Set[str],
    query_ops: Set[str],
    current_time: int,
    top_k: int = 5,
    temporal_weight: float = 0.3,  # 时间权重
    similarity_weight: float = 0.7,  # 相似度权重
) -> List[MemoryItemWithBandit]:
    """
    改进的检索策略：结合相似度和时间距离
    
    最终分数 = similarity_weight × 相似度 + temporal_weight × 时间分数
    
    Args:
        temporal_weight: 时间距离的权重（0-1）
        similarity_weight: 相似度的权重（0-1）
    """
    scored: List[Tuple[float, MemoryItemWithBandit]] = []
    
    for item in items:
        # 1. 计算相似度（原有逻辑）
        pages = set(item.signature[0])
        widgets = set(item.signature[1])
        ops = set(item.signature[2])
        content_sim = (
            0.5 * _jaccard(query_widgets, widgets) +
            0.3 * _jaccard(query_pages, pages) +
            0.2 * _jaccard(query_ops, ops)
        )
        
        # 2. 计算时间分数（时间越近分数越高）
        time_diff = abs(current_time - item.t_end)  # 距离chunk结束时间
        
        # 使用指数衰减：近期的chunk分数高
        # half_life = 5分钟 = 300000ms
        temporal_score = math.exp(-time_diff / 300000)
        
        # 3. 综合分数
        final_score = (
            similarity_weight * content_sim +
            temporal_weight * temporal_score
        )
        
        scored.append((final_score, item, content_sim, temporal_score))
    
    # 排序并返回top-k
    scored.sort(key=lambda x: x[0], reverse=True)
    
    return [item for _, item, _, _ in scored[:top_k]]


def retrieve_hybrid_strategy(
    items: List[MemoryItemWithBandit],
    query_pages: Set[str],
    query_widgets: Set[str],
    query_ops: Set[str],
    current_time: int,
    top_k: int = 5,
) -> List[MemoryItemWithBandit]:
    """
    混合策略：部分按相似度，部分按时间邻近
    
    策略：
    - Top-3 按相似度（找语义相关的）
    - Top-2 按时间邻近（保证连续性）
    """
    # 1. 按相似度排序
    similarity_scored = []
    for item in items:
        pages = set(item.signature[0])
        widgets = set(item.signature[1])
        ops = set(item.signature[2])
        sim = (
            0.5 * _jaccard(query_widgets, widgets) +
            0.3 * _jaccard(query_pages, pages) +
            0.2 * _jaccard(query_ops, ops)
        )
        similarity_scored.append((sim, item))
    
    similarity_scored.sort(key=lambda x: x[0], reverse=True)
    top_similar = [item for _, item in similarity_scored[:3]]
    
    # 2. 按时间距离排序（找最近的）
    temporal_scored = []
    for item in items:
        time_diff = abs(current_time - item.t_end)
        temporal_scored.append((time_diff, item))
    
    temporal_scored.sort(key=lambda x: x[0])  # 时间差越小越好
    top_recent = [item for _, item in temporal_scored[:2]]
    
    # 3. 合并并去重
    result = []
    seen_ids = set()
    
    for item in top_similar + top_recent:
        if item.chunk_id not in seen_ids:
            result.append(item)
            seen_ids.add(item.chunk_id)
        if len(result) >= top_k:
            break
    
    return result


def retrieve_with_temporal_window(
    items: List[MemoryItemWithBandit],
    query_pages: Set[str],
    query_widgets: Set[str],
    query_ops: Set[str],
    current_time: int,
    top_k: int = 5,
    time_window_ms: int = 600000,  # 只在最近10分钟内检索
) -> List[MemoryItemWithBandit]:
    """
    时间窗口约束策略：只检索最近N分钟内的chunk
    
    优势：
    - 避免检索到太久远的chunk
    - 保证检索结果的时间相关性
    """
    # 1. 筛选时间窗口内的chunk
    candidates = [
        item for item in items
        if (current_time - item.t_end) <= time_window_ms
    ]
    
    if not candidates:
        # 如果时间窗口内没有chunk，回退到全局检索
        candidates = items
    
    # 2. 在候选集中按相似度检索
    scored = []
    for item in candidates:
        pages = set(item.signature[0])
        widgets = set(item.signature[1])
        ops = set(item.signature[2])
        sim = (
            0.5 * _jaccard(query_widgets, widgets) +
            0.3 * _jaccard(query_pages, pages) +
            0.2 * _jaccard(query_ops, ops)
        )
        scored.append((sim, item))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    
    return [item for _, item in scored[:top_k]]


def retrieve_causal_chain(
    items: List[MemoryItemWithBandit],
    query_pages: Set[str],
    query_widgets: Set[str],
    query_ops: Set[str],
    current_time: int,
    top_k: int = 5,
) -> List[MemoryItemWithBandit]:
    """
    因果链检索策略：优先检索时间连续的chunk
    
    策略：
    1. 找到最相似的anchor chunk
    2. 包含anchor的前后邻居chunk（保证因果连续性）
    """
    # 1. 找到最相似的chunk作为anchor
    scored = []
    for item in items:
        pages = set(item.signature[0])
        widgets = set(item.signature[1])
        ops = set(item.signature[2])
        sim = (
            0.5 * _jaccard(query_widgets, widgets) +
            0.3 * _jaccard(query_pages, pages) +
            0.2 * _jaccard(query_ops, ops)
        )
        scored.append((sim, item))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    anchor = scored[0][1]  # 最相似的chunk
    
    # 2. 找到anchor的索引
    sorted_items = sorted(items, key=lambda x: x.t_start)
    try:
        anchor_idx = sorted_items.index(anchor)
    except ValueError:
        return [item for _, item in scored[:top_k]]
    
    # 3. 包含anchor及其前后邻居
    result = []
    
    # 前2个
    for i in range(max(0, anchor_idx - 2), anchor_idx):
        result.append(sorted_items[i])
    
    # anchor自己
    result.append(anchor)
    
    # 后2个
    for i in range(anchor_idx + 1, min(len(sorted_items), anchor_idx + 3)):
        result.append(sorted_items[i])
    
    return result[:top_k]


# ============= 对比实验函数 =============

def compare_retrieval_strategies(
    items: List[MemoryItemWithBandit],
    query_pages: Set[str],
    query_widgets: Set[str],
    query_ops: Set[str],
    current_time: int,
    top_k: int = 5,
):
    """
    对比不同检索策略的结果
    """
    print("=" * 80)
    print("🔍 检索策略对比")
    print("=" * 80)
    
    # 策略1：原始相似度检索
    print("\n📌 策略1: 纯相似度检索（当前方法）")
    scored = []
    for item in items:
        pages = set(item.signature[0])
        widgets = set(item.signature[1])
        ops = set(item.signature[2])
        sim = (
            0.5 * _jaccard(query_widgets, widgets) +
            0.3 * _jaccard(query_pages, pages) +
            0.2 * _jaccard(query_ops, ops)
        )
        scored.append((sim, item))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    print("Top-5 chunks:")
    for rank, (sim, item) in enumerate(scored[:top_k], 1):
        time_diff = (current_time - item.t_end) / 1000  # 转换为秒
        print(f"  {rank}. {item.chunk_id}: sim={sim:.3f}, "
              f"时间距离={time_diff:.1f}秒 ({time_diff/60:.1f}分钟)")
    
    # 策略2：时间感知检索
    print("\n📌 策略2: 时间感知检索（相似度70% + 时间30%）")
    result2 = retrieve_with_temporal_awareness(
        items, query_pages, query_widgets, query_ops, current_time, top_k
    )
    for rank, item in enumerate(result2, 1):
        time_diff = (current_time - item.t_end) / 1000
        print(f"  {rank}. {item.chunk_id}: "
              f"时间距离={time_diff:.1f}秒 ({time_diff/60:.1f}分钟)")
    
    # 策略3：混合策略
    print("\n📌 策略3: 混合策略（3个相似 + 2个最近）")
    result3 = retrieve_hybrid_strategy(
        items, query_pages, query_widgets, query_ops, current_time, top_k
    )
    for rank, item in enumerate(result3, 1):
        time_diff = (current_time - item.t_end) / 1000
        print(f"  {rank}. {item.chunk_id}: "
              f"时间距离={time_diff:.1f}秒 ({time_diff/60:.1f}分钟)")
    
    # 策略4：时间窗口约束
    print("\n📌 策略4: 时间窗口约束（只看最近10分钟）")
    result4 = retrieve_with_temporal_window(
        items, query_pages, query_widgets, query_ops, current_time, top_k
    )
    for rank, item in enumerate(result4, 1):
        time_diff = (current_time - item.t_end) / 1000
        print(f"  {rank}. {item.chunk_id}: "
              f"时间距离={time_diff:.1f}秒 ({time_diff/60:.1f}分钟)")
    
    # 策略5：因果链检索
    print("\n📌 策略5: 因果链检索（找相似的+前后邻居）")
    result5 = retrieve_causal_chain(
        items, query_pages, query_widgets, query_ops, current_time, top_k
    )
    for rank, item in enumerate(result5, 1):
        time_diff = (current_time - item.t_end) / 1000
        print(f"  {rank}. {item.chunk_id}: "
              f"时间距离={time_diff:.1f}秒 ({time_diff/60:.1f}分钟)")
    
    print("\n" + "=" * 80)
