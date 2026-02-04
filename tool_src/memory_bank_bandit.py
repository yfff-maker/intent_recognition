"""
Multi-Armed Bandit based Memory Management
基于多臂老虎机的长期记忆动态管理

核心思想：
1. 每个chunk是一个arm
2. 检索=pull arm，获得reward（是否有用）
3. 使用UCB算法平衡探索与利用
4. Non-stationary：价值随时间衰减
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from event_representation import Event


@dataclass
class MemoryItemWithBandit:
    """
    带有Bandit特性的记忆项
    """
    # 原有属性（内容）
    chunk_id: str
    t_start: int
    t_end: int
    summary: str
    features: Dict[str, float]
    signature: Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]
    event_idx_range: Tuple[int, int]
    
    # Bandit相关属性（价值追踪）
    access_count: int = 0              # 被检索次数（pull次数）
    useful_count: int = 0              # 被采用次数（排名top-k且相似度>阈值）
    last_access_time: int = 0          # 最后访问时间（当前异常点timestamp）
    creation_time: int = 0             # 创建时间
    estimated_value: float = 0.0       # 估计价值
    confidence_bound: float = 0.0      # 置信上界（UCB）
    
    # 奖励历史（用于Non-stationary检测）
    reward_history: List[float] = field(default_factory=list)


def chunk_events(events: List[Event], chunk_size: int) -> List[List[Event]]:
    """将事件序列分块"""
    return [events[i : i + chunk_size] for i in range(0, len(events), chunk_size)]


def _top_items(counter: Counter, k: int = 5) -> List[Tuple[str, int]]:
    """返回Counter中频次最高的k项"""
    return [(x, int(c)) for x, c in counter.most_common(k)]


def summarize_chunk(chunk: List[Event], chunk_id: str, creation_time: int = 0) -> MemoryItemWithBandit:
    """
    将事件块摘要化为记忆项
    """
    if not chunk:
        raise ValueError("chunk must be non-empty")

    pages = Counter(e.page for e in chunk if e.page != "None")
    widgets = Counter(e.widget for e in chunk if e.widget != "None")
    ops = Counter(e.op for e in chunk if e.op != "None")

    t_start = chunk[0].t
    t_end = chunk[-1].t
    idx_min = min(e.idx for e in chunk)
    idx_max = max(e.idx for e in chunk)

    top_pages = _top_items(pages, 5)
    top_widgets = _top_items(widgets, 5)
    top_ops = _top_items(ops, 5)

    summary_lines = [
        f"[chunk {chunk_id}] t={t_start}->{t_end} idx={idx_min}->{idx_max}",
        f"- top_pages: {top_pages}",
        f"- top_widgets: {top_widgets}",
        f"- top_ops: {top_ops}",
    ]

    features = {
        "len": float(len(chunk)),
        "unique_pages": float(len(pages)),
        "unique_widgets": float(len(widgets)),
        "unique_ops": float(len(ops)),
    }

    signature = (
        tuple(sorted(set(pages.keys()))[:20]),
        tuple(sorted(set(widgets.keys()))[:20]),
        tuple(sorted(set(ops.keys()))[:20]),
    )

    return MemoryItemWithBandit(
        chunk_id=chunk_id,
        t_start=t_start,
        t_end=t_end,
        summary="\n".join(summary_lines),
        features=features,
        signature=signature,
        event_idx_range=(idx_min, idx_max),
        creation_time=creation_time or t_start,
        estimated_value=0.5,  # 初始中等价值
    )


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard相似度"""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def compute_temporal_decay(chunk: MemoryItemWithBandit, current_time: int, decay_half_life: int = 30000) -> float:
    """
    时间衰减函数（模拟Ebbinghaus遗忘曲线）
    
    Args:
        chunk: 记忆块
        current_time: 当前时间（异常点timestamp）
        decay_half_life: 半衰期（ms），默认30秒
    
    Returns:
        衰减因子 [0, 1]
    """
    time_since_access = current_time - chunk.last_access_time if chunk.last_access_time > 0 else (current_time - chunk.creation_time)
    
    # 强度参数：经常使用的chunk衰减慢
    strength = 1.0 + chunk.useful_count * 0.3
    adjusted_half_life = decay_half_life * strength
    
    # 指数衰减：decay = e^(-t*ln(2)/T_half)
    decay_factor = math.exp(-time_since_access * math.log(2) / adjusted_half_life)
    
    return max(0.01, decay_factor)  # 最低保留1%价值


def recompute_value(chunk: MemoryItemWithBandit, current_time: int) -> float:
    """
    重新计算chunk的估计价值
    
    综合考虑：
    1. 历史效用（被采用率）
    2. 时间衰减
    3. 信息量（稀有度）
    """
    # 1. 基础价值：历史采用率
    if chunk.access_count == 0:
        base_value = 0.5  # 未探索的chunk给中等价值
    else:
        base_value = chunk.useful_count / chunk.access_count
    
    # 2. 时间衰减
    decay = compute_temporal_decay(chunk, current_time)
    
    # 3. 信息量（unique widget数量越多，信息量越大）
    rarity_score = chunk.features.get('unique_widgets', 0) / 20.0  # 归一化到[0,1]
    rarity_score = min(1.0, rarity_score)
    
    # 综合计算
    value = base_value * 0.7 * decay + rarity_score * 0.3
    
    return value


class MemoryBankWithBandit:
    """
    基于多臂老虎机的记忆库
    
    特性：
    1. 动态价值追踪
    2. UCB算法平衡探索与利用
    3. 智能淘汰策略
    4. Non-stationary适应
    """

    def __init__(self, max_items: int, exploration_factor: float = 1.5):
        self.max_items = max_items
        self.exploration_factor = exploration_factor
        self.items: List[MemoryItemWithBandit] = []
        self.total_accesses: int = 0  # 全局pull次数

    def add(self, item: MemoryItemWithBandit) -> None:
        """添加记忆项，超容量时智能淘汰"""
        self.items.append(item)
        if len(self.items) > self.max_items:
            self._evict_with_bandit()

    def _evict_with_bandit(self) -> None:
        """
        基于多臂老虎机的智能淘汰
        
        策略：删除价值最低且置信度低的chunk
        （即：效用低且已被充分探索，不太可能逆袭的chunk）
        """
        if len(self.items) <= self.max_items:
            return
        
        current_time = max((item.last_access_time for item in self.items), default=0)
        
        scored_items = []
        for item in self.items:
            # 重新计算价值
            value = recompute_value(item, current_time)
            
            # 计算置信区间（被充分探索的chunk置信区间小）
            if item.access_count == 0:
                confidence = float('inf')  # 未探索的不删除
            else:
                confidence = self.exploration_factor * math.sqrt(
                    math.log(self.total_accesses + 1) / item.access_count
                )
            
            # 保留分数 = 价值 + 置信度（分数越高越应保留）
            retention_score = value + confidence
            
            scored_items.append((retention_score, item))
        
        # 按保留分数排序，删除分数最低的
        scored_items.sort(key=lambda x: x[0])
        
        num_to_remove = len(self.items) - self.max_items
        for i in range(num_to_remove):
            removed = scored_items[i][1]
            self.items.remove(removed)
            print(f"    [Bandit淘汰] {removed.chunk_id} (retention_score={scored_items[i][0]:.3f}, "
                  f"access={removed.access_count}, useful={removed.useful_count}, "
                  f"value={removed.estimated_value:.3f})")

    def retrieve_with_feedback(
        self,
        query_pages: Set[str],
        query_widgets: Set[str],
        query_ops: Set[str],
        current_time: int,
        top_k: int,
        similarity_threshold: float = 0.1
    ) -> List[MemoryItemWithBandit]:
        """
        检索并提供反馈（核心Bandit交互）
        
        Args:
            query_pages/widgets/ops: 查询特征
            current_time: 当前时间（用于时间衰减）
            top_k: 返回数量
            similarity_threshold: 相似度阈值（高于此值才算有用）
        
        Returns:
            top-k相关的chunk列表
        """
        scored: List[Tuple[float, MemoryItemWithBandit]] = []
        
        for item in self.items:
            # 计算相似度（原有逻辑）
            pages = set(item.signature[0])
            widgets = set(item.signature[1])
            ops = set(item.signature[2])
            sim = (
                0.5 * _jaccard(query_widgets, widgets) +
                0.3 * _jaccard(query_pages, pages) +
                0.2 * _jaccard(query_ops, ops)
            )
            
            # 记录访问（Pull arm）
            item.access_count += 1
            item.last_access_time = current_time
            self.total_accesses += 1
            
            scored.append((sim, item))
        
        # 排序
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # 返回top-k，并更新价值
        results = []
        for rank, (sim, item) in enumerate(scored):
            if rank < top_k and sim > similarity_threshold:
                # Reward +1：被采用且相关
                item.useful_count += 1
                item.reward_history.append(1.0)
                results.append(item)
            elif rank < top_k:
                # Reward 0：被采用但不太相关
                item.reward_history.append(0.0)
                results.append(item)
            else:
                # Reward -0.1：被检索但未被采用（轻微惩罚）
                item.reward_history.append(-0.1)
        
        # 重新计算所有chunk的估计价值
        for item in self.items:
            item.estimated_value = recompute_value(item, current_time)
            
            # 计算UCB上界
            if item.access_count > 0:
                item.confidence_bound = self.exploration_factor * math.sqrt(
                    math.log(self.total_accesses + 1) / item.access_count
                )
        
        return results

    def retrieve(
        self,
        query_pages: Set[str],
        query_widgets: Set[str],
        query_ops: Set[str],
        top_k: int
    ) -> List[MemoryItemWithBandit]:
        """
        兼容原有API的检索方法（无时间参数）
        """
        current_time = max((item.t_end for item in self.items), default=0)
        return self.retrieve_with_feedback(
            query_pages, query_widgets, query_ops, current_time, top_k
        )

    def get_statistics(self) -> Dict:
        """获取记忆库统计信息（用于调试和可视化）"""
        if not self.items:
            return {}
        
        return {
            "total_chunks": len(self.items),
            "total_accesses": self.total_accesses,
            "avg_access_count": sum(item.access_count for item in self.items) / len(self.items),
            "avg_useful_count": sum(item.useful_count for item in self.items) / len(self.items),
            "avg_estimated_value": sum(item.estimated_value for item in self.items) / len(self.items),
            "top_5_valuable_chunks": sorted(
                [(item.chunk_id, item.estimated_value, item.useful_count) for item in self.items],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }

    def promote_stm_to_ltm(
        self,
        stm_window: List[Event],
        chunk_id: str,
        current_time: int,
        initial_value: float = 0.7
    ) -> None:
        """
        将STM窗口提升为LTM chunk
        
        触发条件（在main.py中判断）：
        1. LLM输出高置信度（>0.8）
        2. STM包含稀有操作
        3. STM在多个异常点被复用
        """
        new_chunk = summarize_chunk(stm_window, chunk_id, creation_time=current_time)
        new_chunk.estimated_value = initial_value  # 新提升的chunk给较高初始价值
        new_chunk.useful_count = 1  # 标记为已被使用
        
        self.add(new_chunk)
        print(f"    [STM→LTM] 提升窗口到长期记忆: {chunk_id} (初始价值={initial_value:.2f})")
