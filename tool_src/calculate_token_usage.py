"""
计算Token使用量

用途：估算单个参与者完整流程的Token消耗
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import DataLoader
from anomaly_detector import AnomalyDetector
from config import *
from context_builder import normalize_behavior_sequence
from key_event_selector import select_key_events
from memory_bank import MemoryBank, chunk_events, summarize_chunk
from window_and_compress import build_window, compress_events, format_events_for_prompt
from intent_prompting import build_intent_prompt
from context_builder import find_nearest_event_idx, find_nearest_key_event_pos


def estimate_tokens(text: str) -> int:
    """
    估算Token数量
    
    经验规则：
    - 英文: ~4 chars/token
    - 中文: ~2 chars/token
    - 数字/符号: ~1 char/token
    
    简化估算：取平均值 ~3 chars/token
    """
    return len(text) // 3


def analyze_participant_tokens(p_id: str = "P1"):
    """分析单个参与者的Token使用情况"""
    
    print("=" * 80)
    print(f"🔍 Token使用量分析: {p_id}")
    print("=" * 80)
    
    loader = DataLoader(DATA_DIR)
    detector = AnomalyDetector(config={})
    
    # 1. 加载数据
    raw_seq = loader.load_behavior_sequence(p_id)
    events = normalize_behavior_sequence(raw_seq)
    
    print(f"\n📊 原始数据:")
    print(f"  原始事件数: {len(events)}")
    
    # 估算原始事件格式化后的Token数
    if events:
        sample_event = events[0]
        sample_text = f"- idx={sample_event.idx} t={sample_event.t}->{sample_event.t} page={sample_event.page} widget={sample_event.widget} op={sample_event.op} count=1"
        tokens_per_event = estimate_tokens(sample_text)
        total_raw_tokens = tokens_per_event * len(events)
        print(f"  单个事件Token数: ~{tokens_per_event}")
        print(f"  全部原始事件Token数: ~{total_raw_tokens:,} tokens ({total_raw_tokens/1000:.1f}k)")
    
    # 2. 检测异常
    anomalies = detector.detect_anomalies(raw_seq)
    print(f"\n🔍 异常点:")
    print(f"  检测到异常数: {len(anomalies)}")
    
    # 3. 选择关键事件
    key_events = select_key_events(
        events,
        target_k=KEY_EVENT_TARGET_K,
        num_bins=KEY_EVENT_NUM_BINS,
        top_m_per_bin=KEY_EVENT_TOP_M_PER_BIN,
        near_dt_ms=KEY_EVENT_NEAR_DT_MS,
    )
    print(f"\n⭐ 关键事件:")
    print(f"  选择的关键事件数: {len(key_events)}")
    
    # 4. 构建LTM
    mb = MemoryBank(max_items=MEMORY_MAX_ITEMS)
    chunks = chunk_events(key_events, MEMORY_CHUNK_SIZE)
    print(f"\n🧠 长期记忆:")
    print(f"  LTM chunks数: {len(chunks)}")
    
    ltm_total_tokens = 0
    for ci, ch in enumerate(chunks):
        item = summarize_chunk(ch, chunk_id=f"{p_id}_{ci}")
        mb.add(item)
        ltm_total_tokens += estimate_tokens(item.summary)
    
    print(f"  全部LTM摘要Token数: ~{ltm_total_tokens:,} tokens ({ltm_total_tokens/1000:.1f}k)")
    
    # 5. 模拟单次推理的Token使用
    if anomalies:
        anomaly = anomalies[0]
        timestamp = int(anomaly.get("timestamp", 0))
        task_info = TASK_DEFINITIONS["Task1"]
        
        center_pos = find_nearest_event_idx(events, timestamp)
        if center_pos is not None:
            center_event = events[center_pos]
            key_center_pos = find_nearest_key_event_pos(key_events, center_event)
            
            if key_center_pos is not None:
                print(f"\n📝 单次推理Token分析 (第1个异常点):")
                print(f"  异常时间: {timestamp}ms")
                
                # 检索LTM
                query_pages = {center_event.page} if center_event.page != "None" else set()
                query_widgets = {center_event.widget} if center_event.widget != "None" else set()
                query_ops = {center_event.op} if center_event.op != "None" else set()
                ltm_items = mb.retrieve(query_pages, query_widgets, query_ops, top_k=MEMORY_RETRIEVE_TOP_K)
                
                ltm_text_tokens = sum(estimate_tokens(it.summary) for it in ltm_items)
                print(f"  检索到的LTM Token数: ~{ltm_text_tokens:,} tokens")
                
                # 测试3种策略
                for strategy in ["A", "B", "C"]:
                    win = build_window(
                        key_events=key_events,
                        center_pos=key_center_pos,
                        mode=strategy,
                        window_mode=WINDOW_MODE,
                        strategy_windows=STRATEGY_WINDOWS,
                    )
                    compressed = compress_events(win, merge_consecutive=COMPRESS_MERGE_CONSECUTIVE)
                    stm_text = format_events_for_prompt(compressed, max_lines=PROMPT_MAX_EVENT_LINES)
                    
                    prompt = build_intent_prompt(
                        task_info=task_info,
                        anomaly=anomaly,
                        strategy=strategy,
                        stm_events_text=stm_text,
                        ltm_items=ltm_items,
                        intent_labels=INTENT_LABELS,
                    )
                    
                    prompt_tokens = estimate_tokens(prompt)
                    stm_tokens = estimate_tokens(stm_text)
                    
                    print(f"\n  策略{strategy}:")
                    print(f"    STM窗口事件数: {len(win)}")
                    print(f"    STM Token数: ~{stm_tokens:,} tokens")
                    print(f"    完整Prompt Token数: ~{prompt_tokens:,} tokens ({prompt_tokens/1000:.1f}k)")
    
    # 6. 估算全流程Token使用
    print(f"\n" + "=" * 80)
    print(f"📊 全流程Token估算:")
    print("=" * 80)
    
    if anomalies and center_pos is not None and key_center_pos is not None:
        # 使用策略C的Token数作为代表
        win_c = build_window(
            key_events=key_events,
            center_pos=key_center_pos,
            mode="C",
            window_mode=WINDOW_MODE,
            strategy_windows=STRATEGY_WINDOWS,
        )
        compressed_c = compress_events(win_c, merge_consecutive=COMPRESS_MERGE_CONSECUTIVE)
        stm_text_c = format_events_for_prompt(compressed_c, max_lines=PROMPT_MAX_EVENT_LINES)
        prompt_c = build_intent_prompt(
            task_info=task_info,
            anomaly=anomaly,
            strategy="C",
            stm_events_text=stm_text_c,
            ltm_items=ltm_items,
            intent_labels=INTENT_LABELS,
        )
        
        avg_tokens_per_inference = estimate_tokens(prompt_c)
        num_inferences = len(anomalies) * 3  # 每个异常点3个策略
        total_tokens = avg_tokens_per_inference * num_inferences
        
        print(f"\n每次推理平均Token数（策略C）: ~{avg_tokens_per_inference:,} tokens")
        print(f"异常点数: {len(anomalies)}")
        print(f"推理次数（每个异常×3策略）: {num_inferences}")
        print(f"\n✨ 总Token消耗估算: ~{total_tokens:,} tokens ({total_tokens/1000:.1f}k)")
        
        if total_tokens > 200000:
            print(f"\n⚠️  注意：Token数超过20万！建议:")
            print(f"  1. 减少异常点数量")
            print(f"  2. 减少KEY_EVENT_TARGET_K")
            print(f"  3. 减小STRATEGY_WINDOWS['C']的窗口大小")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # 可以指定参与者ID
    import sys
    p_id = sys.argv[1] if len(sys.argv) > 1 else "P1"
    analyze_participant_tokens(p_id)
