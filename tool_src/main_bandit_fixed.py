"""
修复版本：LTM动态增长，避免数据泄露

关键修复：
1. LTM从空开始
2. 按时间顺序处理异常点
3. 每次只使用当前时间点之前的信息构建LTM
4. 逐步将过去的事件chunking并加入LTM
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from anomaly_detector import AnomalyDetector
from config import *
from context_builder import find_nearest_event_idx, find_nearest_key_event_pos, normalize_behavior_sequence
from data_loader import DataLoader
from intent_prompting import build_intent_prompt, parse_intent_output
from key_event_selector import select_key_events
from llm_client import LLMClient
from memory_bank_bandit import MemoryBankWithBandit, chunk_events, summarize_chunk
from window_and_compress import build_window, compress_events, format_events_for_prompt


def should_add_new_ltm_chunk(
    mb: MemoryBankWithBandit,
    past_key_events: List,
    next_chunk_start_idx: int,
    chunk_size: int
) -> bool:
    """
    判断是否应该将新的chunk加入LTM
    
    策略：
    - 如果有足够的新事件（>=chunk_size），就创建新chunk
    - 或者如果是第一个异常点且LTM为空，也创建初始chunk
    """
    available_events = len(past_key_events) - next_chunk_start_idx
    
    if available_events >= chunk_size:
        return True
    
    # 如果LTM为空且有一些事件，也可以创建（即使不够chunk_size）
    if len(mb.items) == 0 and available_events > 0:
        return True
    
    return False


def main():
    loader = DataLoader(DATA_DIR)
    detector = AnomalyDetector(config={})
    llm = LLMClient(api_key=OPENROUTER_API_KEY, model=LLM_MODEL)

    participants = loader.list_participants()
    
    all_rows = []
    all_stats = []
    
    # Output file paths - 使用fixed后缀
    csv_output_path = os.path.join(OUTPUT_DIR, "intent_inference_results_bandit_fixed.csv" if LLM_TASK == "INTENT" else "inferred_requirements_bandit_fixed.csv")
    xlsx_output_path = os.path.join(OUTPUT_DIR, "intent_inference_results_bandit_fixed.xlsx" if LLM_TASK == "INTENT" else "inferred_requirements_bandit_fixed.xlsx")
    stats_output_path = os.path.join(OUTPUT_DIR, "memory_bank_statistics_fixed.xlsx")
    
    # Remove old CSV if exists (fresh start)
    if os.path.exists(csv_output_path):
        os.remove(csv_output_path)

    for p_id in participants:
        print(f"Processing Participant: {p_id}")

        # 1. Load Data
        raw_seq = loader.load_behavior_sequence(p_id)

        if not raw_seq:
            continue

        # Step 0: unify representation
        events = normalize_behavior_sequence(raw_seq)

        if not events:
            continue

        # 2. Detect Anomalies (anchors)
        anomalies = detector.detect_anomalies(raw_seq)
        print(f"  Found {len(anomalies)} anomalies.")

        # Step 1: key event selection (token control)
        key_events = select_key_events(
            events,
            target_k=KEY_EVENT_TARGET_K,
            num_bins=KEY_EVENT_NUM_BINS,
            top_m_per_bin=KEY_EVENT_TOP_M_PER_BIN,
            near_dt_ms=KEY_EVENT_NEAR_DT_MS,
        )
        
        print(f"  Selected {len(key_events)} key events from {len(events)} raw events")

        # ✅ 修复：LTM从空开始
        mb = MemoryBankWithBandit(max_items=MEMORY_MAX_ITEMS, exploration_factor=1.5)
        ltm_chunk_counter = 0  # 跟踪已经加入LTM的chunk数量
        
        print(f"  ✅ LTM initialized as EMPTY (will grow dynamically)")
        
        # ✅ 修复：按时间顺序处理异常点
        anomalies_sorted = sorted(anomalies, key=lambda a: int(a.get("timestamp", 0)))
        print(f"  ✅ Anomalies sorted by timestamp ({len(anomalies_sorted)} total)")

        for anomaly_idx, anomaly in enumerate(anomalies_sorted, 1):
            task_info = TASK_DEFINITIONS["Task1"]
            timestamp = int(anomaly.get("timestamp", 0))
            
            print(f"\n  --- Anomaly {anomaly_idx}/{len(anomalies_sorted)}: t={timestamp}ms ---")

            # Find center event around anomaly timestamp
            center_pos = find_nearest_event_idx(events, timestamp)
            if center_pos is None:
                continue
            center_event = events[center_pos]

            # Map to key_event position for controllable windows
            key_center_pos = find_nearest_key_event_pos(key_events, center_event)
            if key_center_pos is None:
                continue

            # ✅ 修复：只使用当前时间点之前的key_events
            past_key_events = [e for e in key_events if e.t < timestamp]
            print(f"    Past key events: {len(past_key_events)} (out of {len(key_events)} total)")
            
            # ✅ 修复：动态更新LTM（只包含过去的信息）
            # 检查是否有新的过去事件需要加入LTM
            while should_add_new_ltm_chunk(mb, past_key_events, ltm_chunk_counter * MEMORY_CHUNK_SIZE, MEMORY_CHUNK_SIZE):
                start_idx = ltm_chunk_counter * MEMORY_CHUNK_SIZE
                end_idx = min(start_idx + MEMORY_CHUNK_SIZE, len(past_key_events))
                
                new_chunk = past_key_events[start_idx:end_idx]
                
                if not new_chunk:
                    break
                
                creation_time = new_chunk[0].t if new_chunk else 0
                item = summarize_chunk(new_chunk, chunk_id=f"{p_id}_{ltm_chunk_counter}", creation_time=creation_time)
                mb.add(item)
                
                print(f"    ✅ Added LTM chunk {p_id}_{ltm_chunk_counter}: {len(new_chunk)} events (t={new_chunk[0].t}-{new_chunk[-1].t})")
                ltm_chunk_counter += 1
            
            print(f"    Current LTM size: {len(mb.items)} chunks")

            # Build retrieval query sets from local context
            query_pages = {center_event.page} if center_event.page != "None" else set()
            query_widgets = {center_event.widget} if center_event.widget != "None" else set()
            query_ops = {center_event.op} if center_event.op != "None" else set()
            
            # 检索LTM（此时只包含过去的信息）
            ltm_items = mb.retrieve_with_feedback(
                query_pages, query_widgets, query_ops, 
                current_time=timestamp,
                top_k=MEMORY_RETRIEVE_TOP_K
            )
            
            print(f"    Retrieved {len(ltm_items)} LTM chunks (from {len(mb.items)} available)")

            # For each strategy A/B/C: build window → compress → prompt → infer → parse → store
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

                if LLM_TASK == "INTENT":
                    prompt = build_intent_prompt(
                        task_info=task_info,
                        anomaly=anomaly,
                        strategy=strategy,
                        stm_events_text=stm_text,
                        ltm_items=ltm_items,
                        intent_labels=INTENT_LABELS,
                    )
                    response_text = llm.infer_intent(prompt)
                    parsed = parse_intent_output(response_text)
                    all_rows.append(
                        {
                            "Participant": p_id,
                            "AnchorTimestamp": timestamp,
                            "AnomalyType": anomaly.get("type"),
                            "Strategy": strategy,
                            "Intent": parsed.get("intent"),
                            "Confidence": parsed.get("confidence"),
                            "Reasoning": parsed.get("reasoning", ""),
                            "Evidence": str(parsed.get("evidence")),
                            "Notes": parsed.get("notes", ""),
                            "LTM_Chunks_Available": len(mb.items),  # ✅ 新增：记录当时有多少LTM chunk
                            "LTM_Chunks_Retrieved": len(ltm_items),  # ✅ 新增：记录检索了多少
                            "Prompt": prompt,
                            "RawResponse": response_text,
                        }
                    )
                    
                    # 如果LLM推理置信度高，考虑将STM提升到LTM
                    if parsed.get("confidence", 0) > 0.8 and strategy == "C":
                        mb.promote_stm_to_ltm(
                            stm_window=win,
                            chunk_id=f"{p_id}_promoted_{timestamp}",
                            current_time=timestamp,
                            initial_value=0.7
                        )
                        print(f"    ✅ Promoted STM to LTM (confidence={parsed.get('confidence'):.2f})")
                else:
                    # Fallback
                    prompt = f"[NO_VIDEO] {anomaly.get('description')}\n\n{stm_text}"
                    response_text = llm.infer_requirements(prompt)
                    all_rows.append(
                        {
                            "Participant": p_id,
                            "Timestamp": timestamp,
                            "Anomaly Type": anomaly.get("type"),
                            "Strategy": strategy,
                            "LLM Response": response_text,
                        }
                    )
        
        # 记录最终的记忆库统计
        for item in mb.items:
            all_stats.append({
                "Participant": p_id,
                "ChunkID": item.chunk_id,
                "TimeStart": item.t_start,
                "TimeEnd": item.t_end,
                "AccessCount": item.access_count,
                "UsefulCount": item.useful_count,
                "EstimatedValue": item.estimated_value,
                "RewardHistory": str(item.reward_history[-5:]) if item.reward_history else "[]"
            })
        
        # Immediately save this participant's results to CSV
        if all_rows:
            df_incremental = pd.DataFrame(all_rows)
            write_header = not os.path.exists(csv_output_path)
            df_incremental.to_csv(csv_output_path, mode='a', index=False, header=write_header, encoding='utf-8-sig')
            print(f"\n  ✓ {p_id} 的 {len(all_rows)} 条结果已保存到 {csv_output_path}")
            all_rows = []

    # Final: export to Excel
    if os.path.exists(csv_output_path):
        df_final = pd.read_csv(csv_output_path, encoding='utf-8-sig')
        df_final.to_excel(xlsx_output_path, index=False, engine='openpyxl')
        print(f"\n✅ All results saved to {xlsx_output_path}")
    
    # Save memory statistics
    if all_stats:
        df_stats = pd.DataFrame(all_stats)
        df_stats.to_excel(stats_output_path, index=False, engine='openpyxl')
        print(f"✅ Memory bank statistics saved to {stats_output_path}")


if __name__ == "__main__":
    main()
