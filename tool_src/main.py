import os
import pandas as pd
from config import (
    DATASET_ROOT,
    OUTPUT_DIR,
    LLM_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_SITE_URL,
    OPENROUTER_APP_NAME,
    TASK_DEFINITIONS,
    LLM_INTERACTION_MODE,
    LLM_TASK,
    WINDOW_MODE,
    STRATEGY_WINDOWS,
    KEY_EVENT_TARGET_K,
    KEY_EVENT_NUM_BINS,
    KEY_EVENT_TOP_M_PER_BIN,
    KEY_EVENT_NEAR_DT_MS,
    COMPRESS_MERGE_CONSECUTIVE,
    PROMPT_MAX_EVENT_LINES,
    MEMORY_CHUNK_SIZE,
    MEMORY_MAX_ITEMS,
    MEMORY_RETRIEVE_TOP_K,
    INTENT_LABELS,
)
from data_loader import DataLoader
from anomaly_detector import AnomalyDetector
from llm_client import LLMClient
from event_representation import normalize_behavior_sequence, find_nearest_event_idx
from key_event_selector import select_key_events
from window_and_compress import (
    build_window,
    compress_events,
    find_nearest_key_event_pos,
    format_events_for_prompt,
)
from memory_bank import MemoryBank, chunk_events, summarize_chunk
from intent_prompting import build_intent_prompt, parse_intent_output


def main():
    # Initialize components
    loader = DataLoader(DATASET_ROOT)
    detector = AnomalyDetector()
    llm = LLMClient(
        api_key=OPENROUTER_API_KEY,
        model=LLM_MODEL,
        base_url=OPENROUTER_BASE_URL,
        extra_headers={
            "HTTP-Referer": OPENROUTER_SITE_URL,
            "X-Title": OPENROUTER_APP_NAME,
        },
    )

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    participants = loader.get_participants()

    all_rows = []
    
    # Output file paths (with reasoning)
    csv_output_path = os.path.join(OUTPUT_DIR, "intent_inference_results_with_reasoning.csv" if LLM_TASK == "INTENT" else "inferred_requirements_with_reasoning.csv")
    xlsx_output_path = os.path.join(OUTPUT_DIR, "intent_inference_results_with_reasoning.xlsx" if LLM_TASK == "INTENT" else "inferred_requirements_with_reasoning.xlsx")
    
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

        # Step 4-6: build LTM memory bank from key events
        mb = MemoryBank(max_items=MEMORY_MAX_ITEMS)
        chunks = chunk_events(key_events, MEMORY_CHUNK_SIZE)
        for ci, ch in enumerate(chunks):
            item = summarize_chunk(ch, chunk_id=f"{p_id}_{ci}")
            mb.add(item)

        for anomaly in anomalies:
            # Determine task context (Simplified logic: assume Task1 for demo)
            task_info = TASK_DEFINITIONS["Task1"]
            timestamp = int(anomaly.get("timestamp", 0))

            # Find center event around anomaly timestamp
            center_pos = find_nearest_event_idx(events, timestamp)
            if center_pos is None:
                continue
            center_event = events[center_pos]

            # Map to key_event position for controllable windows
            key_center_pos = find_nearest_key_event_pos(key_events, center_event)
            if key_center_pos is None:
                continue

            # Build retrieval query sets from local context (cheap)
            query_pages = {center_event.page} if center_event.page != "None" else set()
            query_widgets = {center_event.widget} if center_event.widget != "None" else set()
            query_ops = {center_event.op} if center_event.op != "None" else set()
            ltm_items = mb.retrieve(query_pages, query_widgets, query_ops, top_k=MEMORY_RETRIEVE_TOP_K)

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
                            "Prompt": prompt,
                            "RawResponse": response_text,
                        }
                    )
                else:
                    # Fallback to original requirements elicitation prompt (without MP4)
                    # Keep backward compatibility: still run and store LLM response text.
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
        
        # Immediately save this participant's results to CSV (incremental save)
        if all_rows:
            df_incremental = pd.DataFrame(all_rows)
            # First participant: write with header; subsequent: append without header
            write_header = not os.path.exists(csv_output_path)
            df_incremental.to_csv(csv_output_path, mode='a', index=False, header=write_header, encoding='utf-8-sig')
            print(f"  ✓ {p_id} 的 {len(all_rows)} 条结果已保存到 {csv_output_path}")
            all_rows = []  # Clear for next participant

    # All participants processed; CSV already saved incrementally
    print(f"\n{'='*60}")
    print(f"✓ 处理完成！所有结果已保存到: {csv_output_path}")
    
    # Attempt to convert CSV to Excel (if file is not locked)
    if os.path.exists(csv_output_path):
        try:
            df_final = pd.read_csv(csv_output_path, encoding='utf-8-sig')
            df_final.to_excel(xlsx_output_path, index=False)
            print(f"✓ 已同时生成 Excel 版本: {xlsx_output_path}")
        except PermissionError:
            print(f"⚠️  Excel 文件被占用，仅保存了 CSV。请手动转换：{csv_output_path}")
        except Exception as e:
            print(f"⚠️  生成 Excel 时出错: {e}，但 CSV 已完整保存。")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
