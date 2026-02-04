"""
快速测试带reasoning的LLM输出

用途：在运行完整流程前，先测试单个案例，看看reasoning的效果如何
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_participant_data
from anomaly_detector import detect_anomalies
from key_event_selector import select_key_events
from memory_bank import MemoryBank, chunk_events, summarize_chunk
from window_and_compress import build_window, compress_events, format_events_for_prompt
from intent_prompting import build_intent_prompt, parse_intent_output
from llm_client import LLMClient
from config import *


def test_single_case():
    """测试单个案例的reasoning输出"""
    
    print("=" * 80)
    print("🧪 测试带Reasoning的LLM输出")
    print("=" * 80)
    
    # 1. 加载数据（测试P1）
    print("\n📂 加载数据: P1")
    events, task_info = load_participant_data("P1", DATA_DIR)
    print(f"  ✓ 加载了 {len(events)} 个原始事件")
    
    # 2. 检测异常
    print("\n🔍 检测异常...")
    anomalies = detect_anomalies(events)
    print(f"  ✓ 检测到 {len(anomalies)} 个异常")
    
    if not anomalies:
        print("  ❌ 没有异常，无法测试")
        return
    
    # 选择第一个异常进行测试
    anomaly = anomalies[0]
    timestamp = anomaly["timestamp"]
    print(f"\n  选择测试异常:")
    print(f"    时间: {timestamp}")
    print(f"    类型: {anomaly.get('type')}")
    print(f"    描述: {anomaly.get('description')}")
    
    # 3. 选择关键事件
    print(f"\n⭐ 选择关键事件 (目标: {KEY_EVENT_TARGET_K}个)...")
    key_events = select_key_events(
        events=events,
        target_k=KEY_EVENT_TARGET_K,
        num_bins=KEY_EVENT_NUM_BINS,
        top_m_per_bin=KEY_EVENT_TOP_M_PER_BIN,
        near_dt_ms=KEY_EVENT_NEAR_DT_MS,
    )
    print(f"  ✓ 选择了 {len(key_events)} 个关键事件")
    
    # 4. 构建LTM
    print(f"\n🧠 构建长期记忆 (chunk_size={MEMORY_CHUNK_SIZE})...")
    mb = MemoryBank(max_items=MEMORY_MAX_ITEMS)
    chunks = chunk_events(key_events, MEMORY_CHUNK_SIZE)
    print(f"  ✓ 分成 {len(chunks)} 个chunk")
    
    for ci, ch in enumerate(chunks):
        item = summarize_chunk(ch, chunk_id=f"P1_{ci}")
        mb.add(item)
    
    # 5. 为异常点构建上下文（测试策略C - 最大窗口）
    print("\n🔧 构建上下文 (策略C)...")
    
    # 找到异常点在关键事件中的位置
    key_center_pos = 0
    for i, ke in enumerate(key_events):
        if ke.timestamp >= timestamp:
            key_center_pos = i
            break
    
    # 构建STM窗口
    win = build_window(
        key_events=key_events,
        center_pos=key_center_pos,
        mode="C",
        window_mode=WINDOW_MODE,
        strategy_windows=STRATEGY_WINDOWS,
    )
    print(f"  ✓ STM窗口: {len(win)} 个事件")
    
    # 压缩事件
    compressed = compress_events(win, merge_consecutive=COMPRESS_MERGE_CONSECUTIVE)
    stm_text = format_events_for_prompt(compressed, max_lines=PROMPT_MAX_EVENT_LINES)
    print(f"  ✓ 压缩后: {len(compressed)} 个压缩事件")
    
    # 检索LTM
    if win:
        query_pages = {e.page for e in win if e.page}
        query_widgets = {e.widget for e in win if e.widget}
        query_ops = {e.operation for e in win}
        ltm_items = mb.retrieve(query_pages, query_widgets, query_ops, top_k=MEMORY_RETRIEVE_TOP_K)
    else:
        ltm_items = []
    
    print(f"  ✓ 检索到 {len(ltm_items)} 个LTM chunk")
    
    # 6. 构建Prompt
    print("\n📝 构建Prompt...")
    prompt = build_intent_prompt(
        task_info=task_info,
        anomaly=anomaly,
        strategy="C",
        stm_events_text=stm_text,
        ltm_items=ltm_items,
        intent_labels=INTENT_LABELS,
    )
    
    print(f"  ✓ Prompt长度: {len(prompt)} 字符")
    
    # 显示Prompt片段
    print("\n" + "=" * 80)
    print("📄 Prompt预览 (前500字符):")
    print("-" * 80)
    print(prompt[:500])
    print("...")
    print("-" * 80)
    
    # 7. 调用LLM
    print("\n🤖 调用LLM进行推理...")
    llm = LLMClient(api_key=OPENROUTER_API_KEY, model=LLM_MODEL)
    
    try:
        response_text = llm.infer_intent(prompt)
        print(f"  ✓ 收到响应，长度: {len(response_text)} 字符")
        
        # 8. 解析输出
        print("\n📊 解析LLM输出...")
        parsed = parse_intent_output(response_text)
        
        print("\n" + "=" * 80)
        print("🎯 LLM推理结果")
        print("=" * 80)
        print(f"Intent:     {parsed.get('intent')}")
        print(f"Confidence: {parsed.get('confidence')}")
        print(f"\nReasoning:")
        print("-" * 80)
        reasoning = parsed.get('reasoning', '(未生成reasoning)')
        # 自动换行显示
        import textwrap
        for line in textwrap.wrap(reasoning, width=76):
            print(line)
        print("-" * 80)
        
        print(f"\nEvidence:")
        evidence = parsed.get('evidence', [])
        for i, ev in enumerate(evidence, 1):
            print(f"  {i}. 事件 {ev.get('event_idx')}: {ev.get('why')}")
        
        print(f"\nNotes: {parsed.get('notes', '(无)')}")
        
        print("\n" + "=" * 80)
        print("✅ 测试完成！")
        print("=" * 80)
        
        # 评估reasoning质量
        print("\n💡 Reasoning质量评估:")
        if reasoning and len(reasoning) > 50:
            print("  ✅ Reasoning长度合理 (>50字符)")
        else:
            print("  ⚠️  Reasoning可能太短")
        
        if "STM" in reasoning or "LTM" in reasoning or "长期" in reasoning or "短期" in reasoning:
            print("  ✅ Reasoning提到了记忆机制")
        else:
            print("  ⚠️  Reasoning未明确提到STM/LTM")
        
        if any(keyword in reasoning for keyword in ["多次", "重复", "持续", "快速", "行为模式"]):
            print("  ✅ Reasoning包含行为模式分析")
        else:
            print("  ⚠️  Reasoning缺少行为模式分析")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"  ❌ LLM调用失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_single_case()
