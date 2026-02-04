from __future__ import annotations

import json
from typing import Dict, List, Optional

from memory_bank import MemoryItem


def build_intent_prompt(
    task_info: Dict[str, str],
    anomaly: Dict,
    strategy: str,
    stm_events_text: str,
    ltm_items: List[MemoryItem],
    intent_labels: List[str],
) -> str:
    """
    Step 7: LLM Inference prompt.
    Keep the skeleton constant; only stm_events_text changes across A/B/C.
    """
    labels_json = json.dumps(intent_labels, ensure_ascii=False)
    ltm_text = "\n\n".join([it.summary for it in ltm_items]) if ltm_items else "(none)"

    # Note: we intentionally avoid images/MP4 dependency here.
    prompt = f"""
You are an analyst for long-sequence intent inference from interaction logs.
Given the task goal context, the detected abnormal pattern, short-term behavioral evidence (STM), and retrieved long-term memory summaries (LTM),
infer the user's intent and provide evidence references.

### Task Context
- User Task: {task_info.get('objective')}

### Abnormal Pattern (Anchor)
- Type: {anomaly.get('type')}
- Timestamp(ms): {anomaly.get('timestamp')}
- Description: {anomaly.get('description')}

### Strategy
- Strategy: {strategy} (Only the STM context length changes across strategies)

### Short-Term Memory (STM) - Behavior Evidence
{stm_events_text}

### Long-Term Memory (LTM) - Retrieved Summaries (top-k)
{ltm_text}

### Output Schema (MUST be valid JSON)
Choose intent from this closed set: {labels_json}
Return JSON with keys:
- "intent": string (one label from the list)
- "confidence": number (0..1)
- "reasoning": string (详细的推理说明，解释为什么选择这个意图，需要结合STM和LTM中的关键行为模式进行分析，3-5句话)
- "evidence": list of objects, each with:
    - "event_idx": string (e.g., "12" or "12..15")
    - "why": string (short reason grounded in STM/LTM)
- "notes": string (optional)
""".strip()
    return prompt


def parse_intent_output(text: str) -> Dict:
    """
    Robust-ish JSON parser: try direct JSON first, then attempt to extract the first JSON object.
    """
    text = (text or "").strip()
    if not text:
        return {"intent": "Other", "confidence": 0.0, "evidence": [], "notes": "empty_response"}

    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to extract first {...}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass

    return {"intent": "Other", "confidence": 0.0, "evidence": [], "notes": "non_json_response", "raw": text}

