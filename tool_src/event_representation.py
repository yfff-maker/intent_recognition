from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Event:
    """
    Unified event representation used across modules.
    idx is stable and used for evidence referencing in prompts/outputs.
    """

    idx: int
    t: int
    page: str
    module: str
    widget: str
    op: str
    duration: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_str(v: Any, default: str = "None") -> str:
    if v is None:
        return default
    s = str(v)
    return s if s else default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def normalize_behavior_sequence(raw_seq: List[Dict[str, Any]]) -> List[Event]:
    """
    Step 0: Input & unified representation.
    Converts behavior_sequences.json items into a sorted List[Event] and assigns idx.
    """
    events: List[Event] = []
    for i, e in enumerate(raw_seq):
        t = _safe_int(e.get("startTimeTick", 0), 0)
        events.append(
            Event(
                idx=i,
                t=t,
                page=_safe_str(e.get("page", "None")),
                module=_safe_str(e.get("module", "None")),
                widget=_safe_str(e.get("widget", "None")),
                op=_safe_str(e.get("operationId", "None")),
                duration=_safe_int(e.get("duration", 0), 0),
            )
        )

    # Ensure time order; keep stable idx for evidence referencing (idx = original order)
    # If your raw data is already time-sorted, this is a no-op.
    events_sorted = sorted(events, key=lambda x: (x.t, x.idx))
    return events_sorted


def find_nearest_event_idx(events: List[Event], t0: int) -> Optional[int]:
    """
    Finds the position (index in `events` list) of the event whose timestamp is nearest to t0.
    Returns None if events is empty.
    """
    if not events:
        return None
    best_pos = 0
    best_dist = abs(events[0].t - t0)
    for pos in range(1, len(events)):
        d = abs(events[pos].t - t0)
        if d < best_dist:
            best_dist = d
            best_pos = pos
    return best_pos

