from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from event_representation import Event


def _event_sig(e: Event) -> Tuple[str, str, str, str]:
    return (e.page, e.module, e.widget, e.op)


def find_nearest_key_event_pos(key_events: List[Event], center_event: Event) -> Optional[int]:
    """
    Finds nearest key_event position using original idx distance first, then time distance.
    """
    if not key_events:
        return None
    best_pos = 0
    best = (abs(key_events[0].idx - center_event.idx), abs(key_events[0].t - center_event.t))
    for pos in range(1, len(key_events)):
        cand = (abs(key_events[pos].idx - center_event.idx), abs(key_events[pos].t - center_event.t))
        if cand < best:
            best = cand
            best_pos = pos
    return best_pos


def build_window(
    key_events: List[Event],
    center_pos: int,
    mode: str,
    window_mode: str,
    strategy_windows: Dict[str, Dict[str, int]],
) -> List[Event]:
    """
    Step 2: Window Builder.
    Windows are defined on key_events positions to keep length controllable.
    """
    if not key_events:
        return []
    if center_pos < 0 or center_pos >= len(key_events):
        return []

    params = strategy_windows[mode]

    if window_mode == "events":
        left = max(0, center_pos - params["k_left"])
        right = min(len(key_events), center_pos + params["k_right"] + 1)
        return key_events[left:right]

    # time-based window on key_events
    t0 = key_events[center_pos].t
    t_left = t0 - params["time_left_ms"]
    t_right = t0 + params["time_right_ms"]
    return [e for e in key_events if t_left <= e.t <= t_right]


@dataclass
class CompressedEvent:
    idxs: List[int]
    t_start: int
    t_end: int
    page: str
    module: str
    widget: str
    op: str
    count: int
    duration_sum: int
    duration_max: int


def compress_events(events: List[Event], merge_consecutive: bool = True) -> List[CompressedEvent]:
    """
    Step 3: Prompt Compression.
    Merge consecutive repeats with same (page,module,widget,op).
    """
    if not events:
        return []

    out: List[CompressedEvent] = []
    cur: Optional[CompressedEvent] = None

    for e in events:
        if cur is None:
            cur = CompressedEvent(
                idxs=[e.idx],
                t_start=e.t,
                t_end=e.t,
                page=e.page,
                module=e.module,
                widget=e.widget,
                op=e.op,
                count=1,
                duration_sum=e.duration,
                duration_max=e.duration,
            )
            continue

        if merge_consecutive and (cur.page, cur.module, cur.widget, cur.op) == (e.page, e.module, e.widget, e.op):
            cur.idxs.append(e.idx)
            cur.t_end = e.t
            cur.count += 1
            cur.duration_sum += e.duration
            cur.duration_max = max(cur.duration_max, e.duration)
        else:
            out.append(cur)
            cur = CompressedEvent(
                idxs=[e.idx],
                t_start=e.t,
                t_end=e.t,
                page=e.page,
                module=e.module,
                widget=e.widget,
                op=e.op,
                count=1,
                duration_sum=e.duration,
                duration_max=e.duration,
            )

    if cur is not None:
        out.append(cur)
    return out


def format_events_for_prompt(compressed: List[CompressedEvent], max_lines: int) -> str:
    """
    Stable, human-readable evidence text with evidence indices.
    """
    lines: List[str] = []
    for ce in compressed[:max_lines]:
        idx_repr = ce.idxs[0] if len(ce.idxs) == 1 else f"{ce.idxs[0]}..{ce.idxs[-1]}"
        rep = (
            f"- idx={idx_repr} t={ce.t_start}->{ce.t_end} "
            f"page={ce.page} module={ce.module} widget={ce.widget} op={ce.op} "
            f"count={ce.count} dur_sum={ce.duration_sum} dur_max={ce.duration_max}"
        )
        lines.append(rep)
    if len(compressed) > max_lines:
        lines.append(f"- ... truncated: {len(compressed) - max_lines} more compressed lines ...")
    return "\n".join(lines)

