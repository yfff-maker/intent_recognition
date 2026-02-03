from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

from event_representation import Event


@dataclass
class SequenceStats:
    page_counts: Counter
    widget_counts: Counter
    op_counts: Counter


def compute_stats(events: List[Event]) -> SequenceStats:
    return SequenceStats(
        page_counts=Counter(e.page for e in events),
        widget_counts=Counter(e.widget for e in events),
        op_counts=Counter(e.op for e in events),
    )


def importance_score(e: Event, stats: SequenceStats) -> float:
    """
    A lightweight heuristic importance score.
    This is intentionally simple (engineering choice) to support token control.
    """
    score = 0.0

    # Interactions carry more information than "None" placeholders
    if e.widget != "None":
        score += 2.0
    if e.module != "None":
        score += 1.0

    # Page transitions / navigation are often intent-relevant
    if "Enter" in e.page or "Home" in e.page or "Log in" in e.page:
        score += 0.5

    # Long duration events likely indicate hesitation / stuck
    if e.duration >= 1000:
        score += min(2.0, e.duration / 5000.0)

    # Rare widgets / ops can be informative
    w_freq = stats.widget_counts.get(e.widget, 1)
    op_freq = stats.op_counts.get(e.op, 1)
    if e.widget != "None":
        score += 1.0 / max(1.0, w_freq ** 0.5)
    if e.op != "None":
        score += 0.5 / max(1.0, op_freq ** 0.5)

    return score


def select_key_events(
    events: List[Event],
    target_k: int,
    num_bins: int,
    top_m_per_bin: int,
    near_dt_ms: int,
) -> List[Event]:
    """
    Step 1: Key event selection for token control.
    Two-stage:
      - importance scoring
      - time-coverage binning + near-duplicate removal
    """
    if not events:
        return []

    stats = compute_stats(events)
    scored: List[Tuple[float, Event]] = [(importance_score(e, stats), e) for e in events]

    t_min = events[0].t
    t_max = events[-1].t
    span = max(1, t_max - t_min)

    # Bin events by time for coverage
    bins: Dict[int, List[Tuple[float, Event]]] = {b: [] for b in range(num_bins)}
    for s, e in scored:
        b = int(((e.t - t_min) / span) * num_bins)
        b = min(num_bins - 1, max(0, b))
        bins[b].append((s, e))

    picked: List[Event] = []
    for b in range(num_bins):
        candidates = sorted(bins[b], key=lambda x: x[0], reverse=True)[:top_m_per_bin]
        picked.extend([e for _, e in candidates])

    # De-duplicate near events (time proximity + same signature)
    picked_sorted = sorted(picked, key=lambda x: (x.t, x.idx))
    deduped: List[Event] = []
    last_by_sig: Dict[Tuple[str, str, str, str], int] = {}
    for e in picked_sorted:
        sig = (e.page, e.module, e.widget, e.op)
        last_t = last_by_sig.get(sig)
        if last_t is not None and abs(e.t - last_t) <= near_dt_ms:
            continue
        deduped.append(e)
        last_by_sig[sig] = e.t

    # Final cap by importance if still too many
    if len(deduped) > target_k:
        scored_deduped = [(importance_score(e, stats), e) for e in deduped]
        deduped = [e for _, e in sorted(scored_deduped, key=lambda x: x[0], reverse=True)[:target_k]]
        deduped = sorted(deduped, key=lambda x: (x.t, x.idx))

    return deduped

