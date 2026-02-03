from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from event_representation import Event


@dataclass
class MemoryItem:
    chunk_id: str
    t_start: int
    t_end: int
    summary: str
    features: Dict[str, float]
    signature: Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]
    event_idx_range: Tuple[int, int]


def chunk_events(events: List[Event], chunk_size: int) -> List[List[Event]]:
    return [events[i : i + chunk_size] for i in range(0, len(events), chunk_size)]


def _top_items(counter: Counter, k: int = 5) -> List[Tuple[str, int]]:
    return [(x, int(c)) for x, c in counter.most_common(k)]


def summarize_chunk(chunk: List[Event], chunk_id: str) -> MemoryItem:
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

    # Lightweight textual summary (no fancy claims)
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

    return MemoryItem(
        chunk_id=chunk_id,
        t_start=t_start,
        t_end=t_end,
        summary="\n".join(summary_lines),
        features=features,
        signature=signature,
        event_idx_range=(idx_min, idx_max),
    )


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class MemoryBank:
    """
    Step 4-6: LTM storage, update (evict), and retrieval.
    This is an engineering component; keep it lightweight and explainable.
    """

    def __init__(self, max_items: int):
        self.max_items = max_items
        self.items: List[MemoryItem] = []

    def add(self, item: MemoryItem) -> None:
        self.items.append(item)
        if len(self.items) > self.max_items:
            self._evict()

    def _evict(self) -> None:
        # Simple eviction: remove the oldest chunk (smallest t_end)
        self.items.sort(key=lambda x: x.t_end)
        while len(self.items) > self.max_items:
            self.items.pop(0)

    def retrieve(self, query_pages: Set[str], query_widgets: Set[str], query_ops: Set[str], top_k: int) -> List[MemoryItem]:
        scored: List[Tuple[float, MemoryItem]] = []
        for it in self.items:
            pages = set(it.signature[0])
            widgets = set(it.signature[1])
            ops = set(it.signature[2])
            sim = 0.5 * _jaccard(query_widgets, widgets) + 0.3 * _jaccard(query_pages, pages) + 0.2 * _jaccard(query_ops, ops)
            scored.append((sim, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in scored[:top_k] if _ > 0.0]

