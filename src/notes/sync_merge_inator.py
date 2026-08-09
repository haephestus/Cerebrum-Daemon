"""
notes.sync_merge_inator — offline-sync merge engine for notes (gap 1 / stream C).

Pure and deterministic; transport-agnostic (the hub/spoke protocol wires it up).
Three moving parts, matching the design:

  * version vectors decide "newer vs concurrent". A vector is {replica_id: counter};
    a replica bumps only its own slot. Element-wise compare → dominates (one is
    strictly newer) or concurrent (a genuine conflict).
  * pages merge by LWW *only when concurrent* (last_modified wins); otherwise the
    dominating side is taken. The merged page's vector is the join (max per slot)
    so it is causally after both — future compares see it as newer.
  * ink merges by **stroke-id union** (additive, near-conflict-free); a tombstone
    (deleted stroke) on either side wins, so deletes propagate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from models.model_inator import Note, Page


class VectorRelation(str, Enum):
    EQUAL = "equal"
    DOMINATES = "dominates"    # a ≥ b element-wise and a ≠ b  → a is newer
    DOMINATED = "dominated"    # b is newer
    CONCURRENT = "concurrent"  # neither — a real conflict, needs resolution


def compare_vectors(a: dict, b: dict) -> VectorRelation:
    a = a or {}
    b = b or {}
    keys = set(a) | set(b)
    a_ge = all(a.get(k, 0) >= b.get(k, 0) for k in keys)
    b_ge = all(b.get(k, 0) >= a.get(k, 0) for k in keys)
    if a_ge and b_ge:
        return VectorRelation.EQUAL
    if a_ge:
        return VectorRelation.DOMINATES
    if b_ge:
        return VectorRelation.DOMINATED
    return VectorRelation.CONCURRENT


def bump_vector(v: dict, replica_id: str) -> dict:
    """Return a copy with this replica's slot incremented — call on every local
    edit so the vector records who-last-touched-what."""
    out = dict(v or {})
    out[replica_id] = out.get(replica_id, 0) + 1
    return out


def join_vectors(a: dict, b: dict) -> dict:
    """Per-slot max — the merged clock, causally after both inputs."""
    a = a or {}
    b = b or {}
    return {k: max(a.get(k, 0), b.get(k, 0)) for k in set(a) | set(b)}


def merge_ink(local: list, remote: list) -> list:
    """Union strokes by id. A stroke present on both sides keeps one copy; a
    tombstone (`deleted: true`) on EITHER side wins so a delete propagates.
    Strokes without an id are kept as-is (can't be de-duplicated)."""
    by_id: dict[str, dict] = {}
    passthrough: list = []
    for stroke in list(local or []) + list(remote or []):
        sid = stroke.get("id") if isinstance(stroke, dict) else None
        if sid is None:
            passthrough.append(stroke)
            continue
        if sid not in by_id:
            by_id[sid] = dict(stroke)
        elif stroke.get("deleted") or by_id[sid].get("deleted"):
            by_id[sid] = {**by_id[sid], **stroke, "deleted": True}
    return passthrough + list(by_id.values())


@dataclass
class PageMergeResult:
    page: Page
    relation: VectorRelation
    conflicted: bool  # True only when the two sides were concurrent (LWW applied)


def merge_page(local: Page, remote: Page) -> PageMergeResult:
    """Merge two versions of the same page. Document is LWW-on-concurrent (else
    the dominating side); ink always unions; the vector always joins."""
    rel = compare_vectors(
        local.metadata.version_vector, remote.metadata.version_vector
    )
    conflicted = rel is VectorRelation.CONCURRENT

    if rel in (VectorRelation.EQUAL, VectorRelation.DOMINATES):
        winner = local
    elif rel is VectorRelation.DOMINATED:
        winner = remote
    else:  # CONCURRENT → last-writer-wins on wall clock (deterministic tie→local)
        winner = (
            local
            if local.metadata.last_modified >= remote.metadata.last_modified
            else remote
        )

    merged = winner.model_copy(deep=True)
    merged.metadata.version_vector = join_vectors(
        local.metadata.version_vector, remote.metadata.version_vector
    )
    merged.ink = merge_ink(local.ink, remote.ink)
    return PageMergeResult(page=merged, relation=rel, conflicted=conflicted)


@dataclass
class NoteMergeResult:
    note: Note
    conflicted_pages: list[str]  # page_ids where a concurrent conflict was resolved


def merge_note(local: Note, remote: Note) -> NoteMergeResult:
    """Page-wise merge of two note replicas (keyed by page_id). Pages present on
    only one side are carried over. Note-level vector joins."""
    from notes.note_util_inator import note_pages

    local_pages = {p.page_id: p for p in note_pages(local)}
    remote_pages = {p.page_id: p for p in note_pages(remote)}

    merged_pages: list[Page] = []
    conflicted: list[str] = []
    for pid in set(local_pages) | set(remote_pages):
        lp, rp = local_pages.get(pid), remote_pages.get(pid)
        if lp and rp:
            res = merge_page(lp, rp)
            merged_pages.append(res.page)
            if res.conflicted:
                conflicted.append(pid)
        else:
            merged_pages.append((lp or rp).model_copy(deep=True))

    merged_pages.sort(key=lambda p: p.page_index)
    merged = local.model_copy(deep=True)
    merged.pages = merged_pages
    merged.manifest.version_vector = join_vectors(
        local.manifest.version_vector, remote.manifest.version_vector
    )
    return NoteMergeResult(note=merged, conflicted_pages=conflicted)
