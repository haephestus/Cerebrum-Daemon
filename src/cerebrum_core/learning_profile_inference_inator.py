"""
cerebrum_core.learning_profile_inference_inator
===============================================
Turns behavioural signals into learning-profile EVIDENCE rows (the inferred
layer). Hybrid design (see [[notes-and-learning-gaps]] gap 2):

  * STRUCTURAL (deterministic, always on)
      - note analysis  -> concept_map.confused_links imply blurred concept
                          boundaries (pattern-over-detail, breadth-over-depth).
      - engram perf    -> relative strengths across topic-mastery dimensions
                          (factual / applied / conceptual / doctoral) imply a
                          style lean on the abstraction / pattern / depth axes.
                          RELATIVE, not absolute — 'novice everywhere' says
                          nothing about style, so it contributes nothing.
  * LLM ENRICHMENT (optional, feature-flagged, off by default)
      - reads each finding's free-text gap_explanation and emits nuanced axis
        pulls. Structural can't read prose; this is where a finding like
        "compresses the definition ... rather than rigorous understanding"
        -> pattern_detail actually lives.

Every emitted row is {source, axis, value in [-1,1], weight, ref}. Callers
persist them idempotently per `ref` via repo.replace_evidence_for_ref, so
re-analysing the same note (or recomputing perf) REPLACES rather than
duplicates its evidence.
"""

from __future__ import annotations

import math
from typing import Iterable

from cerebrum_core.learning_profile_inator import clamp

SOURCE_NOTE_ANALYSIS = "note_analysis"
SOURCE_ENGRAM_PERF = "engram_performance"
PERF_REF = "_perf"  # single logical ref for the (replaceable) perf signal

# Feature flag for the optional LLM enrichment pass. Hybrid: structural runs
# now; flip this on once the LLM mapper is implemented.
LLM_ENRICH_ENABLED = False

_EPS = 1e-6


def _lean(a: float, b: float) -> float:
    """Scale-invariant relative lean of a vs b in [-1, 1]: +1 = all-a, -1 = all-b,
    0 = equal or both absent. Independent of whether scores are 0..1 or 0..100."""
    a, b = max(0.0, float(a)), max(0.0, float(b))
    if a + b <= _EPS:
        return 0.0
    return clamp((a - b) / (a + b))


def _saturate(x: float, half: float) -> float:
    """0 at x=0, → 1 as x grows; ~0.63 at x = half."""
    return 1.0 - math.exp(-max(0.0, x) / half) if x > 0 else 0.0


# --------------------------------------------------------------------------- #
# STRUCTURAL: note analysis
# --------------------------------------------------------------------------- #
def structural_note_analysis(analysis: dict, ref: str) -> list[dict]:
    """Deterministic axis pulls from an analysis blob's STRUCTURED fields.

    `analysis` is the inner analysis object ({note_overview, chunk_diagnostics}).
    Currently reads only concept_map.confused_links; gap_explanation nuance is
    deliberately left to the LLM enrichment pass.
    """
    overview = (analysis or {}).get("note_overview") or {}
    cmap = overview.get("concept_map") or {}
    confused = cmap.get("confused_links") or []
    n = len(confused)
    if n <= 0:
        return []
    # Blurring concept_a/concept_b => grasps the shape but misses the fine
    # boundary (pattern over detail) and ranges across concepts (breadth).
    return [
        {"source": SOURCE_NOTE_ANALYSIS, "axis": "pattern_detail",
         "value": -0.5, "weight": min(n * 0.4, 1.5), "ref": ref},
        {"source": SOURCE_NOTE_ANALYSIS, "axis": "breadth_depth",
         "value": -0.4, "weight": min(n * 0.3, 1.2), "ref": ref},
    ]


# --------------------------------------------------------------------------- #
# STRUCTURAL: engram performance (topic-mastery aggregates)
# --------------------------------------------------------------------------- #
def structural_engram_performance(topic_masteries: Iterable[dict]) -> list[dict]:
    """Deterministic axis pulls from a user's topic-mastery dimension scores,
    averaged across topics (weighted by engram_count). Uses RELATIVE strengths.

    NOTE: topic mastery only carries factual/applied/conceptual/doctoral, so
    this covers abstraction / pattern / depth. Finer axes (example_discovery
    from originality, precision-vs-connections) need per-engram dimension
    scores — a future accessor + refinement.
    """
    tms = list(topic_masteries or [])
    if not tms:
        return []

    acc = {
        "abstraction_concrete": [0.0, 0.0],  # [sum_w*lean, sum_w]
        "pattern_detail": [0.0, 0.0],
        "breadth_depth": [0.0, 0.0],
    }
    total_engrams = 0
    for tm in tms:
        w = max(1.0, float(tm.get("engram_count", 1) or 1))
        total_engrams += int(tm.get("engram_count", 0) or 0)
        fac = float(tm.get("factual_score", 0.0) or 0.0)
        app = float(tm.get("applied_score", 0.0) or 0.0)
        con = float(tm.get("conceptual_score", 0.0) or 0.0)
        doc = float(tm.get("doctoral_score", 0.0) or 0.0)
        leans = {
            "abstraction_concrete": _lean(con, app),  # abstract vs applied/concrete
            "pattern_detail": _lean(fac, con),        # factual specifics vs conceptual pattern
            "breadth_depth": _lean(doc, fac),         # deep mastery vs surface facts
        }
        for axis, lean in leans.items():
            acc[axis][0] += w * lean
            acc[axis][1] += w

    # Confidence in the perf signal grows with how much practice backs it.
    conf = _saturate(total_engrams, half=8.0)  # ~0.63 at 8 engrams
    rows: list[dict] = []
    for axis, (num, den) in acc.items():
        if den <= 0:
            continue
        value = clamp(num / den)
        weight = round(conf * 3.0, 3)  # perf caps ~3 total weight per axis
        if abs(value) < 0.02 or weight <= 0.0:
            continue
        rows.append({"source": SOURCE_ENGRAM_PERF, "axis": axis,
                     "value": value, "weight": weight, "ref": PERF_REF})
    return rows


# --------------------------------------------------------------------------- #
# OPTIONAL: LLM enrichment seam (Hybrid)
# --------------------------------------------------------------------------- #
def llm_enrich_note_analysis(analysis: dict, ref: str) -> list[dict]:
    """Optional nuance pass over free-text gap_explanations. Off by default.

    TODO: implement via models.model_inator — prompt with the AXES definitions
    + each finding's gap_explanation, parse structured {axis, value, weight}
    pulls. Kept as a seam so Hybrid can be switched on (LLM_ENRICH_ENABLED)
    without touching callers.
    """
    if not LLM_ENRICH_ENABLED:
        return []
    raise NotImplementedError("LLM enrichment pass not implemented yet")


# --------------------------------------------------------------------------- #
# Combiners + drivers
# --------------------------------------------------------------------------- #
def note_analysis_evidence(analysis: dict, ref: str) -> list[dict]:
    return structural_note_analysis(analysis, ref) + llm_enrich_note_analysis(analysis, ref)


def apply_note_analysis(repo, user_id: str, analysis: dict, note_id: str) -> list[dict]:
    """Recompute + persist a note's contribution to the inferred profile.
    Idempotent: replaces any prior evidence for this (note_analysis, note_id)."""
    rows = note_analysis_evidence(analysis, ref=note_id)
    repo.replace_evidence_for_ref(user_id, SOURCE_NOTE_ANALYSIS, note_id, rows)
    return rows


def apply_engram_performance(repo, user_id: str) -> list[dict]:
    """Recompute + persist the engram-performance contribution. Idempotent:
    replaces the single PERF_REF evidence set from current topic mastery."""
    topics = repo.get_all_topic_masteries_for_user(user_id)
    rows = structural_engram_performance(topics)
    repo.replace_evidence_for_ref(user_id, SOURCE_ENGRAM_PERF, PERF_REF, rows)
    return rows
