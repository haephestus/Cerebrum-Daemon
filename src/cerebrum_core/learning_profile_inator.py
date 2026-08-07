"""
cerebrum_core.learning_profile_inator
=====================================
The learning-profile dimension model and the declared/inferred blend.

A learning profile is a small vector of orthogonal axes, each a scalar in
[-1, 1] (0 = no preference). Two layers are stored SEPARATELY (see the
note_engram_repository learning_profile mixin) and never overwrite each other:

  * declared  -- what the user says about how they want to be taught.
                 The Bayesian *prior*.
  * inferred  -- what their behaviour implies, accumulated as a weighted
                 evidence log. The *evidence*.

The EFFECTIVE profile blends them, confidence-gated: with no evidence the
declared prior dominates (cold start); as evidence accumulates the inferred
posterior grows and can override. That is what "weighted toward inferred, but
confidence-gated / dynamic" means in practice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

# The fixed axis set. `neg`/`pos` name the two poles for a value of -1 / +1;
# 0 means no expressed preference. Adding an axis here is all it takes for the
# storage layer to carry it (the DB stores axis strings, not a fixed schema).
AXES: dict[str, dict[str, str]] = {
    "abstraction_concrete": {
        "neg": "concrete, example-first — learns from specific instances",
        "pos": "abstract, principle-first — learns from general rules",
    },
    "breadth_depth": {
        "neg": "breadth-first — wants the wide map before the detail",
        "pos": "depth-first — masters one thing fully before moving on",
    },
    "pattern_detail": {
        "neg": "pattern / top-down — grasps the overall shape, thin on specifics",
        "pos": "detail / bottom-up — rigorous on specifics, builds the pattern up",
    },
    "example_discovery": {
        "neg": "worked-example — prefers to be shown, then reproduce",
        "pos": "discovery — prefers to derive/struggle before being told",
    },
}
AXIS_NAMES: tuple[str, ...] = tuple(AXES.keys())

# Accumulated evidence weight at which the inferred layer reaches ~0.63
# confidence on an axis. Larger = inference overrides the declared prior more
# slowly (needs more behavioural signal to be trusted).
_CONFIDENCE_HALFLIFE = 5.0

# Below this |effective| an axis is treated as "no meaningful preference" and
# omitted from the prompt persona (so we don't emit four "neutral" lines).
_PROMPT_DEADZONE = 0.15


def clamp(v: float) -> float:
    return max(-1.0, min(1.0, float(v)))


def blank_axes() -> dict[str, float]:
    return {a: 0.0 for a in AXIS_NAMES}


def sanitize_declared(axes: Mapping[str, float]) -> dict[str, float]:
    """Keep only known axes, coerce to float, clamp to [-1, 1]. For user input
    (PUT /user/learning-profile) so an unknown axis or out-of-range value can't
    reach storage."""
    clean: dict[str, float] = {}
    for a in AXIS_NAMES:
        if a in axes and axes[a] is not None:
            clean[a] = clamp(axes[a])
    return clean


@dataclass
class AxisEstimate:
    mean: float        # posterior mean in [-1, 1]
    confidence: float  # 0..1, saturates toward 1 as evidence accrues
    n: int             # number of evidence rows on this axis


def infer_posterior(evidence: Iterable[Mapping]) -> dict[str, AxisEstimate]:
    """Aggregate the evidence log into a per-axis posterior.

    mean       = weight-weighted average of the signal values on that axis
    confidence = 1 - exp(-total_weight / halflife)   (0 with no evidence)
    """
    acc: dict[str, list[float]] = {a: [0.0, 0.0, 0] for a in AXIS_NAMES}  # [sum_wv, sum_w, n]
    for e in evidence:
        axis = e["axis"]
        if axis not in acc:
            continue  # unknown/retired axis — ignore, don't crash
        w = float(e.get("weight", 1.0))
        v = clamp(e["value"])
        acc[axis][0] += w * v
        acc[axis][1] += w
        acc[axis][2] += 1

    out: dict[str, AxisEstimate] = {}
    for a in AXIS_NAMES:
        sum_wv, sum_w, n = acc[a]
        mean = clamp(sum_wv / sum_w) if sum_w > 0 else 0.0
        conf = (1.0 - math.exp(-sum_w / _CONFIDENCE_HALFLIFE)) if sum_w > 0 else 0.0
        out[a] = AxisEstimate(mean=mean, confidence=conf, n=int(n))
    return out


def effective_profile(
    declared: Mapping[str, float] | None,
    evidence: Iterable[Mapping],
) -> dict[str, dict]:
    """Confidence-gated blend of declared (prior) and inferred (evidence):

        effective = (1 - c) * declared + c * inferred_mean,   c = confidence

    Cold start (c≈0) → declared; lots of evidence (c→1) → inferred. Returns a
    per-axis dict carrying the effective value plus its provenance, so callers
    (and the UI) can show "you said X, we observe Y".
    """
    declared = declared or {}
    posterior = infer_posterior(evidence)
    result: dict[str, dict] = {}
    for a in AXIS_NAMES:
        d = clamp(declared.get(a, 0.0))
        est = posterior[a]
        c = est.confidence
        eff = clamp((1.0 - c) * d + c * est.mean)
        result[a] = {
            "effective": eff,
            "declared": d,
            "inferred": est.mean,
            "confidence": c,
            "evidence_n": est.n,
        }
    return result


def render_for_prompt(effective: Mapping[str, Mapping]) -> str:
    """Turn the effective profile into a short natural-language persona snippet
    for injection into generation prompts. Near-neutral axes are skipped."""
    lines: list[str] = []
    for a, spec in AXES.items():
        v = effective.get(a, {}).get("effective", 0.0)
        if abs(v) < _PROMPT_DEADZONE:
            continue
        pole = spec["pos"] if v > 0 else spec["neg"]
        strength = "strongly" if abs(v) >= 0.6 else "somewhat"
        lines.append(f"- {strength} {pole}")
    if not lines:
        return (
            "No strong learning-style preferences yet; teach in a balanced, "
            "general way."
        )
    return "Adapt teaching to this learner:\n" + "\n".join(lines)
