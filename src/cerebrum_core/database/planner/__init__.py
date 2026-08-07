"""
cerebrum_core.database.study_plan_registry
=============================================
Composes _RepositoryBase (connection/schema plumbing) with the four
domain mixins into the concrete class used everywhere else in the
codebase. Import path is unchanged from before the split:

    from cerebrum_core.database.study_plan_registry import StudyPlanRegisterInator

so no call site outside this package needs to change.
"""

from __future__ import annotations

from ._base import _RepositoryBase
from .metrics import MetricsMixin
from .phases import PhasesMixin
from .plans import PlansMixin
from .weeks import WeeksMixin


class StudyPlanRegisterInator(
    _RepositoryBase, PlansMixin, PhasesMixin, MetricsMixin, WeeksMixin
):
    """Full study-plan registry: plans, phases, success metrics, and the
    densified weekly/daily task layer, all backed by
    registry/study_plan_registry.db."""

    pass
