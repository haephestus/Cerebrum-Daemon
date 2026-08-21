# Feature: Study planner

**Status**: DONE (P2)

## Goals
Generate structured multi-phase study plans from goals/metrics, then track progress against them.

## Scope
- Service layer `study_planner_inator.py` sits between router and `StudyPlanRegisterInator` — mirrors learning-center discipline: routers never touch registries or ollama callables directly
- Plan lifecycle: `/generate`; get plan; user's all/active plans
- Structure: phases (`incomplete`, `complete`, `densify`), weeks per phase, metrics (`unachieved`, mark achieved)
- Tasks: per-task complete/reopen; aggregate progress endpoint
- Admin review queue endpoint present

## Dependencies
Own SQLite package `database/planner/` (plans/phases/weeks/metrics/progress_service/schema); LLM generation via ollama local/cloud call helpers; [[features/learning-profiles]] personalization.

## Notes
- Densify = re-pack remaining tasks within a phase — useful after missed days; semantics live in planner package.
