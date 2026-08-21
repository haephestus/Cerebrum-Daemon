# Cross-Repo Links — Cerebrum-Daemon

> Projects that share repos or interfaces. Each linked repo needs
> interface documentation so Steward can track cross-repo impacts.

## Linked repos
| Repo | Relationship | Interface | Notes |
|------|-------------|-----------|-------|
| [haephestus/Cerebrum](https://github.com/haephestus/Cerebrum) | sibling product — Flutter client this daemon powers | HTTP JSON API; shapes frozen by contract markers | offline-first; caches notes/images/attempts/mastery/engram content on-device |

## Interface contracts
- Registry of live wire contracts: [[cross-repo/contracts]]
- In-code source of truth: `grep -rn "CROSS-REPO CONTRACT" src` here; same grep under `../Cerebrum/src/lib` there
- Human-readable index maintained in `docs/cross-repo-contracts.md` (repo) — mirrored below in [[cross-repo/contracts]]

Rule: changing any listed shape requires updating both repos in the same change-set.
