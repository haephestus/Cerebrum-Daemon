# Feature: Users, orgs & auth

**Status**: DONE (P0)

## Goals
Multi-user identity for the daemon in both local and cloud modes; org grouping with members.

## Scope
- Account create/get/delete; login; password reset request/verify/update (`routes_user`, `api/auth/{password,reset_token,email}_inator`)
- Org CRUD + member add/remove (`routes_org`)
- Dual transport auth: `X-Daemon-Key` header (local daemon key, minted at first boot, printed to stdout) vs bearer token (cloud/user identity); enforced by `DaemonAuthMiddleware`
- Learning-profile endpoints live here too ([[features/learning-profiles]])

## Dependencies
SQLite `users`/`orgs`/`password_reset` repos in [[research/architecture]]'s database layer.

## Notes
- CORS is added *after* DaemonAuth so preflight OPTIONS answers before key check (middleware order matters; see `create_api_server`).
- Password reset flow is a P3 security-pass target ([[plan/hardening]]).
