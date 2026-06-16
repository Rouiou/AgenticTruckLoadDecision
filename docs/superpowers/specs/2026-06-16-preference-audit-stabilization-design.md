# Preference Audit Stabilization Design

## Goal

Improve hidden-set preference robustness without changing the proven route
planning core.

## Design

- Add deterministic audit correction for weekend delayed rest windows: if an
  all-days base rest window and a weekend-specific delayed-start window share
  the same source preference, keep the weekend start delay but preserve the base
  end hour.
- Strengthen unknown cleanup for carry-over quota wording: once a quota target
  exists, high-risk unknown entries that only restate makeup/deficit semantics
  are removed from per-step Council triggering.

## Safety

- No driver IDs, cargo names, cities, or public preference constants are added.
- The logic uses only compiled policy structure and source text semantics.
- Rest execution, quota scoring, route scoring, and market query breadth remain
  unchanged.

## Acceptance

Public 92-day benchmark should remain at the current stable score shape. Unit
tests must cover both audit corrections.
