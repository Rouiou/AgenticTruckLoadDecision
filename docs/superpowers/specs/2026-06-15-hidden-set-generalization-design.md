# Hidden-Set Generalization Design

## Objective

Raise hidden-set score by reducing preference penalties without overfitting the public D001 driver or materially increasing token use and decision latency.

## Strategy

Treat preferences as three economic primitives:

1. **No-go intervals**: actions may not overlap a time interval. These are deterministic vetoes.
2. **Order rules**: accepting a matching cargo either incurs a known penalty or is forbidden. Known penalties are subtracted from candidate value; explicit hard prohibitions are vetoes.
3. **Deadline obligations**: the driver must reach a location and possibly wait there before a deadline. These act like virtual orders and receive deterministic execution priority as the deadline approaches.

LLM use remains low-frequency. It compiles natural language into audited `machine_ir`; deterministic code evaluates that IR on every candidate and action.

## Phase One Scope

- Enable and regression-test the existing `home_curfew` executor.
- Add generic `order_rules` IR and deterministic matching for:
  - cargo category
  - pickup and delivery region text
  - pickup deadhead distance
  - transport duration
- Support `hard_veto` and `penalty` effects. Penalty effects reduce candidate score by the exact visible penalty rather than banning profitable orders.
- Add audit checks so malformed or hallucinated rules are demoted to `unknown_constraints`.
- Add synthetic hidden-driver tests for curfew and order-rule behavior.
- Keep current market query size and route score unchanged until this preference layer is measured.

## Guardrails

- No sample driver IDs, place names, categories, times, or thresholds in production code.
- Public D001 must not regress in preference penalty.
- New deterministic behavior must be test-first.
- API key is passed only through the process environment.
- Unknown or malformed compiled rules must fail open to Council, not become unsafe deterministic rules.

## Later Phases

- Deadline/location obligations as virtual actions.
- Quality-weighted destination value and shallow two-hop continuation value.
- Adaptive query size only after measuring its effect on income and rest-window compliance.
