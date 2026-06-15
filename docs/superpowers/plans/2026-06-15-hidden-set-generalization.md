# Hidden-Set Generalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce hidden-set preference penalties through audited deterministic curfew and generic order-rule execution.

**Architecture:** Extend the existing low-frequency preference compiler with a generic `order_rules` IR. Audit compiled rules before deterministic evaluation, then apply hard rules as vetoes and penalty rules as exact score deductions. Enable the already-built curfew state machine behind regression tests.

**Tech Stack:** Python 3, standard-library `unittest`, existing simulation API and decision service.

---

### Task 1: Add Synthetic Preference Regression Harness

**Files:**
- Create: `demo/tests/test_preference_execution.py`
- Modify: none

- [ ] Write tests that instantiate `ModelDecisionService` without an external model call and construct `CandidateFact` values directly.
- [ ] Add a failing test proving a matching penalty order rule reduces candidate score by the specified amount.
- [ ] Add a failing test proving a matching hard order rule adds a deterministic veto.
- [ ] Add a passing control test proving non-matching rules do not change candidate value.
- [ ] Run `python3 -m unittest discover -s demo/tests -v` and verify the new behavior tests fail for the expected missing implementation.

### Task 2: Implement Audited Generic Order Rules

**Files:**
- Modify: `demo/agent/model_decision_service.py`
- Test: `demo/tests/test_preference_execution.py`

- [ ] Add `machine_ir.order_rules` to the compiler schema and instructions.
- [ ] Add audit validation for allowed fields, operators, effects, values, and non-negative penalties.
- [ ] Add deterministic field extraction and matching helpers.
- [ ] Add matching hard rules to `_deterministic_vetoes`.
- [ ] Subtract matching penalty rules in `_candidate_score`.
- [ ] Run `python3 -m unittest discover -s demo/tests -v` and verify all order-rule tests pass.

### Task 3: Stabilize and Enable Home Curfew

**Files:**
- Modify: `demo/agent/model_decision_service.py`
- Test: `demo/tests/test_preference_execution.py`

- [ ] Add tests proving a candidate that cannot finish and return home before deadline is vetoed.
- [ ] Add tests proving the executor repositions home before the latest safe departure time.
- [ ] Add tests proving it waits through the quiet interval when already home.
- [ ] Enable `FEATURE_FLAGS["home_curfew"]`.
- [ ] Run `python3 -m unittest discover -s demo/tests -v` and verify all curfew tests pass.

### Task 4: Public Evaluation and Compliance

**Files:**
- Modify only if a regression is discovered.

- [ ] Restore the public cargo dataset through Git LFS.
- [ ] Run a short public simulation to validate compiler output and action legality.
- [ ] Run the full 92-day public D001 simulation.
- [ ] Run `python3 demo/calc_monthly_income.py` and compare net income, preference penalty, token use, and latency against the branch baseline.
- [ ] Run `bash demo/check_compliance.sh`.
- [ ] Keep changes only if D001 does not regress materially and all deterministic tests pass.
