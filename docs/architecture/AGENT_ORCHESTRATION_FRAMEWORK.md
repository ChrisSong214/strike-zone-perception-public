# Autonomous Multi-Agent Quality Orchestration Framework

This document outlines the architecture, roles, and execution protocol of the autonomous **Audit $\to$ Plan $\to$ Implement $\to$ Audit** multi-agent system implemented for `sp26_sagb_dj`.

---

## 1. System Architecture

The orchestration system decomposes complex codebase quality assurance, scientific validation, and bug remediation into four specialized, autonomous subagent roles coordinated in a closed-loop feedback cycle:

```mermaid
flowchart TD
    Orchestrator["Autonomous Orchestrator<br/>(Master Coordination Loop)"]
    Auditor["Auditor Subagent<br/>(Static/Dynamic Audit & pytest)"]
    Planner["Planner Subagent<br/>(Architectural Blueprint & Test Design)"]
    Implementor["Implementor Subagent<br/>(Surgical Edits & Unit Tests)"]

    Orchestrator -->|1. Request Full Audit| Auditor
    Auditor -->|2. Audit Findings & STATUS| Orchestrator
    
    Orchestrator -- "If STATUS: ISSUES_FOUND" -->|3. Pass Findings| Planner
    Planner -->|4. Implementation Blueprint| Orchestrator
    
    Orchestrator -->|5. Pass Blueprint| Implementor
    Implementor -->|6. Execution Log & pytest results| Orchestrator
    
    Orchestrator -->|7. Verification Audit Cycle| Auditor
    
    Orchestrator -- "If STATUS: CLEAN" --> Terminate["Complete & Certified Converged"]
```

---

## 2. Agent Definitions & Roles

### A. Codebase & Mathematical Auditor (`auditor`)
* **Core Responsibilities:**
  - Evaluates mathematical and statistical estimators (Empirical Bayes variance components, bivariate normal CDFs, correlation attenuation under independent noise, coordinate projections, information horizon center-of-mass).
  - Assesses code robustness, zero-division guards, numerical stability ($\epsilon \ge 10^{-12}$), empty DataFrame handling, and NaN propagation.
  - Verifies data contract integrity across SQLite databases (`data/mlb.db`), OOF Parquet caches, and CSV leaderboards (`data/leaderboards/`).
  - Executes dynamic test validation (`pytest tests/ -v`).
  - Evaluates documentation and empirical claims in `README.md` and presentation documents for mathematical honesty.
* **Output Standard:** Produces a structured `AUDIT REPORT` containing a prioritized findings table (`P0` Blocker, `P1` Math/Stat, `P2` Contract/Robustness, `P3` Polish) and ends with an explicit convergence verdict: `STATUS: CLEAN` or `STATUS: ISSUES_FOUND`.

### B. Principal Sabermetrics Architect & Planner (`planner`)
* **Core Responsibilities:**
  - Receives the findings from the `auditor`.
  - Formulates a rigorous, step-by-step implementation blueprint before any code modifications occur.
  - Specifies exact file paths, target functions, parameter signatures, and edge-case guards.
  - Designs explicit unit and regression test specifications to be written in `tests/`.
  - Defines execution sequence ordered by dependency and severity.
* **Output Standard:** Produces an `IMPLEMENTATION PLAN` blueprint with finding-by-finding actionable specifications.

### C. Principal Research Software Engineer (`implementor`)
* **Core Responsibilities:**
  - Ingests the implementation blueprint from the `planner`.
  - Applies surgical code changes using localized edits, preserving documentation integrity and unrelated code.
  - Authors the designed regression tests in `tests/test_*.py`.
  - Executes test suites (`pytest tests/ -v`) to confirm 100% pass rates.
  - Validates SQLite $\leftrightarrow$ CSV parity if leaderboard or model outputs are affected.
* **Output Standard:** Produces an `IMPLEMENTATION REPORT` detailing files modified, regression tests added, and test execution results.

### D. Autonomous Master Orchestrator (`orchestrator`)
* **Core Responsibilities:**
  - Drives the multi-agent state machine across recursive cycles (up to a configurable maximum, default 3).
  - Evaluates audit verdicts: upon receiving `STATUS: CLEAN` with 0 unresolved `P0/P1/P2` findings, halts execution and issues a final Convergence Certificate.
  - Dispatches tasks sequentially between `auditor`, `planner`, and `implementor`.
  - Maintains convergence state and prevents infinite loops or oscillating edits.

---

## 3. Convergence & Termination Criteria

A recursive orchestration run terminates successfully when **all** of the following criteria are satisfied:
1. **Zero Critical/High/Medium Findings:** The `auditor` confirms 0 `P0`, 0 `P1`, and 0 `P2` findings.
2. **100% Test Suite Pass Rate:** All unit tests in `pytest tests/ -v` pass without failures or unexpected warnings.
3. **Data Contract Parity:** All exported CSV leaderboards match the SQLite production database with zero discrepancies (`max_diff < 1e-15`).
4. **No Regression Drift:** Re-auditing in Cycle $k+1$ verifies that fixes from Cycle $k$ did not introduce secondary bugs.

---

## 4. Invocation Guidelines

The orchestrator and specialized subagents are registered in the Antigravity runtime environment and can be invoked either programmatically via the agent tools (`invoke_subagent`, `define_subagent`) or through interactive slash commands:

```python
# Launching the Orchestrator via Python / Antigravity Agent API
invoke_subagent(
    Subagents=[{
        "TypeName": "orchestrator",
        "Role": "Autonomous Quality Orchestrator",
        "Prompt": "Execute the recursive Audit -> Plan -> Implement -> Audit orchestration cycle for sp26_sagb_dj.",
        "Model": "inherit",
        "Workspace": "inherit"
    }]
)
```
