# Presentation & Research Documentation

This directory contains research findings, slide presentations, and pilot analyses for the Trajectory-Forecast Swing Decision (DVA) project.

## Provenance & Historical Artifacts Notice

> [!NOTE]
> **Historical Presentation Artifact:**
> `SAGB Conference Presentation - Chris Song.pdf` represents the historical slide deck delivered at the April 2026 SAGB Conference. It uses early prototype metrics (e.g. static commitment thresholds and heuristic score blends).
>
> **Current Authoritative Pipeline:**
> All current production metrics, models, and leaderboards reflect the upgraded October 2026 methodology:
> - **Empirical Bayes Count-Dependent EV-DVA:** (`ev_dva_eb`, `alignment_dva_eb`) estimated using RE24 count-transition run values and variance-component shrinkage.
> - **Inferred Information-Use Horizons:** ($\tau^*_i$) estimated via continuous Information Center of Mass with closed-form delta-method standard errors.
> - **Deception Over Expected:** (`induced_mirage_chase_oe`) separating trajectory belief shifts from calibration noise.
> - **Boundary-Orthogonal Uncertainty:** ($\sigma_\perp$) controlling for Euclidean distance to the nearest zone boundary.
>
> The authoritative leaderboards reside in [`data/leaderboards/`](../../data/leaderboards/) (e.g., `2025_dva_leaderboard.csv`) and are persisted in SQLite at `data/mlb.db`.

## Contents

- `PRESENTATION_FINDINGS_2020_2025.md`: Comprehensive analytical findings, methodology breakdowns, and Dodgers advance scouting applications across 2020–2025.
- `SAGB Conference Presentation - Chris Song.pdf`: Historical April 2026 conference presentation slide deck.
- `DODGERS_RESEARCH_IDEAS_ONE_WEEK.md`: Prioritized quantitative research directions.
- `RELATED_WORK_FOR_RESEARCH_IDEAS.md`: Literature review and academic positioning.
- `RESEARCH_IDEA_PILOT_EVIDENCE.md`: Empirical pilot evaluation data.
- `research_idea_pilot.py`: Pilot script evaluating research extensions.
