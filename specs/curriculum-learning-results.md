# Curriculum Learning — Run #1 Results

**Date:** 2026-02-25/26
**WandB:** `charles-team/hackmatrix/runs/413203b0`
**Branch:** `curriculum-learning`
**Duration:** ~4.5 hours (635 updates, ~16,800s)

## Summary

Phase 1 (Siphon School) and Phase 2 (Increasing Pressure) succeeded at teaching siphon usage. Phase 3 (Full Game) failed instantly — the agent abandoned all strategic behavior when starting siphons were removed. The curriculum teaches "use siphons" but not "acquire siphons."

## Phase 1: Siphon School (u0–300) — SUCCESS

| Metric | Target | Achieved |
|--------|--------|----------|
| siphon_frac | >3% | **12.1%** (4x target) |
| program_frac | >0.1% | **1.0%** (10x target) |
| highest_stage | — | 9 (full victories) |
| entropy | >1.0 | 1.23 |
| return | >-1.0 to transition | -2.63 (fallback at 300 updates) |

**Observations:**
- Agent learned siphon→resource→program chain within ~50 updates
- Siphon usage stabilized at 12–13% and held steady for 250 updates
- Return plateaued at -2.6, never reaching -1.0 threshold — fallback triggered
- Phase 1 snapshot saved to `phase_1_complete.pkl`

## Phase 2: Increasing Pressure (u300–600) — PARTIAL SUCCESS

| Metric | Target | Achieved |
|--------|--------|----------|
| siphon_frac | >0.5% | **13.5%** (27x target) |
| siphon retention | sustained | stable throughout |
| return | >-1.2 to transition | -3.31 (fallback at 300 updates) |

**Observations:**
- Siphon behavior fully retained through transition (13.5% — actually increased)
- Agent adapted to harder environment: siphon_death_penalty improved from -0.145 to -0.130
- HP recovery tripled (learned to use healing programs)
- KL divergence elevated at 0.036 but stable (spec alarm: 0.10)
- Entropy held at 1.10 (no collapse)
- Return worse than Phase 1 due to more enemies and harsher penalties, but stable
- Phase 2 snapshot saved to `phase_2_complete.pkl`

### Phase 2 adaptation timeline

| Updates into Phase 2 | return | siphon_frac | siphon_death_pen | ep_length |
|-----------------------|--------|-------------|------------------|-----------|
| 5 (u305) | -3.33 | 12.9% | -0.145 | 5.3 |
| 75 (u375) | -3.28 | 13.0% | -0.135 | 5.5 |
| 130 (u430) | -3.21 | 12.6% | -0.128 | 5.5 |
| 220 (u520) | -3.34 | 13.2% | -0.130 | 5.1 |
| 300 (u600) | -3.33 | 13.5% | -0.133 | 5.1 |

Mid-phase improvement (u380–480) followed by regression. Agent found a local optimum but couldn't break through.

## Phase 3: Full Game (u600–635) — FAILURE

| Metric | Phase 2 (u600) | Phase 3 (u610) |
|--------|----------------|----------------|
| siphon_frac | 13.5% | **0.00%** |
| program_frac | 0.41% | **0.00%** |
| move_frac | 86.1% | **100.0%** |
| return | -3.33 | -2.48 |
| entropy | 1.10 | **0.95** (below alarm) |

**Observations:**
- All strategic behavior lost within a single update
- Agent reverted to pure movement (identical to pre-curriculum baseline)
- Return "improved" because agent stopped incurring siphon death penalties
- Entropy declined toward collapse (0.93 by u635)
- KL spiked to 0.062 and rising
- Training stopped at u635 to avoid wasting compute

## Root Cause Analysis

**The curriculum taught "siphon when you have siphons" but not "navigate to corners to acquire siphons."**

In Phases 1–2, the agent started every episode with data siphons in inventory (2 in Phase 1, 1 in Phase 2). It never needed to learn the prerequisite behavior: navigating to corner cells where data siphons spawn on the game board.

When Phase 3 removed starting siphons (starting_data_siphons=0), the agent faced an environment where:
1. Siphon action is invalid without a siphon in inventory
2. Getting a siphon requires walking to a corner cell first
3. The agent has zero experience with corner navigation for siphon acquisition
4. Pure movement gives a better immediate return than the unknown siphon-acquisition path

The agent rationally chose to abandon siphoning entirely.

## What Worked

1. **Curriculum concept is valid** — Phase 1 achieved 1000x improvement in siphon usage vs baseline
2. **Phase transitions preserve behavior** — siphon_frac held through Phase 1→2 transition
3. **Agent adapts to difficulty** — learned to heal, reduced siphon deaths in Phase 2
4. **Infrastructure works** — phase snapshots, rollback CLI, wandb logging all functional

## What Needs Fixing

The Phase 2→3 gap is the single point of failure. Options:

### Option A: Gradual siphon reduction
Add intermediate phases that reduce starting siphons gradually:
- Phase 2a: 1 starting siphon (current Phase 2)
- Phase 2b: 50% chance of starting with 1 siphon
- Phase 2c: 0 starting siphons but reduced enemy pressure

### Option B: Corner navigation reward
Add a reward signal for reaching corner cells (where data siphons spawn), independent of curriculum phase. This teaches the prerequisite "go to corners" behavior that Phase 3 requires.

### Option C: Siphon placement change
Place data siphons on non-corner cells (e.g., adjacent to blocks) so the agent encounters them during normal movement. This is a game environment change rather than a curriculum change.

### Option D: Extended Phase 2 with siphon-finding practice
Keep 1 starting siphon but also spawn extra siphons on the board with a bonus reward for collecting them. The agent learns to navigate to siphons while still having one to use for immediate reinforcement.

## Checkpoints Available

| File | Contents |
|------|----------|
| `phase_1_complete.pkl` | Phase 1 final weights (u300) — strong siphon usage, easy env |
| `phase_2_complete.pkl` | Phase 2 final weights (u600) — adapted siphon usage, medium env |
| `checkpoint_625.pkl` | Phase 3 weights (u625) — collapsed, not useful |

## Next Steps

1. Redesign Phase 2→3 transition (pick an option above)
2. Resume from `phase_2_complete.pkl` with the new transition
3. Do not need to retrain Phases 1–2
