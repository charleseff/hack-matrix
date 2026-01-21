# Plans Directory

Historical implementation plans. These have been completed and their content consolidated into specs.

## Archive

The `archive/` directory contains completed plans preserved for historical reference:

| Plan | Implemented As |
|------|---------------|
| `plan-action-masking.md` | MaskablePPO in `train.py`, action masking in `gym_env.py` |
| `plan-clean-architecture.md` | `tryExecuteAction()`, `ActionResult` in GameState.swift |
| `plan-game-logic-updates.md` | `StartingBonus`, `scheduledTaskInterval` in GameState.swift |
| `plan-visual-cli.md` | `--visual-cli` flag, VisualGameController.swift |
| `reward-system-improvements.md` | RewardCalculator.swift (14 components) |
| `wor-24-part2-observation-space-plan.md` | Observation space in gym_env.py (10+23+40 features) |
| `manual-play-monitor.md` | `manual_play.py`, `observation_utils.py` |

## Current Specs

For active documentation, see [specs/](../specs/).
