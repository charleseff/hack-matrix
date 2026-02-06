# Training Validation Conventions

**Status:** Reference

Conventions for implementing and validating training-related specs. All training specs must follow these conventions and include a `## Training Validation` section.

## Environment Detection

Before launching a training job, check whether a TPU is available:

```bash
python -c "import jax; print(jax.devices())"
```

- **TPU available** (`TpuDevice`): Run full training validation as specified in the spec.
- **CPU/GPU only**: Skip or defer training validation unless the spec provides CPU-specific instructions. Note this in any status update.

The agent may be running on the same machine as the TPU. Training runs happen inside a tmux session. Check for an existing training process before starting a new one:

```bash
pgrep -af train_purejaxrl
```

## Workflow: Commit Before Train

Always commit and push code changes **before** starting a training run. This ensures:

- Wandb records the exact git commit used for each run
- Changes can be reverted if training regresses
- Clean separation between code changes and training evaluation

Sequence: implement → test → commit → train → monitor → evaluate results.

## Running and Monitoring Training

**Launch training in the background** and redirect output to a log file:

```bash
python python/scripts/train_purejaxrl.py [args] > /tmp/training.log 2>&1 &
```

**Monitor by parsing stdout/log output**, not wandb API. Stdout produces the same metrics with far less context overhead:

```bash
tail -20 /tmp/training.log
```

**Determining "enough data"** is the agent's judgment call based on the spec's success criteria. Guidelines:

| Validation type | Typical duration |
|-----------------|-----------------|
| Metric sanity checks (e.g., `reward/step_penalty ≈ -0.01`) | ~5-10 updates |
| Trend validation (e.g., entropy not collapsing) | ~50-200 updates |
| Convergence claims (e.g., return exceeds threshold) | 500+ updates, potentially 1+ hours |

If a metric is clearly not trending toward the target after a reasonable window, stop early and report findings.

**Stopping and resuming training:**

```bash
# Kill current training
kill $(pgrep -f train_purejaxrl)

# Resume from checkpoint (preserves wandb history)
python python/scripts/train_purejaxrl.py --resume --checkpoint-dir checkpoints [args]
```

## Spec Completion With Training Validation

When a spec's success criteria include training outcomes:

- Code changes alone are **not sufficient** to mark the spec Complete — the training validation must also pass
- If training results are inconclusive or not achievable within reason, report findings and leave the spec Active (not Complete)
- If the spec's training criteria are clearly unachievable (e.g., a hyperparameter change makes things worse), document the result and discuss next steps rather than forcing completion

## Per-Spec Training Validation Section

Each training spec should include a `## Training Validation` section with:

1. **Run command** — exact command with recommended arguments
2. **Metrics to monitor** — which stdout/wandb metrics to watch
3. **Success thresholds** — specific numeric criteria (e.g., "entropy > 1.0 at update 500")
4. **Expected duration** — approximate time to reach a verdict
5. **Early stopping signals** — when to abort a run that's clearly failing
