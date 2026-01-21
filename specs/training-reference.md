# RL Training Reference

Reference documentation for training HackMatrix agents with reinforcement learning.

## Training Commands

```bash
# Build Swift binary (headless mode for training)
swift build

# Start training
cd python && source venv/bin/activate
python scripts/train.py --timesteps 100000000 --save-freq 50000

# Resume from checkpoint
python scripts/train.py --resume ./models/MODEL_DIR/best_model.zip

# Train without W&B logging
python scripts/train.py --no-wandb

# Watch trained agent play (requires Xcode build for GUI)
python scripts/watch_trained_agent.py --model ./models/MODEL_DIR/best_model.zip
```

## W&B Monitoring

Training logs to Weights & Biases project `hackmatrix`. Key metrics to watch:

### Critical Metrics

| Metric | Healthy Range | Warning Sign |
|--------|---------------|--------------|
| `train/entropy_loss` | -1.0 to -1.5 | Approaching 0 = collapse |
| `rollout/ep_rew_mean` | Climbing steadily | Flatline for extended period |
| `rollout/ep_len_mean` | Increasing | Stuck at low value |
| `train/policy_gradient_loss` | Non-zero | Near zero = no learning |
| `train/approx_kl` | Non-zero | Zero = no policy updates |

### Episode Metrics (Custom)

| Metric | Description |
|--------|-------------|
| `episode/reward_stage` | Stage completion rewards |
| `episode/reward_kills` | Kill rewards |
| `episode/final_stage` | Stage reached at episode end |
| `episode/final_score` | Score at episode end |

### Entropy Collapse

**Symptoms:**
- `entropy_loss` approaching 0
- `approx_kl` = 0
- `ep_rew_mean` flatlined
- `clip_fraction` = 0

**Solutions:**
1. Stop training immediately
2. Increase `ent_coef` (try 0.15 or 0.2)
3. Start fresh - do NOT resume from collapsed model

**Prevention:**
- Use `ent_coef >= 0.1` for stable exploration
- Monitor entropy regularly during training

## Training Configuration

Current defaults in `train.py`:

```python
learning_rate=3e-4
n_steps=2048
batch_size=64
n_epochs=10
gamma=0.99
gae_lambda=0.95
clip_range=0.2
ent_coef=0.1  # High to prevent entropy collapse
```

## GPU Acceleration

Training has two phases:
1. **Rollout collection** (CPU-bound) - Swift subprocess runs game logic
2. **Training updates** (GPU-bound) - Neural network forward/backward passes

The Swift subprocess always runs on CPU. GPU only accelerates neural network training.

```python
# Check CUDA availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# Enable GPU in training config
device='cuda'
```

## Troubleshooting

### Training Too Slow

**Check:**
- FPS in training output (target: >500)
- CPU usage during rollout (should be high)
- If GPU available but not used, enable `device='cuda'`

### Rewards Seem Wrong

**Verify:**
1. Rebuild Swift binary after reward changes: `swift build`
2. Check reward breakdown in episode info
3. Test manually with `python scripts/manual_play.py`

### Model Won't Load

- Observation space changed = incompatible checkpoints
- Must retrain from scratch after observation space updates

## Key Files

| File | Purpose |
|------|---------|
| `python/scripts/train.py` | Main training script |
| `python/hackmatrix/gym_env.py` | Gymnasium environment wrapper |
| `python/hackmatrix/training_config.py` | Hyperparameter configuration |
| `HackMatrix/GameState.swift` | Game logic |
| `HackMatrix/RewardCalculator.swift` | Reward calculation |
| `HackMatrix/ObservationBuilder.swift` | State encoding |

## Key Learnings

1. **Observation completeness matters** - Missing features prevent learning
2. **Entropy collapse is real** - Use `ent_coef >= 0.1`
3. **Reward shaping helps** - Intermediate rewards guide learning
4. **Credit assignment is hard** - Long episodes (8 stages) slow learning
5. **Don't reward exploits** - Enemy farming breaks training
6. **Manual testing is valuable** - Verify rewards before long runs
