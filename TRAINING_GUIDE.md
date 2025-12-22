# HackMatrix RL Training Guide

## Current Status (as of 2025-12-18)

### ✅ Completed
1. **Fixed observation space** - Added missing features:
   - Player score (what agent maximizes!)
   - Program action indices (which programs available)
   - Transmission spawn counts (risk assessment)

2. **Implemented reward shaping** - Intermediate rewards to guide learning:
   - Siphoning blocks: +0.005 per block
   - Acquiring NEW programs: +0.02 each
   - Score collection: +0.1 per point
   - Stage completion: +1.0
   - Game won (all 8 stages): score × 10.0 + 10 (HUGE!)
   - Death: 0.0

3. **Fixed entropy collapse issue**:
   - Increased `ent_coef` from 0.01 → 0.03 → 0.1
   - Should prevent premature convergence
   - Maintains exploration throughout training

4. **Added --resume functionality** - Can continue from checkpoints

5. **Added reward logging** - For manual testing/verification

### 🎯 Current Training Configuration
```python
# train_maskable_ppo.py settings:
learning_rate=3e-4
n_steps=2048
batch_size=64
n_epochs=10
gamma=0.99
gae_lambda=0.95
clip_range=0.2
ent_coef=0.1  # High exploration to prevent collapse
```

---

## 🚀 How to Start Training (Fresh)

```bash
# 1. Make sure Swift app is built with latest reward shaping
cd /Users/charles/dev/868-hack-2
xcodebuild -scheme HackMatrix -configuration Debug

# 2. Start training
cd python
python train_maskable_ppo.py \
    --timesteps 100000000 \
    --save-freq 50000 \
    --eval-freq 99999999
```

---

## 📊 What to Monitor in TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir logs/
```

### Critical Metrics

1. **`train/entropy_loss`** (MOST IMPORTANT!)
   - Should stay around **-1.0 to -1.5**
   - ❌ If approaching 0 → training will collapse
   - ✅ Stable negative value → healthy exploration

2. **`rollout/ep_rew_mean`**
   - Should climb steadily (but might plateau temporarily)
   - With reward shaping, expect faster initial climb
   - Final performance depends on how far agent progresses

3. **`rollout/ep_len_mean`**
   - Longer episodes = agent surviving more turns
   - Should increase as agent learns

4. **`train/policy_gradient_loss`**
   - Should NOT be near zero (means policy updating)

5. **`train/value_loss`**
   - Should decrease over time but stay > 0

### Warning Signs

| Metric | Bad Sign | What It Means |
|--------|----------|---------------|
| `entropy_loss` | → 0 | Policy collapsed, stop training! |
| `approx_kl` | = 0 | No policy updates happening |
| `clip_fraction` | = 0 | No gradient clipping = no changes |
| `ep_rew_mean` | Flatline for >2M steps | Stuck in local optimum |

---

## 🧪 Manual Testing & Verification

### Test Rewards "Feel Right"

```bash
# Run game with reward logging
cd /Users/charles/dev/868-hack-2
./DerivedData/Build/Products/Debug/HackMatrix.app/Contents/MacOS/HackMatrix
```

**What to verify:**
- Siphon block → See `~0.005` reward
- Acquire program → See `~0.02` bonus
- Collect 10 data points → See `~1.0` reward (10 × 0.1)
- Complete stage → See `~1.0+` reward
- Win game with 100 points → See `1010.0` reward!

**Console output looks like:**
```
🎯 Action: siphon → Reward: 0.025 | Score: 15 | Stage: 1
🎯 Action: direction(up) → Reward: 0.100 | Score: 16 | Stage: 1
🎯 Action: direction(right) → Reward: 1.000 | Score: 16 | Stage: 2
```

---

## 🎮 Watch Trained Agent Play

Once you have a trained model, watch it:

```bash
cd python

# Run test_env.py with visual mode and your trained model
# (need to modify test_env.py to load model - see below)
```

**Quick script to watch agent:**
```python
from hack_env import HackEnv
from sb3_contrib import MaskablePPO

# Load your best model
model = MaskablePPO.load('./models/maskable_ppo_TIMESTAMP/best_model.zip')
env = HackEnv(visual=True)

for episode in range(3):
    obs, info = env.reset()
    total_reward = 0
    done = False

    while not done:
        action_mask = info.get('action_mask')
        action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f'Episode {episode+1}: reward={total_reward:.3f}')

env.close()
```

---

## ⚡ GPU Training (Future)

### CPU vs GPU Bottleneck Check

Your training has two phases:
1. **Rollout collection** (CPU-bound) - Swift subprocess runs game logic
2. **Training updates** (GPU-bound) - Neural network forward/backward passes

**To check if GPU would help:**

```python
# Check current device
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

**To use GPU (if available):**

In `train_maskable_ppo.py`, add `device='cuda'`:
```python
model = MaskablePPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    device='cuda',  # Add this
    tensorboard_log=run_log_dir,
    ...
)
```

**Note:** Swift subprocess (game logic) always runs on CPU. GPU only accelerates neural network training.

---

## 🔍 Troubleshooting

### Problem: Entropy collapsed to 0

**Symptoms:**
- `entropy_loss = 0`
- `approx_kl = 0`
- `ep_rew_mean` flatlined
- Policy not updating

**Solution:**
1. Stop training immediately
2. Increase `ent_coef` (try 0.15 or 0.2)
3. Start fresh (DON'T resume from collapsed model)

### Problem: Training too slow

**Check:**
- FPS in training output (should be >500)
- CPU usage (should be high during rollout)
- If maxing CPU and FPS low → Swift subprocess bottleneck
- If GPU available but not used → add `device='cuda'`

### Problem: Rewards seem wrong

**Verify:**
1. Manually play game and check console logs
2. Ensure Swift app rebuilt after reward changes
3. Check that Python env matches Swift rewards

---

## 💡 Future Improvements (If Current Training Doesn't Work)

### Option 1: Curriculum Learning
Start with easier scenarios, gradually increase difficulty:
- Train on stage 1 only first
- Then stages 1-2
- Gradually add more stages
- Helps with credit assignment

### Option 2: Entropy Coefficient Schedule
Start high, gradually decrease:
```python
# Instead of fixed 0.1, use:
# Start: 0.2 (high exploration)
# End: 0.05 (lower exploration after learning)
```

### Option 3: Different RL Algorithm
If PPO struggles:
- Try DQN (simpler but less sample efficient)
- Try SAC (good for exploration)
- Try Rainbow DQN (more sophisticated)

### Option 4: Imitation Learning
If agent struggles to learn from scratch:
- Record human gameplay
- Use behavioral cloning to bootstrap
- Then fine-tune with RL

---

## 📁 Important Files

| File | Purpose |
|------|---------|
| `python/train_maskable_ppo.py` | Training script |
| `python/hack_env.py` | Gymnasium environment wrapper |
| `HackMatrix/GameState.swift` | Game logic & reward calculation |
| `HackMatrix/ObservationBuilder.swift` | Converts game state → ML observation |
| `HackMatrix/GameCommandProtocol.swift` | JSON communication protocol |

---

## 🎯 Expected Timeline

**Realistic expectations:**

1. **First 1M steps (few hours):**
   - Agent learns basic movement
   - Learns to siphon blocks
   - Maybe completes stage 1 occasionally

2. **1M-5M steps (overnight):**
   - Consistently completes stage 1
   - Starts completing stage 2
   - Learns some program usage

3. **5M-20M steps (days):**
   - Completes multiple stages
   - Strategic program usage
   - Better combat tactics

4. **20M+ steps:**
   - Completing all 8 stages
   - Optimizing for score
   - Near-optimal play

**Note:** These are rough estimates. Could be faster or slower depending on:
- Entropy coefficient (higher = slower convergence)
- Reward structure effectiveness
- Random seed luck
- Hardware speed

---

## 🚨 Important Reminders

1. **NEVER resume from a collapsed model** - entropy can't recover
2. **Monitor entropy_loss regularly** - catch collapse early
3. **Don't add enemy kill rewards** - creates farming exploit
4. **Rebuild Swift app after reward changes** - Python expects matching rewards
5. **Patience is key** - RL takes time, especially for complex games

---

## 📞 Quick Reference Commands

```bash
# Build Swift app
cd /Users/charles/dev/868-hack-2
xcodebuild -scheme HackMatrix -configuration Debug

# Start training (fresh)
cd python
python train_maskable_ppo.py --timesteps 100000000

# Resume training
python train_maskable_ppo.py --resume ./models/maskable_ppo_TIMESTAMP/best_model.zip

# Start TensorBoard
tensorboard --logdir logs/

# Find checkpoints
ls -lt models/maskable_ppo_*/

# Test environment
python test_env.py --visual
```

---

## 🎓 Key Learnings

1. **Observation completeness matters** - Missing score/programs prevented learning
2. **Entropy collapse is real** - Need sufficient `ent_coef` (0.1+)
3. **Reward shaping helps** - Intermediate rewards guide learning
4. **Credit assignment is hard** - Long episodes (8 stages) make learning slow
5. **Don't reward exploits** - Enemy farming would break training
6. **Manual testing valuable** - Verify rewards "feel right" before big training runs

---

Good luck with training! 🚀
