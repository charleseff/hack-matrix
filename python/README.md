# HackMatrix RL Training

Python wrapper and training scripts for training reinforcement learning agents to play HackMatrix.

## Setup

1. **Build the Swift app** (from project root):
   ```bash
   xcodebuild -scheme HackMatrix -configuration Debug build
   ```

2. **Install Python dependencies**:
   ```bash
   cd python
   pip install -r requirements.txt
   ```

3. **Test the environment**:
   ```bash
   python test_env.py
   ```

## Usage

### Test the Environment

```bash
python test_env.py
```

This will:
- Create the environment
- Reset it and show initial observation
- Take 5 random actions
- Show rewards and game state

### Train a PPO Agent

```bash
python train_ppo.py --timesteps 1000000
```

Options:
- `--timesteps`: Total training steps (default: 1,000,000)
- `--save-freq`: Save checkpoint every N steps (default: 10,000)
- `--eval-freq`: Evaluate every N steps (default: 5,000)
- `--log-dir`: TensorBoard log directory (default: ./logs)
- `--model-dir`: Model save directory (default: ./models)

### Monitor Training

```bash
tensorboard --logdir ./logs
```

Then open http://localhost:6006 in your browser.

## Architecture

```
Python (Gymnasium env)
    ↓ JSON over subprocess
Swift (HeadlessGameCLI)
    ↓ Method calls
HeadlessGame
    ↓ State management
GameState
```

### Communication Protocol

Commands (Python → Swift):
- `{"action": "reset"}` - Reset game
- `{"action": "step", "actionIndex": 5}` - Take action
- `{"action": "getValidActions"}` - Get valid action indices

Responses (Swift → Python):
- `{"observation": {...}, "reward": 10.0, "done": false, "info": {}}`
- `{"validActions": [0, 1, 4, 5, 6]}`
- `{"error": "error message"}` - On error

### Observation Space

The observation is a dictionary with three components:

1. **Player state** (9 values):
   - `[row, col, hp, credits, energy, stage, turn, dataSiphons, baseAttack]`

2. **Grid state** (6×6×20):
   - 20 features per cell including enemies, blocks, transmissions, resources

3. **Flags** (1 value):
   - `[showActivated]`
   # todo(cff): does this need to be a flag vs just a player state?

### Action Space

31 discrete actions:
- 0-3: Movement (up, down, left, right)
- 4: Siphon
- 5-30: Programs (in ProgramType.allCases order)

Use `env.get_valid_actions()` to get valid actions for current state.

## Files

- `hack_env.py` - Gymnasium environment wrapper
- `test_env.py` - Test script
- `train_ppo.py` - PPO training script
- `requirements.txt` - Python dependencies

## Tips

- Start with shorter training runs (100k steps) to verify everything works
- Monitor TensorBoard to watch learning progress
- The agent will be slow at first - this is normal!
- Expect initial training to take several hours for meaningful results
- Use `--timesteps 10000` for quick testing
