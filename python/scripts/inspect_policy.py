"""
Inspect the policy's action probabilities for debugging.
Shows which actions the agent prefers in different states.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from hackmatrix import HackEnv


def mask_fn(env):
    return env._get_action_mask()


def get_action_names():
    """Get human-readable action names."""
    return [
        "Move Up", "Move Down", "Move Left", "Move Right", "Siphon",
        # Programs (you can customize these based on your ProgramType enum)
        "Push", "Pull", "Crash", "Warp", "Poly", "Wait", "Debug", "Row", "Col",
        "Undo", "Step", "Siph+", "Exch", "Show", "Reset", "Calm", "D-Bomb",
        "Delay", "Anti-V", "Score", "Reduc", "Atk+", "Hack"
    ]


def inspect_policy(model_path: str, num_steps: int = 10):
    """
    Load a trained model and show action probabilities for N steps.

    Args:
        model_path: Path to saved model checkpoint
        num_steps: Number of steps to inspect
    """
    print(f"Loading model from: {model_path}\n")

    # Create environment
    env = HackEnv(debug=False, info=False)
    env = ActionMasker(env, mask_fn)

    # Load model
    model = MaskablePPO.load(model_path)

    # Get action names
    action_names = get_action_names()

    # Reset environment
    obs, info = env.reset()

    print("=" * 80)
    print("POLICY INSPECTION - Action Probabilities")
    print("=" * 80)

    for step in range(num_steps):
        print(f"\n{'='*80}")
        print(f"STEP {step + 1}")
        print(f"{'='*80}")

        # Get action mask
        action_mask = env.action_masks()

        # Get action probabilities from the policy
        # The model's policy network outputs a distribution over actions
        obs_tensor = {
            key: np.array([value]) for key, value in obs.items()
        }

        # Get the action distribution (this is what predict uses internally)
        # Access the policy's forward pass
        import torch

        model.policy.set_training_mode(False)
        with torch.no_grad():
            # Get features from observation
            features = model.policy.extract_features(obs_tensor)
            # Get latent policy features
            latent_pi = model.policy.mlp_extractor.forward_actor(features)
            # Get action logits
            action_logits = model.policy.action_net(latent_pi)

            # Apply action mask (set invalid actions to very negative logits)
            masked_logits = action_logits.clone()
            masked_logits[0, action_mask == 0] = -1e8

            # Convert to probabilities
            import torch.nn.functional as F
            action_probs = F.softmax(masked_logits, dim=1)[0].detach().cpu().numpy()

        # Show game state
        print(f"\nGame State:")
        print(f"  Stage: {obs['player'][5]:.0f}/8")
        print(f"  Position: ({obs['player'][0]:.0f}, {obs['player'][1]:.0f})")
        print(f"  HP: {obs['player'][2]:.0f}")
        print(f"  Credits: {obs['player'][3]:.0f}")
        print(f"  Energy: {obs['player'][4]:.0f}")

        # Show action probabilities
        print(f"\nAction Probabilities (valid actions only):")
        print(f"{'Action':<15} {'Probability':>12} {'Bar':>20}")
        print("-" * 50)

        # Sort by probability (descending)
        valid_actions = np.where(action_mask)[0]
        sorted_actions = sorted(valid_actions, key=lambda i: -action_probs[i])

        for action_idx in sorted_actions[:10]:  # Show top 10
            prob = action_probs[action_idx]
            action_name = action_names[action_idx] if action_idx < len(action_names) else f"Action {action_idx}"
            bar = "█" * int(prob * 50)  # Visual bar
            print(f"{action_name:<15} {prob*100:>10.2f}%  {bar}")

        # Show entropy (measure of randomness)
        # High entropy = exploring, low entropy = deterministic
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))
        max_entropy = np.log(len(valid_actions))  # Maximum possible entropy
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        print(f"\nPolicy Entropy: {entropy:.3f} (normalized: {normalized_entropy:.1%})")
        print(f"  - 0% = completely deterministic (always picks same action)")
        print(f"  - 100% = completely random (all actions equally likely)")

        # Take action using the policy
        action, _ = model.predict(obs, action_masks=action_mask, deterministic=False)
        chosen_action_name = action_names[action] if action < len(action_names) else f"Action {action}"
        chosen_prob = action_probs[action]

        print(f"\nChosen Action: {chosen_action_name} (probability: {chosen_prob*100:.1f}%)")

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        print(f"Reward: {reward:.2f}")

        if done:
            print(f"\n{'='*80}")
            print(f"EPISODE ENDED after {step + 1} steps")
            print(f"Total reward: {reward:.2f}")
            print(f"{'='*80}")
            break

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect policy action probabilities")
    parser.add_argument("model_path", type=str, help="Path to saved model checkpoint")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps to inspect")

    args = parser.parse_args()

    inspect_policy(args.model_path, args.steps)
