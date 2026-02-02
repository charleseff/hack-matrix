"""
Action-masked PPO implementation for HackMatrix.

This module provides:
- Masked categorical distribution (pure JAX, no distrax dependency)
- Actor-Critic network with Flax
- PPO loss function with action masking support
- Transition dataclass for storing rollout data
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import struct

from hackmatrix.jax_env import NUM_ACTIONS


class MaskedCategorical:
    """Categorical distribution with action masking (pure JAX implementation).

    Replaces distrax.Categorical for compatibility with JAX 0.7.0+.
    Invalid actions have their logits set to a large negative value.
    """

    # Large negative value for masking (not -inf to avoid NaN issues)
    MASK_VALUE = -1e9

    def __init__(self, logits: jax.Array):
        """Initialize with (already masked) logits."""
        self.logits = logits
        # Compute log probabilities using log_softmax for numerical stability
        self._log_probs = jax.nn.log_softmax(logits, axis=-1)
        # Cache probabilities for entropy calculation
        self._probs = jax.nn.softmax(logits, axis=-1)

    def sample(self, seed: jax.Array) -> jax.Array:
        """Sample from the distribution using Gumbel-max trick."""
        # Gumbel-max trick: argmax(logits + Gumbel noise) samples from categorical
        gumbel_noise = jax.random.gumbel(seed, shape=self.logits.shape)
        return jnp.argmax(self.logits + gumbel_noise, axis=-1)

    def log_prob(self, actions: jax.Array) -> jax.Array:
        """Compute log probability of actions."""
        # Handle both batched and unbatched cases
        return jnp.take_along_axis(self._log_probs, actions[..., None], axis=-1).squeeze(-1)

    def entropy(self) -> jax.Array:
        """Compute entropy of the distribution."""
        # Entropy = -sum(p * log(p))
        # Only sum over actions with non-negligible probability
        # Using where to avoid 0 * -inf = nan
        return -jnp.sum(jnp.where(self._probs > 1e-8, self._probs * self._log_probs, 0.0), axis=-1)


@struct.dataclass
class Transition:
    """Single transition for PPO training.

    Stores all data needed to compute PPO loss:
    - obs: Observation at time t
    - action: Action taken at time t
    - reward: Reward received after action
    - done: Episode termination flag
    - log_prob: Log probability of action under policy at time t
    - value: Value estimate at time t
    - action_mask: Valid action mask at time t (for consistent log_prob computation)
    - episode_return: Total return for completed episode (0 if episode not done)
    """

    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    log_prob: jax.Array
    value: jax.Array
    action_mask: jax.Array
    episode_return: jax.Array  # Completed episode return (0 if not done)


def masked_categorical(logits: jax.Array, mask: jax.Array) -> MaskedCategorical:
    """Create categorical distribution with invalid actions masked.

    Invalid actions have their logits set to -inf (zero probability
    after softmax). This ensures the agent can only sample valid actions.

    Args:
        logits: Raw logits from policy network, shape (..., num_actions)
        mask: Boolean mask where True = valid action, shape (..., num_actions)

    Returns:
        MaskedCategorical distribution that only samples from valid actions
    """
    # Set invalid action logits to large negative value (will be ~0 probability after softmax)
    masked_logits = jnp.where(mask, logits, MaskedCategorical.MASK_VALUE)
    return MaskedCategorical(logits=masked_logits)


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO.

    Shared feature extractor with separate actor (policy) and critic (value) heads.

    Architecture:
        Input -> [Dense -> ReLU] x num_layers -> actor_head -> logits
                                               -> critic_head -> value
    """

    action_dim: int = NUM_ACTIONS
    hidden_dim: int = 256
    num_layers: int = 2

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Forward pass.

        Args:
            x: Observation, shape (batch, obs_dim) or (obs_dim,)

        Returns:
            logits: Policy logits, shape (..., action_dim)
            value: Value estimate, shape (..., 1)
        """
        # Shared feature extraction
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)

        # Actor head (policy logits)
        logits = nn.Dense(self.action_dim)(x)

        # Critic head (value)
        value = nn.Dense(1)(x)
        value = value.squeeze(-1)  # Remove last dimension

        return logits, value


def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: Rewards, shape (num_steps, num_envs)
        values: Value estimates, shape (num_steps + 1, num_envs)
            (includes bootstrap value at the end)
        dones: Done flags, shape (num_steps, num_envs)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        advantages: GAE advantages, shape (num_steps, num_envs)
        returns: Returns (advantages + values[:-1]), shape (num_steps, num_envs)
    """

    def _compute_gae_step(carry, inp):
        gae, next_value = carry
        reward, value, done = inp

        # TD error: r + gamma * V(s') * (1 - done) - V(s)
        delta = reward + gamma * next_value * (1.0 - done) - value

        # GAE: delta + gamma * lambda * (1 - done) * GAE(t+1)
        gae = delta + gamma * gae_lambda * (1.0 - done) * gae

        return (gae, value), gae

    # Scan backwards through time
    # values[:-1] are V(s_t), values[-1] is V(s_{T+1}) for bootstrap
    init_carry = (jnp.zeros_like(values[-1]), values[-1])
    inputs = (rewards, values[:-1], dones)

    # Reverse inputs for backward scan
    reversed_inputs = jax.tree.map(lambda x: x[::-1], inputs)

    _, advantages = jax.lax.scan(_compute_gae_step, init_carry, reversed_inputs)

    # Reverse advantages to correct order
    advantages = advantages[::-1]
    returns = advantages + values[:-1]

    return advantages, returns


def ppo_loss(
    params,
    apply_fn,
    obs: jax.Array,
    actions: jax.Array,
    old_log_probs: jax.Array,
    advantages: jax.Array,
    returns: jax.Array,
    action_masks: jax.Array,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
) -> tuple[jax.Array, dict]:
    """Compute PPO loss with action masking.

    Args:
        params: Network parameters
        apply_fn: Network forward function
        obs: Observations, shape (batch,) + obs_shape
        actions: Actions taken, shape (batch,)
        old_log_probs: Log probabilities from rollout, shape (batch,)
        advantages: GAE advantages, shape (batch,)
        returns: Returns for value loss, shape (batch,)
        action_masks: Valid action masks, shape (batch, num_actions)
        clip_eps: PPO clipping epsilon
        vf_coef: Value function loss coefficient
        ent_coef: Entropy bonus coefficient

    Returns:
        total_loss: Scalar loss
        metrics: Dictionary of loss components
    """
    # Forward pass
    logits, values = apply_fn(params, obs)

    # Create masked distribution for this batch
    dist = masked_categorical(logits, action_masks)

    # New log probabilities
    log_probs = dist.log_prob(actions)

    # Entropy bonus (encourage exploration)
    entropy = dist.entropy()

    # Importance sampling ratio
    ratio = jnp.exp(log_probs - old_log_probs)

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Clipped surrogate loss
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

    # Value function loss
    vf_loss = 0.5 * ((values - returns) ** 2).mean()

    # Entropy loss (negative for maximization)
    ent_loss = -entropy.mean()

    # Total loss
    total_loss = pg_loss + vf_coef * vf_loss + ent_coef * ent_loss

    metrics = {
        "pg_loss": pg_loss,
        "vf_loss": vf_loss,
        "ent_loss": ent_loss,
        "entropy": entropy.mean(),
        "total_loss": total_loss,
        "approx_kl": ((ratio - 1) - jnp.log(ratio)).mean(),
        "clip_frac": (jnp.abs(ratio - 1) > clip_eps).mean(),
    }

    return total_loss, metrics


def init_network(
    key: jax.Array,
    obs_shape: tuple[int, ...],
    hidden_dim: int = 256,
    num_layers: int = 2,
) -> tuple[ActorCritic, dict]:
    """Initialize actor-critic network.

    Args:
        key: JAX random key
        obs_shape: Shape of observation
        hidden_dim: Hidden layer dimension
        num_layers: Number of hidden layers

    Returns:
        network: ActorCritic module
        params: Initialized parameters
    """
    network = ActorCritic(
        action_dim=NUM_ACTIONS,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )

    dummy_obs = jnp.zeros(obs_shape)
    params = network.init(key, dummy_obs)

    return network, params
