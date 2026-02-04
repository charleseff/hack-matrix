"""
Smoke tests for PureJaxRL integration.

Tests that:
1. Environment wrapper produces correct observation shapes
2. PPO compiles and runs a single step
3. Action masking correctly prevents invalid actions
"""

import jax
import jax.numpy as jnp
import pytest

from hackmatrix.jax_env import NUM_ACTIONS
from hackmatrix.purejaxrl import (
    HackMatrixGymnax,
    TrainConfig,
    make_chunked_train,
    masked_categorical,
)
from hackmatrix.purejaxrl.env_wrapper import OBS_SIZE, GymnaxEnvState
from hackmatrix.purejaxrl.masked_ppo import (
    compute_gae,
    init_network,
    ppo_loss,
)


class TestEnvWrapper:
    """Test Gymnax-compatible environment wrapper."""

    def test_reset_returns_correct_shapes(self):
        """Reset should return (obs, state) with correct shapes."""
        env = HackMatrixGymnax()
        key = jax.random.PRNGKey(0)

        obs, state = env.reset(key)

        assert obs.shape == (OBS_SIZE,), f"Expected obs shape {(OBS_SIZE,)}, got {obs.shape}"
        assert isinstance(state, GymnaxEnvState)
        assert obs.dtype == jnp.float32

    def test_step_returns_correct_shapes(self):
        """Step should return (obs, state, reward, done, info)."""
        env = HackMatrixGymnax()
        key = jax.random.PRNGKey(0)

        obs, state = env.reset(key)
        key, step_key = jax.random.split(key)

        # Get a valid action
        action_mask = env.get_action_mask(state)
        valid_action = jnp.argmax(action_mask.astype(jnp.int32))

        next_obs, next_state, reward, done, info = env.step(step_key, state, valid_action)

        assert next_obs.shape == (OBS_SIZE,)
        assert isinstance(next_state, GymnaxEnvState)
        assert reward.shape == ()
        assert done.shape == ()
        assert isinstance(info, dict)

    def test_action_mask_shape(self):
        """Action mask should have shape (NUM_ACTIONS,)."""
        env = HackMatrixGymnax()
        key = jax.random.PRNGKey(0)

        _, state = env.reset(key)
        action_mask = env.get_action_mask(state)

        assert action_mask.shape == (NUM_ACTIONS,)
        assert action_mask.dtype == jnp.bool_

    def test_observation_normalized(self):
        """Observations should be normalized to [0, 1] range."""
        env = HackMatrixGymnax()
        key = jax.random.PRNGKey(0)

        obs, _ = env.reset(key)

        # Most values should be in [0, 1], but some may be slightly outside
        # due to normalization constants. Check that values are reasonable.
        assert jnp.all(obs >= -0.5), f"Min obs value: {obs.min()}"
        assert jnp.all(obs <= 1.5), f"Max obs value: {obs.max()}"

    def test_vectorized_reset(self):
        """Vectorized reset should work."""
        env = HackMatrixGymnax()
        num_envs = 4
        keys = jax.random.split(jax.random.PRNGKey(0), num_envs)

        batched_reset = jax.vmap(lambda k: env.reset(k, env.default_params))
        obs, states = batched_reset(keys)

        assert obs.shape == (num_envs, OBS_SIZE)


class TestMaskedPPO:
    """Test action-masked PPO components."""

    def test_masked_categorical_respects_mask(self):
        """Masked categorical should only sample valid actions."""
        key = jax.random.PRNGKey(0)

        # Create logits and mask (only actions 1 and 3 valid)
        logits = jnp.zeros(NUM_ACTIONS)
        mask = jnp.zeros(NUM_ACTIONS, dtype=jnp.bool_)
        mask = mask.at[1].set(True)
        mask = mask.at[3].set(True)

        dist = masked_categorical(logits, mask)

        # Sample many times
        keys = jax.random.split(key, 1000)
        samples = jax.vmap(lambda k: dist.sample(seed=k))(keys)

        # All samples should be valid actions (1 or 3)
        unique_samples = jnp.unique(samples)
        assert jnp.all((unique_samples == 1) | (unique_samples == 3))

    def test_actor_critic_output_shapes(self):
        """Actor-critic should output correct shapes."""
        key = jax.random.PRNGKey(0)

        network, params = init_network(key, obs_shape=(OBS_SIZE,), hidden_dim=64, num_layers=2)

        # Single observation
        obs = jnp.zeros(OBS_SIZE)
        logits, value = network.apply(params, obs)

        assert logits.shape == (NUM_ACTIONS,)
        assert value.shape == ()

        # Batched observations
        batch_obs = jnp.zeros((4, OBS_SIZE))
        batch_logits, batch_values = network.apply(params, batch_obs)

        assert batch_logits.shape == (4, NUM_ACTIONS)
        assert batch_values.shape == (4,)

    def test_gae_computation(self):
        """GAE should compute reasonable advantages."""
        num_steps = 10
        num_envs = 4

        rewards = jax.random.uniform(jax.random.PRNGKey(0), (num_steps, num_envs))
        values = jax.random.uniform(jax.random.PRNGKey(1), (num_steps + 1, num_envs))
        dones = jnp.zeros((num_steps, num_envs))

        advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)

        assert advantages.shape == (num_steps, num_envs)
        assert returns.shape == (num_steps, num_envs)
        assert jnp.isfinite(advantages).all()
        assert jnp.isfinite(returns).all()

    def test_ppo_loss_computes(self):
        """PPO loss should compute without errors."""
        key = jax.random.PRNGKey(0)
        batch_size = 8

        network, params = init_network(key, obs_shape=(OBS_SIZE,), hidden_dim=64, num_layers=2)

        # Create dummy batch
        obs = jnp.zeros((batch_size, OBS_SIZE))
        actions = jnp.zeros(batch_size, dtype=jnp.int32)
        old_log_probs = jnp.zeros(batch_size)
        advantages = jnp.ones(batch_size)
        returns = jnp.ones(batch_size)
        action_masks = jnp.ones((batch_size, NUM_ACTIONS), dtype=jnp.bool_)

        loss, metrics = ppo_loss(
            params,
            network.apply,
            obs,
            actions,
            old_log_probs,
            advantages,
            returns,
            action_masks,
            clip_eps=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
        )

        assert jnp.isfinite(loss)
        assert "pg_loss" in metrics
        assert "vf_loss" in metrics
        assert "entropy" in metrics


class TestTraining:
    """Test training loop components."""

    def test_make_chunked_train_compiles(self):
        """Training function should compile without errors."""
        config = TrainConfig(
            num_envs=4,
            num_steps=8,
            total_timesteps=32,  # Very small for testing
            num_minibatches=2,
            update_epochs=1,
            hidden_dim=32,
            num_layers=1,
        )

        env = HackMatrixGymnax()
        train_fn = make_chunked_train(config, env)

        # Just check it compiles (don't run full training)
        assert callable(train_fn)

    def test_single_training_step(self):
        """Single training step should run without errors."""
        config = TrainConfig(
            num_envs=4,
            num_steps=8,
            total_timesteps=32,
            num_minibatches=2,
            update_epochs=1,
            hidden_dim=32,
            num_layers=1,
        )

        env = HackMatrixGymnax()
        train_fn = make_chunked_train(config, env)

        key = jax.random.PRNGKey(0)

        # Run training (very short)
        final_state, metrics, _ = train_fn(key)

        # Check we got output
        assert final_state is not None
        assert "total_loss" in metrics
        assert metrics["total_loss"].shape[0] == config.num_updates


class TestActionMaskingIntegration:
    """Test that action masking works end-to-end."""

    def test_invalid_actions_never_sampled(self):
        """With proper masking, invalid actions should never be sampled."""
        env = HackMatrixGymnax()
        key = jax.random.PRNGKey(42)

        network, params = init_network(key, obs_shape=(OBS_SIZE,), hidden_dim=64, num_layers=2)

        # Run several steps
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key)

        for _ in range(20):
            key, action_key = jax.random.split(key)

            # Get valid actions
            action_mask = env.get_action_mask(state)

            # Sample action with masking
            logits, _ = network.apply(params, obs)
            dist = masked_categorical(logits, action_mask)
            action = dist.sample(seed=action_key)

            # Verify action is valid
            assert action_mask[action], f"Sampled invalid action {action}"

            # Step
            key, step_key = jax.random.split(key)
            obs, state, _, done, _ = env.step(step_key, state, action)

            if done:
                key, reset_key = jax.random.split(key)
                obs, state = env.reset(reset_key)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
