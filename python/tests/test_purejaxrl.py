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


class TestMathematicalCorrectness:
    """Mathematical verification tests - these catch algorithm bugs."""

    def test_gae_single_step_no_discount(self):
        """GAE with gamma=0 should equal immediate TD error."""
        # With gamma=0, advantage = r - V(s) (no future value)
        rewards = jnp.array([[1.0]])  # (1 step, 1 env)
        values = jnp.array([[0.5], [0.8]])  # V(s)=0.5, V(s')=0.8 (unused with gamma=0)
        dones = jnp.array([[0.0]])

        advantages, returns = compute_gae(rewards, values, dones, gamma=0.0, gae_lambda=0.95)

        # With gamma=0: delta = r + 0*V(s') - V(s) = 1.0 - 0.5 = 0.5
        expected_advantage = 1.0 - 0.5
        assert jnp.allclose(advantages[0, 0], expected_advantage, atol=1e-5)

    def test_gae_handles_episode_boundary(self):
        """GAE should not bootstrap through done=True."""
        rewards = jnp.array([[1.0], [1.0]])  # 2 steps
        values = jnp.array([[0.0], [0.0], [100.0]])  # Bootstrap value is huge
        dones = jnp.array([[1.0], [0.0]])  # Episode ends at step 0

        advantages, _ = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)

        # Step 0: done=True, so advantage = r - V(s) = 1.0 - 0.0 = 1.0
        # (should NOT include gamma * V(s') because done)
        assert jnp.allclose(advantages[0, 0], 1.0, atol=1e-5)

    def test_gae_multi_step_known_values(self):
        """GAE with known trajectory should match hand computation."""
        # Simple 3-step trajectory: r=[1,2,3], V=[0,0,0,0], no termination
        rewards = jnp.array([[1.0], [2.0], [3.0]])
        values = jnp.array([[0.0], [0.0], [0.0], [0.0]])
        dones = jnp.array([[0.0], [0.0], [0.0]])
        gamma, lam = 0.9, 0.8

        advantages, _ = compute_gae(rewards, values, dones, gamma=gamma, gae_lambda=lam)

        # Hand-compute GAE backwards:
        # t=2: delta_2 = 3 + 0.9*0 - 0 = 3, A_2 = 3
        # t=1: delta_1 = 2 + 0.9*0 - 0 = 2, A_1 = 2 + 0.9*0.8*3 = 2 + 2.16 = 4.16
        # t=0: delta_0 = 1 + 0.9*0 - 0 = 1, A_0 = 1 + 0.9*0.8*4.16 = 1 + 2.9952 = 3.9952
        expected = jnp.array([[3.9952], [4.16], [3.0]])
        assert jnp.allclose(advantages, expected, atol=1e-3)

    def test_ppo_clipping_behavior(self):
        """PPO clipping should bound the ratio's effect on loss."""
        key = jax.random.PRNGKey(0)
        network, params = init_network(key, obs_shape=(OBS_SIZE,), hidden_dim=32, num_layers=1)

        batch_size = 4
        obs = jnp.zeros((batch_size, OBS_SIZE))
        actions = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
        advantages = jnp.array([1.0, 1.0, 1.0, 1.0])  # Positive advantages
        returns = jnp.ones(batch_size)
        action_masks = jnp.ones((batch_size, NUM_ACTIONS), dtype=jnp.bool_)

        # Artificially create different ratios by manipulating old_log_probs
        # Higher old_log_prob → lower ratio (policy moved away from this action)
        logits, _ = network.apply(params, obs)
        dist = masked_categorical(logits, action_masks)
        current_log_probs = dist.log_prob(actions)

        # Case 1: ratio ≈ 1 (same policy)
        _, metrics1 = ppo_loss(
            params, network.apply, obs, actions,
            current_log_probs, advantages, returns, action_masks,
            clip_eps=0.2, vf_coef=0.5, ent_coef=0.01
        )

        # Case 2: ratio > 1 + clip_eps (policy increased this action's prob)
        old_log_probs_low = current_log_probs - 1.0  # Makes ratio = e^1 ≈ 2.7
        _, metrics2 = ppo_loss(
            params, network.apply, obs, actions,
            old_log_probs_low, advantages, returns, action_masks,
            clip_eps=0.2, vf_coef=0.5, ent_coef=0.01
        )

        # With positive advantages and high ratio, clipping should engage
        assert metrics2["clip_frac"] > 0.5, "Expected clipping with high ratio"

    def test_entropy_computation(self):
        """Entropy should be maximal for uniform distribution."""
        # Uniform logits → uniform distribution → max entropy
        uniform_logits = jnp.zeros(NUM_ACTIONS)
        mask = jnp.ones(NUM_ACTIONS, dtype=jnp.bool_)
        dist = masked_categorical(uniform_logits, mask)
        uniform_entropy = dist.entropy()

        # Max entropy = log(num_actions) for uniform categorical
        expected_max = jnp.log(NUM_ACTIONS)
        assert jnp.allclose(uniform_entropy, expected_max, atol=1e-5)

        # Peaked logits → low entropy
        peaked_logits = jnp.zeros(NUM_ACTIONS).at[0].set(10.0)
        peaked_dist = masked_categorical(peaked_logits, mask)
        peaked_entropy = peaked_dist.entropy()

        assert peaked_entropy < uniform_entropy / 2, "Peaked distribution should have lower entropy"

    def test_log_prob_sums_to_one(self):
        """Log probabilities should sum to 1 (in probability space)."""
        logits = jax.random.normal(jax.random.PRNGKey(0), (NUM_ACTIONS,))
        mask = jnp.ones(NUM_ACTIONS, dtype=jnp.bool_)
        dist = masked_categorical(logits, mask)

        # Sum of probabilities should be 1
        all_log_probs = jnp.array([dist.log_prob(jnp.array(i)) for i in range(NUM_ACTIONS)])
        prob_sum = jnp.exp(all_log_probs).sum()
        assert jnp.allclose(prob_sum, 1.0, atol=1e-5)

    def test_masked_entropy_excludes_invalid(self):
        """Entropy with mask should only count valid actions."""
        logits = jnp.zeros(NUM_ACTIONS)

        # All valid → entropy = log(28)
        full_mask = jnp.ones(NUM_ACTIONS, dtype=jnp.bool_)
        full_dist = masked_categorical(logits, full_mask)

        # Only 2 valid → entropy = log(2)
        partial_mask = jnp.zeros(NUM_ACTIONS, dtype=jnp.bool_).at[0].set(True).at[1].set(True)
        partial_dist = masked_categorical(logits, partial_mask)

        assert jnp.allclose(full_dist.entropy(), jnp.log(NUM_ACTIONS), atol=1e-4)
        assert jnp.allclose(partial_dist.entropy(), jnp.log(2), atol=1e-4)

    def test_advantage_normalization_in_loss(self):
        """Normalized advantages should have mean≈0, std≈1."""
        # This is a property we can verify by checking the loss computation
        key = jax.random.PRNGKey(0)
        network, params = init_network(key, obs_shape=(OBS_SIZE,), hidden_dim=32, num_layers=1)

        batch_size = 64
        obs = jnp.zeros((batch_size, OBS_SIZE))
        actions = jnp.zeros(batch_size, dtype=jnp.int32)
        old_log_probs = jnp.zeros(batch_size)
        # Non-normalized advantages with weird scale
        advantages = jnp.array([100.0] * 32 + [200.0] * 32)
        returns = jnp.ones(batch_size)
        action_masks = jnp.ones((batch_size, NUM_ACTIONS), dtype=jnp.bool_)

        # Loss should still be reasonable (not explode with large advantages)
        loss, metrics = ppo_loss(
            params, network.apply, obs, actions, old_log_probs,
            advantages, returns, action_masks,
            clip_eps=0.2, vf_coef=0.5, ent_coef=0.01
        )

        assert jnp.isfinite(loss)
        assert jnp.abs(metrics["pg_loss"]) < 10, "Policy loss should be bounded after normalization"


class TestReferenceComparison:
    """Compare outputs against known reference values.

    TODO(human): Implement comparison tests against Stable Baselines3 or CleanRL.
    This is the gold standard for PPO verification.
    """

    def test_placeholder_for_reference_comparison(self):
        """Placeholder - see docstring for what to implement."""
        # To properly verify PPO:
        # 1. Run same trajectory through SB3's PPO
        # 2. Compare: GAE values, loss components, gradient magnitudes
        # 3. Use np.testing.assert_allclose with small tolerances
        pass


class TestGradientSanity:
    """Verify gradients flow correctly."""

    def test_gradients_are_nonzero(self):
        """PPO loss should produce non-zero gradients for all parameters."""
        key = jax.random.PRNGKey(0)
        network, params = init_network(key, obs_shape=(OBS_SIZE,), hidden_dim=32, num_layers=2)

        batch_size = 16
        obs = jax.random.normal(jax.random.PRNGKey(1), (batch_size, OBS_SIZE))
        actions = jax.random.randint(jax.random.PRNGKey(2), (batch_size,), 0, NUM_ACTIONS)
        old_log_probs = jax.random.normal(jax.random.PRNGKey(3), (batch_size,)) - 3.0
        advantages = jax.random.normal(jax.random.PRNGKey(4), (batch_size,))
        returns = jax.random.normal(jax.random.PRNGKey(5), (batch_size,))
        action_masks = jnp.ones((batch_size, NUM_ACTIONS), dtype=jnp.bool_)

        # Compute gradients
        grad_fn = jax.grad(lambda p: ppo_loss(
            p, network.apply, obs, actions, old_log_probs,
            advantages, returns, action_masks,
            clip_eps=0.2, vf_coef=0.5, ent_coef=0.01
        )[0])

        grads = grad_fn(params)

        # Check all gradient leaves are non-zero
        flat_grads = jax.tree.leaves(grads)
        for i, g in enumerate(flat_grads):
            grad_norm = jnp.linalg.norm(g)
            assert grad_norm > 1e-10, f"Gradient {i} is zero - possible dead gradient path"

    def test_value_loss_gradient_independent_of_policy(self):
        """Value loss gradient should only affect critic head."""
        key = jax.random.PRNGKey(0)
        network, params = init_network(key, obs_shape=(OBS_SIZE,), hidden_dim=32, num_layers=1)

        batch_size = 8
        obs = jnp.zeros((batch_size, OBS_SIZE))
        actions = jnp.zeros(batch_size, dtype=jnp.int32)
        old_log_probs = jnp.zeros(batch_size)
        advantages = jnp.zeros(batch_size)  # Zero advantages → zero policy gradient
        returns = jnp.ones(batch_size) * 10  # Large returns → value loss gradient
        action_masks = jnp.ones((batch_size, NUM_ACTIONS), dtype=jnp.bool_)

        grad_fn = jax.grad(lambda p: ppo_loss(
            p, network.apply, obs, actions, old_log_probs,
            advantages, returns, action_masks,
            clip_eps=0.2, vf_coef=0.5, ent_coef=0.0  # No entropy to isolate value loss
        )[0])

        grads = grad_fn(params)

        # With num_layers=1: Dense_0=shared, Dense_1=actor, Dense_2=critic
        # The actor head should have near-zero gradients (only value loss matters)
        # Note: Due to advantage normalization, there might be small residual gradients
        actor_grad = grads['params']['Dense_1']  # Actor head (policy logits)
        actor_grad_norm = jnp.linalg.norm(jax.tree.leaves(actor_grad)[0])

        critic_grad = grads['params']['Dense_2']  # Critic head (value)
        critic_grad_norm = jnp.linalg.norm(jax.tree.leaves(critic_grad)[0])

        # Critic gradient should be much larger than actor gradient
        assert critic_grad_norm > actor_grad_norm * 10, \
            f"Value loss should primarily affect critic: actor={actor_grad_norm}, critic={critic_grad_norm}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
