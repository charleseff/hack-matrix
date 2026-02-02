"""
PureJaxRL training loop for HackMatrix.

This module provides training functions:
- make_train() - Monolithic JIT (original, for testing/comparison)
- make_train_chunk() - JIT-compiled chunk for use in custom loops
- make_chunked_train() - Python loop + JIT chunks with logging callbacks
- init_runner_state() - Initialize training state outside JIT boundary
"""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from .config import TrainConfig
from .env_wrapper import OBS_SIZE, GymnaxEnvState, HackMatrixGymnax
from .masked_ppo import (
    Transition,
    compute_gae,
    init_network,
    masked_categorical,
    ppo_loss,
)


class RunnerState(NamedTuple):
    """State carried through training loop."""

    train_state: TrainState
    env_state: GymnaxEnvState
    obs: jax.Array
    key: jax.Array
    update_step: int
    episode_returns: jax.Array  # Per-env accumulator for episode returns


# ============================================================================
# Initialization (outside JIT)
# ============================================================================


def init_runner_state(
    config: TrainConfig,
    env: HackMatrixGymnax,
    key: jax.Array,
    start_step: int = 0,
    checkpoint_path: str | None = None,
) -> RunnerState:
    """Initialize runner state outside JIT boundary.

    This enables passing state between Python and JAX for chunked training.

    Args:
        config: Training configuration
        env: Environment instance
        key: JAX random key
        start_step: Initial update step (for resuming training)
        checkpoint_path: Optional path to checkpoint directory to resume from

    Returns:
        runner_state: Initialized RunnerState ready for training
    """
    # Initialize network
    key, init_key = jax.random.split(key)
    network, params = init_network(
        init_key,
        obs_shape=(OBS_SIZE,),
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
    )

    # Initialize optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate),
    )

    train_state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx,
    )

    # Load checkpoint if provided
    if checkpoint_path is not None:
        from .checkpointing import load_checkpoint

        train_state, loaded_step, _ = load_checkpoint(checkpoint_path, train_state)
        # Use loaded step as start_step (overrides parameter)
        start_step = loaded_step
        print(f"Resuming from step {start_step}")

    # Initialize environments (vectorized)
    key, *env_keys = jax.random.split(key, config.num_envs + 1)
    env_keys = jnp.array(env_keys)

    reset_fn = jax.vmap(lambda k: env.reset(k, env.default_params))
    obs, env_states = reset_fn(env_keys)

    # Initialize episode return accumulators (one per env)
    episode_returns = jnp.zeros(config.num_envs)

    return RunnerState(
        train_state=train_state,
        env_state=env_states,
        obs=obs,
        key=key,
        update_step=start_step,
        episode_returns=episode_returns,
    )


def make_train(config: TrainConfig, env: HackMatrixGymnax = None):
    """Create JIT-compiled training function.

    Args:
        config: Training configuration
        env: Optional environment instance (creates new one if None)

    Returns:
        train_fn: Function that runs full training loop
            train_fn(key) -> (final_runner_state, metrics)
    """
    if env is None:
        env = HackMatrixGymnax()

    def train(key: jax.Array) -> tuple[RunnerState, dict]:
        """Run complete training loop.

        Args:
            key: JAX random key

        Returns:
            final_state: Final runner state with trained parameters
            metrics: Dictionary of training metrics
        """
        # Initialize network
        key, init_key = jax.random.split(key)
        network, params = init_network(
            init_key,
            obs_shape=(OBS_SIZE,),
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
        )

        # Initialize optimizer
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.learning_rate),
        )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=tx,
        )

        # Initialize environments
        key, *env_keys = jax.random.split(key, config.num_envs + 1)
        env_keys = jnp.array(env_keys)

        # Vectorized reset
        reset_fn = jax.vmap(lambda k: env.reset(k, env.default_params))
        obs, env_states = reset_fn(env_keys)

        # Initialize episode return accumulators
        episode_returns = jnp.zeros(config.num_envs)

        # Initial runner state
        runner_state = RunnerState(
            train_state=train_state,
            env_state=env_states,
            obs=obs,
            key=key,
            update_step=0,
            episode_returns=episode_returns,
        )

        # Training loop
        def _update_step(runner_state: RunnerState, _) -> tuple[RunnerState, dict]:
            """Single PPO update step."""
            train_state, env_state, obs, key, update_step, episode_returns = runner_state

            # Collect rollout
            key, rollout_key = jax.random.split(key)

            def _env_step(carry, _):
                """Single environment step."""
                env_state, obs, key, ep_returns = carry

                key, action_key, step_key = jax.random.split(key, 3)

                # Get action mask for all envs
                get_mask_fn = jax.vmap(env.get_action_mask)
                action_masks = get_mask_fn(env_state)

                # Forward pass
                logits, values = train_state.apply_fn(train_state.params, obs)

                # Sample actions with masking
                def sample_action(logit, mask, k):
                    dist = masked_categorical(logit, mask)
                    action = dist.sample(seed=k)
                    log_prob = dist.log_prob(action)
                    return action, log_prob

                action_keys = jax.random.split(action_key, config.num_envs)
                actions, log_probs = jax.vmap(sample_action)(logits, action_masks, action_keys)

                # Step environments
                step_keys = jax.random.split(step_key, config.num_envs)
                step_fn = jax.vmap(lambda k, s, a: env.step(k, s, a, env.default_params))
                next_obs, next_env_state, rewards, dones, infos = step_fn(
                    step_keys, env_state, actions
                )

                # Track episode returns
                # Add reward to running total
                new_ep_returns = ep_returns + rewards
                # Record completed episode returns (will be 0 for non-done envs)
                completed_returns = jnp.where(dones, new_ep_returns, 0.0)
                # Reset accumulator for done envs
                ep_returns = jnp.where(dones, 0.0, new_ep_returns)

                # Handle episode resets
                def handle_reset(done, next_s, next_o, k):
                    reset_obs, reset_state = env.reset(k, env.default_params)
                    return jax.lax.cond(
                        done,
                        lambda: (reset_state, reset_obs),
                        lambda: (next_s, next_o),
                    )

                reset_keys = jax.random.split(step_key, config.num_envs)
                env_state, obs = jax.vmap(handle_reset)(dones, next_env_state, next_obs, reset_keys)

                transition = Transition(
                    obs=obs,
                    action=actions,
                    reward=rewards,
                    done=dones,
                    log_prob=log_probs,
                    value=values,
                    action_mask=action_masks,
                    episode_return=completed_returns,
                )

                return (env_state, next_obs, key, ep_returns), transition

            # Collect num_steps transitions
            (env_state, obs, key, episode_returns), transitions = jax.lax.scan(
                _env_step,
                (env_state, obs, rollout_key, episode_returns),
                None,
                length=config.num_steps,
            )

            # Compute advantages
            # Get bootstrap value
            _, last_values = train_state.apply_fn(train_state.params, obs)

            # Stack values with bootstrap
            values = jnp.concatenate([transitions.value, last_values[None]], axis=0)

            advantages, returns = compute_gae(
                rewards=transitions.reward,
                values=values,
                dones=transitions.done,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
            )

            # Flatten batch for PPO updates
            def flatten_batch(x):
                return x.reshape(-1, *x.shape[2:]) if x.ndim > 2 else x.reshape(-1)

            batch = Transition(
                obs=flatten_batch(transitions.obs),
                action=flatten_batch(transitions.action),
                reward=flatten_batch(transitions.reward),
                done=flatten_batch(transitions.done),
                log_prob=flatten_batch(transitions.log_prob),
                value=flatten_batch(transitions.value),
                action_mask=flatten_batch(transitions.action_mask),
                episode_return=flatten_batch(transitions.episode_return),
            )
            advantages_flat = flatten_batch(advantages)
            returns_flat = flatten_batch(returns)

            # PPO update epochs
            key, ppo_key = jax.random.split(key)

            def _ppo_epoch(carry, _):
                """Single PPO epoch over all minibatches."""
                train_state, key = carry
                key, perm_key = jax.random.split(key)

                # Shuffle batch
                batch_size = config.batch_size
                perm = jax.random.permutation(perm_key, batch_size)

                def permute(x):
                    return x[perm]

                shuffled_batch = jax.tree.map(permute, batch)
                shuffled_advantages = advantages_flat[perm]
                shuffled_returns = returns_flat[perm]

                # Split into minibatches
                def reshape_minibatch(x):
                    return x.reshape(config.num_minibatches, -1, *x.shape[1:])

                minibatch_data = jax.tree.map(reshape_minibatch, shuffled_batch)
                minibatch_advantages = reshape_minibatch(shuffled_advantages)
                minibatch_returns = reshape_minibatch(shuffled_returns)

                def _update_minibatch(train_state, minibatch_idx):
                    """Update on single minibatch."""
                    mb = jax.tree.map(lambda x: x[minibatch_idx], minibatch_data)
                    mb_advantages = minibatch_advantages[minibatch_idx]
                    mb_returns = minibatch_returns[minibatch_idx]

                    def loss_fn(params):
                        return ppo_loss(
                            params,
                            train_state.apply_fn,
                            mb.obs,
                            mb.action,
                            mb.log_prob,
                            mb_advantages,
                            mb_returns,
                            mb.action_mask,
                            config.clip_eps,
                            config.vf_coef,
                            config.ent_coef,
                        )

                    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                        train_state.params
                    )
                    train_state = train_state.apply_gradients(grads=grads)

                    return train_state, metrics

                train_state, metrics = jax.lax.scan(
                    _update_minibatch,
                    train_state,
                    jnp.arange(config.num_minibatches),
                )

                return (train_state, key), metrics

            (train_state, key), epoch_metrics = jax.lax.scan(
                _ppo_epoch,
                (train_state, ppo_key),
                None,
                length=config.update_epochs,
            )

            # Average metrics across epochs and minibatches
            metrics = jax.tree.map(lambda x: x.mean(), epoch_metrics)

            # Add rollout info
            metrics["mean_reward"] = transitions.reward.mean()
            done_count = transitions.done.sum()
            metrics["mean_episode_length"] = (~transitions.done).sum() / jnp.maximum(done_count, 1)

            # Episode return metrics (only from completed episodes)
            # Sum of all completed episode returns / number of completed episodes
            total_episode_returns = transitions.episode_return.sum()
            metrics["mean_episode_return"] = total_episode_returns / jnp.maximum(done_count, 1)
            metrics["num_episodes"] = done_count

            new_runner_state = RunnerState(
                train_state=train_state,
                env_state=env_state,
                obs=obs,
                key=key,
                update_step=update_step + 1,
                episode_returns=episode_returns,
            )

            return new_runner_state, metrics

        # Run training loop
        final_state, all_metrics = jax.lax.scan(
            _update_step,
            runner_state,
            None,
            length=config.num_updates,
        )

        return final_state, all_metrics

    return jax.jit(train)


# ============================================================================
# Chunked training (Python loop + JIT chunks)
# ============================================================================


def _make_update_step(config: TrainConfig, env: HackMatrixGymnax):
    """Create the PPO update step function.

    This is factored out so it can be shared between make_train and
    make_train_chunk, ensuring identical behavior.

    Args:
        config: Training configuration
        env: Environment instance

    Returns:
        _update_step: Function(runner_state, _) -> (new_runner_state, metrics)
    """

    def _update_step(runner_state: RunnerState, _) -> tuple[RunnerState, dict]:
        """Single PPO update step."""
        train_state, env_state, obs, key, update_step, episode_returns = runner_state

        # Collect rollout
        key, rollout_key = jax.random.split(key)

        def _env_step(carry, _):
            """Single environment step."""
            env_state, obs, key, ep_returns = carry

            key, action_key, step_key = jax.random.split(key, 3)

            # Get action mask for all envs
            get_mask_fn = jax.vmap(env.get_action_mask)
            action_masks = get_mask_fn(env_state)

            # Forward pass
            logits, values = train_state.apply_fn(train_state.params, obs)

            # Sample actions with masking
            def sample_action(logit, mask, k):
                dist = masked_categorical(logit, mask)
                action = dist.sample(seed=k)
                log_prob = dist.log_prob(action)
                return action, log_prob

            action_keys = jax.random.split(action_key, config.num_envs)
            actions, log_probs = jax.vmap(sample_action)(logits, action_masks, action_keys)

            # Step environments
            step_keys = jax.random.split(step_key, config.num_envs)
            step_fn = jax.vmap(lambda k, s, a: env.step(k, s, a, env.default_params))
            next_obs, next_env_state, rewards, dones, infos = step_fn(step_keys, env_state, actions)

            # Track episode returns
            # Add reward to running total
            new_ep_returns = ep_returns + rewards
            # Record completed episode returns (will be 0 for non-done envs)
            completed_returns = jnp.where(dones, new_ep_returns, 0.0)
            # Reset accumulator for done envs
            ep_returns = jnp.where(dones, 0.0, new_ep_returns)

            # Handle episode resets
            def handle_reset(done, next_s, next_o, k):
                reset_obs, reset_state = env.reset(k, env.default_params)
                return jax.lax.cond(
                    done,
                    lambda: (reset_state, reset_obs),
                    lambda: (next_s, next_o),
                )

            reset_keys = jax.random.split(step_key, config.num_envs)
            env_state, obs = jax.vmap(handle_reset)(dones, next_env_state, next_obs, reset_keys)

            transition = Transition(
                obs=obs,
                action=actions,
                reward=rewards,
                done=dones,
                log_prob=log_probs,
                value=values,
                action_mask=action_masks,
                episode_return=completed_returns,
            )

            return (env_state, next_obs, key, ep_returns), transition

        # Collect num_steps transitions
        (env_state, obs, key, episode_returns), transitions = jax.lax.scan(
            _env_step,
            (env_state, obs, rollout_key, episode_returns),
            None,
            length=config.num_steps,
        )

        # Compute advantages
        _, last_values = train_state.apply_fn(train_state.params, obs)
        values = jnp.concatenate([transitions.value, last_values[None]], axis=0)

        advantages, returns = compute_gae(
            rewards=transitions.reward,
            values=values,
            dones=transitions.done,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )

        # Flatten batch for PPO updates
        def flatten_batch(x):
            return x.reshape(-1, *x.shape[2:]) if x.ndim > 2 else x.reshape(-1)

        batch = Transition(
            obs=flatten_batch(transitions.obs),
            action=flatten_batch(transitions.action),
            reward=flatten_batch(transitions.reward),
            done=flatten_batch(transitions.done),
            log_prob=flatten_batch(transitions.log_prob),
            value=flatten_batch(transitions.value),
            action_mask=flatten_batch(transitions.action_mask),
            episode_return=flatten_batch(transitions.episode_return),
        )
        advantages_flat = flatten_batch(advantages)
        returns_flat = flatten_batch(returns)

        # PPO update epochs
        key, ppo_key = jax.random.split(key)

        def _ppo_epoch(carry, _):
            """Single PPO epoch over all minibatches."""
            train_state, key = carry
            key, perm_key = jax.random.split(key)

            # Shuffle batch
            batch_size = config.batch_size
            perm = jax.random.permutation(perm_key, batch_size)

            def permute(x):
                return x[perm]

            shuffled_batch = jax.tree.map(permute, batch)
            shuffled_advantages = advantages_flat[perm]
            shuffled_returns = returns_flat[perm]

            # Split into minibatches
            def reshape_minibatch(x):
                return x.reshape(config.num_minibatches, -1, *x.shape[1:])

            minibatch_data = jax.tree.map(reshape_minibatch, shuffled_batch)
            minibatch_advantages = reshape_minibatch(shuffled_advantages)
            minibatch_returns = reshape_minibatch(shuffled_returns)

            def _update_minibatch(train_state, minibatch_idx):
                """Update on single minibatch."""
                mb = jax.tree.map(lambda x: x[minibatch_idx], minibatch_data)
                mb_advantages = minibatch_advantages[minibatch_idx]
                mb_returns = minibatch_returns[minibatch_idx]

                def loss_fn(params):
                    return ppo_loss(
                        params,
                        train_state.apply_fn,
                        mb.obs,
                        mb.action,
                        mb.log_prob,
                        mb_advantages,
                        mb_returns,
                        mb.action_mask,
                        config.clip_eps,
                        config.vf_coef,
                        config.ent_coef,
                    )

                (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                    train_state.params
                )
                train_state = train_state.apply_gradients(grads=grads)

                return train_state, metrics

            train_state, metrics = jax.lax.scan(
                _update_minibatch,
                train_state,
                jnp.arange(config.num_minibatches),
            )

            return (train_state, key), metrics

        (train_state, key), epoch_metrics = jax.lax.scan(
            _ppo_epoch,
            (train_state, ppo_key),
            None,
            length=config.update_epochs,
        )

        # Average metrics across epochs and minibatches
        metrics = jax.tree.map(lambda x: x.mean(), epoch_metrics)

        # Add rollout info
        metrics["mean_reward"] = transitions.reward.mean()
        done_count = transitions.done.sum()
        metrics["mean_episode_length"] = (~transitions.done).sum() / jnp.maximum(done_count, 1)

        # Episode return metrics (only from completed episodes)
        # Sum of all completed episode returns / number of completed episodes
        total_episode_returns = transitions.episode_return.sum()
        metrics["mean_episode_return"] = total_episode_returns / jnp.maximum(done_count, 1)
        metrics["num_episodes"] = done_count

        new_runner_state = RunnerState(
            train_state=train_state,
            env_state=env_state,
            obs=obs,
            key=key,
            update_step=update_step + 1,
            episode_returns=episode_returns,
        )

        return new_runner_state, metrics

    return _update_step


def make_train_chunk(
    config: TrainConfig,
    env: HackMatrixGymnax = None,
    chunk_size: int = 10,
):
    """Create JIT-compiled training chunk function.

    This function runs a fixed number of PPO updates and returns metrics.
    It does NOT initialize the network or environments - that must be done
    externally via init_runner_state() and passed in via runner_state.

    Args:
        config: Training configuration
        env: Environment instance (creates new one if None)
        chunk_size: Number of updates per chunk (statically compiled)

    Returns:
        train_chunk_fn: JIT-compiled function
            train_chunk_fn(runner_state) -> (new_runner_state, chunk_metrics)
            where chunk_metrics has shape {metric_name: (chunk_size,)}
    """
    if env is None:
        env = HackMatrixGymnax()

    _update_step = _make_update_step(config, env)

    def train_chunk(runner_state: RunnerState) -> tuple[RunnerState, dict]:
        """Run chunk_size PPO updates."""
        final_state, chunk_metrics = jax.lax.scan(
            _update_step,
            runner_state,
            None,
            length=chunk_size,
        )
        return final_state, chunk_metrics

    return jax.jit(train_chunk)


def aggregate_chunk_metrics(chunk_metrics: dict) -> dict:
    """Aggregate metrics across a chunk for logging.

    Args:
        chunk_metrics: Dict with values of shape (chunk_size,)

    Returns:
        aggregated: Dict with scalar values (mean across chunk)
    """
    return jax.tree.map(lambda x: float(jnp.mean(x)), chunk_metrics)


def concatenate_metrics(chunks: list[dict]) -> dict:
    """Concatenate metrics from multiple chunks.

    Args:
        chunks: List of metric dicts, each with shape (chunk_size,)

    Returns:
        all_metrics: Dict with values of shape (total_updates,)
    """
    if not chunks:
        return {}

    keys = chunks[0].keys()
    return {k: jnp.concatenate([c[k] for c in chunks], axis=0) for k in keys}


def make_chunked_train(
    config: TrainConfig,
    env: HackMatrixGymnax = None,
    chunk_size: int | None = None,
    log_fn: Callable[[dict, int], None] | None = None,
    checkpoint_fn: Callable[[RunnerState, int], None] | None = None,
    start_step: int = 0,
    checkpoint_path: str | None = None,
):
    """Create chunked training function with Python logging loop.

    This wraps train_chunk_fn in a Python loop, enabling logging/checkpointing
    between JIT-compiled chunks while maintaining high performance.

    Args:
        config: Training configuration
        env: Environment instance (created if None)
        chunk_size: Updates per chunk (default: config.log_interval)
        log_fn: Called after each chunk with (aggregated_metrics, update_step)
        checkpoint_fn: Called periodically with (runner_state, update_step)
        start_step: Initial update step for resuming (offsets step counter)
        checkpoint_path: Optional path to checkpoint directory to resume from

    Returns:
        train_fn: Function(key) -> (final_state, all_metrics)
    """
    if env is None:
        env = HackMatrixGymnax()

    if chunk_size is None:
        chunk_size = config.log_interval

    # Create the JIT-compiled chunk function
    train_chunk_fn = make_train_chunk(config, env, chunk_size)

    def train(key: jax.Array) -> tuple[RunnerState, dict]:
        """Run training with logging between chunks.

        Args:
            key: JAX random key

        Returns:
            final_state: Final RunnerState
            all_metrics: Dictionary of metrics, each with shape (num_updates,)
        """
        # Initialize state OUTSIDE JIT
        runner_state = init_runner_state(
            config, env, key, start_step=start_step, checkpoint_path=checkpoint_path
        )

        # Calculate number of chunks
        num_updates = config.num_updates
        num_full_chunks = num_updates // chunk_size
        remainder = num_updates % chunk_size

        # Collect all metrics
        all_chunk_metrics = []

        for chunk_idx in range(num_full_chunks):
            # Run JIT-compiled chunk
            runner_state, chunk_metrics = train_chunk_fn(runner_state)

            # Block on metrics for logging (transfer from device)
            chunk_metrics_np = jax.tree.map(lambda x: jax.device_get(x), chunk_metrics)
            all_chunk_metrics.append(chunk_metrics_np)

            # Log aggregated metrics
            if log_fn is not None:
                aggregated = aggregate_chunk_metrics(chunk_metrics_np)
                log_fn(aggregated, int(runner_state.update_step))

            # Checkpoint if requested
            if checkpoint_fn is not None:
                checkpoint_fn(runner_state, int(runner_state.update_step))

        # Handle remainder (if num_updates not divisible by chunk_size)
        if remainder > 0:
            # Create a separate JIT-compiled function for the remainder
            remainder_chunk_fn = make_train_chunk(config, env, remainder)
            runner_state, remainder_metrics = remainder_chunk_fn(runner_state)

            remainder_metrics_np = jax.tree.map(lambda x: jax.device_get(x), remainder_metrics)
            all_chunk_metrics.append(remainder_metrics_np)

            if log_fn is not None:
                aggregated = aggregate_chunk_metrics(remainder_metrics_np)
                log_fn(aggregated, int(runner_state.update_step))

            if checkpoint_fn is not None:
                checkpoint_fn(runner_state, int(runner_state.update_step))

        # Concatenate all metrics
        all_metrics = concatenate_metrics(all_chunk_metrics)

        return runner_state, all_metrics

    return train


def evaluate(
    train_state: TrainState,
    env: HackMatrixGymnax,
    num_episodes: int,
    key: jax.Array,
) -> dict:
    """Evaluate trained policy.

    Args:
        train_state: Trained model state
        env: Environment instance
        num_episodes: Number of episodes to evaluate
        key: JAX random key

    Returns:
        metrics: Evaluation metrics (mean reward, mean length, etc.)
    """

    def _eval_episode(key):
        """Run single evaluation episode."""
        key, reset_key = jax.random.split(key)
        obs, env_state = env.reset(reset_key, env.default_params)

        def _step(carry, _):
            env_state, obs, key, total_reward, done = carry
            key, action_key, step_key = jax.random.split(key, 3)

            # Get action (greedy)
            action_mask = env.get_action_mask(env_state)
            logits, _ = train_state.apply_fn(train_state.params, obs)

            # Greedy action from masked distribution
            masked_logits = jnp.where(action_mask, logits, -1e9)
            action = jnp.argmax(masked_logits)

            # Step
            next_obs, next_env_state, reward, next_done, _ = env.step(
                step_key, env_state, action, env.default_params
            )

            # Only accumulate if not already done
            total_reward = jax.lax.cond(
                done,
                lambda: total_reward,
                lambda: total_reward + reward,
            )

            return (next_env_state, next_obs, key, total_reward, done | next_done), None

        max_steps = 1000
        init_carry = (env_state, obs, key, 0.0, False)
        (_, _, _, total_reward, _), _ = jax.lax.scan(_step, init_carry, None, length=max_steps)

        return total_reward

    # Run episodes
    episode_keys = jax.random.split(key, num_episodes)
    rewards = jax.vmap(_eval_episode)(episode_keys)

    return {
        "mean_reward": rewards.mean(),
        "std_reward": rewards.std(),
        "min_reward": rewards.min(),
        "max_reward": rewards.max(),
    }
