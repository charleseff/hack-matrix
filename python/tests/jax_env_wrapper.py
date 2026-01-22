"""
JAX Environment Wrapper implementing EnvInterface.

This wrapper adapts the pure functional JAX environment to the EnvInterface
protocol for parity testing.

Why this design:
- set_state() converts GameState dataclass to JAX EnvState for deterministic testing
- get_internal_state() exposes hidden state for implementation tests
- The interface matches SwiftEnvWrapper for test compatibility
"""

import numpy as np
import jax
import jax.numpy as jnp

from .env_interface import (
    EnvInterface,
    GameState,
    Observation,
    StepResult,
    InternalState,
    InternalEnemy,
    GRID_SIZE,
)

# Import the JAX environment modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hackmatrix import jax_env
from hackmatrix.jax_env import get_observation
from hackmatrix.jax_state import (
    EnvState,
    Player,
    create_empty_state,
    MAX_ENEMIES,
    MAX_TRANSMISSIONS,
    NUM_PROGRAMS,
    ENEMY_TYPE_TO_INT,
    ENEMY_INT_TO_TYPE,
    BLOCK_EMPTY,
    BLOCK_DATA,
    BLOCK_PROGRAM,
    BLOCK_QUESTION,
    PLAYER_MAX_HP,
    DEFAULT_SCHEDULED_TASK_INTERVAL,
)
from hackmatrix.jax_observation import Observation as JaxObservation


class JaxEnvWrapper:
    """JAX environment wrapper implementing EnvInterface.

    Provides full set_state() support for deterministic testing.
    """

    def __init__(self, seed: int = 0):
        """
        Initialize the JAX environment wrapper.

        Args:
            seed: Random seed for JAX PRNG.
        """
        self.key = jax.random.PRNGKey(seed)
        self.state = None
        self._initialized = False

    def _convert_observation(self, jax_obs: JaxObservation) -> Observation:
        """Convert JAX Observation to EnvInterface Observation."""
        return Observation(
            player=np.array(jax_obs.player_state, dtype=np.float32),
            programs=np.array(jax_obs.programs, dtype=np.int32),
            grid=np.array(jax_obs.grid, dtype=np.float32),
        )

    # MARK: - EnvInterface Implementation

    def reset(self) -> Observation:
        """Reset the environment to initial state."""
        self.key, subkey = jax.random.split(self.key)
        self.state, jax_obs = jax_env.reset(subkey)
        self._initialized = True
        return self._convert_observation(jax_obs)

    def step(self, action: int) -> StepResult:
        """Execute an action in the environment."""
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self.key, subkey = jax.random.split(self.key)
        self.state, jax_obs, reward, done = jax_env.step(
            self.state, jnp.int32(action), subkey
        )

        return StepResult(
            observation=self._convert_observation(jax_obs),
            reward=float(reward),
            done=bool(done),
            info={}
        )

    def get_valid_actions(self) -> list[int]:
        """Get list of valid action indices for current state."""
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        mask = jax_env.get_valid_actions(self.state)
        return [i for i, valid in enumerate(np.array(mask)) if valid]

    def set_state(self, state: GameState) -> Observation:
        """Set the complete game state for test setup.

        Converts GameState dataclass to JAX EnvState.
        """
        self.key, subkey = jax.random.split(self.key)

        # Start with empty state
        jax_state = create_empty_state(subkey)

        # Convert player
        player = Player(
            row=jnp.int32(state.player.row),
            col=jnp.int32(state.player.col),
            hp=jnp.int32(state.player.hp),
            credits=jnp.int32(state.player.credits),
            energy=jnp.int32(state.player.energy),
            data_siphons=jnp.int32(state.player.dataSiphons),
            attack_damage=jnp.int32(state.player.attackDamage),
            score=jnp.int32(state.player.score),
        )

        # Convert enemies to fixed-size array
        enemies = jnp.zeros((MAX_ENEMIES, 8), dtype=jnp.int32)
        enemy_mask = jnp.zeros(MAX_ENEMIES, dtype=jnp.bool_)

        for i, e in enumerate(state.enemies[:MAX_ENEMIES]):
            enemy_type = ENEMY_TYPE_TO_INT.get(e.type, 0)
            enemy_data = jnp.array([
                enemy_type,
                e.row,
                e.col,
                e.hp,
                0,  # disabled_turns
                int(e.stunned),
                int(e.spawnedFromSiphon),
                int(e.isFromScheduledTask),
            ], dtype=jnp.int32)
            enemies = enemies.at[i].set(enemy_data)
            enemy_mask = enemy_mask.at[i].set(True)

        # Convert transmissions to fixed-size array
        transmissions = jnp.zeros((MAX_TRANSMISSIONS, 6), dtype=jnp.int32)
        trans_mask = jnp.zeros(MAX_TRANSMISSIONS, dtype=jnp.bool_)

        for i, t in enumerate(state.transmissions[:MAX_TRANSMISSIONS]):
            enemy_type = ENEMY_TYPE_TO_INT.get(t.enemyType, 0)
            trans_data = jnp.array([
                t.row,
                t.col,
                t.turnsRemaining,
                enemy_type,
                0,  # spawned_from_siphon (not in interface)
                0,  # is_from_scheduled_task (not in interface)
            ], dtype=jnp.int32)
            transmissions = transmissions.at[i].set(trans_data)
            trans_mask = trans_mask.at[i].set(True)

        # Convert grid - initialize arrays
        grid_block_type = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.int32)
        grid_block_points = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.int32)
        grid_block_program = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.int32)
        grid_block_spawn_count = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.int32)
        grid_block_siphoned = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.bool_)

        # Process blocks
        for block in state.blocks:
            row, col = block.row, block.col
            if block.type == "data":
                grid_block_type = grid_block_type.at[row, col].set(BLOCK_DATA)
                grid_block_points = grid_block_points.at[row, col].set(block.points)
                grid_block_spawn_count = grid_block_spawn_count.at[row, col].set(block.spawnCount)
            elif block.type == "program":
                grid_block_type = grid_block_type.at[row, col].set(BLOCK_PROGRAM)
                # Convert action index (5-27) to program index (0-22)
                if block.programActionIndex is not None:
                    prog_idx = block.programActionIndex - 5
                    grid_block_program = grid_block_program.at[row, col].set(prog_idx)
                grid_block_spawn_count = grid_block_spawn_count.at[row, col].set(block.spawnCount)
            elif block.type == "question":
                grid_block_type = grid_block_type.at[row, col].set(BLOCK_QUESTION)
                grid_block_spawn_count = grid_block_spawn_count.at[row, col].set(block.spawnCount)

            grid_block_siphoned = grid_block_siphoned.at[row, col].set(block.siphoned)

        # Process resources
        grid_resources_credits = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.int32)
        grid_resources_energy = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.int32)
        grid_data_siphon = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.bool_)

        for resource in state.resources:
            row, col = resource.row, resource.col
            grid_resources_credits = grid_resources_credits.at[row, col].set(resource.credits)
            grid_resources_energy = grid_resources_energy.at[row, col].set(resource.energy)
            grid_data_siphon = grid_data_siphon.at[row, col].set(resource.dataSiphon)

        # Convert owned programs (action indices 5-27 to boolean array)
        owned_programs = jnp.zeros(NUM_PROGRAMS, dtype=jnp.bool_)
        for action_idx in state.owned_programs:
            if 5 <= action_idx <= 27:
                prog_idx = action_idx - 5
                owned_programs = owned_programs.at[prog_idx].set(True)

        # Exit at (5, 5) by default for tests
        exit_row, exit_col = 5, 5
        grid_exit = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.bool_)
        grid_exit = grid_exit.at[exit_row, exit_col].set(True)

        # Build complete state
        self.state = jax_state.replace(
            player=player,
            grid_block_type=grid_block_type,
            grid_block_points=grid_block_points,
            grid_block_program=grid_block_program,
            grid_block_spawn_count=grid_block_spawn_count,
            grid_block_siphoned=grid_block_siphoned,
            grid_siphon_center=jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.bool_),
            grid_resources_credits=grid_resources_credits,
            grid_resources_energy=grid_resources_energy,
            grid_data_siphon=grid_data_siphon,
            grid_exit=grid_exit,
            enemies=enemies,
            enemy_mask=enemy_mask,
            transmissions=transmissions,
            trans_mask=trans_mask,
            owned_programs=owned_programs,
            stage=jnp.int32(state.stage),
            turn=jnp.int32(state.turn),
            show_activated=jnp.bool_(state.showActivated),
            scheduled_tasks_disabled=jnp.bool_(state.scheduledTasksDisabled),
            step_active=jnp.bool_(False),
            atk_plus_uses_this_stage=jnp.int32(0),
            exit_row=jnp.int32(exit_row),
            exit_col=jnp.int32(exit_col),
            next_scheduled_task_turn=jnp.int32(DEFAULT_SCHEDULED_TASK_INTERVAL),
            scheduled_task_interval=jnp.int32(DEFAULT_SCHEDULED_TASK_INTERVAL),
            pending_siphon_transmissions=jnp.int32(0),
            prev_hp=jnp.int32(state.player.hp),
            prev_score=jnp.int32(state.player.score),
            prev_credits=jnp.int32(state.player.credits),
            prev_energy=jnp.int32(state.player.energy),
        )

        self._initialized = True

        # Build and return observation
        jax_obs = get_observation(self.state)
        return self._convert_observation(jax_obs)

    def get_internal_state(self) -> InternalState:
        """Get internal state for implementation-level testing.

        Exposes hidden state not visible in observations.
        """
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call reset() or set_state() first.")

        # Extract enemy information
        enemies = []
        for i in range(MAX_ENEMIES):
            if self.state.enemy_mask[i]:
                enemy_data = self.state.enemies[i]
                enemy_type_int = int(enemy_data[0])
                enemies.append(InternalEnemy(
                    row=int(enemy_data[1]),
                    col=int(enemy_data[2]),
                    type=ENEMY_INT_TO_TYPE.get(enemy_type_int, "unknown"),
                    hp=int(enemy_data[3]),
                    disabled_turns=int(enemy_data[4]),
                    is_stunned=bool(enemy_data[5]),
                    spawned_from_siphon=bool(enemy_data[6]),
                    is_from_scheduled_task=bool(enemy_data[7]),
                ))

        return InternalState(
            scheduled_task_interval=int(self.state.scheduled_task_interval),
            next_scheduled_task_turn=int(self.state.next_scheduled_task_turn),
            pending_siphon_transmissions=int(self.state.pending_siphon_transmissions),
            turn_count=int(self.state.turn),
            enemies=enemies,
        )

    # MARK: - Cleanup

    def close(self):
        """Clean up resources (no-op for JAX)."""
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
