"""
JIT-compatible siphon quality check for reward parity.

Implements checkForBetterSiphonPosition from Swift GameState.swift.
Computes siphon yields across all 36 grid positions simultaneously,
then checks if any position strictly dominates the player's position.

Strict dominance = exact same blocks + exact same programs +
(>= credits AND >= energy AND at least one strictly >).

Why this matters:
  Penalizes suboptimal siphon positioning to encourage strategic placement.
  A player who siphons at a position where they could have collected more
  resources (same blocks, more credits/energy) gets a proportional penalty.

Penalty formula (from RewardCalculator.swift):
  missed_value = missed_credits * 0.05 + missed_energy * 0.05
  penalty = -0.5 * missed_value
"""

import jax
import jax.numpy as jnp

from .state import (
    BLOCK_DATA,
    BLOCK_EMPTY,
    BLOCK_PROGRAM,
    GRID_SIZE,
    NUM_PROGRAMS,
    EnvState,
)

# Cross pattern: center + 4 cardinal neighbors
_CROSS_OFFSETS = jnp.array(
    [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.int32
)
_NUM_CROSS = 5

# Pre-compute cross neighbor row/col indices for every grid position.
# Shape: (6, 6, 5) — for each center (r,c), the 5 cross cell coordinates.
_ROW_GRID, _COL_GRID = jnp.meshgrid(
    jnp.arange(GRID_SIZE), jnp.arange(GRID_SIZE), indexing="ij"
)
_CROSS_ROWS = _ROW_GRID[:, :, None] + _CROSS_OFFSETS[None, None, :, 0]
_CROSS_COLS = _COL_GRID[:, :, None] + _CROSS_OFFSETS[None, None, :, 1]

# Static bounds mask: which cross cells are within the 6x6 grid
_IN_BOUNDS = (
    (_CROSS_ROWS >= 0)
    & (_CROSS_ROWS < GRID_SIZE)
    & (_CROSS_COLS >= 0)
    & (_CROSS_COLS < GRID_SIZE)
)

# Clipped indices for safe array gathering (out-of-bounds reads harmless due to mask)
_SAFE_ROWS = jnp.clip(_CROSS_ROWS, 0, GRID_SIZE - 1)
_SAFE_COLS = jnp.clip(_CROSS_COLS, 0, GRID_SIZE - 1)


def compute_all_siphon_yields(
    state: EnvState,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute siphon yield for every grid position simultaneously.

    Matches Swift calculateSiphonYieldAt() for all 36 positions at once.
    A position on a block yields zeros (can't stand on a block to siphon).

    Returns:
        credits: (6,6) int32 — total credits from non-block cells in cross
        energy: (6,6) int32 — total energy from non-block cells in cross
        block_values: (6,6,5) int32 — sorted data block points (ascending, zero-padded)
        programs: (6,6,23) bool — set of unowned programs in cross
    """
    # Gather grid properties for all cross cells: shape (6, 6, 5)
    cell_block_type = state.grid_block_type[_SAFE_ROWS, _SAFE_COLS]
    cell_siphoned = state.grid_block_siphoned[_SAFE_ROWS, _SAFE_COLS]
    cell_block_points = state.grid_block_points[_SAFE_ROWS, _SAFE_COLS]
    cell_block_program = state.grid_block_program[_SAFE_ROWS, _SAFE_COLS]
    cell_credits = state.grid_resources_credits[_SAFE_ROWS, _SAFE_COLS]
    cell_energy = state.grid_resources_energy[_SAFE_ROWS, _SAFE_COLS]

    # Active cross cell: in bounds AND not already siphoned
    active = _IN_BOUNDS & ~cell_siphoned

    has_block = cell_block_type != BLOCK_EMPTY
    is_data = cell_block_type == BLOCK_DATA
    is_program = cell_block_type == BLOCK_PROGRAM

    # Credits/energy from non-block cells in cross
    resource_mask = active & ~has_block
    credits = jnp.sum(cell_credits * resource_mask, axis=2)
    energy = jnp.sum(cell_energy * resource_mask, axis=2)

    # Data block values: sorted ascending (zeros pad the front, values at the end)
    data_mask = active & is_data
    data_values = jnp.where(data_mask, cell_block_points, 0)
    block_values = jnp.sort(data_values, axis=2)

    # Program sets: one-hot encode program indices, mask by active program blocks,
    # then take union across the 5 cross cells. Exclude already-owned programs.
    prog_one_hot = jax.nn.one_hot(cell_block_program, NUM_PROGRAMS, dtype=jnp.bool_)
    prog_masked = prog_one_hot & (active & is_program)[:, :, :, None]
    prog_union = jnp.any(prog_masked, axis=2)  # (6, 6, 23)
    programs = prog_union & ~state.owned_programs[None, None, :]

    # Invalidate positions where center cell has a block (can't stand on block)
    center_has_block = state.grid_block_type != BLOCK_EMPTY
    credits = jnp.where(center_has_block, 0, credits)
    energy = jnp.where(center_has_block, 0, energy)
    block_values = jnp.where(center_has_block[:, :, None], 0, block_values)
    programs = jnp.where(center_has_block[:, :, None], False, programs)

    return credits, energy, block_values, programs


def check_siphon_optimality(
    state: EnvState,
) -> tuple[jnp.bool_, jnp.int32, jnp.int32]:
    """Check if a strictly better siphon position exists than the player's.

    Must be called BEFORE the siphon action modifies state (grid_block_siphoned
    and owned_programs change after siphon). This matches Swift's
    checkForBetterSiphonPosition() which runs before performSiphon().

    Returns:
        found_better: bool — whether any strictly better position exists
        missed_credits: int32 — max credits difference (best - player's)
        missed_energy: int32 — max energy difference (best - player's)
    """
    credits, energy, block_values, programs = compute_all_siphon_yields(state)

    # Player's yield at current position
    pr, pc = state.player.row, state.player.col
    player_credits = credits[pr, pc]
    player_energy = energy[pr, pc]
    player_blocks = block_values[pr, pc]   # (5,)
    player_programs = programs[pr, pc]     # (23,)

    # Structure match: exact same sorted data block values AND exact same program set
    same_blocks = jnp.all(block_values == player_blocks[None, None, :], axis=2)
    same_programs = jnp.all(programs == player_programs[None, None, :], axis=2)
    same_structure = same_blocks & same_programs

    # Strict resource dominance: >= both AND > at least one
    ge_credits = credits >= player_credits
    ge_energy = energy >= player_energy
    gt_credits = credits > player_credits
    gt_energy = energy > player_energy
    strictly_better = same_structure & ge_credits & ge_energy & (gt_credits | gt_energy)

    found_better = jnp.any(strictly_better)

    # Max missed resources across all strictly-better positions
    best_credits = jnp.max(jnp.where(strictly_better, credits, player_credits))
    best_energy = jnp.max(jnp.where(strictly_better, energy, player_energy))
    missed_credits = best_credits - player_credits
    missed_energy = best_energy - player_energy

    return found_better, missed_credits, missed_energy
