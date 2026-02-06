"""
JIT-compatible BFS pathfinding on 6x6 grid.

Matches Swift Pathfinding.findDistance() for reward parity:
- BFS from start to target on a 6x6 grid
- Obstacles: all blocks (grid_block_type != 0), both siphoned and unsiphoned
- Returns shortest path distance, or -1 if no path exists
- Target cell is NOT checked as an obstacle (BFS matches target before block check)

Why BFS over Manhattan:
  Manhattan distance gives misleading reward signals when blocks create walls.
  BFS accounts for actual walkable paths, matching Swift's pathfinding exactly.

JIT strategy:
  Wavefront expansion using jax.lax.scan with fixed iteration count.
  Each iteration expands the frontier one BFS layer using grid-level array ops
  (pad + slice for cardinal shifts). No dynamic queue indexing, no while_loop.
  Compiles much faster than queue-based BFS under vmap.
"""

import jax
import jax.numpy as jnp

from .state import GRID_SIZE

# Maximum BFS depth on grid (all cells minus start)
_MAX_BFS_DEPTH = GRID_SIZE * GRID_SIZE - 1  # 35


def bfs_distance(
    start_row: jnp.int32,
    start_col: jnp.int32,
    target_row: jnp.int32,
    target_col: jnp.int32,
    grid_block_type: jax.Array,
) -> jnp.int32:
    """Compute BFS shortest-path distance from start to target.

    Args:
        start_row, start_col: Starting position.
        target_row, target_col: Target position.
        grid_block_type: (6,6) int32 array. Non-zero = block = obstacle.

    Returns:
        Shortest path distance (int32). Returns -1 if no path exists.

    Matches Swift Pathfinding.findDistance():
    - Obstacles are any cell with grid_block_type != 0
    - Target cell itself is reachable even if it has a block
      (Swift checks target match before block check)
    - Cardinal movement only (up, down, left, right)
    """
    # Obstacle mask: blocks impede movement, but target is always reachable
    obstacles = (grid_block_type != 0).at[target_row, target_col].set(False)

    # Distance grid: -1 = unvisited, 0 = start
    dist = jnp.full((GRID_SIZE, GRID_SIZE), -1, dtype=jnp.int32).at[start_row, start_col].set(0)
    frontier = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.bool_).at[start_row, start_col].set(True)

    def expand_wave(carry, step_num):
        dist, frontier = carry
        # Expand frontier to cardinal neighbors via padding + slicing
        padded = jnp.pad(frontier, 1, constant_values=False)
        neighbors = (
            padded[2:, 1:-1]    # cell above was in frontier
            | padded[:-2, 1:-1]  # cell below was in frontier
            | padded[1:-1, 2:]   # cell left was in frontier
            | padded[1:-1, :-2]  # cell right was in frontier
        )
        new_frontier = neighbors & (dist == -1) & ~obstacles
        dist = jnp.where(new_frontier, step_num, dist)
        return (dist, new_frontier), None

    (dist, _), _ = jax.lax.scan(
        expand_wave, (dist, frontier), jnp.arange(1, _MAX_BFS_DEPTH + 1)
    )

    return dist[target_row, target_col]
