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
  Fixed-size queue (36 entries = all cells), head/tail pointers, visited mask
  as a flat boolean array. Uses jax.lax.while_loop with bounded iteration.
"""

import jax
import jax.numpy as jnp

from .state import GRID_SIZE

# Total cells on the grid (maximum BFS queue size)
_NUM_CELLS = GRID_SIZE * GRID_SIZE  # 36

# Cardinal direction offsets: up, down, left, right
_DIRECTION_OFFSETS = jnp.array(
    [[1, 0], [-1, 0], [0, -1], [0, 1]], dtype=jnp.int32
)


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
    # Early return: start == target
    same_pos = (start_row == target_row) & (start_col == target_col)

    # BFS state: queue of (row, col) pairs, distance array, visited mask
    # Queue: fixed-size array of flat indices, with head/tail pointers
    # Distance: per-cell distance from start
    queue = jnp.zeros(_NUM_CELLS, dtype=jnp.int32)
    distances = jnp.full(_NUM_CELLS, -1, dtype=jnp.int32)
    visited = jnp.zeros(_NUM_CELLS, dtype=jnp.bool_)

    start_idx = start_row * GRID_SIZE + start_col
    queue = queue.at[0].set(start_idx)
    distances = distances.at[start_idx].set(0)
    visited = visited.at[start_idx].set(True)

    target_idx = target_row * GRID_SIZE + target_col

    # BFS loop state: (queue, distances, visited, head, tail, found, result)
    init_state = (queue, distances, visited, jnp.int32(0), jnp.int32(1), jnp.bool_(False), jnp.int32(-1))

    def cond_fn(loop_state):
        _queue, _distances, _visited, head, tail, found, _result = loop_state
        return (head < tail) & ~found

    def body_fn(loop_state):
        queue, distances, visited, head, tail, found, result = loop_state

        # Dequeue current cell
        current_idx = queue[head]
        current_row = current_idx // GRID_SIZE
        current_col = current_idx % GRID_SIZE
        current_dist = distances[current_idx]
        head = head + 1

        # Explore 4 cardinal neighbors
        def process_neighbor(carry, direction):
            queue, distances, visited, tail, found, result = carry
            new_row = current_row + direction[0]
            new_col = current_col + direction[1]
            new_idx = new_row * GRID_SIZE + new_col

            in_bounds = (new_row >= 0) & (new_row < GRID_SIZE) & (new_col >= 0) & (new_col < GRID_SIZE)
            not_visited = ~visited[jnp.clip(new_idx, 0, _NUM_CELLS - 1)]
            valid = in_bounds & not_visited & ~found

            # Check if this neighbor IS the target (checked before block check, matching Swift)
            is_target = (new_row == target_row) & (new_col == target_col)
            found_target = valid & is_target

            # Block check: only blocks movement to non-target cells
            is_blocked = grid_block_type[
                jnp.clip(new_row, 0, GRID_SIZE - 1),
                jnp.clip(new_col, 0, GRID_SIZE - 1),
            ] != 0
            can_enqueue = valid & ~is_target & ~is_blocked

            # Update result if target found
            new_dist = current_dist + 1
            result = jnp.where(found_target, new_dist, result)
            found = found | found_target

            # Enqueue neighbor if walkable and not target
            safe_idx = jnp.clip(new_idx, 0, _NUM_CELLS - 1)
            queue = jnp.where(can_enqueue, queue.at[tail].set(safe_idx), queue)
            distances = jnp.where(can_enqueue, distances.at[safe_idx].set(new_dist), distances)
            visited = jnp.where(can_enqueue, visited.at[safe_idx].set(True), visited)
            tail = jnp.where(can_enqueue, tail + 1, tail)

            return (queue, distances, visited, tail, found, result), None

        (queue, distances, visited, tail, found, result), _ = jax.lax.scan(
            process_neighbor,
            (queue, distances, visited, tail, found, result),
            _DIRECTION_OFFSETS,
        )

        return (queue, distances, visited, head, tail, found, result)

    _queue, _distances, _visited, _head, _tail, _found, result = jax.lax.while_loop(
        cond_fn, body_fn, init_state
    )

    # same_pos → 0, otherwise BFS result (which is -1 if not found)
    return jnp.where(same_pos, jnp.int32(0), result)
