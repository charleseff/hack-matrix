"""
SQLite database for storing training episode history.
Allows historical analysis of training runs and reward breakdown trends.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class TrainingDB:
    """SQLite database for training episode history."""

    def __init__(self, db_path: str = "training_history.db"):
        """
        Initialize the training database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        """Create database tables if they don't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY,
                run_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                episode_num INTEGER NOT NULL,
                timestep INTEGER NOT NULL,
                total_reward REAL NOT NULL,
                reward_stage REAL DEFAULT 0,
                reward_kills REAL DEFAULT 0,
                reward_distance REAL DEFAULT 0,
                reward_score REAL DEFAULT 0,
                reward_data_siphon REAL DEFAULT 0,
                reward_victory REAL DEFAULT 0,
                reward_death REAL DEFAULT 0,
                programs_used INTEGER DEFAULT 0,
                highest_stage INTEGER DEFAULT 1,
                steps INTEGER DEFAULT 0,
                action_moves INTEGER DEFAULT 0,
                action_siphons INTEGER DEFAULT 0,
                action_programs INTEGER DEFAULT 0
            )
        """)

        # Index for efficient queries by run_id
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodes_run_id
            ON episodes(run_id)
        """)

        self.conn.commit()

    def log_episode(self, run_id: str, episode_num: int, timestep: int, stats: Dict[str, Any]):
        """
        Log an episode to the database.

        Args:
            run_id: Unique identifier for the training run
            episode_num: Episode number within the run
            timestep: Current timestep in training
            stats: Episode statistics from gym_env.py
        """
        breakdown = stats.get("reward_breakdown", {})
        actions = stats.get("action_counts", {})

        total_reward = sum(breakdown.values()) if breakdown else 0

        self.conn.execute("""
            INSERT INTO episodes (
                run_id, timestamp, episode_num, timestep, total_reward,
                reward_stage, reward_kills, reward_distance, reward_score,
                reward_data_siphon, reward_victory, reward_death,
                programs_used, highest_stage, steps,
                action_moves, action_siphons, action_programs
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            datetime.now().isoformat(),
            episode_num,
            timestep,
            total_reward,
            breakdown.get("stage", 0),
            breakdown.get("kills", 0),
            breakdown.get("distance", 0),
            breakdown.get("score", 0),
            breakdown.get("dataSiphon", 0),
            breakdown.get("victory", 0),
            breakdown.get("death", 0),
            stats.get("programs_used", 0),
            stats.get("highest_stage", 1),
            stats.get("steps", 0),
            actions.get("move", 0),
            actions.get("siphon", 0),
            actions.get("program", 0),
        ))
        self.conn.commit()

    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        """
        Get summary statistics for a training run.

        Args:
            run_id: Unique identifier for the training run

        Returns:
            Dictionary with summary statistics
        """
        cursor = self.conn.execute("""
            SELECT
                COUNT(*) as episode_count,
                AVG(total_reward) as avg_reward,
                MAX(total_reward) as max_reward,
                AVG(highest_stage) as avg_stage,
                MAX(highest_stage) as max_stage,
                AVG(steps) as avg_steps,
                SUM(reward_victory) as total_victories
            FROM episodes
            WHERE run_id = ?
        """, (run_id,))

        row = cursor.fetchone()
        if row:
            return {
                "episode_count": row[0],
                "avg_reward": row[1],
                "max_reward": row[2],
                "avg_stage": row[3],
                "max_stage": row[4],
                "avg_steps": row[5],
                "total_victories": row[6] if row[6] else 0,
            }
        return {}

    def get_recent_episodes(self, run_id: str, limit: int = 100) -> list:
        """
        Get recent episodes for a training run.

        Args:
            run_id: Unique identifier for the training run
            limit: Maximum number of episodes to return

        Returns:
            List of episode dictionaries
        """
        cursor = self.conn.execute("""
            SELECT
                episode_num, timestep, total_reward, highest_stage, steps,
                reward_stage, reward_kills, reward_distance, reward_score,
                reward_data_siphon, reward_victory, reward_death
            FROM episodes
            WHERE run_id = ?
            ORDER BY episode_num DESC
            LIMIT ?
        """, (run_id, limit))

        return [
            {
                "episode_num": row[0],
                "timestep": row[1],
                "total_reward": row[2],
                "highest_stage": row[3],
                "steps": row[4],
                "reward_breakdown": {
                    "stage": row[5],
                    "kills": row[6],
                    "distance": row[7],
                    "score": row[8],
                    "dataSiphon": row[9],
                    "victory": row[10],
                    "death": row[11],
                }
            }
            for row in cursor.fetchall()
        ]

    def close(self):
        """Close the database connection."""
        self.conn.close()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except:
            pass
