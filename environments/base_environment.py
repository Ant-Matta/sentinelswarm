import numpy as np
import yaml
from enum import IntEnum


class CellState(IntEnum):
    UNKNOWN = 0
    FREE = 1
    OCCUPIED = 2
    CONTESTED = 3


class Cell:
    """A single cell in the environment grid."""

    def __init__(self):
        self.state = CellState.UNKNOWN
        self.confidence = 0.0
        self.last_observed = -1      # timestep of last observation, -1 = never
        self.observed_by = []        # list of scout IDs that have observed this cell
        self.anomaly_likelihood = 0.0

    def to_dict(self):
        return {
            "state": int(self.state),
            "confidence": self.confidence,
            "last_observed": self.last_observed,
            "observed_by": self.observed_by.copy(),
            "anomaly_likelihood": self.anomaly_likelihood
        }


class BaseEnvironment:
    """
    2D grid environment for SentinelSwarm simulation.

    The environment holds two maps:
    - ground_truth: the real environment, never visible to agents
    - Agents build their own world models from Scout observations

    Origin (0,0) is at the Sentinel's position (bottom-centre by default).
    """

    def __init__(self, config_path=None, width=50, height=50, seed=42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if config_path:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            self.width = config.get("width", width)
            self.height = config.get("height", height)
            self.seed = config.get("seed", seed)
        else:
            self.width = width
            self.height = height

        # Sentinel sits at bottom-centre, just outside the environment
        self.sentinel_position = (self.width // 2, -1)

        # Entry points — subclasses define these
        self.entry_points = []

        # Ground truth grid — 2D array of booleans (True = occupied/wall)
        self.ground_truth = np.zeros((self.width, self.height), dtype=bool)

        # Timestep counter
        self.timestep = 0

    def build(self):
        """
        Construct the environment. Subclasses implement this
        to place walls, obstacles, and define entry points.
        """
        raise NotImplementedError("Subclasses must implement build()")

    def is_valid(self, x, y):
        """Check if a position is within grid bounds."""
        return 0 <= x < self.width and 0 <= y < self.height

    def is_passable(self, x, y):
        """Check if a position is within bounds and not a wall."""
        return self.is_valid(x, y) and not self.ground_truth[x, y]

    def get_neighbours(self, x, y):
        """Return passable neighbouring cells (4-directional)."""
        neighbours = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if self.is_passable(nx, ny):
                neighbours.append((nx, ny))
        return neighbours

    def true_cell_state(self, x, y):
        """Return the ground truth state of a cell."""
        if not self.is_valid(x, y):
            return CellState.OCCUPIED   # out of bounds treated as wall
        return CellState.OCCUPIED if self.ground_truth[x, y] else CellState.FREE

    def step(self):
        """Advance the environment by one timestep."""
        self.timestep += 1

    def reset(self):
        """Reset the environment to its initial state."""
        self.timestep = 0
        self.ground_truth = np.zeros((self.width, self.height), dtype=bool)
        self.rng = np.random.default_rng(self.seed)
        self.build()

    def get_ground_truth_dict(self):
        """Serialise ground truth for logging."""
        return {
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "sentinel_position": self.sentinel_position,
            "entry_points": self.entry_points,
            "walls": [
                (int(x), int(y))
                for x in range(self.width)
                for y in range(self.height)
                if self.ground_truth[x, y]
            ]
        }

    def __repr__(self):
        return (
            f"BaseEnvironment(width={self.width}, height={self.height}, "
            f"seed={self.seed}, timestep={self.timestep})"
        )