import numpy as np
import yaml
from enum import IntEnum


class CellState(IntEnum):
    UNKNOWN = 0
    FREE = 1
    OCCUPIED = 2
    CONTESTED = 3


class TerrainType(IntEnum):
    NORMAL = 0      # standard traversal
    RUBBLE = 1      # collapsed rock — high traversal cost
    SMOOTH = 2      # smooth basalt — low LIDAR reflectivity
    THERMAL = 3     # thermally active zone — heat signature


class Cell:
    """A single cell in the environment grid."""

    def __init__(self):
        self.state = CellState.UNKNOWN
        self.confidence = 0.0
        self.last_observed = -1
        self.observed_by = []
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

    Supports:
    - Occupancy (wall / free)
    - Terrain type (normal, rubble, smooth basalt)
    - Thermal field (scalar heat signature per cell)
    - Surface albedo (affects LIDAR confidence)

    Two maps exist simultaneously:
    - ground_truth: real environment, never visible to agents
    - Agents build world models from Scout observations only
    """

    # Traversal energy costs per terrain type
    TERRAIN_COSTS = {
        TerrainType.NORMAL: 1.0,
        TerrainType.RUBBLE: 3.0,
        TerrainType.SMOOTH: 1.0,
        TerrainType.THERMAL: 1.2,
    }

    # LIDAR confidence multiplier per terrain type
    TERRAIN_ALBEDO = {
        TerrainType.NORMAL: 1.0,
        TerrainType.RUBBLE: 0.8,
        TerrainType.SMOOTH: 0.5,   # dark basalt absorbs laser
        TerrainType.THERMAL: 0.9,
    }

    def __init__(self, config_path=None, width=100, height=100, seed=42):
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

        # Sentinel sits outside — at the skylight entry point
        self.sentinel_position = (self.width // 2, -1)

        # Entry points
        self.entry_points = []

        # Ground truth occupancy — True = wall
        self.ground_truth = np.ones((self.width, self.height), dtype=bool)

        # Terrain type per cell
        self.terrain = np.full(
            (self.width, self.height),
            TerrainType.NORMAL,
            dtype=np.int8
        )

        # Thermal field — scalar heat value per cell, 0.0 to 1.0
        self.thermal_field = np.zeros((self.width, self.height), dtype=float)

        # Thermal anomaly locations (ground truth)
        self.thermal_anomalies = []

        self.timestep = 0

    def build(self):
        raise NotImplementedError("Subclasses must implement build()")

    def is_valid(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def is_passable(self, x, y):
        return self.is_valid(x, y) and not self.ground_truth[x, y]

    def get_traversal_cost(self, x, y):
        """Energy cost to move into this cell."""
        if not self.is_valid(x, y):
            return float('inf')
        return self.TERRAIN_COSTS[TerrainType(self.terrain[x, y])]

    def get_albedo(self, x, y):
        """LIDAR confidence multiplier for this cell's surface."""
        if not self.is_valid(x, y):
            return 1.0
        return self.TERRAIN_ALBEDO[TerrainType(self.terrain[x, y])]

    def get_thermal_reading(self, x, y, sensor_range=4):
        """
        Return thermal gradient detectable from position (x,y).
        Scouts detect heat within sensor_range cells.
        Returns 0.0 if no anomaly nearby, up to 1.0 at anomaly centre.
        """
        max_reading = 0.0
        for ax, ay in self.thermal_anomalies:
            dist = abs(x - ax) + abs(y - ay)
            if dist <= sensor_range:
                strength = 1.0 - (dist / sensor_range)
                max_reading = max(max_reading, strength)
        return max_reading

    def get_neighbours(self, x, y):
        neighbours = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if self.is_passable(nx, ny):
                neighbours.append((nx, ny))
        return neighbours

    def true_cell_state(self, x, y):
        if not self.is_valid(x, y):
            return CellState.OCCUPIED
        return CellState.OCCUPIED if self.ground_truth[x, y] else CellState.FREE

    def step(self):
        self.timestep += 1

    def reset(self):
        self.timestep = 0
        self.ground_truth = np.ones((self.width, self.height), dtype=bool)
        self.terrain = np.full(
            (self.width, self.height),
            TerrainType.NORMAL,
            dtype=np.int8
        )
        self.thermal_field = np.zeros((self.width, self.height), dtype=float)
        self.thermal_anomalies = []
        self.rng = np.random.default_rng(self.seed)
        self.build()

    def get_ground_truth_dict(self):
        return {
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "sentinel_position": self.sentinel_position,
            "entry_points": self.entry_points,
            "thermal_anomalies": self.thermal_anomalies,
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