import numpy as np
from environments.base_environment import BaseEnvironment, TerrainType


class EnvironmentC(BaseEnvironment):
    """
    Environment C — Lava Tube Analogue.

    A procedurally generated cave system modelling a subterranean
    lava tube, valid as both a terrestrial SAR scenario and a
    lunar/Martian exploration analogue.

    Features:
    - Organic irregular tunnel walls (no rectangular rooms)
    - Single skylight entry — narrow vertical shaft from surface
    - Branching side passages — some dead ends, some loops
    - Collapse zones — rubble fields with high traversal cost
    - Dark basalt surfaces — reduced LIDAR confidence
    - Thermal anomalies — heat signatures for Scout detection
    - Communication dead zones — rock density attenuates signal

    Design principles:
    - No prior environmental knowledge assumed
    - Topology stress-tests relay chain formation
    - Thermal sensor modality motivates anomaly-driven exploration
    """

    def __init__(self, seed=42, num_anomalies=3):
        self.num_anomalies = num_anomalies
        super().__init__(width=100, height=100, seed=seed)
        self.build()

    def build(self):
        """Procedurally generate the lava tube environment."""
        # Start with all walls
        self.ground_truth[:, :] = True
        self.terrain[:, :] = TerrainType.NORMAL

        # Carve main tube
        self._carve_main_tube()

        # Carve side passages
        self._carve_side_passages()

        # Add collapse zones (rubble)
        self._add_collapse_zones()

        # Add smooth basalt regions (low albedo)
        self._add_basalt_zones()

        # Place thermal anomalies
        self._place_thermal_anomalies()

        # Set entry point — skylight at top centre
        entry_x = self.width // 2
        entry_y = 0
        self.entry_points = [(entry_x, entry_y)]
        self.sentinel_position = (entry_x, -1)

        # Carve skylight shaft — narrow vertical entry
        for y in range(0, 8):
            for x in range(entry_x - 1, entry_x + 2):
                if self.is_valid(x, y):
                    self.ground_truth[x, y] = False

    def _carve_main_tube(self):
        """
        Carve the primary lava tube using a random walk.
        Produces an organic curved tunnel rather than straight corridors.
        """
        # Start near the top, walk toward the bottom
        cx = self.width // 2
        cy = 10

        tube_width = 8
        heading = 1   # 1 = downward, slight lateral drift

        while cy < self.height - 10:
            # Carve circular cross-section at current position
            for dx in range(-tube_width, tube_width + 1):
                for dy in range(-tube_width // 2, tube_width // 2 + 1):
                    nx, ny = cx + dx, cy + dy
                    # Organic shape — elliptical with noise
                    dist = (dx**2 / tube_width**2 +
                            dy**2 / (tube_width // 2)**2)
                    noise = self.rng.uniform(-0.2, 0.2)
                    if dist + noise < 1.0 and self.is_valid(nx, ny):
                        self.ground_truth[nx, ny] = False

            # Advance position with organic drift
            lateral_drift = self.rng.integers(-2, 3)
            cx = np.clip(cx + lateral_drift, tube_width + 2,
                         self.width - tube_width - 2)

            # Vary vertical advance
            vertical_step = self.rng.integers(2, 5)
            cy += vertical_step

            # Occasionally narrow the tube
            if self.rng.random() < 0.1:
                tube_width = max(4, tube_width - 1)
            elif self.rng.random() < 0.05:
                tube_width = min(10, tube_width + 1)

    def _carve_side_passages(self):
        """
        Carve branching side passages off the main tube.
        Some are dead ends, some loop back to the main tube.
        Creates relay chain pressure — Scouts in side passages
        need relay nodes at the junction to maintain contact.
        """
        num_passages = self.rng.integers(4, 7)

        for _ in range(num_passages):
            # Find a random passable cell in the main tube
            attempts = 0
            while attempts < 100:
                sx = self.rng.integers(10, self.width - 10)
                sy = self.rng.integers(15, self.height - 15)
                if self.is_passable(sx, sy):
                    break
                attempts += 1
            else:
                continue

            # Choose a lateral direction
            direction = self.rng.choice([-1, 1])
            passage_length = self.rng.integers(10, 25)
            passage_width = self.rng.integers(2, 5)

            cx, cy = sx, sy
            for step in range(passage_length):
                for dx in range(-passage_width, passage_width + 1):
                    for dy in range(-1, 2):
                        nx = cx + dx
                        ny = cy + dy
                        if self.is_valid(nx, ny):
                            self.ground_truth[nx, ny] = False

                # Move laterally with slight vertical drift
                cx += direction
                cy += self.rng.integers(-1, 2)
                cx = np.clip(cx, 2, self.width - 3)
                cy = np.clip(cy, 2, self.height - 3)

    def _add_collapse_zones(self):
        """
        Add rubble collapse zones — passable but costly.
        Represents sections where the cave ceiling has partially
        fallen, creating difficult terrain.
        """
        num_collapses = self.rng.integers(3, 6)

        for _ in range(num_collapses):
            # Find a passable region
            attempts = 0
            while attempts < 50:
                cx = self.rng.integers(10, self.width - 10)
                cy = self.rng.integers(15, self.height - 15)
                if self.is_passable(cx, cy):
                    break
                attempts += 1
            else:
                continue

            # Spread rubble in an irregular patch
            radius = self.rng.integers(3, 7)
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = cx + dx, cy + dy
                    if not self.is_valid(nx, ny):
                        continue
                    dist = (dx**2 + dy**2) ** 0.5
                    if dist < radius and self.is_passable(nx, ny):
                        if self.rng.random() < 0.7:
                            self.terrain[nx, ny] = TerrainType.RUBBLE

    def _add_basalt_zones(self):
        """
        Mark regions of smooth dark basalt.
        These cells have reduced LIDAR albedo —
        confidence in observations from these regions is lower.
        """
        num_zones = self.rng.integers(4, 8)

        for _ in range(num_zones):
            attempts = 0
            while attempts < 50:
                cx = self.rng.integers(5, self.width - 5)
                cy = self.rng.integers(5, self.height - 5)
                if self.is_passable(cx, cy):
                    break
                attempts += 1
            else:
                continue

            radius = self.rng.integers(4, 10)
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = cx + dx, cy + dy
                    if not self.is_valid(nx, ny):
                        continue
                    dist = (dx**2 + dy**2) ** 0.5
                    if dist < radius and self.is_passable(nx, ny):
                        if self.terrain[nx, ny] == TerrainType.NORMAL:
                            self.terrain[nx, ny] = TerrainType.SMOOTH

    def _place_thermal_anomalies(self):
        """
        Place thermal anomalies in the cave.
        These represent geological heat sources (lava tube context)
        or survivors (SAR context).
        Scouts detect them via thermal gradient within sensor range.
        """
        placed = 0
        attempts = 0

        while placed < self.num_anomalies and attempts < 200:
            ax = self.rng.integers(10, self.width - 10)
            ay = self.rng.integers(20, self.height - 10)

            if not self.is_passable(ax, ay):
                attempts += 1
                continue

            # Ensure minimum separation between anomalies
            too_close = any(
                abs(ax - ex) + abs(ay - ey) < 15
                for ex, ey in self.thermal_anomalies
            )
            if too_close:
                attempts += 1
                continue

            self.thermal_anomalies.append((ax, ay))
            self.terrain[ax, ay] = TerrainType.THERMAL

            # Spread thermal field as gradient
            for dx in range(-8, 9):
                for dy in range(-8, 9):
                    nx, ny = ax + dx, ay + dy
                    if self.is_valid(nx, ny):
                        dist = (dx**2 + dy**2) ** 0.5
                        heat = max(0.0, 1.0 - dist / 8.0)
                        self.thermal_field[nx, ny] = max(
                            self.thermal_field[nx, ny], heat
                        )

            placed += 1
            attempts += 1

    def __repr__(self):
        return (
            f"EnvironmentC(lava_tube, width={self.width}, "
            f"height={self.height}, seed={self.seed}, "
            f"anomalies={len(self.thermal_anomalies)})"
        )