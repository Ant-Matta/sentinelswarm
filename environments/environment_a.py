import numpy as np
from environments.base_environment import BaseEnvironment


class EnvironmentA(BaseEnvironment):
    """
    Environment A — Single Entry Point, Linear Layout.

    A building-like structure with one entrance at the bottom centre.
    Internal rooms and corridors force Scouts deeper into the structure,
    creating natural relay chain pressure as distance from Sentinel grows.

    Layout:
    - Outer perimeter walls
    - Single entry point at bottom centre (Sentinel's position)
    - Internal rooms connected by corridors
    - Communication dead zones in far rooms
    """

    def __init__(self, seed=42):
        super().__init__(width=50, height=50, seed=seed)
        self.entry_points = [(self.width // 2, 0)]  # single bottom-centre entry
        self.build()

    def build(self):
        """Construct the single-entry linear environment."""
        # Fill entire grid with walls first
        self.ground_truth[:, :] = True

        # Carve out the main interior space
        self._carve_rect(2, 2, 47, 47)

        # Add internal room dividers
        self._add_rooms()

        # Ensure entry point is always passable
        ex, ey = self.entry_points[0]
        self.ground_truth[ex, ey] = False
        self.ground_truth[ex, ey + 1] = False

    def _carve_rect(self, x1, y1, x2, y2):
        """Carve a rectangular free space into the grid."""
        self.ground_truth[x1:x2, y1:y2] = False

    def _add_rooms(self):
        """
        Add internal walls to create rooms and corridors.
        Rooms are connected by single-cell doorways, forcing
        Scouts to navigate through chokepoints.
        """
        # Horizontal divider 1 — creates lower and upper zones
        self._add_wall(2, 16, 47, 16)
        # Doorway in divider 1
        self.ground_truth[24, 16] = False
        self.ground_truth[25, 16] = False

        # Horizontal divider 2 — creates mid and far zones
        self._add_wall(2, 32, 47, 32)
        # Doorway in divider 2
        self.ground_truth[12, 32] = False
        self.ground_truth[38, 32] = False

        # Vertical divider in lower zone — creates two entry rooms
        self._add_wall(24, 2, 24, 15)
        # Doorway in vertical divider
        self.ground_truth[24, 8] = False

        # Vertical divider in upper zone — creates two far rooms
        self._add_wall(24, 33, 24, 47)
        # Doorway in vertical divider
        self.ground_truth[24, 40] = False

    def _add_wall(self, x1, y1, x2, y2):
        """Add a wall segment between two points (axis-aligned only)."""
        if x1 == x2:
            # Vertical wall
            for y in range(min(y1, y2), max(y1, y2) + 1):
                if self.is_valid(x1, y):
                    self.ground_truth[x1, y] = True
        elif y1 == y2:
            # Horizontal wall
            for x in range(min(x1, x2), max(x1, x2) + 1):
                if self.is_valid(x, y1):
                    self.ground_truth[x, y1] = True

    def get_zones(self):
        """
        Return named zones for analysis and task allocation.
        Zones correspond to the structural regions created by dividers.
        """
        return {
            "entry_left": {"x": (2, 23), "y": (0, 15)},
            "entry_right": {"x": (25, 47), "y": (0, 15)},
            "mid_left": {"x": (2, 23), "y": (17, 31)},
            "mid_right": {"x": (25, 47), "y": (17, 31)},
            "far_left": {"x": (2, 23), "y": (33, 47)},
            "far_right": {"x": (25, 47), "y": (33, 47)},
        }

    def __repr__(self):
        return (
            f"EnvironmentA(single_entry, width={self.width}, "
            f"height={self.height}, seed={self.seed})"
        )