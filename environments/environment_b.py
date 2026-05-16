from environments.base_environment import BaseEnvironment


class EnvironmentB(BaseEnvironment):
    """
    Environment B — Multiple Entry Points, Distributed Layout.

    A building-like structure with two entrances: bottom-centre and top-centre.
    An open central atrium connects side rooms, allowing Scouts to distribute
    immediately without congestion. Tests whether distributed entry improves
    coverage speed and whether the Sentinel adapts its command posture.

    Layout:
    - Outer perimeter walls
    - Two entry points: bottom-centre and top-centre
    - Central open atrium
    - Side rooms off the atrium
    - Communication challenges in corner rooms
    """

    def __init__(self, seed=42):
        super().__init__(width=50, height=50, seed=seed)
        self.entry_points = [
            (self.width // 2, 0),       # south entry
            (self.width // 2, self.height - 1)   # north entry
        ]
        self.build()

    def build(self):
        """Construct the multi-entry distributed environment."""
        # Fill entire grid with walls
        self.ground_truth[:, :] = True

        # Carve main interior
        self._carve_rect(2, 2, 47, 47)

        # Add internal structure
        self._add_rooms()

        # Ensure both entry points are passable
        for ex, ey in self.entry_points:
            self.ground_truth[ex, ey] = False
            if ey == 0:
                self.ground_truth[ex, ey + 1] = False
            else:
                self.ground_truth[ex, ey - 1] = False

    def _carve_rect(self, x1, y1, x2, y2):
        """Carve a rectangular free space into the grid."""
        self.ground_truth[x1:x2, y1:y2] = False

    def _add_rooms(self):
        """
        Create a central atrium with side rooms.
        Two horizontal walls create upper and lower wings
        connected through the open central atrium.
        """
        # Upper horizontal divider — separates north wing from atrium
        self._add_wall(2, 34, 22, 34)
        self._add_wall(27, 34, 47, 34)
        # Doorways into north wing
        self.ground_truth[12, 34] = False
        self.ground_truth[37, 34] = False

        # Lower horizontal divider — separates south wing from atrium
        self._add_wall(2, 16, 22, 16)
        self._add_wall(27, 16, 47, 16)
        # Doorways into south wing
        self.ground_truth[12, 16] = False
        self.ground_truth[37, 16] = False

        # Vertical dividers in south wing
        self._add_wall(24, 2, 24, 15)
        self.ground_truth[24, 8] = False

        # Vertical dividers in north wing
        self._add_wall(24, 35, 24, 47)
        self.ground_truth[24, 41] = False

    def _add_wall(self, x1, y1, x2, y2):
        """Add a wall segment between two points (axis-aligned only)."""
        if x1 == x2:
            for y in range(min(y1, y2), max(y1, y2) + 1):
                if self.is_valid(x, y):
                    self.ground_truth[x1, y] = True
        elif y1 == y2:
            for x in range(min(x1, x2), max(x1, x2) + 1):
                if self.is_valid(x, y1):
                    self.ground_truth[x, y1] = True

    def is_valid(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def get_zones(self):
        """
        Return named zones for analysis and task allocation.
        """
        return {
            "south_left": {"x": (2, 23), "y": (0, 15)},
            "south_right": {"x": (25, 47), "y": (0, 15)},
            "atrium": {"x": (2, 47), "y": (17, 33)},
            "north_left": {"x": (2, 23), "y": (35, 47)},
            "north_right": {"x": (25, 47), "y": (35, 47)},
        }

    def __repr__(self):
        return (
            f"EnvironmentB(multi_entry, width={self.width}, "
            f"height={self.height}, seed={self.seed})"
        )