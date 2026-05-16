import numpy as np
import heapq
from agents.base_agent import BaseAgent
from sensors.lidar import LidarSensor


class BehaviouralState:
    IDLE = "idle"
    EXPLORING = "exploring"
    RETURNING = "returning"
    RELAYING = "relaying"
    LOST_CONTACT = "lost_contact"
    MISSION_LOST = "mission_lost"


class Scout(BaseAgent):
    """
    Scout agent — cheap, expendable, mobile.

    Enters the unknown environment, conducts local LIDAR sensing,
    relays observations back toward the Sentinel.

    The Scout does not maintain a global map. It holds only:
    - Its current believed position (with drift)
    - A local observation buffer (queued for transmission)
    - Its current energy level
    - Its current behavioural state
    """

    # Energy costs per action
    COST_MOVE = 1.0
    COST_SCAN = 0.5
    COST_TRANSMIT = 2.0
    SAFETY_MARGIN = 0.15      # fraction of max energy kept in reserve

    def __init__(
        self,
        position,
        max_energy=200.0,
        max_range=8,
        drift_rate=0.05,
        seed=None
    ):
        super().__init__(position, agent_type="scout")

        # Position tracking
        self.true_position = list(position)
        self.believed_position = list(position)

        # Energy
        self.max_energy = max_energy
        self.energy = max_energy

        # Localisation drift
        self.drift_rate = drift_rate
        self.cumulative_drift = [0.0, 0.0]

        # Random state
        self.rng = np.random.default_rng(seed)

        # Sensor
        self.lidar = LidarSensor(max_range=max_range)

        # Observation buffer
        self.observation_buffer = []
        self.max_buffer_size = 100

        # Behavioural state
        self.state = BehaviouralState.IDLE

        # Current directive from Sentinel
        self.current_directive = None
        self.target_position = None

        # Mission log
        self.positions_visited = [tuple(position)]
        self.timestep = 0

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def move(self, direction, environment):
        """
        Attempt to move one cell in the given direction.

        Args:
            direction: one of "up", "down", "left", "right"
            environment: BaseEnvironment instance

        Returns:
            True if move succeeded, False if blocked
        """
        if self.energy <= 0:
            self.active = False
            return False

        dx, dy = {
            "up":    (0, -1),
            "down":  (0,  1),
            "left":  (-1, 0),
            "right": (1,  0),
        }.get(direction, (0, 0))

        nx = self.true_position[0] + dx
        ny = self.true_position[1] + dy

        if not environment.is_passable(nx, ny):
            return False

        # Update true position
        self.true_position[0] = nx
        self.true_position[1] = ny

        # Update believed position with drift
        drift_x = self.rng.normal(0, self.drift_rate)
        drift_y = self.rng.normal(0, self.drift_rate)
        self.cumulative_drift[0] += drift_x
        self.cumulative_drift[1] += drift_y

        self.believed_position[0] = self.true_position[0] + self.cumulative_drift[0]
        self.believed_position[1] = self.true_position[1] + self.cumulative_drift[1]

        # Deplete energy
        self.energy -= self.COST_MOVE
        self.positions_visited.append(tuple(self.true_position))
        self.timestep += 1

        if self.energy <= 0:
            self.active = False

        return True

    def move_toward(self, target, environment):
        """
        Move one step toward target position using simple
        axis-aligned movement. Returns True if moved.
        """
        tx, ty = target
        cx, cy = self.true_position

        dx = tx - cx
        dy = ty - cy

        if dx == 0 and dy == 0:
            return False

        if abs(dx) >= abs(dy):
            primary = "right" if dx > 0 else "left"
            secondary = "down" if dy > 0 else "up"
        else:
            primary = "down" if dy > 0 else "up"
            secondary = "right" if dx > 0 else "left"

        if self.move(primary, environment):
            return True
        if self.move(secondary, environment):
            return True

        return False

    def find_path(self, target, environment):
        """
        A* pathfinding from current position to target.
        Returns list of (x,y) waypoints, or empty list if no path found.
        """
        start = tuple(self.true_position)
        goal = tuple(target)

        if start == goal:
            return []

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = current[0] + dx, current[1] + dy
                neighbour = (nx, ny)

                if not environment.is_passable(nx, ny):
                    continue

                tentative_g = g_score[current] + 1

                if tentative_g < g_score.get(neighbour, float('inf')):
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative_g
                    f = tentative_g + heuristic(neighbour, goal)
                    heapq.heappush(open_set, (f, neighbour))

        return []

    # ------------------------------------------------------------------
    # Sensing
    # ------------------------------------------------------------------

    def scan(self, environment):
        """
        Perform a LIDAR scan and add observations to buffer.
        Returns the list of new observations.
        """
        if self.energy <= self.COST_SCAN:
            return []

        observations, _ = self.lidar.scan(
            tuple(self.true_position), environment, self.rng
        )

        self.energy -= self.COST_SCAN

        for obs in observations:
            if len(self.observation_buffer) < self.max_buffer_size:
                self.observation_buffer.append(obs)

        return observations

    # ------------------------------------------------------------------
    # Localisation correction
    # ------------------------------------------------------------------

    def correct_drift(self, reference_position):
        """
        Snap believed position to a known reference point.
        Called when Scout passes through a known location.
        """
        self.believed_position[0] = float(reference_position[0])
        self.believed_position[1] = float(reference_position[1])
        self.cumulative_drift = [0.0, 0.0]

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------

    @property
    def energy_fraction(self):
        return self.energy / self.max_energy

    @property
    def energy_critical(self):
        return self.energy_fraction <= self.SAFETY_MARGIN

    def estimated_return_cost(self, sentinel_position):
        """
        Estimate energy needed to return to Sentinel.
        Uses Manhattan distance as approximation.
        """
        sx, sy = sentinel_position
        cx, cy = self.true_position
        distance = abs(cx - sx) + abs(cy - sy)
        return distance * self.COST_MOVE * 1.2

    def can_return(self, sentinel_position):
        """Check if Scout has enough energy to return safely."""
        return self.energy > self.estimated_return_cost(sentinel_position)

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def flush_buffer(self):
        """Return and clear the observation buffer."""
        obs = self.observation_buffer.copy()
        self.observation_buffer = []
        return obs

    # ------------------------------------------------------------------
    # Status reporting
    # ------------------------------------------------------------------

    def get_status(self):
        """Return current Scout status as a dict."""
        return {
            "id": self.id,
            "position": tuple(self.true_position),
            "believed_position": tuple(self.believed_position),
            "energy": self.energy,
            "energy_fraction": self.energy_fraction,
            "energy_critical": self.energy_critical,
            "state": self.state,
            "buffer_size": len(self.observation_buffer),
            "active": self.active,
            "timestep": self.timestep,
        }

    def __repr__(self):
        return (
            f"Scout(id={self.id}, pos={self.true_position}, "
            f"energy={self.energy:.1f}/{self.max_energy}, "
            f"state={self.state}, active={self.active})"
        )