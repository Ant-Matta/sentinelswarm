import numpy as np
import heapq
from agents.base_agent import BaseAgent
from sensors.lidar import LidarSensor
from sensors.localisation import LocalisationModule


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

    Position estimation is fully delegated to LocalisationModule,
    keeping movement logic clean and the localisation stack swappable.
    """

    COST_MOVE = 1.0
    COST_SCAN = 0.5
    COST_TRANSMIT = 2.0
    SAFETY_MARGIN = 0.15

    def __init__(
        self,
        position,
        max_energy=200.0,
        max_range=8,
        drift_rate=0.05,
        seed=None
    ):
        super().__init__(position, agent_type="scout")

        # Localisation — fully delegated
        self.localisation = LocalisationModule(
            initial_position=position,
            drift_rate=drift_rate,
            seed=seed
        )

        # Energy
        self.max_energy = max_energy
        self.energy = max_energy

        # Sensor
        self.lidar = LidarSensor(max_range=max_range)

        # Observation buffer
        self.observation_buffer = []
        self.max_buffer_size = 100

        # Behavioural state
        self.state = BehaviouralState.IDLE

        # Current directive
        self.current_directive = None
        self.target_position = None

        # Mission log
        self.positions_visited = [tuple(position)]
        self.timestep = 0

        # Service record
        self.service_record = {
            "deployed_at": tuple(position),
            "observations_contributed": 0,
            "distance_travelled": 0,
            "fate": "active",
            "last_known_position": tuple(position),
            "data_lost_on_casualty": 0,
            "timestep_deployed": 0,
            "timestep_end": None,
        }

    # ------------------------------------------------------------------
    # Position properties — delegate to localisation
    # ------------------------------------------------------------------

    @property
    def true_position(self):
        return self.localisation.true_position

    @property
    def believed_position(self):
        return self.localisation.believed_position

    @property
    def position_confidence(self):
        return self.localisation.position_confidence

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    DIRECTION_DELTAS = {
        "up":    (0, -1),
        "down":  (0,  1),
        "left":  (-1, 0),
        "right": (1,  0),
    }

    def move(self, direction, environment):
        """
        Attempt to move one cell in the given direction.
        Returns True if move succeeded, False if blocked.
        """
        if self.energy <= 0:
            self.active = False
            return False

        dx, dy = self.DIRECTION_DELTAS.get(direction, (0, 0))
        nx = self.localisation.x + dx
        ny = self.localisation.y + dy

        if not environment.is_passable(nx, ny):
            return False

        # Delegate position update to localisation module
        self.localisation.update(dx, dy)

        # Energy and logging
        # Terrain-aware energy cost
        terrain_cost = environment.get_traversal_cost(nx, ny)
        self.energy -= self.COST_MOVE * terrain_cost
        self.positions_visited.append(tuple(self.localisation.true_position))
        self.service_record["distance_travelled"] += 1
        self.service_record["last_known_position"] = tuple(
            self.localisation.true_position
        )
        self.timestep += 1

        if self.energy <= 0:
            self.active = False

        return True

    def move_toward(self, target, environment):
        """Move one step toward target using axis-aligned movement."""
        tx, ty = target
        cx, cy = self.localisation.true_position

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
        """A* pathfinding to target. Returns waypoint list."""
        start = tuple(self.localisation.true_position)
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

            for ddx, ddy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = current[0] + ddx, current[1] + ddy
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
        """Perform LIDAR scan and buffer observations."""
        if self.energy <= self.COST_SCAN:
            return []

        observations, _ = self.lidar.scan(
            tuple(self.localisation.true_position),
            environment,
            self.localisation.rng
        )

        self.energy -= self.COST_SCAN

        for obs in observations:
            if len(self.observation_buffer) < self.max_buffer_size:
                self.observation_buffer.append(obs)

        return observations

    def scan_thermal(self, environment):
        """
        Sample the thermal field at current position.
        Returns thermal reading 0.0-1.0 and flags anomaly
        if reading exceeds detection threshold.
        """
        x, y = self.localisation.true_position
        reading = environment.get_thermal_reading(x, y)

        if reading > 0.2:
            # Detectable thermal gradient
            anomaly_obs = {
                "type": "thermal",
                "position": (x, y),
                "reading": reading,
                "confidence": reading
            }
            return reading, anomaly_obs

        return reading, None

    # ------------------------------------------------------------------
    # Localisation correction
    # ------------------------------------------------------------------

    def correct_at_reference(self, known_position):
        """Delegate drift correction to localisation module."""
        self.localisation.correct_at_reference(known_position)

    def get_range_to(self, other_scout):
        """Estimate distance to another Scout via localisation module."""
        return self.localisation.get_range_to(
            other_scout.believed_position
        )

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
        sx, sy = sentinel_position
        cx, cy = self.localisation.true_position
        distance = abs(cx - sx) + abs(cy - sy)
        return distance * self.COST_MOVE * 1.2

    def can_return(self, sentinel_position):
        return self.energy > self.estimated_return_cost(sentinel_position)

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def flush_buffer(self):
        """Return and clear observation buffer."""
        obs = self.observation_buffer.copy()
        self.service_record["observations_contributed"] += len(obs)
        self.observation_buffer = []
        return obs

    # ------------------------------------------------------------------
    # Service record
    # ------------------------------------------------------------------

    def mark_returned(self, timestep):
        self.service_record["fate"] = "returned"
        self.service_record["timestep_end"] = timestep

    def mark_lost(self, timestep):
        self.service_record["fate"] = "lost"
        self.service_record["timestep_end"] = timestep
        self.service_record["data_lost_on_casualty"] = len(
            self.observation_buffer
        )
        self.active = False

    # ------------------------------------------------------------------
    # Status reporting
    # ------------------------------------------------------------------

    def get_status(self):
        """Return full Scout status for Sentinel registry update."""
        loc = self.localisation.get_status()
        return {
            "id": self.id,
            "position": loc["true_position"],
            "believed_position": loc["believed_position"],
            "position_confidence": loc["position_confidence"],
            "drift_magnitude": loc["drift_magnitude"],
            "steps_since_correction": loc["steps_since_correction"],
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
            f"Scout(id={self.id}, "
            f"pos={tuple(self.localisation.true_position)}, "
            f"conf={self.position_confidence:.2f}, "
            f"energy={self.energy:.0f}/{self.max_energy}, "
            f"state={self.state})"
        )