import numpy as np
from agents.base_agent import BaseAgent
from environments.base_environment import CellState


class Sentinel(BaseAgent):
    """
    Sentinel agent — stationary, expensive, outside the operational zone.

    Responsibilities:
    - Maintain the global probabilistic world model
    - Fuse incoming Scout observations (Bayesian update)
    - Track Scout registry (position, energy, state)
    - Issue directives to Scouts (navigate, return, hold)
    - Identify frontier cells for exploration
    - Manage energy-aware return decisions

    The Sentinel never enters the environment.
    Its position is fixed at sentinel_position throughout the mission.
    """

    DECISION_INTERVAL = 5          # timesteps between decision cycles
    CONTACT_LOSS_THRESHOLD = 15    # timesteps before flagging contact lost
    MISSION_LOST_THRESHOLD = 45    # timesteps before declaring Scout mission lost
    CONTRADICTION_THRESHOLD = 0.2  # confidence gap to trigger contradiction handling
    CONTRADICTION_PENALTY = 0.7    # confidence multiplier on contradiction
    DECAY_RATE = 0.997             # confidence decay per timestep
    MAX_BUFFER_PROCESS = 20        # max observations processed per Scout per cycle

    def __init__(self, position, environment):
        super().__init__(position, agent_type="sentinel")
        self.environment = environment

        # World model — dict mapping (x,y) -> cell dict
        self.world_model = {}
        self._initialise_world_model()

        # Scout registry
        self.scout_registry = {}

        # Claimed frontiers — cells already assigned to a Scout
        self.claimed_frontiers = {}   # scout_id -> target (x,y)

        # Mission timestep
        self.timestep = 0

        # Directive log
        self.directives_issued = []

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialise_world_model(self):
        """Set all cells to unknown at mission start."""
        for x in range(self.environment.width):
            for y in range(self.environment.height):
                self.world_model[(x, y)] = {
                    "state": CellState.UNKNOWN,
                    "confidence": 0.0,
                    "last_observed": -1,
                    "observed_by": [],
                    "anomaly_likelihood": 0.0
                }

    # ------------------------------------------------------------------
    # Scout registry
    # ------------------------------------------------------------------

    def register_scout(self, scout):
        """Add a Scout to the registry."""
        self.scout_registry[scout.id] = {
            "id": scout.id,
            "last_known_position": tuple(scout.true_position),
            "last_contact_timestep": self.timestep,
            "energy_level": scout.energy_fraction,
            "state": scout.state,
            "current_directive": None,
            "contact_lost": False,
            "mission_lost": False,
        }

    def update_scout_status(self, scout_id, status):
        """Update registry from a Scout status report."""
        if scout_id not in self.scout_registry:
            return
        reg = self.scout_registry[scout_id]
        reg["last_known_position"] = status["position"]
        reg["last_contact_timestep"] = self.timestep
        reg["energy_level"] = status["energy_fraction"]
        reg["state"] = status["state"]
        reg["contact_lost"] = False

    # ------------------------------------------------------------------
    # Map fusion
    # ------------------------------------------------------------------

    def fuse_observations(self, scout_id, observations):
        """
        Process a batch of observations from a Scout.
        Updates the world model using Bayesian confidence combination.
        """
        processed = 0
        for obs in observations:
            if processed >= self.MAX_BUFFER_PROCESS:
                break

            cell_key = tuple(obs["cell"])
            if not self.environment.is_valid(*cell_key):
                continue

            cell = self.world_model[cell_key]
            incoming_state = obs["state"]
            incoming_conf = obs["confidence"]

            if cell["state"] == CellState.UNKNOWN:
                # First observation — accept directly
                cell["state"] = incoming_state
                cell["confidence"] = incoming_conf

            elif cell["state"] == incoming_state:
                # Consistent — strengthen confidence
                cell["confidence"] = self._combine_confidences(
                    cell["confidence"], incoming_conf
                )

            else:
                # Contradiction
                self._handle_contradiction(cell, incoming_state, incoming_conf)

            cell["last_observed"] = self.timestep
            if scout_id not in cell["observed_by"]:
                cell["observed_by"].append(scout_id)

            processed += 1

    def _combine_confidences(self, existing, incoming):
        """Bayesian confidence combination."""
        return 1.0 - ((1.0 - existing) * (1.0 - incoming))

    def _handle_contradiction(self, cell, new_state, new_conf):
        """
        Resolve conflicting observations.
        High confidence new observation overwrites.
        Similar confidence marks cell as contested.
        """
        diff = new_conf - cell["confidence"]

        if diff > self.CONTRADICTION_THRESHOLD:
            # New observation significantly more confident
            cell["state"] = new_state
            cell["confidence"] = new_conf - cell["confidence"]

        elif abs(diff) <= self.CONTRADICTION_THRESHOLD:
            # Similar confidence — contested, needs reinvestigation
            cell["state"] = CellState.CONTESTED
            cell["confidence"] = 0.3

        else:
            # Existing more confident — retain but penalise
            cell["confidence"] *= self.CONTRADICTION_PENALTY

    def apply_confidence_decay(self):
        """Decay confidence of all observed cells each timestep."""
        for cell in self.world_model.values():
            if cell["state"] != CellState.UNKNOWN:
                cell["confidence"] *= self.DECAY_RATE
                # If confidence drops very low, revert toward unknown
                if cell["confidence"] < 0.1:
                    cell["state"] = CellState.UNKNOWN
                    cell["confidence"] = 0.0

    # ------------------------------------------------------------------
    # Frontier identification
    # ------------------------------------------------------------------

    def get_frontiers(self):
        """
        Identify frontier cells — unknown or low-confidence cells
        adjacent to known free cells.

        Returns list of (score, (x,y)) tuples, sorted descending.
        """
        frontiers = []
        claimed = set(self.claimed_frontiers.values())

        for x in range(self.environment.width):
            for y in range(self.environment.height):
                cell = self.world_model[(x, y)]

                # Skip walls and high-confidence known cells
                if cell["state"] == CellState.OCCUPIED and cell["confidence"] > 0.7:
                    continue

                is_frontier = (
                    cell["state"] == CellState.UNKNOWN or
                    cell["state"] == CellState.CONTESTED or
                    (cell["state"] == CellState.FREE and cell["confidence"] < 0.4)
                )

                if not is_frontier:
                    continue

                if not self.environment.is_passable(x, y):
                    continue

                # Check adjacency to known free space
                has_known_neighbour = False
                unknown_neighbours = 0
                for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                    nx, ny = x+dx, y+dy
                    if not self.environment.is_valid(nx, ny):
                        continue
                    n = self.world_model[(nx, ny)]
                    if n["state"] == CellState.FREE and n["confidence"] > 0.5:
                        has_known_neighbour = True
                    if n["state"] == CellState.UNKNOWN:
                        unknown_neighbours += 1

                if not has_known_neighbour and cell["state"] != CellState.UNKNOWN:
                    continue

                score = self._frontier_score(x, y, cell, unknown_neighbours)
                frontiers.append((score, (x, y)))

        frontiers.sort(reverse=True)
        return frontiers

    def _frontier_score(self, x, y, cell, unknown_neighbours):
        """Score a frontier cell for exploration priority."""
        uncertainty_value = 1.0 - cell["confidence"]
        adjacency_bonus = unknown_neighbours / 8.0

        # Distance from environment centre (prefer central exploration)
        cx = self.environment.width / 2
        cy = self.environment.height / 2
        dist_from_centre = np.sqrt((x - cx)**2 + (y - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        centrality = 1.0 - (dist_from_centre / max_dist)

        return (
            0.5 * uncertainty_value +
            0.3 * adjacency_bonus +
            0.2 * centrality
        )

    # ------------------------------------------------------------------
    # Decision engine
    # ------------------------------------------------------------------

    def run_decision_cycle(self, scouts):
        """
        Periodic decision cycle — runs every DECISION_INTERVAL timesteps.

        Priority order:
        1. Emergency — critical energy, contact lost
        2. Tactical — relay chain management, contested cells
        3. Strategic — frontier assignment
        """
        directives = {}

        # Priority 1 — Emergency
        for scout in scouts:
            if not scout.active:
                continue
            reg = self.scout_registry.get(scout.id)
            if not reg:
                continue

            if scout.energy_critical or not scout.can_return(self.position):
                directives[scout.id] = {
                    "type": "RETURN",
                    "urgency": "critical",
                    "target": self.environment.entry_points[0]
                }
                reg["current_directive"] = directives[scout.id]
                continue

            # Check contact loss
            since_contact = self.timestep - reg["last_contact_timestep"]
            if since_contact > self.MISSION_LOST_THRESHOLD:
                reg["mission_lost"] = True
                continue
            elif since_contact > self.CONTACT_LOSS_THRESHOLD:
                reg["contact_lost"] = True

        # Priority 2 — Tactical (contested cells)
        contested = [
            pos for pos, cell in self.world_model.items()
            if cell["state"] == CellState.CONTESTED
        ]

        # Priority 3 — Strategic frontier assignment
        frontiers = self.get_frontiers()
        frontier_idx = 0

        for scout in scouts:
            if not scout.active:
                continue
            if scout.id in directives:
                continue  # already has emergency directive

            reg = self.scout_registry.get(scout.id)
            if not reg:
                continue

            # Assign contested cell first if available
            if contested:
                target = contested.pop(0)
                directives[scout.id] = {
                    "type": "NAVIGATE",
                    "target": target,
                    "priority": 2,
                    "reason": "contested_resolution"
                }
                self.claimed_frontiers[scout.id] = target
                reg["current_directive"] = directives[scout.id]
                continue

            # Assign frontier
            while frontier_idx < len(frontiers):
                score, pos = frontiers[frontier_idx]
                frontier_idx += 1
                if pos not in self.claimed_frontiers.values():
                    directives[scout.id] = {
                        "type": "NAVIGATE",
                        "target": pos,
                        "priority": 3,
                        "reason": "frontier_exploration"
                    }
                    self.claimed_frontiers[scout.id] = pos
                    reg["current_directive"] = directives[scout.id]
                    break

        self.directives_issued.append({
            "timestep": self.timestep,
            "directives": directives
        })

        return directives

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def calculate_coverage(self):
        """Percentage of passable cells with known state."""
        known = 0
        total = 0
        for x in range(self.environment.width):
            for y in range(self.environment.height):
                if self.environment.is_passable(x, y):
                    total += 1
                    cell = self.world_model[(x, y)]
                    if cell["state"] != CellState.UNKNOWN:
                        known += 1
        return (known / total * 100) if total > 0 else 0.0

    def calculate_fidelity(self):
        """
        Map fidelity — how closely world model matches ground truth.
        Rewards confident correct beliefs.
        Penalises confident incorrect beliefs.
        """
        scored = 0.0
        total_weight = 0.0

        for x in range(self.environment.width):
            for y in range(self.environment.height):
                cell = self.world_model[(x, y)]
                true_state = self.environment.true_cell_state(x, y)

                if cell["state"] == CellState.UNKNOWN:
                    weight = 0.5
                    score = 0.0
                elif cell["state"] == true_state:
                    weight = cell["confidence"]
                    score = cell["confidence"]
                else:
                    weight = cell["confidence"]
                    score = -cell["confidence"]

                scored += score * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0
        raw = scored / total_weight
        return max(0.0, (raw + 1.0) / 2.0 * 100.0)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, scouts):
        """
        Advance Sentinel by one timestep.
        Collects observations, applies decay, runs decision cycle.
        """
        self.timestep += 1
        self.apply_confidence_decay()

        # Collect observations from all active Scouts
        for scout in scouts:
            if not scout.active:
                continue
            if scout.id not in self.scout_registry:
                self.register_scout(scout)

            # Update registry
            self.update_scout_status(scout.id, scout.get_status())

            # Fuse observations
            obs = scout.flush_buffer()
            if obs:
                self.fuse_observations(scout.id, obs)

        # Run decision cycle periodically
        directives = {}
        if self.timestep % self.DECISION_INTERVAL == 0:
            directives = self.run_decision_cycle(scouts)

        return directives

    def __repr__(self):
        coverage = self.calculate_coverage()
        return (
            f"Sentinel(pos={self.position}, "
            f"timestep={self.timestep}, "
            f"coverage={coverage:.1f}%)"
        )