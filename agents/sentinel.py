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
    DEPLOYMENT_GAP = 40            # timesteps between Scout deployments
    MIN_FRONTIERS_TO_DEPLOY = 15   # minimum new frontiers before deploying next Scout

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
        self.casualty_log = []

        # Deployment queue
        self.deployment_queue = []      # Scouts waiting to be deployed
        self.deployed_scouts = []       # Scouts currently on mission
        self.last_deployment_time = -1  # timestep of last deployment
        self.deployment_count = 0       # how many Scouts have been deployed

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
            "believed_position": tuple(scout.believed_position),
            "position_confidence": 1.0,
            "drift_magnitude": 0.0,
            "last_contact_timestep": self.timestep,
            "energy_level": scout.energy_fraction,
            "state": scout.state,
            "current_directive": None,
            "contact_lost": False,
            "mission_lost": False,
        }

    def queue_scout(self, scout):
        """
        Add a Scout to the deployment queue.
        Scouts wait here until the Sentinel decides to deploy them.
        """
        self.deployment_queue.append(scout)

    def should_deploy_next(self):
        """
        Sentinel decision: is it time to deploy the next Scout?

        Conditions:
        1. There are Scouts waiting in the queue
        2. Enough time has passed since last deployment
        3. The world model has enough new frontier data to give
           the next Scout a meaningfully different assignment
        4. At least one deployed Scout is healthy and transmitting

        Returns True if deployment should proceed.
        """
        if not self.deployment_queue:
            return False

        # Always deploy first Scout immediately
        if self.deployment_count == 0:
            return True

        # Time gap condition
        time_since_last = self.timestep - self.last_deployment_time
        if time_since_last < self.DEPLOYMENT_GAP:
            return False

        # Frontier condition — enough unexplored space to justify new Scout
        frontiers = self.get_frontiers()
        if len(frontiers) < self.MIN_FRONTIERS_TO_DEPLOY:
            return False

        # Health condition — at least one active Scout reporting in
        active_reporting = [
            reg for reg in self.scout_registry.values()
            if not reg["contact_lost"] and not reg["mission_lost"]
        ]
        if self.deployment_count > 0 and not active_reporting:
            return False

        return True

    def deploy_next_scout(self):
        """
        Deploy the next Scout from the queue.
        Returns the deployed Scout, or None if queue is empty.
        """
        if not self.deployment_queue:
            return None

        scout = self.deployment_queue.pop(0)
        self.deployed_scouts.append(scout)
        self.register_scout(scout)
        self.last_deployment_time = self.timestep
        self.deployment_count += 1

        print(
            f"  [T:{self.timestep:04d}] Sentinel deploys Scout S{scout.id} "
            f"— {len(self.deployment_queue)} remaining in queue"
        )

        return scout

    def update_scout_status(self, scout_id, status):
        """Update registry from a Scout status report."""
        if scout_id not in self.scout_registry:
            return
        reg = self.scout_registry[scout_id]
        reg["last_known_position"] = status["position"]
        reg["believed_position"] = status["believed_position"]
        reg["position_confidence"] = status["position_confidence"]
        reg["drift_magnitude"] = status["drift_magnitude"]
        reg["last_contact_timestep"] = self.timestep
        reg["energy_level"] = status["energy_fraction"]
        reg["state"] = status["state"]
        reg["contact_lost"] = False

    def clear_claimed_frontier(self, scout_id):
        """Release a Scout's claimed frontier when reached or reassigned."""
        if scout_id in self.claimed_frontiers:
            del self.claimed_frontiers[scout_id]

    def scouts_are_clustered(self, scout_a_id, scout_b_id, threshold=10):
        """
        Determine whether two Scouts are clustered too closely.

        Only trusted when both Scouts have high position confidence.
        Avoids false positives from drift-induced position errors.
        """
        reg_a = self.scout_registry.get(scout_a_id)
        reg_b = self.scout_registry.get(scout_b_id)

        if not reg_a or not reg_b:
            return False

        # Don't trust clustering detection if positions are uncertain
        combined_confidence = min(
            reg_a["position_confidence"],
            reg_b["position_confidence"]
        )
        if combined_confidence < 0.5:
            return False

        ax, ay = reg_a["last_known_position"]
        bx, by = reg_b["last_known_position"]
        distance = abs(ax - bx) + abs(ay - by)

        return distance < threshold

    def get_dispersal_target(self, scout_id, other_scout_ids):
        """
        Find a frontier target that maximises distance from other Scouts.
        Used when clustering is detected — no environmental prior needed.
        """
        reg = self.scout_registry.get(scout_id)
        if not reg:
            return None

        other_positions = [
            self.scout_registry[sid]["last_known_position"]
            for sid in other_scout_ids
            if sid in self.scout_registry and sid != scout_id
        ]

        frontiers = self.get_frontiers()
        best_score = -1
        best_pos = None

        claimed = set(self.claimed_frontiers.values())

        for _, pos in frontiers:
            if pos in claimed:
                continue

            if not other_positions:
                return pos

            # Score by minimum distance from all other Scouts
            min_dist = min(
                abs(pos[0] - ox) + abs(pos[1] - oy)
                for ox, oy in other_positions
            )

            if min_dist > best_score:
                best_score = min_dist
                best_pos = pos

        return best_pos    

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

        # Current Scout positions for isolation scoring
        scout_positions = [
            reg["last_known_position"]
            for reg in self.scout_registry.values()
            if not reg["mission_lost"] and not reg["contact_lost"]
        ]

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

                score = self._frontier_score(x, y, cell, unknown_neighbours, scout_positions)
                frontiers.append((score, (x, y)))

        frontiers.sort(reverse=True)
        return frontiers

    def _frontier_score(self, x, y, cell, unknown_neighbours, scout_positions=None):
        """
        Score a frontier cell for exploration priority.

        Weights:
        - Uncertainty value: how unknown is this cell
        - Adjacency bonus: how much unknown space lies beyond it
        - Isolation bonus: how far is it from current Scout positions
        """
        uncertainty_value = 1.0 - cell["confidence"]
        adjacency_bonus = unknown_neighbours / 8.0

        # Isolation bonus — prefer cells far from current Scout positions
        if scout_positions and len(scout_positions) > 0:
            min_dist = min(
                abs(x - sx) + abs(y - sy)
                for sx, sy in scout_positions
            )
            max_possible = self.environment.width + self.environment.height
            isolation = min(min_dist / max_possible, 1.0)
        else:
            isolation = 0.5

        return (
            0.5 * uncertainty_value +
            0.3 * adjacency_bonus +
            0.2 * isolation
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
                if not reg["mission_lost"]:
                    reg["mission_lost"] = True
                    scout.mark_lost(self.timestep)
                    self.casualty_log.append({
                        "scout_id": scout.id,
                        "timestep": self.timestep,
                        "last_known_position": reg["last_known_position"],
                        "observations_contributed": scout.service_record["observations_contributed"],
                        "data_lost": scout.service_record["data_lost_on_casualty"],
                    })
                continue
            elif since_contact > self.CONTACT_LOSS_THRESHOLD:
                reg["contact_lost"] = True

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

    def step(self, deployed_scouts):
        """
        Advance Sentinel by one timestep.

        Only processes Scouts that have been formally deployed.
        Queued Scouts are held at entry until deployment conditions met.
        """
        self.timestep += 1
        self.apply_confidence_decay()

        # Collect observations from deployed Scouts only
        for scout in deployed_scouts:
            if not scout.active:
                continue
            if scout.id not in self.scout_registry:
                self.register_scout(scout)

            self.update_scout_status(scout.id, scout.get_status())

            obs = scout.flush_buffer()
            if obs:
                self.fuse_observations(scout.id, obs)

        # Run decision cycle periodically
        directives = {}
        if self.timestep % self.DECISION_INTERVAL == 0:
            directives = self.run_decision_cycle(deployed_scouts)

        return directives

    def __repr__(self):
        coverage = self.calculate_coverage()
        return (
            f"Sentinel(pos={self.position}, "
            f"timestep={self.timestep}, "
            f"coverage={coverage:.1f}%)"
        )