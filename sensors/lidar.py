import numpy as np


class LidarSensor:
    """
    Simulated 2D LIDAR sensor for Scout agents.

    Casts rays in N directions from the Scout's position.
    Returns distance to nearest obstacle and whether each
    ray hit a wall or reached maximum range.

    Models:
    - Finite range (max_range cells)
    - Gaussian distance noise
    - Angular resolution (num_rays)
    """

    def __init__(self, max_range=8, num_rays=36, noise_std=0.2):
        self.max_range = max_range
        self.num_rays = num_rays
        self.noise_std = noise_std

        # Precompute ray angles
        self.angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

    def scan(self, position, environment, rng=None):
        """
        Perform a LIDAR scan from the given position.

        Args:
            position: (x, y) tuple — Scout's current position
            environment: BaseEnvironment instance
            rng: numpy random generator for noise

        Returns:
            observations: list of dicts, one per ray:
                {
                    "cell": (x, y),
                    "state": CellState,
                    "confidence": float,
                    "distance": float
                }
            hits: list of (x, y) wall cells detected
        """
        from environments.base_environment import CellState

        if rng is None:
            rng = np.random.default_rng()

        px, py = position
        observations = {}
        hits = []

        for angle in self.angles:
            dx = np.cos(angle)
            dy = np.sin(angle)

            # Step along ray
            for step in range(1, self.max_range + 1):
                rx = int(round(px + dx * step))
                ry = int(round(py + dy * step))

                if not environment.is_valid(rx, ry):
                    break

                # Add noise to distance measurement
                noise = rng.normal(0, self.noise_std) if self.noise_std > 0 else 0
                measured_distance = step + noise

                # Confidence decreases with distance
                confidence = max(0.3, 1.0 - (step / self.max_range) * 0.6)

                # Apply surface albedo to confidence
                albedo = environment.get_albedo(rx, ry)
                adjusted_confidence = confidence * albedo

                if environment.ground_truth[rx, ry]:
                    hits.append((rx, ry))
                    obs_key = (rx, ry)
                    if obs_key not in observations:
                        observations[obs_key] = {
                            "cell": (rx, ry),
                            "state": CellState.OCCUPIED,
                            "confidence": adjusted_confidence,
                            "distance": measured_distance
                        }
                    break
                else:
                    obs_key = (rx, ry)
                    if obs_key not in observations:
                        observations[obs_key] = {
                            "cell": (rx, ry),
                            "state": CellState.FREE,
                            "confidence": adjusted_confidence,
                            "distance": measured_distance
                        }

        return list(observations.values()), hits

    def get_visible_cells(self, position, environment):
        """
        Return all cells visible from position without noise.
        Used for debug visualisation.
        """
        observations, _ = self.scan(position, environment, rng=np.random.default_rng())
        return [obs["cell"] for obs in observations]