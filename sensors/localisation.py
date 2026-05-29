import numpy as np


class LocalisationModule:
    """
    Manages Scout position estimation.

    Models dead reckoning with drift accumulation and position confidence
    tracking. Designed to be extended with SLAM and inter-agent ranging
    in future phases.

    Architecture note:
        This module is intentionally separated from Scout movement logic
        so that the localisation stack can be swapped independently.
        Current implementation: dead reckoning baseline.
        Planned extensions: SLAM feature matching, UWB inter-agent ranging.
    """

    def __init__(self, initial_position, drift_rate=0.05, seed=None):
        self.true_position = list(initial_position)
        self.believed_position = list(initial_position)
        self.cumulative_drift = [0.0, 0.0]
        self.drift_rate = drift_rate
        self.rng = np.random.default_rng(seed)

        # Confidence in believed position — 1.0 = certain, 0.0 = lost
        self.position_confidence = 1.0

        # Tracking how long since last correction
        self.steps_since_correction = 0

        # Confidence decay per uncorrected step
        self.confidence_decay_per_step = 0.005

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, dx, dy):
        """
        Update position estimate after one movement step.

        Args:
            dx: true x displacement (-1, 0, or 1)
            dy: true y displacement (-1, 0, or 1)
        """
        # Update true position
        self.true_position[0] += dx
        self.true_position[1] += dy

        # Accumulate drift noise
        drift_x = self.rng.normal(0, self.drift_rate)
        drift_y = self.rng.normal(0, self.drift_rate)
        self.cumulative_drift[0] += drift_x
        self.cumulative_drift[1] += drift_y

        # Believed position = true position + accumulated drift
        self.believed_position[0] = (
            self.true_position[0] + self.cumulative_drift[0]
        )
        self.believed_position[1] = (
            self.true_position[1] + self.cumulative_drift[1]
        )

        # Confidence degrades with each uncorrected step
        self.steps_since_correction += 1
        self.position_confidence = max(
            0.1,
            1.0 - (self.steps_since_correction * self.confidence_decay_per_step)
        )

    # ------------------------------------------------------------------
    # Correction
    # ------------------------------------------------------------------

    def correct_at_reference(self, known_position):
        """
        Correct believed position at a known reference point.

        Called when Scout passes through a landmark whose global
        position is known — entry point, Sentinel position, or
        any cell with verified ground truth.

        Resets drift and restores full position confidence.
        """
        self.believed_position = list(known_position)
        self.true_position = list(known_position)
        self.cumulative_drift = [0.0, 0.0]
        self.position_confidence = 1.0
        self.steps_since_correction = 0

    # ------------------------------------------------------------------
    # Inter-agent ranging
    # ------------------------------------------------------------------

    def get_range_to(self, other_believed_position, noise_std=0.5):
        """
        Estimate distance to another agent from believed positions.

        Current implementation: Euclidean distance with optional noise.
        Future implementation: UWB time-of-flight ranging model.

        Args:
            other_believed_position: (x, y) of the other agent
            noise_std: standard deviation of ranging noise in cells

        Returns:
            Estimated distance as float
        """
        dx = self.believed_position[0] - other_believed_position[0]
        dy = self.believed_position[1] - other_believed_position[1]
        true_range = (dx**2 + dy**2) ** 0.5

        # Add ranging noise
        noise = self.rng.normal(0, noise_std)
        return max(0.0, true_range + noise)

    def get_combined_confidence(self, other_confidence):
        """
        Combined positional confidence between this Scout and another.
        Used by Sentinel to decide whether to trust clustering detection.
        """
        return min(self.position_confidence, other_confidence)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def x(self):
        return self.true_position[0]

    @property
    def y(self):
        return self.true_position[1]

    @property
    def believed_x(self):
        return self.believed_position[0]

    @property
    def believed_y(self):
        return self.believed_position[1]

    @property
    def drift_magnitude(self):
        """Total accumulated drift distance."""
        return (
            self.cumulative_drift[0]**2 +
            self.cumulative_drift[1]**2
        ) ** 0.5

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self):
        """Return localisation state as dict for Sentinel reporting."""
        return {
            "true_position": tuple(self.true_position),
            "believed_position": tuple(self.believed_position),
            "position_confidence": self.position_confidence,
            "cumulative_drift": tuple(self.cumulative_drift),
            "drift_magnitude": self.drift_magnitude,
            "steps_since_correction": self.steps_since_correction
        }

    def __repr__(self):
        return (
            f"LocalisationModule("
            f"true={tuple(self.true_position)}, "
            f"believed={tuple(int(b) for b in self.believed_position)}, "
            f"confidence={self.position_confidence:.2f}, "
            f"drift={self.drift_magnitude:.2f})"
        )