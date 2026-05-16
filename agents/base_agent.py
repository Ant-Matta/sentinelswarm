class BaseAgent:
    """
    Shared properties for all agents in SentinelSwarm.

    Both Sentinel and Scout inherit from this.
    """

    # Class-level ID counter
    _id_counter = 0

    def __init__(self, position, agent_type="base"):
        self.id = BaseAgent._id_counter
        BaseAgent._id_counter += 1

        self.position = position          # (x, y) tuple
        self.agent_type = agent_type
        self.active = True                # False when dead or mission complete

    @classmethod
    def reset_id_counter(cls):
        """Reset ID counter between missions."""
        cls._id_counter = 0

    def __repr__(self):
        return (
            f"{self.agent_type.capitalize()}Agent("
            f"id={self.id}, position={self.position}, active={self.active})"
        )