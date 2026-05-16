import pygame
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environments.environment_a import EnvironmentA
from environments.environment_b import EnvironmentB
from visualisation.renderer import Renderer
from agents.scout import Scout, BehaviouralState
from agents.base_agent import BaseAgent


def build_world_model(all_observations):
    """Build a simple world model dict from accumulated observations."""
    world_model = {}
    for obs in all_observations:
        cell = obs["cell"]
        if cell not in world_model:
            world_model[cell] = obs.copy()
        else:
            # Keep highest confidence observation
            if obs["confidence"] > world_model[cell]["confidence"]:
                world_model[cell] = obs.copy()
    return world_model


def main():
    BaseAgent.reset_id_counter()

    env = EnvironmentA(seed=42)
    renderer = Renderer(env, cell_px=12)
    clock = pygame.time.Clock()

    # Spawn three Scouts at the entry point
    entry = env.entry_points[0]
    scouts = [
        Scout(position=entry, max_energy=300, seed=i)
        for i in range(3)
    ]

    # Simple random walk targets for demonstration
    import random
    random.seed(42)

    all_observations = []
    timestep = 0

    print("SentinelSwarm — baseline visualisation running.")
    print("Scouts exploring Environment A. Press ESC to exit.")

    running = True
    while running:
        running = renderer.handle_events()

        # Each Scout takes one step and scans
        for scout in scouts:
            if not scout.active:
                continue

            # Simple frontier: move toward a random passable cell
            if scout.target_position is None or scout.true_position == list(scout.target_position):
                while True:
                    tx = random.randint(0, env.width - 1)
                    ty = random.randint(0, env.height - 1)
                    if env.is_passable(tx, ty):
                        scout.target_position = (tx, ty)
                        break

            scout.move_toward(scout.target_position, env)
            new_obs = scout.scan(env)
            all_observations.extend(new_obs)

            # Return to entry if energy critical
            if scout.energy_critical:
                scout.target_position = entry
                if tuple(scout.true_position) == entry:
                    scout.energy = scout.max_energy  # recharge

        world_model = build_world_model(all_observations)

        scout_states = [
            {
                "id": s.id,
                "position": tuple(s.true_position),
                "energy": s.energy_fraction
            }
            for s in scouts
        ]

        # Simple coverage and fidelity
        known = sum(1 for c in world_model.values()
                    if c["state"] != 0)
        total = sum(1 for x in range(env.width)
                    for y in range(env.height)
                    if not env.ground_truth[x, y])
        coverage = (known / total * 100) if total > 0 else 0

        metrics = {
            "timestep": timestep,
            "fidelity": 0.0,
            "coverage": coverage
        }

        renderer.render(world_model, scout_states, metrics)
        clock.tick(30)
        timestep += 1

    renderer.close()


if __name__ == "__main__":
    main()