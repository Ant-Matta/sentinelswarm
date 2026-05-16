import pygame
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environments.environment_a import EnvironmentA
from environments.environment_b import EnvironmentB
from visualisation.renderer import Renderer


def main():
    # Toggle between A and B here
    env = EnvironmentA(seed=42)
    # env = EnvironmentB(seed=42)

    renderer = Renderer(env, cell_px=12)
    clock = pygame.time.Clock()

    print(f"Running: {env}")
    print(f"Entry points: {env.entry_points}")
    print(f"Sentinel position: {env.sentinel_position}")
    print("Press ESC or close window to exit.")

    running = True
    while running:
        running = renderer.handle_events()
        renderer.render(
            world_model={},
            scout_states=[],
            metrics={"timestep": 0, "fidelity": 0.0, "coverage": 0.0}
        )
        clock.tick(30)

    renderer.close()


if __name__ == "__main__":
    main()