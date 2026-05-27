import pygame
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environments.environment_a import EnvironmentA
from visualisation.renderer import Renderer
from agents.scout import Scout, BehaviouralState
from agents.sentinel import Sentinel
from agents.base_agent import BaseAgent

def print_roll_of_honour(scouts, sentinel, final_metrics):
    """Print the mission debrief and Roll of Honour to terminal."""
    width = 48
    line = "═" * width

    print(f"\n{line}")
    print(f"{'MISSION DEBRIEF':^{width}}")
    print(f"{'SentinelSwarm — Environment A':^{width}}")
    print(line)

    returned = 0
    lost = 0
    total_obs = 0
    total_dist = 0

    for scout in scouts:
        rec = scout.service_record
        fate = rec["fate"]
        sid = scout.id

        if fate == "returned" or fate == "active":
            returned += 1
            fate_str = "Returned safely"
        else:
            lost += 1
            fate_str = "Lost in service"

        total_obs += rec["observations_contributed"]
        total_dist += rec["distance_travelled"]

        print(f"\n  Scout S{sid} — {fate_str}")
        print(f"    Observations contributed : {rec['observations_contributed']}")
        print(f"    Distance travelled       : {rec['distance_travelled']} cells")

        if fate == "lost":
            print(f"    Last known position      : {rec['last_known_position']}")
            print(f"    Data lost on casualty    : {rec['data_lost_on_casualty']} obs")
        else:
            print(f"    Final energy             : {scout.energy_fraction*100:.0f}%")

    print(f"\n{line}")
    print(f"  Scouts deployed  : {len(scouts)}")
    print(f"  Returned safely  : {returned}")
    print(f"  Lost in service  : {lost}")
    print(f"  Total observations: {total_obs}")
    print(f"  Total distance   : {total_dist} cells")
    print(f"\n  Final fidelity   : {final_metrics['fidelity']:.1f}%")
    print(f"  Final coverage   : {final_metrics['coverage']:.1f}%")
    print(f"  Mission duration : T:{final_metrics['timestep']}")
    print(f"\n{line}")
    print(f"{'They mapped the unknown so we':^{width}}")
    wouldnt = "wouldn't have to."
    print(f"{wouldnt:^{width}}")
    print(f"{line}\n")


def main():
    BaseAgent.reset_id_counter()

    env = EnvironmentA(seed=42)
    renderer = Renderer(env, cell_px=12)
    clock = pygame.time.Clock()

    sentinel = Sentinel(position=env.sentinel_position, environment=env)

    # Disable decay for Phase 1 — static environment doesn't need it
    sentinel.DECAY_RATE = 1.0

    entry = env.entry_points[0]
    scouts = [
        Scout(position=entry, max_energy=500, seed=i)
        for i in range(3)
    ]

    for scout in scouts:
        sentinel.register_scout(scout)

    scout_paths = {s.id: [] for s in scouts}

    print("SentinelSwarm — mission started.")
    print(f"Environment: {env}")
    print("Press ESC to exit.")

    running = True
    mission_complete = False
    final_metrics = None

    while running:
        running = renderer.handle_events()

        if mission_complete:
            # Hold final frame — mission is done
            renderer.render(sentinel.world_model, [], final_metrics)
            clock.tick(10)
            continue

        # Check termination — all Scouts dead or returned with no energy
        all_inactive = all(
            not s.active or s.energy_critical
            for s in scouts
        )

        if all_inactive:
            mission_complete = True
            final_metrics = {
                "timestep": sentinel.timestep,
                "fidelity": sentinel.calculate_fidelity(),
                "coverage": sentinel.calculate_coverage()
            }
            # Mark all active scouts as returned
            for scout in scouts:
                if scout.active and scout.service_record["fate"] == "active":
                    scout.mark_returned(sentinel.timestep)
            print_roll_of_honour(scouts, sentinel, final_metrics)
            continue

        # Sentinel step
        directives = sentinel.step(scouts)

        for scout in scouts:
            if not scout.active:
                continue

            # Force return if energy critical
            if scout.energy_critical and scout.state != BehaviouralState.RETURNING:
                scout.state = BehaviouralState.RETURNING
                scout.target_position = entry
                scout_paths[scout.id] = scout.find_path(entry, env)

            # Apply Sentinel directive if not already returning
            elif scout.state != BehaviouralState.RETURNING:
                directive = directives.get(scout.id)
                if directive:
                    new_target = directive["target"]
                    if new_target != scout.target_position:
                        scout.target_position = new_target
                        scout_paths[scout.id] = scout.find_path(new_target, env)
                    if directive["type"] == "RETURN":
                        scout.state = BehaviouralState.RETURNING
                    else:
                        scout.state = BehaviouralState.EXPLORING

            # Follow path
            path = scout_paths[scout.id]
            if path:
                next_step = path[0]
                scout.move_toward(next_step, env)
                if tuple(scout.true_position) == next_step:
                    scout_paths[scout.id] = path[1:]
            elif scout.target_position:
                scout.move_toward(scout.target_position, env)

            # Scan
            scout.scan(env)

            # Recharge and reset at entry
            if tuple(scout.true_position) == entry:
                scout.energy = scout.max_energy
                scout.state = BehaviouralState.IDLE
                scout.correct_drift(entry)
                scout.target_position = None
                scout_paths[scout.id] = []

        scout_states = [
            {
                "id": s.id,
                "position": tuple(s.true_position),
                "energy": s.energy_fraction
            }
            for s in scouts
            if s.active
        ]

        metrics = {
            "timestep": sentinel.timestep,
            "fidelity": sentinel.calculate_fidelity(),
            "coverage": sentinel.calculate_coverage()
        }

        renderer.render(sentinel.world_model, scout_states, metrics)
        clock.tick(30)

    renderer.close()

    if final_metrics:
        print("\nFinal results logged. Close window to exit.")


if __name__ == "__main__":
    main()