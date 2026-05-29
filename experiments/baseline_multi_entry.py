import pygame
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environments.environment_b import EnvironmentB
from visualisation.renderer import Renderer
from agents.scout import Scout, BehaviouralState
from agents.sentinel import Sentinel
from agents.base_agent import BaseAgent


def print_roll_of_honour(scouts, sentinel, final_metrics):
    width = 48
    line = "═" * width

    print(f"\n{line}")
    print(f"{'MISSION DEBRIEF':^{width}}")
    print(f"{'SentinelSwarm — Environment B':^{width}}")
    print(f"{'Staggered Deployment Protocol':^{width}}")
    print(line)

    returned = 0
    lost = 0
    total_obs = 0
    total_dist = 0

    for scout in scouts:
        rec = scout.service_record
        fate = rec["fate"]
        sid = scout.id

        if fate in ("returned", "active"):
            returned += 1
            fate_str = "Returned safely"
        else:
            lost += 1
            fate_str = "Lost in service"

        total_obs += rec["observations_contributed"]
        total_dist += rec["distance_travelled"]

        print(f"\n  Scout S{sid} — {fate_str}")
        print(f"    Deployed at              : T:{rec['timestep_deployed']:04d}")
        print(f"    Observations contributed : {rec['observations_contributed']}")
        print(f"    Distance travelled       : {rec['distance_travelled']} cells")

        if fate == "lost":
            print(f"    Last known position      : {rec['last_known_position']}")
            print(f"    Data lost on casualty    : {rec['data_lost_on_casualty']} obs")
        else:
            print(f"    Final energy             : {scout.energy_fraction*100:.0f}%")

    print(f"\n{line}")
    print(f"  Scouts deployed   : {len(scouts)}")
    print(f"  Returned safely   : {returned}")
    print(f"  Lost in service   : {lost}")
    print(f"  Total observations: {total_obs}")
    print(f"  Total distance    : {total_dist} cells")
    print(f"\n  Final fidelity    : {final_metrics['fidelity']:.1f}%")
    print(f"  Final coverage    : {final_metrics['coverage']:.1f}%")
    print(f"  Mission duration  : T:{final_metrics['timestep']}")
    wouldnt = "wouldn't have to."
    print(f"\n{line}")
    print(f"{'They mapped the unknown so we':^{width}}")
    print(f"{wouldnt:^{width}}")
    print(f"{line}\n")


def main():
    BaseAgent.reset_id_counter()

    env = EnvironmentB(seed=42)
    renderer = Renderer(env, cell_px=12)
    clock = pygame.time.Clock()

    sentinel = Sentinel(position=env.sentinel_position, environment=env)
    sentinel.DECAY_RATE = 1.0

    # Environment B has two entry points
    # Sentinel sits at south entry by default
    south_entry = env.entry_points[0]
    north_entry = env.entry_points[1]

    # Create Scouts — assign to different entry points
    # Scout 0 — south entry (same as Sentinel)
    # Scout 1 — south entry, deployed later
    # Scout 2 — north entry, deployed later from opposite side
    scout_configs = [
        {"entry": south_entry, "seed": 0},
        {"entry": south_entry, "seed": 1},
        {"entry": north_entry, "seed": 2},
    ]

    all_scouts = [
        Scout(position=cfg["entry"], max_energy=500, seed=cfg["seed"])
        for cfg in scout_configs
    ]

    # Store each Scout's assigned entry point
    scout_entries = {
        all_scouts[i].id: scout_configs[i]["entry"]
        for i in range(len(all_scouts))
    }

    # Queue all Scouts
    for scout in all_scouts:
        sentinel.queue_scout(scout)

    deployed_scouts = []
    scout_paths = {}

    print("SentinelSwarm — Environment B, multi-entry staggered deployment.")
    print(f"Environment: {env}")
    print(f"Entry points: South {south_entry}, North {north_entry}")
    print(f"Scout S0, S1 → South entry")
    print(f"Scout S2     → North entry")
    print(f"Deployment gap: {sentinel.DEPLOYMENT_GAP} timesteps")
    print("Press ESC to exit.\n")

    running = True
    mission_complete = False
    final_metrics = None

    while running:
        running = renderer.handle_events()

        if mission_complete:
            renderer.render(sentinel.world_model, [], final_metrics)
            clock.tick(10)
            continue

        # Sentinel deployment decision
        if sentinel.should_deploy_next():
            new_scout = sentinel.deploy_next_scout()
            if new_scout:
                new_scout.service_record["timestep_deployed"] = sentinel.timestep
                new_scout.state = BehaviouralState.EXPLORING
                deployed_scouts.append(new_scout)
                scout_paths[new_scout.id] = []

        # Termination check
        if deployed_scouts:
            all_inactive = all(
                not s.active or s.energy_critical
                for s in deployed_scouts
            ) and not sentinel.deployment_queue

            if all_inactive:
                mission_complete = True
                final_metrics = {
                    "timestep": sentinel.timestep,
                    "fidelity": sentinel.calculate_fidelity(),
                    "coverage": sentinel.calculate_coverage()
                }
                for scout in deployed_scouts:
                    if scout.active and scout.service_record["fate"] == "active":
                        scout.mark_returned(sentinel.timestep)
                print_roll_of_honour(all_scouts, sentinel, final_metrics)
                continue

        # Sentinel step
        sentinel.step(deployed_scouts)

        # Scout behaviour
        for scout in deployed_scouts:
            if not scout.active:
                continue

            entry = scout_entries[scout.id]

            # Force return if energy critical
            if scout.energy_critical and scout.state != BehaviouralState.RETURNING:
                scout.state = BehaviouralState.RETURNING
                scout.target_position = entry
                scout_paths[scout.id] = scout.find_path(entry, env)

            # Assign new target if needed
            elif scout.state != BehaviouralState.RETURNING and not scout.target_position:
                other_ids = [
                    s.id for s in deployed_scouts
                    if s.id != scout.id and s.active
                ]

                clustered = any(
                    sentinel.scouts_are_clustered(scout.id, oid)
                    for oid in other_ids
                )

                if clustered:
                    target = sentinel.get_dispersal_target(scout.id, other_ids)
                else:
                    exclude = {
                        tuple(s.target_position)
                        for s in deployed_scouts
                        if s.target_position and s.id != scout.id
                    }
                    frontiers = sentinel.get_frontiers()
                    target = None
                    for _, pos in frontiers:
                        if pos not in exclude:
                            target = pos
                            break

                if target:
                    scout.target_position = target
                    scout_paths[scout.id] = scout.find_path(target, env)
                    scout.state = BehaviouralState.EXPLORING
                    sentinel.claimed_frontiers[scout.id] = target

            # Follow path
            path = scout_paths.get(scout.id, [])
            if path:
                next_step = path[0]
                scout.move_toward(next_step, env)
                if tuple(scout.true_position) == next_step:
                    scout_paths[scout.id] = path[1:]
            elif scout.target_position:
                scout.move_toward(scout.target_position, env)

            # Scan
            scout.scan(env)

            # Recharge at own entry point
            if tuple(scout.true_position) == entry:
                scout.energy = scout.max_energy
                scout.state = BehaviouralState.IDLE
                scout.correct_at_reference(entry)
                scout.target_position = None
                scout_paths[scout.id] = []
                sentinel.clear_claimed_frontier(scout.id)

            # Clear target when reached
            elif scout.target_position and \
                    tuple(scout.true_position) == tuple(scout.target_position):
                sentinel.clear_claimed_frontier(scout.id)
                scout.target_position = None
                scout_paths[scout.id] = []

        # Render
        scout_states = [
            {
                "id": s.id,
                "position": tuple(s.true_position),
                "energy": s.energy_fraction
            }
            for s in deployed_scouts if s.active
        ]

        # Show queued Scouts waiting at their entry points
        for i, s in enumerate(all_scouts):
            if s not in deployed_scouts:
                scout_states.append({
                    "id": s.id,
                    "position": scout_configs[i]["entry"],
                    "energy": 1.0
                })

        metrics = {
            "timestep": sentinel.timestep,
            "fidelity": sentinel.calculate_fidelity(),
            "coverage": sentinel.calculate_coverage()
        }

        renderer.render(sentinel.world_model, scout_states, metrics)
        clock.tick(30)

    renderer.close()


if __name__ == "__main__":
    main()