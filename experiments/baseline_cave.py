import pygame
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environments.environment_c import EnvironmentC
from visualisation.renderer import Renderer
from agents.scout import Scout, BehaviouralState
from agents.sentinel import Sentinel
from agents.base_agent import BaseAgent


def print_roll_of_honour(scouts, sentinel, final_metrics):
    width = 48
    line = "═" * width

    print(f"\n{line}")
    print(f"{'MISSION DEBRIEF':^{width}}")
    print(f"{'SentinelSwarm — Environment C':^{width}}")
    print(f"{'Lava Tube Analogue':^{width}}")
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
        print(f"    Thermal detections       : {rec.get('thermal_detections', 0)}")

        if fate == "lost":
            print(f"    Last known position      : {rec['last_known_position']}")
            print(f"    Data lost on casualty    : {rec['data_lost_on_casualty']} obs")
        else:
            print(f"    Final energy             : {scout.energy_fraction*100:.0f}%")

    print(f"\n{line}")
    print(f"  Scouts deployed   : {len(scouts)}")
    print(f"  Returned safely   : {returned}")
    print(f"  Lost in service   : {lost}")
    print(f"  Thermal anomalies : {len(sentinel.environment.thermal_anomalies)}")
    print(f"  Anomalies detected: {final_metrics.get('anomalies_detected', 0)}")
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

    env = EnvironmentC(seed=42, num_anomalies=3)
    renderer = Renderer(env, cell_px=8)   # smaller cells for larger grid

    clock = pygame.time.Clock()

    sentinel = Sentinel(position=env.sentinel_position, environment=env)
    sentinel.DECAY_RATE = 1.0

    entry = env.entry_points[0]

    # Five Scouts for the larger 100x100 environment
    all_scouts = [
        Scout(position=entry, max_energy=800, max_range=10, seed=i)
        for i in range(5)
    ]

    # Add thermal_detections to service records
    for scout in all_scouts:
        scout.service_record["thermal_detections"] = 0
        sentinel.queue_scout(scout)

    deployed_scouts = []
    scout_paths = {}
    anomalies_detected = set()

    print("SentinelSwarm — Environment C, lava tube analogue.")
    print(f"Environment: {env}")
    print(f"Thermal anomalies placed: {len(env.thermal_anomalies)}")
    print(f"Locations (hidden from agents): {env.thermal_anomalies}")
    print(f"5 Scouts queued. Deployment gap: {sentinel.DEPLOYMENT_GAP} timesteps")
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

        # Sentinel deployment
        if sentinel.should_deploy_next():
            new_scout = sentinel.deploy_next_scout()
            if new_scout:
                new_scout.service_record["timestep_deployed"] = sentinel.timestep
                new_scout.state = BehaviouralState.EXPLORING
                deployed_scouts.append(new_scout)
                scout_paths[new_scout.id] = []

        # Termination
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
                    "coverage": sentinel.calculate_coverage(),
                    "anomalies_detected": len(anomalies_detected)
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

            # Force return if energy critical
            if scout.energy_critical and scout.state != BehaviouralState.RETURNING:
                scout.state = BehaviouralState.RETURNING
                scout.target_position = entry
                scout_paths[scout.id] = scout.find_path(entry, env)

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

            # LIDAR scan
            scout.scan(env)

            # Thermal scan
            reading, thermal_obs = scout.scan_thermal(env)
            if thermal_obs and reading > 0.5:
                # Strong thermal detection — log it
                pos = tuple(scout.true_position)
                if pos not in anomalies_detected:
                    # Check if genuinely near an anomaly
                    for ax, ay in env.thermal_anomalies:
                        if abs(pos[0]-ax) + abs(pos[1]-ay) < 6:
                            anomalies_detected.add((ax, ay))
                            scout.service_record["thermal_detections"] += 1
                            print(
                                f"  [T:{sentinel.timestep:04d}] "
                                f"Scout S{scout.id} detects thermal anomaly "
                                f"near {(ax,ay)} — reading: {reading:.2f}"
                            )
                            break

            # Recharge at entry
            if tuple(scout.true_position) == entry:
                scout.energy = scout.max_energy
                scout.state = BehaviouralState.IDLE
                scout.correct_at_reference(entry)
                scout.target_position = None
                scout_paths[scout.id] = []
                sentinel.clear_claimed_frontier(scout.id)

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

        for s in all_scouts:
            if s not in deployed_scouts:
                scout_states.append({
                    "id": s.id,
                    "position": entry,
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