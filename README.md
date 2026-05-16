# SentinelSwarm

**A federated hierarchical autonomy architecture for cooperative environmental mapping under partial observability and communication constraints.**

---

## What Is This

SentinelSwarm investigates how a network of heterogeneous autonomous agents — differentiated by cost, capability, and risk profile — can cooperate to build the most accurate possible model of an unknown environment under resource and communication constraints.

The system comprises three tiers:

- **Human Handler** — sets mission parameters, receives synthesised intelligence, retains override authority
- **Sentinel Node** — stationary, outside the operational zone, maintains the global probabilistic world model, issues directives to Scouts, never enters the hazard zone
- **Scout Agents** — cheap, expendable, mobile, enter the unknown environment, conduct local sensing, relay observations back through the communication network

The map is not the end product. The end product is a living, confidence-weighted world model that supports human decision-making, survives agent failure, and improves continuously as long as any Scout remains operational.

---

## Core Research Question

Does a hierarchical Sentinel-Scout architecture with federated peer synchronisation produce superior map fidelity, mission robustness, and resource efficiency compared to independent or fully distributed swarm approaches under equivalent constraints?

---

## Project Status

**Phase 1 — In Progress**
- [ ] Base simulation environment
- [ ] Scout and Sentinel agents
- [ ] Communication protocol with degradation
- [ ] Map fusion engine
- [ ] Sentinel decision logic
- [ ] Logging and replay system
- [ ] Visualisation
- [ ] Baseline experiments

---

## Getting Started

```bash
git clone https://github.com/Ant-Matta/sentinelswarm.git
cd sentinelswarm
pip install -r requirements.txt
python experiments/baseline_single_entry.py
```

---

## Architecture

See `docs/architecture.md` for full design decisions.
See `docs/problem_statement.md` for the formal problem statement.

---

## License

MIT
