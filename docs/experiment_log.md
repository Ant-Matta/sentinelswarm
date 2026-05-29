# SentinelSwarm — Experiment Log

---

## Experiment 001 — Environment A Baseline
**Date:** 17 May 2026  
**Configuration:**
- Environment: A (single entry, linear, 50×50)
- Scouts: 3, max energy 500 each
- Deployment: Simultaneous
- Frontier scoring: Centrist heuristic

**Result:**
- Fidelity: 75.2%
- Coverage: 44.0%
- Duration: T:308

**Notes:** Scouts clustered in upper-centre region. Left rooms largely
unmapped. Frontier scoring biased toward environment centre.

---

## Experiment 002 — Environment A, Staggered Deployment
**Date:** 17 May 2026  
**Configuration:**
- Environment: A (single entry, linear, 50×50)
- Scouts: 3, max energy 500 each
- Deployment: Staggered, gap=40 timesteps
- Deployment condition: >15 frontiers available
- Frontier scoring: Isolation-weighted

**Result:**
- Fidelity: 91.0%
- Coverage: 85.6%
- Duration: T:606

**Notes:** Staggered deployment nearly doubled coverage vs simultaneous.
Each Scout deployed into a progressively more informed world model,
naturally exploring different regions without environmental prior knowledge.
All three Scouts returned safely.

---

## Experiment 003 — Environment B Baseline
**Date:** 27 May 2026  
**Configuration:**
- Environment: B (multi-entry, open atrium, 50×50)
- Scouts: 3, max energy 500 each
- Scout S0, S1: south entry
- Scout S2: north entry
- Deployment: Staggered, gap=40 timesteps
- Frontier scoring: Isolation-weighted

**Result:**
- Fidelity: 91.0%
- Coverage: 86.5%
- Duration: T:928

**Notes:** Same fidelity and near-identical coverage as Environment A
despite structurally different topology and dual entry points.
Multi-entry was significantly slower (T:928 vs T:606).
Scout S2 (north entry) ended at 100% energy — environment largely
mapped by S1 before S2 could contribute meaningfully.
Scout S3 travelled only 1 cell.

**Hypothesis:** 91% fidelity may represent a sensor ceiling for this
LIDAR configuration in 50×50 grids — wall-adjacent corners unreachable
regardless of Scout trajectory. Coverage ceiling similarly constrained
by total energy budget (3 × 500 = 1500 units) rather than entry topology.

**Next experiment:** Vary Scout energy budget and measure coverage scaling.
Does doubling energy double coverage or hit diminishing returns?

---

## Planned Experiments

- **Exp 004** — Energy scaling study: run Environment A with energy 
  budgets of 250, 500, 750, 1000 per Scout. Plot coverage vs energy.
- **Exp 005** — Scout count study: vary 1, 2, 3, 4, 5 Scouts at fixed 
  total energy budget. Does more Scouts always mean better coverage?
- **Exp 006** — Forced Scout failure: kill Scout mid-mission, measure 
  graceful degradation and coverage loss.
- **Exp 007** — Lost Scout recon protocol (Phase 1b).