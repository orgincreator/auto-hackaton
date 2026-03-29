# 🚀 AMSS — Autonomous Modular Space Station

> A space station that builds itself. No astronaut. No remote control. No single point of failure.

AMSS is a multi-layer simulation of autonomous orbital assembly — combining real orbital mechanics, distributed consensus, and machine learning into a single cohesive system.

---

## What's Inside

| File | What it does |
|------|-------------|
| `amss_docking_sim.py` | Physics-based docking simulation using real orbital mechanics + Raft consensus |
| `flet_docking_sim_live.py` | Interactive visual simulation of multiple modules assembling autonomously |
| `this_is_another_for_the_docking.py` | CNN + RL policy network for vision-based autonomous docking (in development) |

---

## The Three Layers

### Layer 1 — Real Orbital Navigation (`amss_docking_sim.py`)

Physics-grounded docking simulation in the Hill reference frame.

- **Clohessy-Wiltshire (CW) equations** — the same relative motion model used in real rendezvous missions
- **MPC-style controller** — closed-form optimal thrust law with phase-dependent gains (APPROACH → CAPTURE → LOCK → HANDSHAKE)
- **RK4 integrator** — 4th-order Runge-Kutta for accurate state propagation
- **Raft consensus** — if the Control Module fails mid-mission, surviving modules elect a new leader and continue
- Produces a full mission report plot: trajectory, distance, speed, thrust commands, and mission log

**Result from simulation:** 666s duration · 46mm final distance · 0.008 mm/s final speed · Status: COMPLETE

### Layer 2 — Distributed Visual Assembly (`flet_docking_sim_live.py`)

Real-time interactive simulation of multiple Flet modules assembling around a central hub.

- Modules orbit, approach, negotiate, vote, and lock in — no central controller
- **Raft-inspired leader election** with health-based voting — leader failure triggers instant re-election
- Full docking state machine per module: `LAUNCH → APPROACH → NEGOTIATE → DOCK → LOCKED`
- Energy bars, vote badges (YES/NO), leader crown, orbital trails
- Interactive controls: pause/play, rewind, new generation, zoom/pan, speed and spread sliders

### Layer 3 — Learning to Dock (`this_is_another_for_the_docking.py`) *(in development)*

Machine learning layer being integrated on top of the physics simulation.

- **DockingCNN** — 3-layer convolutional network that estimates relative state (x, z, vx, vz) from 128×128 RGB simulation frames
- **DockingPolicy** — MLP reinforcement learning agent (6→64→64→2) that learns thrust commands end-to-end
- **DockingTrainer** — Adam-based training loop with MSE loss
- Goal: replace the hand-coded MPC controller entirely with a learned policy

---

## Quickstart

### Requirements

```bash
pip install numpy matplotlib torch
```

### Run the physics simulation (generates a plot)

```bash
python amss_docking_sim.py
```

### Run the interactive visual simulation

```bash
python flet_docking_sim_live.py
```

**Controls:**

| Key / Input | Action |
|-------------|--------|
| `SPACE` | Pause / Resume |
| `R` | Rewind to t=0 |
| `N` | New generation |
| Scroll | Zoom in/out |
| Click + Drag | Pan canvas |
| Speed slider | 1x – 10x real-time |
| Spread / Vel sliders | Adjust initial conditions |

---

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│              AMSS Simulation Stack           │
├─────────────────────────────────────────────┤
│  Layer 3: Machine Learning                  │
│  DockingCNN (vision) + DockingPolicy (RL)   │
├─────────────────────────────────────────────┤
│  Layer 2: Distributed Consensus             │
│  Raft Election · State Machine · Peer Votes │
├─────────────────────────────────────────────┤
│  Layer 1: Orbital Physics                   │
│  CW Equations · MPC Controller · RK4        │
└─────────────────────────────────────────────┘
```

---

## Key Parameters

```python
# Orbit
R_ORBIT  = 6.771e6   # ~400 km LEO
N        = 0.001107  # mean motion [rad/s]

# Docking thresholds
CAPTURE_DIST  = 0.50   # [m]
LOCK_DIST     = 0.05   # [m]
HANDSHAKE_VEL = 0.005  # [m/s]

# Fault injection
FAILURE_STEP = 300     # Control Module fails at t=600s
```

---

## Built For

This project was built for a hackathon under the theme of **next-generation autonomous space infrastructure**. It is a proof of concept for modular self-assembling stations where no single failure can stop the mission.

---

## Roadmap

- [x] CW physics engine + MPC controller
- [x] Raft-style consensus with fault injection
- [x] Interactive multi-module visual simulation
- [ ] Close the CNN + RL training loop
- [ ] Full 3D / 6-DOF docking with attitude control
- [ ] Hardware-in-the-loop integration

---

## License

MIT
