"""
AMSS – Autonomous Modular Space Station
2D Orbital Docking Simulation
======================================================
Architecture grounded in the AMSS Full Report:
  - Clohessy-Wiltshire (CW) relative orbital mechanics
  - Analytical MPC-style controller (closed-form optimal thrust)
  - State Machine: APPROACH → CAPTURE → LOCK → HANDSHAKE → COMPLETE
  - Raft-like consensus leader election on Control Module failure
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

# ── Constants ────────────────────────────────────────────────────────────────
MU       = 3.986004418e14
R_ORBIT  = 6.771e6
N        = np.sqrt(MU / R_ORBIT**3)   # ~0.001107 rad/s  (LEO ~400 km)
DT       = 2.0                         # timestep [s]
SIM_STEPS = 1500

CAPTURE_DIST  = 0.50    # [m]
LOCK_DIST     = 0.05    # [m]
HANDSHAKE_VEL = 0.005   # [m/s]
FAILURE_STEP  = 300     # step index of Control Module failure

# ── Enumerations ─────────────────────────────────────────────────────────────
class DockingState(Enum):
    APPROACH   = auto()
    CAPTURE    = auto()
    LOCK       = auto()
    HANDSHAKE  = auto()
    COMPLETE   = auto()
    ABORTED    = auto()

class RaftRole(Enum):
    FOLLOWER  = auto()
    CANDIDATE = auto()
    LEADER    = auto()

STATE_COLORS = {
    DockingState.APPROACH  : "#4FC3F7",
    DockingState.CAPTURE   : "#FFD54F",
    DockingState.LOCK      : "#FF8A65",
    DockingState.HANDSHAKE : "#81C784",
    DockingState.COMPLETE  : "#CE93D8",
    DockingState.ABORTED   : "#EF5350",
}

# ── CW propagator (RK4) ───────────────────────────────────────────────────────
def cw_propagate(s: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    def deriv(sv, u):
        x, z, vx, vz = sv
        return np.array([vx, vz,
                         2*N*vz + u[0],
                         -2*N*vx + 3*N**2*z + u[1]])
    k1 = deriv(s, u)
    k2 = deriv(s + .5*dt*k1, u)
    k3 = deriv(s + .5*dt*k2, u)
    k4 = deriv(s +    dt*k3, u) 
    return s + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

# ── Analytical MPC-style controller ──────────────────────────────────────────
def mpc_thrust(state: np.ndarray, phase: DockingState) -> np.ndarray:
    """
    Closed-form proportional-derivative thrust law that mimics MPC
    optimal behaviour for linear CW dynamics.
    Gains are tuned per phase for approach vs precision docking.
    """
    pos = state[:2]
    vel = state[2:]
    dist = float(np.linalg.norm(pos))

    if phase == DockingState.APPROACH:
        kp, kd, max_t = 0.0008, 0.015, 0.05
    elif phase == DockingState.CAPTURE:
        kp, kd, max_t = 0.003,  0.08,  0.008
    elif phase == DockingState.LOCK:
        kp, kd, max_t = 0.000,  0.5,   0.002   # pure velocity braking
    else:
        return np.zeros(2)

    # CW coupling correction: along-track acceleration bleeds into radial
    cw_coupling = np.array([-2*N*vel[1], 2*N*vel[0]])
    thrust = -kp * pos - kd * vel - 0.1 * cw_coupling
    return np.clip(thrust, -max_t, max_t)

# ── Raft Consensus ─────────────────────────────────────────────────────────
@dataclass
class RaftModule:
    module_id: int
    role:      RaftRole = RaftRole.FOLLOWER
    term:      int      = 0
    votes:     int      = 0

class RaftCluster:
    def __init__(self, n: int = 5):
        self.modules = [RaftModule(i) for i in range(n)]
        self.modules[0].role = RaftRole.LEADER
        self.alive   = [True]*n
        self.leader  = 0
        self.log: List[str] = []

    def fail_leader(self):
        self.alive[self.leader] = False
        self.modules[self.leader].role = RaftRole.FOLLOWER
        self.log.append(f"[FAULT]  Control Module {self.leader} FAILED → Raft election triggered")

    def elect(self) -> int:
        survivors = [m for m in self.modules if self.alive[m.module_id]]
        new_term = max(m.term for m in survivors) + 1
        for m in survivors:
            m.term = new_term; m.role = RaftRole.CANDIDATE; m.votes = 0

        np.random.seed(7)
        health = {m.module_id: float(np.random.uniform(0.55, 1.0)) for m in survivors}
        self.log.append(f"[RAFT]   Term {new_term} | candidates {[m.module_id for m in survivors]}")
        self.log.append(f"[RAFT]   Health { {k:f'{v:.2f}' for k,v in health.items()} }")

        best_id = max(health, key=health.__getitem__)
        for m in survivors:
            self.modules[best_id].votes += 1

        winner = self.modules[best_id]
        winner.role = RaftRole.LEADER
        for m in survivors:
            if m.module_id != best_id: m.role = RaftRole.FOLLOWER
        self.leader = best_id
        self.log.append(f"[RAFT]   Module {best_id} elected LEADER "
                        f"({winner.votes}/{len(survivors)} votes, h={health[best_id]:.2f})")
        self.log.append(f"[RAFT]   Station OPERATIONAL. Active leader: Module {best_id}")
        return best_id

# ── Docking State Machine ─────────────────────────────────────────────────
class DockingStateMachine:
    def __init__(self, pos0, vel0, raft: RaftCluster):
        self.s    = np.array([pos0[0], pos0[1], vel0[0], vel0[1]], dtype=float)
        self.phase = DockingState.APPROACH
        self.raft  = raft
        self.failure_handled = False
        self.step_count = 0

        self.h_state  : List[np.ndarray]    = [self.s.copy()]
        self.h_thrust : List[np.ndarray]    = [np.zeros(2)]
        self.h_phase  : List[DockingState]  = [self.phase]
        self.h_dist   : List[float]         = [float(np.linalg.norm(pos0))]
        self.mission_log: List[str]         = []

    @property
    def dist(self):  return float(np.linalg.norm(self.s[:2]))
    @property
    def speed(self): return float(np.linalg.norm(self.s[2:]))

    def _log(self, msg): self.mission_log.append(f"[t={self.step_count*DT:5.0f}s]  {msg}")

    def _transition(self):
        p = self.phase
        if   p == DockingState.APPROACH  and self.dist  < CAPTURE_DIST:
            self.phase = DockingState.CAPTURE
            self._log(f"APPROACH  -> CAPTURE    dist={self.dist*100:.2f} cm")
        elif p == DockingState.CAPTURE   and self.dist  < LOCK_DIST:
            self.phase = DockingState.LOCK
            self._log(f"CAPTURE   -> LOCK       dist={self.dist*100:.2f} cm")
        elif p == DockingState.LOCK      and self.speed < HANDSHAKE_VEL:
            self.phase = DockingState.HANDSHAKE
            self._log(f"LOCK      -> HANDSHAKE  speed={self.speed*1000:.2f} mm/s")
        elif p == DockingState.HANDSHAKE:
            self.phase = DockingState.COMPLETE
            self._log("HANDSHAKE -> COMPLETE   Power+data handshake nominal ✓")

    def _check_failure(self):
        if self.step_count == FAILURE_STEP and not self.failure_handled:
            self._log("*** CONTROL MODULE FAILURE DETECTED ***")
            self.raft.fail_leader()
            new = self.raft.elect()
            self._log(f"*** Raft consensus complete — Module {new} now controls approach ***")
            self.failure_handled = True

    def step(self):
        if self.phase in (DockingState.COMPLETE, DockingState.ABORTED): return
        self._check_failure()
        u = mpc_thrust(self.s, self.phase)
        self.s = cw_propagate(self.s, u, DT)
        self.step_count += 1
        self._transition()
        self.h_state.append(self.s.copy())
        self.h_thrust.append(u.copy())
        self.h_phase.append(self.phase)
        self.h_dist.append(self.dist)

    def run(self):
        for _ in range(SIM_STEPS):
            self.step()
            if self.phase == DockingState.COMPLETE: break
        return self.phase

# ── Plotting ──────────────────────────────────────────────────────────────
def plot_results(sim: DockingStateMachine):
    states  = np.array(sim.h_state)
    thrusts = np.array(sim.h_thrust)
    dists   = np.array(sim.h_dist)
    phases  = sim.h_phase
    t       = np.arange(len(states)) * DT

    fig = plt.figure(figsize=(18, 12), facecolor="#0D1117")
    fig.suptitle("AMSS — Autonomous Modular Space Station: Docking Simulation",
                 color="white", fontsize=15, fontweight="bold", y=0.985)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)
    ax_traj = fig.add_subplot(gs[0:2, 0:2])
    ax_dist = fig.add_subplot(gs[0, 2])
    ax_spd  = fig.add_subplot(gs[1, 2])
    ax_thr  = fig.add_subplot(gs[2, 0])
    ax_log  = fig.add_subplot(gs[2, 1:])

    for ax in [ax_traj, ax_dist, ax_spd, ax_thr, ax_log]:
        ax.set_facecolor("#161B22")
        ax.tick_params(colors="#C9D1D9")
        for sp in ax.spines.values(): sp.set_edgecolor("#30363D")

    # Trajectory coloured by phase
    ax_traj.set_title("Hill-Frame Trajectory  (Along-Track vs Radial)", color="white", fontsize=11)
    ax_traj.set_xlabel("Along-Track  x  [m]", color="#8B949E")
    ax_traj.set_ylabel("Radial  z  [m]", color="#8B949E")
    seg = 0
    for i in range(1, len(phases)):
        if phases[i] != phases[i-1] or i == len(phases)-1:
            c = STATE_COLORS.get(phases[i-1], "white")
            ax_traj.plot(states[seg:i, 0], states[seg:i, 1], color=c, lw=1.8)
            seg = i
    if sim.failure_handled and FAILURE_STEP < len(states):
        fx, fz = states[FAILURE_STEP, 0], states[FAILURE_STEP, 1]
        ax_traj.axvline(fx, color="#FF6B6B", ls="--", lw=1.2, alpha=0.8)
        ax_traj.annotate("  CM Failure", xy=(fx, fz), color="#FF6B6B", fontsize=8)
    ax_traj.plot(0, 0, "o",  color="#CE93D8", ms=12, zorder=5, label="Docking Hub (target)")
    ax_traj.plot(states[0,0], states[0,1], "^", color="#4FC3F7", ms=10, zorder=5, label="Module Start")
    ax_traj.plot(states[-1,0], states[-1,1], "s", color="#81C784", ms=10, zorder=5, label="Final Position")
    patches = [mpatches.Patch(color=v, label=k.name) for k,v in STATE_COLORS.items() if k != DockingState.ABORTED]
    h, l = ax_traj.get_legend_handles_labels()
    ax_traj.legend(handles=patches+h, labels=[p.get_label() for p in patches]+l,
                   fontsize=7, loc="upper right", facecolor="#21262D", labelcolor="white")
    ax_traj.grid(True, color="#21262D", lw=0.5)

    # Distance
    ax_dist.set_title("Relative Distance", color="white", fontsize=10)
    ax_dist.set_xlabel("Time [s]", color="#8B949E", fontsize=8)
    ax_dist.set_ylabel("Distance [m]", color="#8B949E", fontsize=8)
    ax_dist.semilogy(t, np.clip(dists, 1e-6, None), color="#4FC3F7", lw=1.5)
    ax_dist.axhline(CAPTURE_DIST, color="#FFD54F", lw=1, ls="--", label="Capture")
    ax_dist.axhline(LOCK_DIST,    color="#FF8A65", lw=1, ls="--", label="Lock")
    if sim.failure_handled: ax_dist.axvline(FAILURE_STEP*DT, color="#FF6B6B", lw=1, ls="--")
    ax_dist.legend(fontsize=7, facecolor="#21262D", labelcolor="white")
    ax_dist.grid(True, color="#21262D", lw=0.5)

    # Speed
    speeds = np.sqrt(states[:,2]**2 + states[:,3]**2)
    ax_spd.set_title("Relative Speed", color="white", fontsize=10)
    ax_spd.set_xlabel("Time [s]", color="#8B949E", fontsize=8)
    ax_spd.set_ylabel("Speed [m/s]", color="#8B949E", fontsize=8)
    ax_spd.plot(t, speeds, color="#81C784", lw=1.5)
    ax_spd.axhline(HANDSHAKE_VEL, color="#CE93D8", lw=1, ls="--", label="Handshake vel")
    if sim.failure_handled: ax_spd.axvline(FAILURE_STEP*DT, color="#FF6B6B", lw=1, ls="--")
    ax_spd.legend(fontsize=7, facecolor="#21262D", labelcolor="white")
    ax_spd.grid(True, color="#21262D", lw=0.5)

    # Thrust
    ax_thr.set_title("Controller Thrust Commands  (MPC-style)", color="white", fontsize=10)
    ax_thr.set_xlabel("Time [s]", color="#8B949E", fontsize=8)
    ax_thr.set_ylabel("Thrust [m/s²]", color="#8B949E", fontsize=8)
    ax_thr.plot(t, thrusts[:,0], color="#4FC3F7", lw=1.2, label="Tx along-track")
    ax_thr.plot(t, thrusts[:,1], color="#FFD54F", lw=1.2, label="Tz radial")
    if sim.failure_handled: ax_thr.axvline(FAILURE_STEP*DT, color="#FF6B6B", lw=1, ls="--", label="CM Failure")
    ax_thr.legend(fontsize=7, facecolor="#21262D", labelcolor="white")
    ax_thr.grid(True, color="#21262D", lw=0.5)

    # Mission log
    ax_log.set_title("Mission Log  (State Machine + Raft Consensus)", color="white", fontsize=10)
    ax_log.axis("off")
    all_logs = sim.mission_log + sim.raft.log
    ax_log.text(0.01, 0.98, "\n".join(all_logs), transform=ax_log.transAxes,
                color="#C9D1D9", fontsize=7.8, va="top", fontfamily="monospace",
                bbox=dict(facecolor="#21262D", edgecolor="#30363D", boxstyle="round,pad=0.4"))

    color = "#81C784" if sim.phase == DockingState.COMPLETE else "#EF5350"
    fig.text(0.5, 0.003,
             f"Status: {sim.phase.name}  |  Duration: {sim.step_count*DT:.0f} s  |"
             f"  Final dist: {sim.dist*1000:.3f} mm  |  Final speed: {sim.speed*1000:.3f} mm/s",
             ha="center", color=color, fontsize=12, fontweight="bold")

    out = r"C:\Users\Administrator\Downloads\amss_docking_sim.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved -> {out}")

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("  AMSS — Autonomous Docking Simulation")
    print("  CW + Analytical MPC + State Machine + Raft")
    print("="*60)

    pos0 = np.array([200.0, 50.0])
    vel0 = np.array([-0.15,  0.0])

    raft = RaftCluster(n=5)
    sim  = DockingStateMachine(pos0, vel0, raft)

    print(f"\n  Initial separation : {np.linalg.norm(pos0):.1f} m")
    print(f"  CM failure injected: step {FAILURE_STEP} (t={FAILURE_STEP*DT:.0f}s)\n")

    result = sim.run()

    print("--- Mission Log ---")
    for l in sim.mission_log: print(l)
    print("\n--- Raft Log ---")
    for l in raft.log: print(l)
    print(f"\n  Result : {result.name}")
    print(f"  Steps  : {sim.step_count}  ({sim.step_count*DT:.0f} s)")
    print(f"  Dist   : {sim.dist*1000:.3f} mm")
    print(f"  Speed  : {sim.speed*1000:.3f} mm/s\n")

    plot_results(sim)

main()
