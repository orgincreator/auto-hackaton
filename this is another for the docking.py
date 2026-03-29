"""
AMSS — Autonomous Modular Space Station
Flet Docking Simulation  (Python / Matplotlib interactive)
=============================================================
Controls
--------
  SPACE       pause / resume
  R           rewind to t=0
  N           new generation
  scroll      zoom in/out on canvas
  click+drag  pan canvas
  Speed slider   1x – 10x real-time rate
  Spread/Vel/Flets sliders  adjust initial conditions (resets sim)

Run:  python flet_docking_sim_live.py
Requires: numpy, matplotlib  (pip install numpy matplotlib)
"""

from enum import Enum, auto

import numpy as np
import matplotlib
matplotlib.use('TkAgg')          # works on Windows/Mac/Linux with a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon, Circle
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import matplotlib.gridspec as gridspec
import math, time

#for computer vision model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# ── Palette ──────────────────────────────────────────────────────────────────
BG        = "#040d1a"
PANEL_BG  = "#0D1117"
CARD_BG   = "#161B22"
BORDER    = "#30363D"
TEXT      = "#C9D1D9"
TEXT_DIM  = "#8B949E"

FLET_COLS = ['#378ADD','#1D9E75','#EF9F27','#D4537E','#7F77DD',
             '#D85A30','#639922','#BA7517','#E24B4A','#5DCAA5',
             '#AFA9EC','#F09595']

ROLE_COL  = {
    'LAUNCH'   : '#378ADD',
    'APPROACH' : '#7F77DD',
    'NEGOTIATE': '#1D9E75',
    'DOCK'     : '#EF9F27',
    'LOCKED'   : '#CE93D8',
    'LOW'      : '#E24B4A',
    'LEADER'   : '#FFD700',
}

# ── World constants ───────────────────────────────────────────────────────────
W, H    = 600, 420          # canvas world units
CX, CY  = W/2, H/2
TOTAL_STEPS = 400
HUB_R   = 20
DOCK_R  = 32
LOCK_R  = 18
MU = 8000.0   # gravitational parameter (tune this)
DT = 0.09    # time step
DTS = 2.0  # real seconds per sim step at 1x speed
MUN = 3.986004418e14  # Earth's gravitational parameter (m^3/s^2)
R_ORBIT  = 6.771e6
N        = np.sqrt(MUN / R_ORBIT**3)   # ~0.001107 rad/s  (LEO ~400 km)
POLICY   = None  # placeholder for learned policy
#__for the computer vision part, we can generate synthetic images of the simulation state and train a CNN to predict the control actions (velocity adjustments) needed for docking. This would involve rendering the simulation state to an image, creating a dataset of image-control pairs, and training a model on this data.
class DockingState(Enum):
    APPROACH  = auto()
    CAPTURE   = auto()
    LOCK      = auto()
    NONE      = auto()

def role_to_docking_state(role: str) -> DockingState:
    """Map flet role string to DockingState for MPC controller."""
    return {
        'LAUNCH':    DockingState.APPROACH,
        'APPROACH':  DockingState.APPROACH,
        'NEGOTIATE': DockingState.CAPTURE,
        'DOCK':      DockingState.LOCK,
    }.get(role, DockingState.NONE)

class DockingCNN(nn.Module):
    """Simple CNN for docking control prediction"""
    
    def __init__(self):
        super(DockingCNN, self).__init__()
        
        # Convolution layers
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 4)  # output: x, z, vx, vz
        
    def forward(self, x):
        # Input: (batch, 3, 128, 128)
        x = F.relu(self.conv1(x))  # → (8, 64, 64)
        x = F.relu(self.conv2(x))  # → (16, 32, 32)
        x = F.relu(self.conv3(x))  # → (32, 16, 16)
        
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # regression output
        return x


class DockingTrainer:
    """Simple trainer for the CNN"""
    
    def __init__(self, model, lr=0.001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
    def train_step(self, images, targets):
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        outputs = self.model(images)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict(self, image):
        """Predict control from image"""
        self.model.eval()
        with torch.no_grad():
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            output = self.model(image)
        return output.cpu().numpy()


                                                             # Simple usage example

class DockingPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2),  nn.Tanh()
        )
    
    def forward(self, x):
        return self.net(x)
    
    def get_thrust(self, obs, max_t=0.05):
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32)
            return self.net(t).numpy() * max_t
        
def submain():
    # Create model
    model = DockingCNN()
    trainer = DockingTrainer(model)
    
    # Dummy training data (replace with real data)
    dummy_images = torch.randn(10, 3, 128, 128)
    dummy_targets = torch.randn(10, 4)
    
    # Train
    loss = trainer.train_step(dummy_images, dummy_targets)
    print(f"Training loss: {loss:.4f}")
    
    # Predict
    test_image = torch.randn(1, 3, 128, 128)
    prediction = trainer.predict(test_image)
    print(f"Predicted control: {prediction}")


# ── RNG ──────────────────────────────────────────────────────────────────────
def make_rng(seed):
    s = [int(seed) & 0xFFFFFFFF]
    def rand():
        s[0] = (s[0]*1664525 + 1013904223) & 0xFFFFFFFF
        return s[0] / 4294967296.0
    return rand

# ── Flet data ─────────────────────────────────────────────────────────────────
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
def make_flets(spread, vel, n, generation,dt=DTS):
    flets = []
    for i in range(n):
        ang = i * math.pi * 2 / n
        d   = spread * (0.8 + 0.4 * ((i*137) % 100 / 100))
        x   = CX + d * math.cos(ang)
        y   = CY + d * math.sin(ang)
        v = math.sqrt(MU / d)
        flets.append({
            # circular orbital velocity
            'vx': -v* math.cos(ang),   # direct inward toward hub
            'vy': -v * math.sin(ang),
            'id': i, 'x': x, 'y': y,
            'energy': 1.0, 'role': 'LAUNCH',
            'is_leader': i == 0,
            'rot': 0.0, 'target_rot': 0.0,
            'trail': [],
            'propose_timer': 0, 'vote_result': None, 'vote_timer': 0,
            'rand': make_rng(generation*1000 + i*7),
        })
    return flets
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

def clone_flets(fs):
    out = []
    for f in fs:
        c = dict(f)
        c['trail'] = list(f['trail'])
        c['rand']  = f['rand']          # share rng — same stream across clone
        out.append(c)
    return out

def step_flets(fs, t, log_fn):
    nf = clone_flets(fs)
    for f in nf:
        if f['role'] == 'LOCKED':
            continue

        dx   = CX - f['x'];  dy   = CY - f['y']
        dist = math.hypot(dx, dy)

        if f['role'] in ('LAUNCH', 'APPROACH'):
            leader = next((x for x in nf if x['is_leader'] and x['id'] != f['id']), None)
            tx, ty = CX, CY
            if leader and f['role'] == 'APPROACH':
                n_flets = len(nf)
                a = f['id'] * 2 * math.pi / n_flets
                tx = CX + DOCK_R * math.cos(a)
                ty = CY + DOCK_R * math.sin(a)
                        # CW state vector: position & velocity relative to dock target
            cw_s = np.array([f['x'] - CX,
                                f['y'] - CY,
                                f['vx'],
                                f['vy']])
            # Offset the position error to guide toward port, not hub centre
            port_offset = 0 #np.array([tx - CX, ty - CY, 0.0, 0.0])
            cw_s_error  = cw_s - port_offset   # error relative to port IN hub fram
            phase  = role_to_docking_state(f['role'])
            if POLICY is not None:
                dist  = float(np.linalg.norm(cw_s[:2]))
                obs   = np.array([cw_s[0], cw_s[1], cw_s[2], cw_s[3],
                                dist, f['energy']])
                thrust = POLICY.get_thrust(obs)
            else:
                thrust = mpc_thrust(cw_s, phase)
            cw_s   = cw_propagate(cw_s, thrust, DT)
            f['x']  = cw_s[0] + CX
            f['y']  = cw_s[1] + CY   
            f['vx'] = float(cw_s[2])    
            f['vy'] = float(cw_s[3])
            f['energy'] = max(0.0, f['energy'] - 0.0008)
            if t > 30:
                f['role'] = 'APPROACH'
            port_dist = math.hypot(f['x'] - tx, f['y'] - ty)
            if port_dist < 8:             # arrived at dock port → negotiate
                f['role'] = 'NEGOTIATE'


        if f['role'] == 'NEGOTIATE':
            f['vx'] *= 0.88;  f['vy'] *= 0.88
            f['energy'] = max(0.0, f['energy'] - 0.0004)
            f['propose_timer'] += 1
            if f['propose_timer'] > 15:
                f['vote_result']   = 'accept' if f['rand']() > 0.3 else 'reject'
                f['vote_timer']    = 20
                f['propose_timer'] = 0
            if f['vote_result'] == 'accept' and f['vote_timer'] <= 0:
                f['role'] = 'DOCK'
            if f['vote_timer'] > 0:
                f['vote_timer'] -= 1
            if dist < HUB_R + 12:
                f['role'] = 'DOCK'

        if f['role'] == 'DOCK':
                        # LOCK phase: pure velocity braking via MPC
            cw_s = np.array([f['x'] - CX,
                             f['y'] - CY,
                             f['vx'],
                             f['vy']])
            thrust = mpc_thrust(cw_s, DockingState.LOCK)
            cw_s   = cw_propagate(cw_s, thrust, DT)

            f['x']  = cw_s[0] + CX
            f['y']  = cw_s[1] + CY
            f['vx'] = float(cw_s[2])
            f['vy'] = float(cw_s[3])

            f['energy'] = max(0.0, f['energy'] - 0.0002)
            spd = math.hypot(f['vx'], f['vy'])
            if spd < 0.3 and dist < HUB_R + LOCK_R + 20:
                f['role'] = 'LOCKED'
                
                f['x'] += f['vx'];  f['y'] += f['vy']

        f['trail'].append((f['x'], f['y']))
        if len(f['trail']) > 40:
            f['trail'].pop(0)
        f['rot'] += (f['target_rot'] - f['rot']) * 0.05
        if f['role'] == 'LOCKED':
            f['target_rot'] = math.atan2(f['y'] - CY, f['x'] - CX)

    # Raft election every 120 steps
    if t % 120 == 60:
        alive = [f for f in nf if f['energy'] > 0.1]
        if alive:
            for f in nf:
                f['is_leader'] = False
            best = max(alive, key=lambda f: f['energy'])
            best['is_leader'] = True
            log_fn(f"[t={t*2}s] Raft election — Flet {best['id']} elected leader")

    return nf

def calc_score(fs):
    docked = [f for f in fs if f['role'] in ('LOCKED', 'DOCK')]
    if len(docked) < 2:
        return 0.0
    avg_e  = sum(f['energy'] for f in docked) / len(docked)
    cx     = sum(f['x'] for f in docked) / len(docked)
    cy     = sum(f['y'] for f in docked) / len(docked)
    spread = sum(math.hypot(f['x']-cx, f['y']-cy) for f in docked) / len(docked)
    stab   = max(0.0, 1 - spread/60)
    return (0.5*stab + 0.3*avg_e + 0.2*(len(docked)/len(fs))) * 100

def compute_reward(pos_err, vel, thrust, phase):
    dist  = float(np.linalg.norm(pos_err))
    speed = float(np.linalg.norm(vel))
    fuel  = float(np.dot(thrust, thrust))

    r  = -1.0  * dist**2
    r -= -0.5  * speed**2
    r -= 0.01  * fuel

    if phase == DockingState.LOCK:
        r += 50       # getting to final approach
    if dist < 8 and speed < 0.3:
        r += 1000     # successful dock
    if dist > 400:
        r -= 500      # flew off screen — abort

    return r

def train_policy(policy, episodes=2000, sim_steps=500):
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    
    for episode in range(episodes):
        # randomise starting position and velocity
        ang   = np.random.uniform(0, 2*math.pi)
        dist0 = np.random.uniform(80, 180)
        pos0  = np.array([dist0 * math.cos(ang),
                          dist0 * math.sin(ang)])
        vel0  = np.array([-pos0[1], pos0[0]]) * math.sqrt(MUN) / dist0**1.5

        # initial CW state relative to hub
        s = np.array([pos0[0], pos0[1], vel0[0], vel0[1]])

        log_probs = []
        rewards   = []

        for step in range(sim_steps):
            # build observation
            dist  = float(np.linalg.norm(s[:2]))
            obs   = np.array([s[0], s[1], s[2], s[3],
                              dist, 1.0])   # 1.0 = full energy placeholder

            # get thrust from policy (add noise for exploration)
            t_obs  = torch.tensor(obs, dtype=torch.float32)
            raw    = policy.net(t_obs)
            dist_  = torch.distributions.Normal(raw, 0.1)
            action = dist_.sample()
            thrust = action.detach().numpy() * 0.05

            log_prob = dist_.log_prob(action).sum()
            log_probs.append(log_prob)

            # world model step
            s = cw_propagate(s, thrust, DT)

            # reward
            phase = DockingState.APPROACH if dist > DOCK_R else DockingState.LOCK
            r = compute_reward(s[:2], s[2:], thrust, phase)
            rewards.append(r)

            # done?
            if np.linalg.norm(s[:2]) < 8 and np.linalg.norm(s[2:]) < 0.3:
                break
            if np.linalg.norm(s[:2]) > 400:
                break

        # REINFORCE update
        G, returns = 0, []
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = -sum(lp * G for lp, G in zip(log_probs, returns))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 100 == 0:
            print(f"Episode {episode}  last_reward={sum(rewards):.1f}")


def flet_color(f):
    if f['energy'] < 0.25:
        return ROLE_COL['LOW']
    if f['is_leader']:
        return ROLE_COL['LEADER']
    return ROLE_COL.get(f['role'], FLET_COLS[f['id'] % len(FLET_COLS)])

# ── Hex drawing helper ────────────────────────────────────────────────────────
def hex_verts(cx, cy, r, rot):
    pts = []
    for i in range(6):
        a = rot + math.pi/6 + i*math.pi/3
        pts.append((cx + r*math.cos(a), cy + r*math.sin(a)))
    return pts

# ── Main simulation class ─────────────────────────────────────────────────────
class FletSim:
    def __init__(self):
        self.generation = 1
        self.best_score = 0.0
        self.spread  = 100
        self.vel     = 8
        self.n_flets = 5
        self.speed   = 3          # steps per second
        self.playing = False
        self.log_lines = []

        self.flets    = []
        self.history  = []
        self.step_idx = 0
        self._init_sim()

        self._build_figure()
        self._connect_events()

        self._last_time  = time.time()
        self._frame_acc  = 0.0

    # ── Sim init ──────────────────────────────────────────────────────────────
    def _init_sim(self):
        self.step_idx  = 0
        self.flets     = make_flets(self.spread, self.vel, self.n_flets, self.generation)
        self.history   = [clone_flets(self.flets)]
        self.log_lines = []
        self._add_log(f"[GEN {self.generation}] {self.n_flets} Flets launched  spread={self.spread}  vel={self.vel}")

    def _add_log(self, msg):
        self.log_lines.append(msg)
        if len(self.log_lines) > 60:
            self.log_lines = self.log_lines[-60:]

    # ── Figure layout ─────────────────────────────────────────────────────────
    def _build_figure(self):
        plt.rcParams.update({
            'figure.facecolor': PANEL_BG,
            'axes.facecolor':   BG,
            'text.color':       TEXT,
            'axes.labelcolor':  TEXT_DIM,
            'xtick.color':      TEXT_DIM,
            'ytick.color':      TEXT_DIM,
            'axes.edgecolor':   BORDER,
        })

        self.fig = plt.figure(figsize=(14, 8), facecolor=PANEL_BG)
        self.fig.canvas.manager.set_window_title('AMSS — Flet Docking Simulation')

        gs = gridspec.GridSpec(
            4, 3,
            figure=self.fig,
            left=0.01, right=0.99,
            top=0.95,  bottom=0.22,
            hspace=0.35, wspace=0.25,
        )

        # Main canvas
        self.ax = self.fig.add_subplot(gs[0:3, 0:2])
        self.ax.set_xlim(0, W);  self.ax.set_ylim(0, H)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ax.set_facecolor(BG)

        # Status panel
        self.ax_status = self.fig.add_subplot(gs[0, 2])
        self.ax_status.axis('off')
        self.ax_status.set_facecolor(CARD_BG)

        # Log panel
        self.ax_log = self.fig.add_subplot(gs[1:3, 2])
        self.ax_log.axis('off')
        self.ax_log.set_facecolor(CARD_BG)

        # Title
        self.fig.text(0.5, 0.975,
            'AMSS — Flet Autonomous Docking Simulation',
            ha='center', va='top', fontsize=13, fontweight='bold',
            color='#7ec8e3', fontfamily='monospace')

        self._build_controls()

        # Camera state for zoom/pan
        self._cam_x   = 0.0
        self._cam_y   = 0.0
        self._cam_z   = 1.0
        self._pan_start = None

        # Zoom text
        self._zoom_txt = self.ax.text(
            0.99, 0.01, '100%', transform=self.ax.transAxes,
            ha='right', va='bottom', fontsize=8, color='#7ec8e388',
            fontfamily='monospace')

    def _build_controls(self):
        fig = self.fig

        # ── Row 1: buttons ────────────────────────────────────────────────────
        bw, bh = 0.07, 0.035
        by = 0.155

        axbtn = lambda x: fig.add_axes([x, by, bw, bh])

        self.btn_rewind = Button(axbtn(0.04),  '⏮ Rewind',  color=CARD_BG, hovercolor=BORDER)
        self.btn_pause  = Button(axbtn(0.125), '⏸ Pause',   color='#0d3a6e', hovercolor=BORDER)
        self.btn_play   = Button(axbtn(0.21),  '▶ Play',    color=CARD_BG, hovercolor=BORDER)
        self.btn_newgen = Button(axbtn(0.295), '+ Gen',     color=CARD_BG, hovercolor=BORDER)
        self.btn_restart= Button(axbtn(0.38),  '↺ Restart', color=CARD_BG, hovercolor=BORDER)
        self.btn_zoomin = Button(axbtn(0.465), '+ Zoom',    color=CARD_BG, hovercolor=BORDER)
        self.btn_zoomout= Button(axbtn(0.55),  '- Zoom',    color=CARD_BG, hovercolor=BORDER)
        self.btn_resetv = Button(axbtn(0.635), '⊙ View',   color=CARD_BG, hovercolor=BORDER)

        for btn in [self.btn_rewind, self.btn_pause, self.btn_play,
                    self.btn_newgen, self.btn_restart,
                    self.btn_zoomin, self.btn_zoomout, self.btn_resetv]:
            btn.label.set_color(TEXT)
            btn.label.set_fontsize(9)
            btn.label.set_fontfamily('monospace')

        self.btn_pause.label.set_color('#7ec8e3')

        # ── Timeline slider ───────────────────────────────────────────────────
        ax_tl = fig.add_axes([0.04, 0.115, 0.66, 0.02])
        ax_tl.set_facecolor(CARD_BG)
        self.sl_timeline = Slider(ax_tl, 't', 0, TOTAL_STEPS, valinit=0,
                                  color='#378ADD', track_color=BORDER)
        self.sl_timeline.label.set_color(TEXT_DIM)
        self.sl_timeline.valtext.set_color(TEXT)

        # ── Parameter sliders ─────────────────────────────────────────────────
        def make_slider(left, label, vmin, vmax, vinit):
            ax = fig.add_axes([left, 0.07, 0.14, 0.02])
            ax.set_facecolor(CARD_BG)
            sl = Slider(ax, label, vmin, vmax, valinit=vinit,
                        valstep=1, color='#378ADD', track_color=BORDER)
            sl.label.set_color(TEXT_DIM)
            sl.label.set_fontsize(8)
            sl.valtext.set_color(TEXT)
            sl.valtext.set_fontsize(8)
            return sl

        self.sl_speed  = make_slider(0.04,  'Speed',  1, 10,  3)
        self.sl_spread = make_slider(0.22,  'Spread', 40, 180, 100)
        self.sl_vel    = make_slider(0.40,  'Vel',    0,  30,  8)
        self.sl_flets  = make_slider(0.58,  'Flets',  3,  12,  5)

    # ── Event wiring ──────────────────────────────────────────────────────────
    def _connect_events(self):
        self.btn_play.on_clicked(self._on_play)
        self.btn_pause.on_clicked(self._on_pause)
        self.btn_rewind.on_clicked(self._on_rewind)
        self.btn_newgen.on_clicked(self._on_newgen)
        self.btn_restart.on_clicked(self._on_restart)
        self.btn_zoomin.on_clicked(lambda e: self._zoom(1.3))
        self.btn_zoomout.on_clicked(lambda e: self._zoom(1/1.3))
        self.btn_resetv.on_clicked(lambda e: self._reset_view())

        self.sl_timeline.on_changed(self._on_timeline)
        self.sl_speed.on_changed(lambda v: None)
        self.sl_spread.on_changed(self._on_param_change)
        self.sl_vel.on_changed(self._on_param_change)
        self.sl_flets.on_changed(self._on_param_change)

        self.fig.canvas.mpl_connect('scroll_event',      self._on_scroll)
        self.fig.canvas.mpl_connect('button_press_event',self._on_mdown)
        self.fig.canvas.mpl_connect('motion_notify_event',self._on_mmove)
        self.fig.canvas.mpl_connect('button_release_event',self._on_mup)
        self.fig.canvas.mpl_connect('key_press_event',   self._on_key)

    # ── Camera ────────────────────────────────────────────────────────────────
    def _apply_camera(self):
        cx, cy, cz = self._cam_x, self._cam_y, self._cam_z
        half_w = (W/2) / cz
        half_h = (H/2) / cz
        self.ax.set_xlim(CX - cx/cz - half_w, CX - cx/cz + half_w)
        self.ax.set_ylim(CY - cy/cz - half_h, CY - cy/cz + half_h)
        self._zoom_txt.set_text(f'{int(self._cam_z*100)}%')

    def _zoom(self, factor, mx=CX, my=CY):
        nz = max(0.25, min(10.0, self._cam_z * factor))
        self._cam_x += (mx - CX) * (self._cam_z - nz)
        self._cam_y += (my - CY) * (self._cam_z - nz)
        self._cam_z  = nz
        self._apply_camera()
        if not self.playing:
            self._draw()

    def _reset_view(self):
        self._cam_x = self._cam_y = 0.0
        self._cam_z = 1.0
        self._apply_camera()
        if not self.playing:
            self._draw()

    # ── Mouse events ──────────────────────────────────────────────────────────
    def _on_scroll(self, ev):
        if ev.inaxes != self.ax:
            return
        factor = 1.12 if ev.step > 0 else 0.89
        self._zoom(factor, ev.xdata or CX, ev.ydata or CY)

    def _on_mdown(self, ev):
        if ev.inaxes == self.ax and ev.button == 1:
            self._pan_start = (ev.xdata, ev.ydata, self._cam_x, self._cam_y)

    def _on_mmove(self, ev):
        if self._pan_start and ev.inaxes == self.ax and ev.xdata:
            sx, sy, cx0, cy0 = self._pan_start
            self._cam_x = cx0 - (ev.xdata - sx) * self._cam_z
            self._cam_y = cy0 - (ev.ydata - sy) * self._cam_z
            self._apply_camera()
            if not self.playing:
                self._draw()

    def _on_mup(self, ev):
        self._pan_start = None

    def _on_key(self, ev):
        if ev.key == ' ':
            if self.playing:
                self._on_pause(None)
            else:
                self._on_play(None)
        elif ev.key == 'r':
            self._on_rewind(None)
        elif ev.key == 'n':
            self._on_newgen(None)

    # ── Button callbacks ──────────────────────────────────────────────────────
    def _on_play(self, ev):
        self.playing = True
        self._last_time = time.time()
        self._frame_acc = 0.0
        self.btn_pause.color     = CARD_BG
        self.btn_pause.hovercolor= BORDER
        self.btn_play.color      = '#0d3a6e'
        self.btn_play.label.set_color('#7ec8e3')
        self.btn_pause.label.set_color(TEXT)

    def _on_pause(self, ev):
        self.playing = False
        self.btn_play.color      = CARD_BG
        self.btn_play.label.set_color(TEXT)
        self.btn_pause.color     = '#0d3a6e'
        self.btn_pause.label.set_color('#7ec8e3')

    def _on_rewind(self, ev):
        self._on_pause(None)
        self.step_idx = 0
        self.flets    = clone_flets(self.history[0])
        self.sl_timeline.set_val(0)
        self._draw()

    def _on_restart(self, ev):
        self._on_pause(None)
        self._init_sim()
        self.sl_timeline.set_val(0)
        self._draw()

    def _on_newgen(self, ev):
        self._on_pause(None)
        self.generation += 1
        self._init_sim()
        self.sl_timeline.set_val(0)
        self._draw()

    def _on_param_change(self, val):
        self.spread  = int(self.sl_spread.val)
        self.vel     = int(self.sl_vel.val)
        self.n_flets = int(self.sl_flets.val)
        self._on_pause(None)
        self._init_sim()
        self.sl_timeline.set_val(0)
        self._draw()

    def _on_timeline(self, val):
        target = int(val)
        if target < len(self.history):
            self.step_idx = target
            self.flets    = clone_flets(self.history[target])
        else:
            # step forward to target
            while self.step_idx < target and self.step_idx < TOTAL_STEPS:
                self._do_step()
        self._draw()

    # ── Simulation step ───────────────────────────────────────────────────────
    def _do_step(self):
        all_docked = all(f['role'] == 'LOCKED' for f in self.flets)
        if self.step_idx >= TOTAL_STEPS or all_docked:
            return False
        prev_roles = {f['id']: f['role'] for f in self.flets}
        self.flets  = step_flets(self.flets, self.step_idx + 1, self._add_log)
        self.step_idx += 1
        # store in history
        if self.step_idx >= len(self.history):
            self.history.append(clone_flets(self.flets))
        for f in self.flets:
            if f['role'] != prev_roles[f['id']]:
                self._add_log(f"[t={self.step_idx*2}s] Flet {f['id']}: "
                              f"{prev_roles[f['id']]} → {f['role']}")
        score = calc_score(self.flets)
        if score > self.best_score:
            self.best_score = score
        return True

    # ── Drawing ───────────────────────────────────────────────────────────────
    def _draw(self):
        ax = self.ax
        ax.cla()
        ax.set_facecolor(BG)
        ax.axis('off')
        self._apply_camera()

        # Stars
        rng = np.random.default_rng(42 + self.generation)
        sx  = rng.uniform(0, W, 80)
        sy  = rng.uniform(0, H, 80)
        ss  = rng.uniform(0.5, 2.0, 80)
        ax.scatter(sx, sy, s=ss, c='white', alpha=0.15, zorder=0)

        # Range rings
        for r, a in [(70, 0.08), (130, 0.05), (200, 0.03)]:
            ring = plt.Circle((CX, CY), r, color='#378ADD', fill=False,
                              alpha=a, linewidth=0.8, zorder=1)
            ax.add_patch(ring)

        # Trails
        for f in self.flets:
            if len(f['trail']) > 1:
                xs = [p[0] for p in f['trail']]
                ys = [p[1] for p in f['trail']]
                col = FLET_COLS[f['id'] % len(FLET_COLS)]
                ax.plot(xs, ys, color=col, alpha=0.3, linewidth=0.8, zorder=2)

        # Dock connections
        locked = [f for f in self.flets if f['role'] in ('LOCKED', 'DOCK')]
        for i, fa in enumerate(locked):
            for fb in locked[i+1:]:
                d = math.hypot(fa['x']-fb['x'], fa['y']-fb['y'])
                if d < 65:
                    ax.plot([fa['x'], fb['x']], [fa['y'], fb['y']],
                            color='#EF9F27', alpha=0.45, linewidth=1.5, zorder=3)

        # Hub
        hub = plt.Circle((CX, CY), 16, color='#0d2040', zorder=4)
        ax.add_patch(hub)
        hub_ring = plt.Circle((CX, CY), 16, color='#378ADD', fill=False,
                               linewidth=1.5, zorder=5)
        ax.add_patch(hub_ring)
        ax.text(CX, CY, 'HUB', ha='center', va='center',
                fontsize=7, fontweight='bold', color='#7ec8e3',
                fontfamily='monospace', zorder=6)

        # Flets
        for f in self.flets:
            col  = flet_color(f)
            verts = hex_verts(f['x'], f['y'], 14, f['rot'])

            # filled hex
            hex_patch = Polygon(verts, closed=True,
                                facecolor=col+'2a',
                                edgecolor=col,
                                linewidth=2.5 if f['role']=='LOCKED' else 1.5,
                                zorder=7)
            ax.add_patch(hex_patch)

            # connectors (dots + wedges on alternating faces)
            for i in range(6):
                a  = f['rot'] + math.pi/6 + i*math.pi/3
                mx = f['x'] + 14*math.cos(a)
                my = f['y'] + 14*math.sin(a)
                if i % 2 == 0:
                    dot = plt.Circle((mx, my), 2.5, color=col, zorder=8)
                    ax.add_patch(dot)
                else:
                    nx = f['x'] + 9*math.cos(a)
                    ny = f['y'] + 9*math.sin(a)
                    wx = [nx,
                          mx + 3.5*math.cos(a + math.pi/2),
                          mx + 3.5*math.cos(a - math.pi/2)]
                    wy = [ny,
                          my + 3.5*math.sin(a + math.pi/2),
                          my + 3.5*math.sin(a - math.pi/2)]
                    wedge = Polygon(list(zip(wx, wy)), closed=True,
                                   facecolor=col, zorder=8)
                    ax.add_patch(wedge)

            # leader crown dot
            if f['is_leader']:
                crown = plt.Circle((f['x'], f['y']-19), 3.5,
                                   color='#FFD700', zorder=9)
                ax.add_patch(crown)

            # vote badge
            if f['vote_timer'] > 0:
                vcol = '#1D9E75' if f['vote_result']=='accept' else '#E24B4A'
                label = 'YES' if f['vote_result']=='accept' else 'NO'
                ax.text(f['x'], f['y']-23, label,
                        ha='center', va='bottom', fontsize=7,
                        fontweight='bold', color=vcol,
                        fontfamily='monospace', zorder=10)

            # Flet ID
            ax.text(f['x'], f['y']+17, f"F{f['id']}",
                    ha='center', va='top', fontsize=7,
                    color=col, fontfamily='monospace', zorder=10)

        # HUD overlay (data coords so it moves with camera)
        score  = calc_score(self.flets)
        n_lock = sum(1 for f in self.flets if f['role']=='LOCKED')
        xlim   = ax.get_xlim();  ylim = ax.get_ylim()
        ax.text(xlim[0]+4, ylim[1]-6,
                f"t={self.step_idx*2}s  docked:{n_lock}/{len(self.flets)}  score:{score:.1f}",
                fontsize=8, color='#7ec8e3', fontfamily='monospace',
                va='top', zorder=11)

        self._draw_panel()
        self._draw_log()

        self.fig.canvas.draw_idle()

    def _draw_panel(self):
        ax = self.ax_status
        ax.cla(); ax.axis('off'); ax.set_facecolor(CARD_BG)
        ax.set_xlim(0,1); ax.set_ylim(0,1)

        ax.text(0.05, 1.10, 'FLET STATUS', fontsize=8, color=TEXT_DIM,
                va='top', fontfamily='monospace', transform=ax.transAxes)

        n = len(self.flets)
        row_h = min(0.13, 0.85/max(n,1))
        for i, f in enumerate(self.flets):
            y  = 0.87 - i*row_h
            col = flet_color(f)
            # dot
            ax.add_patch(plt.Circle((0.07, y+0.03), 0.025,
                                    color=col, transform=ax.transAxes,
                                    clip_on=False))
            star = ' ★' if f['is_leader'] else ''
            ax.text(0.14, y+0.02, f"F{f['id']}{star} {f['role']}",
                    fontsize=7.5, color=col,
                    fontfamily='monospace', transform=ax.transAxes)
            # energy bar bg
            bar_x, bar_y, bar_w, bar_h = 0.14, y-0.025, 0.82, 0.022
            ax.add_patch(mpatches.FancyBboxPatch(
                (bar_x, bar_y), bar_w, bar_h,
                boxstyle='round,pad=0', facecolor=BORDER,
                transform=ax.transAxes, clip_on=False))
            # energy bar fill
            ecol = '#1D9E75' if f['energy']>0.6 else '#EF9F27' if f['energy']>0.3 else '#E24B4A'
            fill = max(0.001, f['energy']) * bar_w
            ax.add_patch(mpatches.FancyBboxPatch(
                (bar_x, bar_y), fill, bar_h,
                boxstyle='round,pad=0', facecolor=ecol,
                transform=ax.transAxes, clip_on=False))

        # metrics
        score  = calc_score(self.flets)
        n_lock = sum(1 for f in self.flets if f['role']=='LOCKED')
        by = 0.04
        ax.text(0.05, by,
                f"Gen {self.generation}   Score {score:.1f}   Best {self.best_score:.1f}\n"
                f"Docked: {n_lock}/{len(self.flets)}",
                fontsize=7.5, color=TEXT_DIM,
                fontfamily='monospace', transform=ax.transAxes, va='bottom')

    def _draw_log(self):
        ax = self.ax_log
        ax.cla(); ax.axis('off'); ax.set_facecolor(CARD_BG)
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.text(0.05, 0.97, 'MISSION LOG', fontsize=8, color=TEXT_DIM,
                va='top', fontfamily='monospace', transform=ax.transAxes)
        lines = self.log_lines[-22:]
        for i, line in enumerate(reversed(lines)):
            y = 0.88 - i*0.042
            if y < 0.01:
                break
            ax.text(0.05, y, line, fontsize=6.8, color=TEXT_DIM,
                    fontfamily='monospace', transform=ax.transAxes, va='top')

    # ── Animation tick ────────────────────────────────────────────────────────
    def _tick(self, frame):
        if not self.playing:
            return
        now = time.time()
        dt  = now - self._last_time
        self._last_time = now
        self._frame_acc += dt

        speed = max(1, int(self.sl_speed.val))
        ms_per_step = 1.0 / speed

        advanced = False
        while self._frame_acc >= ms_per_step and self.step_idx < TOTAL_STEPS:
            self._frame_acc -= ms_per_step
            if not self._do_step():
                break
            advanced = True

        if advanced:
            # update timeline without triggering callback
            self.sl_timeline.eventson = False
            self.sl_timeline.set_val(self.step_idx)
            self.sl_timeline.eventson = True
            self._draw()

        if self.step_idx >= TOTAL_STEPS:
            self._on_pause(None)
            score = calc_score(self.flets)
            self._add_log(f"[GEN {self.generation}] Complete — score:{score:.1f}")
            self._draw()

    # ── Run ───────────────────────────────────────────────────────────────────
    def run(self):
        self._draw()
        self.anim = FuncAnimation(
            self.fig, self._tick,
            interval=33,         # ~30 fps tick
            cache_frame_data=False
        )
        plt.show()


if __name__ == '__main__':
    print("=" * 55)
    print("  AMSS — Flet Docking Simulation")
    print("  Controls: SPACE=pause/play  R=rewind  N=new gen")
    print("  Scroll=zoom  Drag=pan  Sliders=adjust params")
    print("=" * 55)
    sim = FletSim()
    # policy = DockingPolicy()
    # train_policy(policy, episodes=2000)
    # torch.save(policy.state_dict(), 'docking_policy.pth')
    # POLICY = policy

    # uncomment to load a previously trained policy
    # POLICY = DockingPolicy()
    # POLICY.load_state_dict(torch.load('docking_policy.pth'))
    sim.run()
