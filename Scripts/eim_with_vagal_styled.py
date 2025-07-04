# eim_with_vagal_styled.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

# ─── 1) Baseline Engram–Immune parameters ───────────────────────────────────────
params = {
    'zeta1':      0.1,
    'beta':       0.5,
    'Lambda1':    0.05,
    'kappa':      0.2,
    'y0':         0.1,
    'omega':      1.0,
    'delta':      0.3,
    'lambda_val': 0.1,
    'Lambda2':    0.04,
    'zeta2':      0.1,
    'xi':         0.2,
    'eta':        0.1,
    'Gamma':      0.05
}

# ─── 2) Build v(t) template (days→hours) & normalized suppression ────────────
days       = np.array([0, 42, 56, 84])
suppression = np.array([0.00, 1.00, 0.052, 0.791])
hours      = days * 24
v_interp   = PchipInterpolator(hours, suppression, extrapolate=False)

# ─── 3) Define ODEs with non-negativity & proportional vagal suppression ─────
def eim_vagal(t, state, p, mu):
    x, y, z, u = state
    x_ = max(x, 0)
    y_ = max(y, 0)
    u_ = max(u, 0)
    v_t = np.nan_to_num(v_interp(t))  # outside range → 0

    dxdt = (1 - z)*(x_ + p['zeta1']*x_**p['beta']) + p['Lambda1']*u_
    dydt = (
        x_
        - (p['kappa']*x_*(y_ + p['y0']))/(p['omega'] + x_)
        - p['delta']*y_*(1 + p['lambda_val']*z)
        + p['Lambda2']*u_
        - mu * v_t * y_                # proportional suppression
    )
    dzdt = (y_ + p['y0'])*(u_ + p['zeta2']*u_**p['beta']) + p['xi']*y_ - p['eta']*z
    dudt = -(y_ + p['y0'])*(u_ + p['zeta2']*u_**p['beta']) + p['Gamma']*u_
    return [dxdt, dydt, dzdt, dudt]

# ─── 4) Simulation setup ───────────────────────────────────────────────────────
t_start, t_end = 0, 84*24                         # 0–84 days in hours
t_eval         = np.linspace(t_start, t_end, 600) # time grid for output
init_state     = [0, 0, 0, 1.0]                   # x=y=z=0, u=1

# ─── 5) Run baseline vs VNS (mu=0.9) ─────────────────────────────────────────
sol_base = solve_ivp(eim_vagal, (t_start, t_end), init_state,
                     args=(params, 0.0), t_eval=t_eval)
sol_vns  = solve_ivp(eim_vagal, (t_start, t_end), init_state,
                     args=(params, 0.9), t_eval=t_eval)

# ─── 6) Plot x(t), y(t), z(t) with polished styling ────────────────────────────
colors = {'baseline': '#4C72B0', 'vns': '#DD8452'}
labels = {'baseline': 'Baseline (μ=0)', 'vns': 'VNS (μ=0.9)'}

fig, axes = plt.subplots(3,1, figsize=(8,12), sharex=True)

# cytokine y(t)
axes[0].plot(sol_base.t, sol_base.y[1], color=colors['baseline'], lw=2, label=labels['baseline'])
axes[0].plot(sol_vns.t,  sol_vns.y[1],  color=colors['vns'],      lw=2, linestyle='--', label=labels['vns'])
axes[0].set_ylabel('y(t) – Cytokine', fontsize=12)
axes[0].legend(loc='upper left', bbox_to_anchor=(1.01,1))
axes[0].grid(True, linestyle=':', color='#CCCCCC')

# engram x(t)
axes[1].plot(sol_base.t, sol_base.y[0], color=colors['baseline'], lw=2)
axes[1].plot(sol_vns.t,  sol_vns.y[0],  color=colors['vns'],      lw=2, linestyle='--')
axes[1].set_ylabel('x(t) – Engram', fontsize=12)
axes[1].grid(True, linestyle=':', color='#CCCCCC')

# feedback z(t)
axes[2].plot(sol_base.t, sol_base.y[2], color=colors['baseline'], lw=2)
axes[2].plot(sol_vns.t,  sol_vns.y[2],  color=colors['vns'],      lw=2, linestyle='--')
axes[2].set_ylabel('z(t) – Feedback', fontsize=12)
axes[2].set_xlabel('Time (hours)', fontsize=12)
axes[2].grid(True, linestyle=':', color='#CCCCCC')

# soften spines
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

# ─── 7) Compute AUC & % suppression ───────────────────────────────────────────
from numpy import trapz
auc_base = trapz(sol_base.y[1], sol_base.t)
auc_vns  = trapz(sol_vns.y[1],      sol_vns.t)
supp_pct = 100*(auc_base - auc_vns)/auc_base

print(f"AUC baseline = {auc_base:.1f}")
print(f"AUC VNS      = {auc_vns:.1f}")
print(f"Suppression  = {supp_pct:.1f} %")

