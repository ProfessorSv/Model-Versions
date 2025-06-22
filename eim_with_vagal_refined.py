# eim_with_vagal_refined.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

# 1) Baseline EIM parameters
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

# 2) Build v(t) template (days→hours) & normalised suppression
days = np.array([0, 42, 56, 84])
suppression = np.array([0.00, 1.00, 0.052, 0.791])
hours = days * 24
v_interp = PchipInterpolator(hours, suppression, extrapolate=False)

# 3) EIM ODEs with vagal suppression term
def eim_vagal(t, state, p, mu):
    x, y, z, u = state
    x_ = max(x, 0); u_ = max(u, 0)
    v_t = np.nan_to_num(v_interp(t))  # safe outside-data-range → 0
    dxdt = (1 - z)*(x_ + p['zeta1']*x_**p['beta']) + p['Lambda1']*u
    dydt = (
        x
        - (p['kappa']*x*(y + p['y0']))/(p['omega'] + x)
        - p['delta']*y*(1 + p['lambda_val']*z)
        + p['Lambda2']*u
        - mu * v_t
    )
    dzdt = (y + p['y0'])*(u_ + p['zeta2']*u_**p['beta']) + p['xi']*y - p['eta']*z
    dudt = -(y + p['y0'])*(u_ + p['zeta2']*u_**p['beta']) + p['Gamma']*u
    return [dxdt, dydt, dzdt, dudt]

# 4) Simulation parameters
t_start, t_end = 0, 84*24
t_eval = np.linspace(t_start, t_end, 500)
init_state = [0, 0, 0, 1.0]

# 5) Run baseline (μ=0) and VNS (μ=0.9)
sol_base = solve_ivp(eim_vagal, (t_start, t_end), init_state,
                     args=(params, 0.0), t_eval=t_eval)
sol_vns  = solve_ivp(eim_vagal, (t_start, t_end), init_state,
                     args=(params, 0.9), t_eval=t_eval)

# 6) Plot y(t), x(t) & z(t)
fig, axes = plt.subplots(3,1, figsize=(8,12), sharex=True)

axes[0].plot(sol_base.t, sol_base.y[1], label='Baseline')
axes[0].plot(sol_vns.t,  sol_vns.y[1],  label='VNS (μ=0.9)')
axes[0].set_ylabel('y(t) – Cytokine')
axes[0].legend()

axes[1].plot(sol_base.t, sol_base.y[0], label='Baseline')
axes[1].plot(sol_vns.t,  sol_vns.y[0],  label='VNS (μ=0.9)')
axes[1].set_ylabel('x(t) – Engram signal')

axes[2].plot(sol_base.t, sol_base.y[2], label='Baseline')
axes[2].plot(sol_vns.t,  sol_vns.y[2],  label='VNS (μ=0.9)')
axes[2].set_ylabel('z(t) – Feedback signal')
axes[2].set_xlabel('Time (hours)')
axes[2].legend()

plt.tight_layout()
plt.show()

# 7) Compute AUC & suppression %
from numpy import trapz
auc_base = trapz(sol_base.y[1], sol_base.t)
auc_vns  = trapz(sol_vns.y[1],      sol_vns.t)
supp_pct = 100 * (auc_base - auc_vns) / auc_base
print(f"AUC baseline = {auc_base:.1f}")
print(f"AUC VNS      = {auc_vns:.1f}")
print(f"Suppression  = {supp_pct:.1f} %")
