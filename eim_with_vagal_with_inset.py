# eim_with_vagal_with_inset.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# 1) Baseline parameters
params = {
    'zeta1': 0.1,  'beta': 0.5,
    'Lambda1': 0.05, 'kappa': 0.2,
    'y0': 0.1,    'omega': 1.0,
    'delta': 0.3, 'lambda_val': 0.1,
    'Lambda2': 0.04, 'zeta2': 0.1,
    'xi': 0.2,    'eta': 0.1,
    'Gamma': 0.05
}

# 2) Build v(t)
days = np.array([0, 42, 56, 84])
suppression = np.array([0.0, 1.0, 0.052, 0.791])
hours = days * 24
v_interp = PchipInterpolator(hours, suppression, extrapolate=False)

# 3) ODEs with non-negativity & proportional suppression
def eim_vagal(t, state, p, mu):
    x, y, z, u = state
    x_ = max(x, 0); y_ = max(y, 0); u_ = max(u, 0)
    v_t = np.nan_to_num(v_interp(t))
    dxdt = (1 - z)*(x_ + p['zeta1']*x_**p['beta']) + p['Lambda1']*u_
    dydt = (
        x_
        - (p['kappa']*x_*(y_ + p['y0']))/(p['omega'] + x_)
        - p['delta']*y_*(1 + p['lambda_val']*z)
        + p['Lambda2']*u_
        - mu * v_t * y_
    )
    dzdt = (y_ + p['y0'])*(u_ + p['zeta2']*u_**p['beta']) + p['xi']*y_ - p['eta']*z
    dudt = -(y_ + p['y0'])*(u_ + p['zeta2']*u_**p['beta']) + p['Gamma']*u_
    return [dxdt, dydt, dzdt, dudt]

# 4) Simulation
t_start, t_end = 0, 84*24
t_eval = np.linspace(t_start, t_end, 600)
init = [0, 0, 0, 1.0]
sol_base = solve_ivp(eim_vagal, (t_start, t_end), init, args=(params, 0.0), t_eval=t_eval)
sol_vns  = solve_ivp(eim_vagal, (t_start, t_end), init, args=(params, 0.9), t_eval=t_eval)

# … after sol_base, sol_vns are computed …

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

fig, axes = plt.subplots(3,1,figsize=(8,12), sharex=True)
colors = {'baseline':'#4C72B0','vns':'#DD8452'}
labels = {'baseline':'Baseline','vns':'VNS μ=0.9'}

for ax, var in zip(axes, [('y',1),('x',0),('z',2)]):
    name, idx = var
    # main 0–10 day view
    ax.plot(sol_base.t/24, sol_base.y[idx], color=colors['baseline'], lw=2, label=labels['baseline'])
    ax.plot(sol_vns.t/24,  sol_vns.y[idx],  color=colors['vns'],     lw=2, linestyle='--', label=labels['vns'])
    ax.set_xlim(0,10)                              # zoom to first 10 days
    ax.set_ylabel(f'{name}(t)', fontsize=12)
    ax.grid(True, linestyle=':', color='#CCCCCC')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01,1))
    # inset showing full 84 days
    ins = inset_axes(ax, width="30%", height="20%", loc='upper right')
    ins.plot(sol_base.t/24, sol_base.y[idx], color=colors['baseline'], lw=1)
    ins.plot(sol_vns.t/24,  sol_vns.y[idx],  color=colors['vns'],     lw=1, linestyle='--')
    ins.set_xlim(0,84); ins.set_xticks([0,42,84]); ins.set_xticklabels(['0','42','84'])
    ins.set_ylim(ax.get_ylim())
    ins.spines['top'].set_visible(False); ins.spines['right'].set_visible(False)

axes[-1].set_xlabel('Time (days)', fontsize=12)
plt.tight_layout()
plt.show()



# 6) AUC suppression
from numpy import trapz
auc_base = trapz(sol_base.y[1], sol_base.t)
auc_vns  = trapz(sol_vns.y[1], sol_vns.t)
print(f"Suppression = {100*(auc_base-auc_vns)/auc_base:.1f} %")
