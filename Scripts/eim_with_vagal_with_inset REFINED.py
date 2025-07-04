import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# 1) Base parameters
params = {
    'zeta1': 0.1,  'beta': 0.5,
    'Lambda1': 0.05, 'kappa': 0.2,
    'y0': 0.1,    'omega': 1.0,
    'delta': 0.3, 'lambda_val': 0.1,
    'Lambda2': 0.04, 'zeta2': 0.1,
    'xi': 0.2,    'eta': 0.1,
    'Gamma': 0.05
}

# 2) v(t) template
days        = np.array([0, 42, 56, 84])
suppression = np.array([0.0, 1.0, 0.052, 0.791])
hours       = days * 24
v_interp    = PchipInterpolator(hours, suppression, extrapolate=False)

# 3) ODE system
def eim_vagal(t, s, p, mu):
    x, y, z, u = s
    x_, y_, u_ = max(x,0), max(y,0), max(u,0)
    v_t = np.nan_to_num(v_interp(t))
    dxdt = (1 - z)*(x_ + p['zeta1']*x_**p['beta']) + p['Lambda1']*u_
    dydt = (x_
            - (p['kappa']*x_*(y_+p['y0']))/(p['omega']+x_)
            - p['delta']*y_*(1+p['lambda_val']*z)
            + p['Lambda2']*u_
            - mu*v_t*y_)
    dzdt = (y_+p['y0'])*(u_+p['zeta2']*u_**p['beta']) + p['xi']*y_ - p['eta']*z
    dudt = -(y_+p['y0'])*(u_+p['zeta2']*u_**p['beta']) + p['Gamma']*u_
    return [dxdt, dydt, dzdt, dudt]

# 4) Simulations
t0, t1 = 0, 84*24
t_eval = np.linspace(t0, t1, 600)
y0     = [0,0,0,1.0]

sol_b = solve_ivp(eim_vagal, (t0,t1), y0, args=(params, 0.0), t_eval=t_eval)
sol_v = solve_ivp(eim_vagal, (t0,t1), y0, args=(params, 0.9), t_eval=t_eval)

# 5) Plot with zoomed main + inset
colors = {'base':'#4C72B0','vns':'#DD8452'}
labels = {'base':'Baseline','vns':'VNS μ=0.9'}
fig, axes = plt.subplots(3,1,figsize=(8,12), sharex=True)

for ax, var in zip(axes, [('y',1),('x',0),('z',2)]):
    name, idx = var
    # main 0–10 day view
    ax.plot(sol_b.t/24, sol_b.y[idx], color=colors['base'], lw=2, label=labels['base'])
    ax.plot(sol_v.t/24, sol_v.y[idx], color=colors['vns'], lw=2, linestyle='--', label=labels['vns'])
    ax.set_xlim(0,4)
    ax.set_ylabel(f"{name}(t)", fontsize=12)
    ax.grid(True, linestyle=':', color='#CCCCCC')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01,1))
    # inset for full 84 days
    ins = inset_axes(ax, width="30%", height="20%", loc='upper right')
    ins.plot(sol_b.t/24, sol_b.y[idx], color=colors['base'], lw=1)
    ins.plot(sol_v.t/24, sol_v.y[idx], color=colors['vns'], lw=1, linestyle='--')
    ins.set_xlim(0,84)
    ins.set_xticks([0,42,84])
    ins.set_xticklabels(['0','42','84'])
    ins.set_ylim(ax.get_ylim())
    ins.spines['top'].set_visible(False)
    ins.spines['right'].set_visible(False)



axes[-1].set_xlabel('Time (days)', fontsize=12)

# single tight_layout call
plt.show()

# … after you have sol_base and sol_vns …

print("\nTime (days) |  y_base  |  y_VNS")
print("---------------------------------")
for ti, yb, yv in zip(sol_base.t, sol_base.y[1], sol_vns.y[1]):
    # only print every 24 h
    if abs((ti/24) - round(ti/24)) < 1e-6:
        print(f"{ti/24:8.0f}       | {yb:7.4f} | {yv:7.4f}")


# 5) AUC & suppression
from numpy import trapezoid
auc_b = trapezoid(sol_b.y[1], sol_b.t)
auc_v = trapezoid(sol_v.y[1], sol_v.t)
print(f"Suppression = {100*(auc_b-auc_v)/auc_b:.1f}%")
