# =====================  SHAPE-FIT CORE EIM PARAMETERS WITH κ FIT  =====================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# 1) Load IL-4 data and normalise to peak = 1
data         = pd.read_csv(r'C:\Users\celal\Desktop\Model-Versions\il4_hdm.csv')
t_data       = data['time'].values
y_data       = data['il4'].values
y_data_norm  = y_data / np.max(y_data)

# 2) Core EIM ODEs (x, y, z, u)
def EIM_core(t, state, p):
    x, y, z, u = state
    ζ1, β     = p['zeta1'],    p['beta']
    Λ1, κ     = p['Lambda1'],  p['kappa']
    y0, ω     = p['y0'],       p['omega']
    δ, λv     = p['delta'],    p['lambda_val']
    Λ2, ζ2    = p['Lambda2'],  p['zeta2']
    ξ, η, Γ   = p['xi'],       p['eta'], p['Gamma']

    x_, u_    = max(x, 0), max(u, 0)

    dxdt = (1 - z)*(x_ + ζ1*x_**β) + Λ1 * u
    dydt = x - (κ * x * (y + y0)) / (ω + x) - δ * y * (1 + λv * z) + Λ2 * u
    dzdt = (y + y0)*(u_ + ζ2 * u_**β) + ξ * y - η * z
    dudt = -(y + y0)*(u_ + ζ2 * u_**β) + Γ * u

    return [dxdt, dydt, dzdt, dudt]

# 3) Baseline core parameters
params_base = {
    'zeta1': 0.1,   'beta': 0.5,    'Lambda1': 0.05, 'kappa': 0.2,
    'y0':    0.1,   'omega': 1.0,   'delta':   0.3,  'lambda_val': 0.1,
    'Lambda2':0.04,'zeta2': 0.1,    'xi':      0.2,  'eta':        0.1,
    'Gamma': 0.05
}

# 4) Simulator for y(t) at data time points, with variable u0
def simulate_core(p, u0):
    sol = solve_ivp(
        fun=lambda t, s: EIM_core(t, s, p),
        t_span=(t_data.min(), t_data.max()),
        y0=[0, 0, 0, u0],
        t_eval=t_data
    )
    return sol.y[1]  # immune response y(t)

# 5) Residuals including κ
def residuals_shape(x):
    δ, Λ1, Λ2, κ_val, u0 = x
    p = params_base.copy()
    p.update(delta=δ, Lambda1=Λ1, Lambda2=Λ2, kappa=κ_val)
    y_pred = simulate_core(p, u0)
    return (y_pred / np.max(y_pred)) - y_data_norm

# 6) Fit δ, Λ1, Λ2, κ, and u₀
init = [
    params_base['delta'],
    params_base['Lambda1'],
    params_base['Lambda2'],
    params_base['kappa'],  # include kappa
    1.0                     # initial u0
]
bounds = ([0,0,0,0,0], [np.inf]*5)
res = least_squares(residuals_shape, init, bounds=bounds)
δ_fit, Λ1_fit, Λ2_fit, κ_fit, u0_fit = res.x

print(f"Shape fit → δ={δ_fit:.3f}, Λ₁={Λ1_fit:.3f}, Λ₂={Λ2_fit:.3f}, κ={κ_fit:.3f}, u₀={u0_fit:.2f}")

# 7) Simulate fitted core and plot normalised curves
y_fit_norm = simulate_core(
    {**params_base, 'delta': δ_fit, 'Lambda1': Λ1_fit, 'Lambda2': Λ2_fit, 'kappa': κ_fit},
    u0_fit
)
y_fit_norm /= np.max(y_fit_norm)

plt.figure(figsize=(6,4))
plt.plot(t_data, y_data_norm, 'o-', label='Data (normalised)')
plt.plot(t_data, y_fit_norm, 's--', label='Model (normalised)')
plt.xlabel('Time (h)')
plt.ylabel('Normalised IL-4')
plt.title('Core Shape Fit: Normalised IL-4 vs Normalised y(t)')
plt.legend()
plt.tight_layout()
plt.show()

# 8) Residuals of shape fit
res_shape_vals = y_fit_norm - y_data_norm
plt.figure(figsize=(6,4))
plt.plot(t_data, res_shape_vals, 'o-')
plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Time (h)')
plt.ylabel('Residual (model − data)')
plt.title('Residuals of Core Shape Fit')
plt.tight_layout()
plt.show()

# 9) Compute normalized fit metrics
rmse_norm   = np.sqrt(np.mean((y_fit_norm - y_data_norm)**2))
ss_res_norm = np.sum((y_data_norm - y_fit_norm)**2)
ss_tot_norm = np.sum((y_data_norm - np.mean(y_data_norm))**2)
r2_norm     = 1 - ss_res_norm/ss_tot_norm

print(f"Normalized RMSE = {rmse_norm:.3f}")
print(f"Normalized R²   = {r2_norm:.3f}")
