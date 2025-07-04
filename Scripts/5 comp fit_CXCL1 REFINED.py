# fit_cxcl1_refined.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# 1) Load & normalise CXCL1 data
data   = pd.read_csv(r'C:\Users\celal\Desktop\Model-Versions\data\CXCL1_hdm.csv')
t_data = data['time'].values
y_raw  = data['CXCL1'].values
y_adj  = np.maximum(y_raw - y_raw[0], 0)
y_norm = y_adj / np.max(y_adj)

# 2) Core Engram–Immune ODEs
def EIM_core(t, state, p):
    x, y, z, u = state
    ζ1, β     = p['zeta1'],    p['beta']
    Λ1, κ     = p['Lambda1'],  p['kappa']
    y0, ω     = p['y0'],       p['omega']
    δ, λv     = p['delta'],    p['lambda_val']
    Λ2, ζ2    = p['Lambda2'],  p['zeta2']
    ξ, η, Γ   = p['xi'],       p['eta'], p['Gamma']

    x_, u_ = max(x, 0), max(u, 0)
    dxdt = (1 - z)*(x_ + ζ1*x_**β) + Λ1*u
    dydt = x - (κ*x*(y + y0))/(ω + x) - δ*y*(1 + λv*z) + Λ2*u
    dzdt = (y + y0)*(u_ + ζ2*u_**β) + ξ*y - η*z
    dudt = -(y + y0)*(u_ + ζ2*u_**β) + Γ*u
    return [dxdt, dydt, dzdt, dudt]

# 3) Baseline parameters
params_base = {
    'zeta1':      0.1,
    'beta':       0.5,
    'Lambda1':    0.05,  # will be overridden to 0 in CXCL1 fit
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

# 4) Simulator for y(t)
def simulate_core(p, u0):
    sol = solve_ivp(
        fun=lambda t, s: EIM_core(t, s, p),
        t_span=(t_data.min(), t_data.max()),
        y0=[0, 0, 0, u0],
        t_eval=t_data
    )
    return sol.y[1]  # immune response y(t)

# 5) Refined residuals (drop neural drive Λ1)
def residuals_cxcl1(x):
    δ, Λ2, κ_val, u0 = x
    p = params_base.copy()
    p.update(delta=δ, Lambda1=0, Lambda2=Λ2, kappa=κ_val)
    y_pred = simulate_core(p, u0)
    return (y_pred/np.max(y_pred)) - y_norm

# 6) Fit δ, Λ2, κ, u₀
init_guess = [params_base['delta'],
              params_base['Lambda2'],
              params_base['kappa'],
              1.0]
bounds = ([0,0,0,0], [np.inf]*4)
res = least_squares(residuals_cxcl1, init_guess, bounds=bounds)
δ_fit, Λ2_fit, κ_fit, u0_fit = res.x
print(f"Refined CXCL1 → δ={δ_fit:.3f}, Λ₂={Λ2_fit:.3f}, κ={κ_fit:.3f}, u₀={u0_fit:.2f}")

# 7) Simulate & normalise fitted model
p_fit = params_base.copy()
p_fit.update(delta=δ_fit, Lambda1=0, Lambda2=Λ2_fit, kappa=κ_fit)
y_fit = simulate_core(p_fit, u0_fit)
y_fit_norm = y_fit / np.max(y_fit)

# 8) Plot data vs. refined model
plt.figure(figsize=(6,4))
plt.plot(t_data, y_norm,    'o-',  label='CXCL1 Data')
plt.plot(t_data, y_fit_norm,'s--', label='Model (Λ₁=0)')
plt.xlabel('Time (h post‐challenge)')
plt.ylabel('Normalised CXCL1')
plt.title('Refined Shape‐Fit: CXCL1 (Innate‐Only)')
plt.legend()
plt.tight_layout()
plt.show()

# 8) Plot residuals to check for systematic deviations
resid = y_fit_norm - y_norm
plt.figure(figsize=(6,4))
plt.plot(t_data, resid, 'o-')
plt.axhline(0, linestyle='--', color='gray')
plt.xlabel('Time (h)')
plt.ylabel('Residual (model − data)')
plt.title('Residuals of Normalized Shape Fit')
plt.tight_layout(); plt.show()

# 9) Residuals and metrics
resid = y_fit_norm - y_norm
rmse = np.sqrt(np.mean(resid**2))
ss_res = np.sum(resid**2)
ss_tot = np.sum((y_norm - np.mean(y_norm))**2)
r2 = 1 - ss_res/ss_tot
print(f"Normalized RMSE = {rmse:.3f}")
print(f"Normalized R²   = {r2:.3f}")
