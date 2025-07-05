import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# 1) Load and normalize IL-4 data
data       = pd.read_csv(r'C:\Users\celal\Desktop\Model-Versions\data\il4_hdm.csv')
t_data     = data['time'].values       # e.g. [0, 2, 4, 8, 24]
y_data     = data['il4'].values        # corresponding pg/ml values
y_norm     = y_data / np.max(y_data)   # peak → 1

# 2) Core Engram-Immune ODEs
def eim_core(t, state, p):
    x, y, z, u = state
    ζ1, β     = p['zeta1'],    p['beta']
    Λ1, κ     = p['Lambda1'],  p['kappa']
    y0, ω     = p['y0'],       p['omega']
    δ, λv     = p['delta'],    p['lambda_val']
    Λ2, ζ2    = p['Lambda2'],  p['zeta2']
    ξ, η, Γ   = p['xi'],       p['eta'], p['Gamma']
    x_, u_    = max(x,0), max(u,0)
    dx = (1 - z)*(x_ + ζ1*x_**β) + Λ1*u
    dy = x - (κ*x*(y+y0))/(ω + x) - δ*y*(1 + λv*z) + Λ2*u
    dz = (y+y0)*(u_ + ζ2*u_**β) + ξ*y - η*z
    du = -(y+y0)*(u_ + ζ2*u_**β) + Γ*u
    return [dx, dy, dz, du]

# 3) Baseline parameters (add 'tau')
params = {
    'zeta1':0.1,   'beta':0.5,   'Lambda1':0.05, 'kappa':0.2,
    'y0':0.1,      'omega':1.0,  'delta':0.3,    'lambda_val':0.1,
    'Lambda2':0.04,'zeta2':0.1,  'xi':0.2,       'eta':0.1,
    'Gamma':0.05,  'tau':1.0      # initial time-scale guess
}

# 4) Simulator using scaled time and variable u0
def simulate(p, u0):
    tau   = p['tau']
    t_mod = t_data / tau
    sol = solve_ivp(lambda tt, ss: eim_core(tt, ss, p),
                    (t_mod.min(), t_mod.max()),
                    [0, 0, 0, u0],
                    t_eval=t_mod)
    return sol.y[1]

# 5) Residuals including baseline offset b
def residuals_shape(x):
    δ_val, Λ1_val, Λ2_val, u0_val, tau_val, b_val = x
    p = params.copy()
    p.update(delta=δ_val,
             Lambda1=Λ1_val,
             Lambda2=Λ2_val,
             tau=tau_val)
    y_pred = simulate(p, u0_val)
    y_pred_norm = y_pred / np.max(y_pred)
    return (y_pred_norm + b_val) - y_norm

# 6) Fit δ, Λ1, Λ2, u0, tau, and baseline offset b
init_guess = [
    params['delta'],
    params['Lambda1'],
    params['Lambda2'],
    1.0,            # u0
    1.0,            # tau
    y_norm[0]       # b, start at observed normalized baseline
]
lower = [0, 0, 0, 0, 0.1, 0]
upper = [np.inf, np.inf, np.inf, np.inf, 10, 1]
res = least_squares(residuals_shape, init_guess, bounds=(lower, upper))
δ_fit, Λ1_fit, Λ2_fit, u0_fit, tau_fit, b_fit = res.x
print(f"Fitted δ={δ_fit:.4f}, Λ₁={Λ1_fit:.4f}, Λ₂={Λ2_fit:.4f}, u₀={u0_fit:.2f}, τ={tau_fit:.2f}, b={b_fit:.3f}")

# 7) Compute normalized fit & metrics
p_fit      = {**params, 'delta':δ_fit, 'Lambda1':Λ1_fit, 'Lambda2':Λ2_fit, 'tau':tau_fit}
y_fit      = simulate(p_fit, u0_fit)
y_fit_norm = y_fit / np.max(y_fit) + b_fit

rmse_norm  = np.sqrt(np.mean((y_fit_norm - y_norm)**2))
ss_res     = np.sum((y_norm - y_fit_norm)**2)
ss_tot     = np.sum((y_norm - np.mean(y_norm))**2)
r2_norm    = 1 - ss_res/ss_tot
print(f"Normalized RMSE = {rmse_norm:.3f}")
print(f"Normalized R²   = {r2_norm:.3f}")

# 8) Plot normalized data vs model + baseline
plt.figure(figsize=(6,4))
plt.plot(t_data, y_norm,     'o-', label='Data (norm)')
plt.plot(t_data, y_fit_norm, 's--', label='Model + baseline')
plt.xlabel('Time (h post-challenge)')
plt.ylabel('Normalized IL-4')
plt.title('Shape Fit with Time Scaling & Baseline Offset')
plt.text(0.05, 0.95,
         f'RMSE={rmse_norm:.3f}, R²={r2_norm:.3f}',
         transform=plt.gca().transAxes,
         va='top',
         bbox=dict(facecolor='white', alpha=0.7))
plt.legend(); plt.tight_layout(); plt.show()

# 9) Residuals
resid = y_fit_norm - y_norm
plt.figure(figsize=(6,4))
plt.plot(t_data, resid, 'o-')
plt.axhline(0, ls='--', c='k')
plt.xlabel('Time (h post-challenge)')
plt.ylabel('Residual (norm)')
plt.title('Residuals with Baseline Offset')
plt.tight_layout(); plt.show()
