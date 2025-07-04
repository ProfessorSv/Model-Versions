# overlay_chemokines_refined.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# 1) Core EIM ODEs
def EIM_core(t, state, p):
    x, y, z, u = state
    ζ1, β     = p['zeta1'],    p['beta']
    Λ1, κ     = p['Lambda1'],  p['kappa']
    y0, ω     = p['y0'],       p['omega']
    δ, λv     = p['delta'],    p['lambda_val']
    Λ2, ζ2    = p['Lambda2'],  p['zeta2']
    ξ, η, Γ   = p['xi'],       p['eta'], p['Gamma']
    x_, u_    = max(x,0), max(u,0)
    dxdt = (1 - z)*(x_ + ζ1*x_**β) + Λ1*u
    dydt = x - (κ*x*(y + y0))/(ω + x) - δ*y*(1 + λv*z) + Λ2*u
    dzdt = (y + y0)*(u_ + ζ2*u_**β) + ξ*y - η*z
    dudt = -(y + y0)*(u_ + ζ2*u_**β) + Γ*u
    return [dxdt, dydt, dzdt, dudt]

# 2) Baseline parameters
params_base = {
    'zeta1':      0.1,
    'beta':       0.5,
    'Lambda1':    0.05,  # will be zeroed for CXCL1
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

# 3) Simulator
def simulate_core(p, u0, t_eval):
    sol = solve_ivp(lambda t,s: EIM_core(t,s,p),
                    (t_eval.min(), t_eval.max()),
                    [0,0,0,u0],
                    t_eval=t_eval)
    return sol.y[1]

# 4) Residuals: full vs CXCL1‐only
def residuals_full(x, y_norm, t_data):
    δ, Λ1, Λ2, κ_val, u0 = x
    p = params_base.copy()
    p.update(delta=δ, Lambda1=Λ1, Lambda2=Λ2, kappa=κ_val)
    y_pred = simulate_core(p, u0, t_data)
    return (y_pred/np.max(y_pred)) - y_norm

def residuals_cxcl1(x, y_norm, t_data):
    δ, Λ2, κ_val, u0 = x
    p = params_base.copy()
    p.update(delta=δ, Lambda1=0, Lambda2=Λ2, kappa=κ_val)
    y_pred = simulate_core(p, u0, t_data)
    return (y_pred/np.max(y_pred)) - y_norm

# 5) Metrics
def compute_metrics(y_pred, y_obs):
    rmse = np.sqrt(np.mean((y_pred-y_obs)**2))
    ss_res = np.sum((y_obs-y_pred)**2)
    ss_tot = np.sum((y_obs-np.mean(y_obs))**2)
    r2 = 1 - ss_res/ss_tot
    return rmse, r2

# 6) Load & normalise helper
def load_norm(path, col):
    df = pd.read_csv(path)
    t  = df['time'].values
    y  = df[col].values
    y_adj = np.maximum(y - y[0], 0)
    return t, y_adj/np.max(y_adj)

# Chemokine files & column names
chemokines = {
    'CXCL1':    (r'C:\Users\celal\Desktop\Model-Versions\data\CXCL1_hdm.csv',   'CXCL1'),
    'Eotaxin1': (r'C:\Users\celal\Desktop\Model-Versions\data\Eotaxin-1_hdm.csv','Eotaxin-1'),
    'TARC':     (r'C:\Users\celal\Desktop\Model-Versions\data\TARC_hdm.csv',    'TARC')
}

# Prepare summary
summary = {'Param':['δ','Λ₁','Λ₂','κ','u₀','RMSE','R²']}

plt.figure(figsize=(6,4))
colors = {'CXCL1':'tab:blue','Eotaxin1':'tab:green','TARC':'tab:orange'}

for name,(path,col) in chemokines.items():
    t_data, y_norm = load_norm(path, col)
    if name=='CXCL1':
        init = [0.3, 0.04, 0.2, 1.0]            # δ, Λ2, κ, u0
        bounds = ([0,0,0,0],[np.inf]*4)
        res = least_squares(residuals_cxcl1, init, bounds=bounds,
                            args=(y_norm, t_data))
        δ, Λ2, κ, u0 = res.x
        Λ1 = 0.0
    else:
        init = [0.3, 0.05, 0.04, 0.2, 1.0]       # δ, Λ1, Λ2, κ, u0
        bounds = ([0,0,0,0,0],[np.inf]*5)
        res = least_squares(residuals_full, init, bounds=bounds,
                            args=(y_norm, t_data))
        δ, Λ1, Λ2, κ, u0 = res.x

    # Simulate & normalise
    p = params_base.copy()
    p.update(delta=δ, Lambda1=Λ1, Lambda2=Λ2, kappa=κ)
    y_fit = simulate_core(p, u0, t_data)
    y_fit_norm = y_fit/np.max(y_fit)

    # Metrics
    rmse, r2 = compute_metrics(y_fit_norm, y_norm)

    # Plot
    plt.plot(t_data, y_norm,    'o-',  color=colors[name], label=f'{name} data')
    plt.plot(t_data, y_fit_norm,'s--', color=colors[name], label=f'{name} model')

    summary[name] = [δ, Λ1, Λ2, κ, u0, rmse, r2]

# Finalise
plt.xlabel('Time (h post-challenge)')
plt.ylabel('Normalised concentration')
plt.title('Overlay: CXCL1 (refined), Eotaxin-1 & TARC')
plt.legend()
plt.tight_layout()
plt.show()

# Print summary
df = pd.DataFrame(summary)
print('\n=== Chemokine Parameter & Metrics ===')
print(df.round(3))
