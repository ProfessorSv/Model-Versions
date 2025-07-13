# SHAPE-FIT CORE EIM PARAMETERS WITH κ FIT  

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp       # ODE solver
from scipy.optimize import least_squares    # non-linear fitting

# 1) Load IL-13 data and normalise to peak = 1
data         = pd.read_csv(r'C:\Users\celal\Desktop\Model-Versions\data\il13_hdm.csv')
t_data       = data['time'].values        # [0, 2, 4, 8, 24] hours post-challenge
y_data       = data['il13'].values         # corresponding IL-13 concentrations (pg/ml)
y_data_norm  = y_data / np.max(y_data)    # scale so the highest point = 1

# 2) Define the core Engram-Immune ODE system (x,y,z,u)
def EIM_core(t, state, p):
    x, y, z, u = state
    # unpack parameters
    ζ1, β     = p['zeta1'],    p['beta']         # engram nonlinearity
    Λ1, κ     = p['Lambda1'],  p['kappa']        # strength of neural drive and feedback
    y0, ω     = p['y0'],       p['omega']        # basal immune tone & receptor saturation
    δ, λv     = p['delta'],    p['lambda_val']   # immune decay & feedback modulation
    Λ2, ζ2    = p['Lambda2'],  p['zeta2']        # allergen→immune drive & lesion nonlinearity
    ξ, η, Γ   = p['xi'],       p['eta'], p['Gamma']  # feedback dynamics & lesion clearance

    # ensure we don’t take weird powers of negative numbers
    x_, u_    = max(x, 0), max(u, 0)

    # ODEs for each compartment:
    dxdt = (1 - z)*(x_ + ζ1*x_**β) + Λ1*u
    dydt = x - (κ*x*(y + y0))/(ω + x) - δ*y*(1 + λv*z) + Λ2*u
    dzdt = (y + y0)*(u_ + ζ2*u_**β) + ξ*y - η*z
    dudt = -(y + y0)*(u_ + ζ2*u_**β) + Γ*u

    return [dxdt, dydt, dzdt, dudt]

# 3) Set the starting (“baseline”) parameter values
params_base = {
    'zeta1': 0.1,   # nonlinearity in engram activation
    'beta': 0.5,    # fractional-power cut-off exponent
    'Lambda1': 0.05,# neural (engr​am) drive to immune response
    'kappa': 0.2,   # feedback strength of immune onto itself
    'y0': 0.1,      # basal immune activity
    'omega': 1.0,   # receptor saturation constant
    'delta': 0.3,   # immune decay rate
    'lambda_val': 0.1,# modulation of decay by feedback
    'Lambda2': 0.04,# allergen (u) → immune drive
    'zeta2': 0.1,   # lesion nonlinearity
    'xi': 0.2,      # feedback activation via immune
    'eta': 0.1,     # feedback decay
    'Gamma': 0.05   # lesion clearance (self-replication term)
}

# 4) Simulator: solve the ODEs over your time points, given an initial lesion u0
def simulate_core(p, u0):
    sol = solve_ivp(
        fun=lambda t, s: EIM_core(t, s, p),
        t_span=(t_data.min(), t_data.max()),
        y0=[0, 0, 0, u0],  # start with x=y=z=0, lesion=u0
        t_eval=t_data      # output at exactly your measured hours
    )
    return sol.y[1]      # return only y(t) (the immune response)

# 5) Define residuals for fitting δ, Λ₁, Λ₂, κ, and u₀
def residuals_shape(x):
    δ, Λ1, Λ2, κ_val, u0 = x
    p = params_base.copy()
    # update the five fitted params
    p.update(delta=δ, Lambda1=Λ1, Lambda2=Λ2, kappa=κ_val)
    y_pred = simulate_core(p, u0)
    # normalize predicted peak → 1 and compare to normalized data
    return (y_pred / np.max(y_pred)) - y_data_norm

# 6) Run the least-squares optimizer
init = [
    params_base['delta'],     # initial guess for δ
    params_base['Lambda1'],   # Λ₁
    params_base['Lambda2'],   # Λ₂
    params_base['kappa'],     # κ
    1.0                       # u₀
]
bounds = ([0,0,0,0,0], [np.inf]*5)
res = least_squares(residuals_shape, init, bounds=bounds)
δ_fit, Λ1_fit, Λ2_fit, κ_fit, u0_fit = res.x
print(f"Shape fit → δ={δ_fit:.3f}, Λ₁={Λ1_fit:.3f}, Λ₂={Λ2_fit:.3f}, κ={κ_fit:.3f}, u₀={u0_fit:.2f}")

# 7) Simulate with the fitted parameters and plot normalized curves
y_fit_norm = simulate_core(
    {**params_base, 'delta': δ_fit, 'Lambda1': Λ1_fit, 'Lambda2': Λ2_fit, 'kappa': κ_fit},
    u0_fit
) / np.max(simulate_core(
    {**params_base, 'delta': δ_fit, 'Lambda1': Λ1_fit, 'Lambda2': Λ2_fit, 'kappa': κ_fit},
    u0_fit
))

plt.figure(figsize=(6,4))
plt.plot(t_data, y_data_norm, 'o-', label='IL-13 Data (normalised)')
plt.plot(t_data, y_fit_norm, 's--', label='EIM Fit (normalised)')
plt.xlabel('Time (h post-challenge)')
plt.ylabel('Normalised IL-13')
plt.title('Normalized Shape Fit of EIM (IL-13 vs y(t))')
plt.legend(); plt.tight_layout(); plt.show()

# 8) Plot residuals to check for systematic deviations
resid = y_fit_norm - y_data_norm
plt.figure(figsize=(6,4))
plt.plot(t_data, resid, 'o-')
plt.axhline(0, linestyle='--', color='gray')
plt.xlabel('Time (h)')
plt.ylabel('Residual (model − data)')
plt.title('IL-13 Residuals of Normalized Shape Fit')
plt.tight_layout(); plt.show()

# 9) Compute and print fit metrics
rmse_norm   = np.sqrt(np.mean(resid**2))
ss_res_norm = np.sum(resid**2)
ss_tot_norm = np.sum((y_data_norm - np.mean(y_data_norm))**2)
r2_norm     = 1 - ss_res_norm/ss_tot_norm

print(f"Normalized RMSE = {rmse_norm:.3f}")
print(f"Normalized R²   = {r2_norm:.3f}")
