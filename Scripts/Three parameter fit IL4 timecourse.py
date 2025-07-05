# =====================  THREE-PARAMETER FIT SCRIPT  =====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# 1) Load IL-4 time-course data
data   = pd.read_csv(r'C:\Users\celal\Desktop\Model-Versions\data\il4_hdm.csv')
t_data = data['time'].values
y_data = data['il4'].values

# 2) Define the Engram-Immune ODE system
def EIM_ODEs(t, state, p):
    x, y, z, u = state
    ζ1, β       = p['zeta1'],    p['beta']
    Λ1, κ       = p['Lambda1'],  p['kappa']
    y0, ω       = p['y0'],       p['omega']
    δ, λv       = p['delta'],    p['lambda_val']
    Λ2, ζ2      = p['Lambda2'],  p['zeta2']
    ξ, η, Γ     = p['xi'],       p['eta'], p['Gamma']
    
    x_ = max(x, 0)
    u_ = max(u, 0)
    
    dxdt = (1 - z)*(x_ + ζ1*x_**β) + Λ1*u
    dydt = x - (κ*x*(y+y0))/(ω + x) - δ*y*(1 + λv*z) + Λ2*u
    dzdt = (y+y0)*(u_ + ζ2*u_**β) + ξ*y - η*z
    dudt = -(y+y0)*(u_ + ζ2*u_**β) + Γ*u
    
    return [dxdt, dydt, dzdt, dudt]

# 3) Baseline parameters
params_base = {
    'zeta1': 0.1,   'beta': 0.5,     'Lambda1': 0.05, 'kappa': 0.2,
    'y0': 0.1,      'omega': 1.0,    'delta': 0.3,    'lambda_val': 0.1,
    'Lambda2': 0.04,'zeta2': 0.1,    'xi': 0.2,       'eta': 0.1,
    'Gamma': 0.05
}

# 4) Simulator at the IL-4 measurement times
initial_state = [0, 0, 0, 1.0]
def simulate(p):
    sol = solve_ivp(lambda t, s: EIM_ODEs(t, s, p),
                    (t_data.min(), t_data.max()),
                    initial_state,
                    t_eval=t_data)
    return sol.y[1]  # return the y variable

# 5) Residuals function fitting δ, Λ₁, and Λ₂
def residuals(params):
    δ_val, Λ1_val, Λ2_val = params
    p = params_base.copy()
    p['delta']   = δ_val
    p['Lambda1'] = Λ1_val
    p['Lambda2'] = Λ2_val
    return simulate(p) - y_data

# 6) Perform the three-parameter fit
initial_guess = [
    params_base['delta'],
    params_base['Lambda1'],
    params_base['Lambda2']
]
bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
res = least_squares(residuals, initial_guess, bounds=bounds)
δ_fit, Λ1_fit, Λ2_fit = res.x
print(f"Fitted δ = {δ_fit:.4f},  Λ₁ = {Λ1_fit:.4f},  Λ₂ = {Λ2_fit:.4f}")

# 7) Generate fitted curve and compute fit metrics
p_fit = {**params_base, 'delta': δ_fit, 'Lambda1': Λ1_fit, 'Lambda2': Λ2_fit}
y_fit = simulate(p_fit)

rmse = np.sqrt(np.mean((y_fit - y_data)**2))
ss_res = np.sum((y_data - y_fit)**2)
ss_tot = np.sum((y_data - np.mean(y_data))**2)
r2     = 1 - ss_res/ss_tot
print(f"RMSE = {rmse:.1f} pg/ml")
print(f"R²   = {r2:.3f}")

# 8) Plot observed data vs fitted model
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(t_data, y_data, color='black', label='Observed IL-4')
ax.plot(t_data, y_fit, color='C1', label='Model fit')
ax.set_xlabel('Time (h post-challenge)')
ax.set_ylabel('IL-4 (pg/ml)')
ax.set_title('Fit of Engram-Immune Model to IL-4 Data')
textstr = (f'δ={δ_fit:.4f}, Λ₁={Λ1_fit:.4f}, Λ₂={Λ2_fit:.4f}\n'
           f'RMSE={rmse:.1f}, R²={r2:.2f}')
ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
        va='top', bbox=dict(facecolor='white', alpha=0.7))
ax.legend()
plt.tight_layout()
plt.show()

# 9) Plot residuals vs time
resid = y_fit - y_data
fig2, ax2 = plt.subplots(figsize=(6,4))
ax2.plot(t_data, resid, 'o-', label='Residuals')
ax2.axhline(0, linestyle='--', color='k')
ax2.set_xlabel('Time (h post-challenge)')
ax2.set_ylabel('Residual (model − data) pg/ml')
ax2.set_title('Residuals of Fit vs Time')
ax2.legend()
plt.tight_layout()
plt.show()
# =====================  END SCRIPT  =====================
