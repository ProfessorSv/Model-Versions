import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# 1) Load the IL-4 time-course data
data   = pd.read_csv(r'C:\Users\celal\Desktop\Model-Versions\il4_hdm.csv')
t_data = data['time'].values    # e.g. [2,4,8,24]
y_data = data['il4'].values     # e.g. [50,90,55,20]

# 2) Extended ODE system: (x, y, z, u, I4)
def EIM_plus_IL4(t, state, p):
    x, y, z, u, I4 = state

    # Unpack core EIM params
    zeta1 = p['zeta1']; beta   = p['beta']
    Lambda1 = p['Lambda1']; kappa = p['kappa']
    y0    = p['y0'];    omega  = p['omega']
    delta = p['delta']; lambda_v = p['lambda_val']
    Lambda2= p['Lambda2']; zeta2=p['zeta2']
    xi    = p['xi'];   eta    = p['eta']; Gamma = p['Gamma']
    # Unpack IL-4 params
    alpha4 = p['alpha4']; gamma4 = p['gamma4']

    # Ensure we never take a fractional power of negative
    x_ = max(x,0); u_ = max(u,0)

    # Core EIM equations
    dxdt = (1 - z)*(x_ + zeta1*x_**beta) + Lambda1*u
    dydt = x - (kappa*x*(y+y0))/(omega + x) - delta*y*(1 + lambda_v*z) + Lambda2*u
    dzdt = (y+y0)*(u_ + zeta2*u_**beta) + xi*y - eta*z
    dudt = -(y+y0)*(u_ + zeta2*u_**beta) + Gamma*u

    # New IL-4 compartment
    # dI4/dt = alpha4 * y  -  gamma4 * I4
    dI4dt = alpha4 * y - gamma4 * I4

    return [dxdt, dydt, dzdt, dudt, dI4dt]

# 3) Baseline parameters (keep core EIM as before; add dummy IL-4 params)
params = {
    'zeta1':0.1, 'beta':0.5, 'Lambda1':0.05, 'kappa':0.2,
    'y0':0.1, 'omega':1.0, 'delta':0.3, 'lambda_val':0.1,
    'Lambda2':0.04,'zeta2':0.1,'xi':0.2, 'eta':0.1,
    'Gamma':0.05,
    # IL-4 production & clearance (to be fitted)
    'alpha4': 1.0,   # initial guess
    'gamma4': 0.1    # initial guess
}

# 4) Simulator at the data time points (with I4)
initial_state = [0, 0, 0, 1.0, 0.0]  # x, y, z, u, I4(0)=0
def simulate_IL4(p):
    sol = solve_ivp(
        fun=lambda t,s: EIM_plus_IL4(t, s, p),
        t_span=(t_data.min(), t_data.max()),
        y0=initial_state,
        t_eval=t_data
    )
    return sol.y[4]  # return I4 over time

# 5) Residuals function fitting alpha4 & gamma4
def residuals_IL4(x):
    a4, g4 = x
    p = params.copy()
    p['alpha4'] = a4
    p['gamma4'] = g4
    return simulate_IL4(p) - y_data

# 6) Run the fit
initial_guess = [params['alpha4'], params['gamma4']]
bounds = ([0, 0], [np.inf, np.inf])
res = least_squares(residuals_IL4, initial_guess, bounds=bounds)
alpha4_fit, gamma4_fit = res.x
print(f"\nFitted α₄ = {alpha4_fit:.4f}, γ₄ = {gamma4_fit:.4f}")

# 7) Generate fitted IL-4 curve & compute metrics
params['alpha4'] = alpha4_fit
params['gamma4'] = gamma4_fit
y_fit = simulate_IL4(params)

rmse = np.sqrt(np.mean((y_fit - y_data)**2))
ss_res = np.sum((y_data - y_fit)**2)
ss_tot = np.sum((y_data - np.mean(y_data))**2)
r2 = 1 - ss_res/ss_tot
print(f"RMSE = {rmse:.1f} pg/ml")
print(f"R²   = {r2:.3f}\n")

# 8) Plot data vs fitted IL-4
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(t_data, y_data, color='black', label='Observed IL-4')
ax.plot(t_data, y_fit, color='C1', label='Model IL-4')
ax.set_xlabel('Time (h)')
ax.set_ylabel('IL-4 (pg/ml)')
ax.set_title('EIM + IL-4 Compartment Fit')
txt = (f'α₄={alpha4_fit:.3f}, γ₄={gamma4_fit:.3f}\n'
       f'RMSE={rmse:.1f}, R²={r2:.2f}')
ax.text(0.05,0.95, txt, transform=ax.transAxes,
        va='top', bbox=dict(facecolor='white', alpha=0.7))
ax.legend(); plt.tight_layout(); plt.show()

# 9) Residuals vs time
resid = y_fit - y_data
fig2, ax2 = plt.subplots(figsize=(6,4))
ax2.plot(t_data, resid, 'o-', label='Residuals')
ax2.axhline(0, linestyle='--', color='k')
ax2.set_xlabel('Time (h)')
ax2.set_ylabel('Residual (model−data)')
ax2.set_title('Residuals of IL-4 Fit')
ax2.legend(); plt.tight_layout(); plt.show()
