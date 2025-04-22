# =====================  COMPLETE ±10 % SENSITIVITY SCAN SCRIPT  =====================
# 1) Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 2) Engram‑Immune Model ODEs
def EIM_ODEs(t, state, params):
    x, y, z, u = state
    
    # Unpack parameters
    zeta1   = params['zeta1']
    beta    = params['beta']
    Lambda1 = params['Lambda1']
    kappa   = params['kappa']
    y0      = params['y0']
    omega   = params['omega']
    delta   = params['delta']
    lambda_val = params['lambda_val']
    Lambda2 = params['Lambda2']
    zeta2   = params['zeta2']
    xi      = params['xi']
    eta     = params['eta']
    Gamma   = params['Gamma']
    
    # Ensure non‑negative values for fractional powers
    x_nonneg = max(x, 0)
    u_nonneg = max(u, 0)
    f_u = u                               # simple f(u) = u
    
    dxdt = (1 - z) * (x_nonneg + zeta1 * (x_nonneg ** beta)) + Lambda1 * u
    dydt = x - (kappa * x * (y + y0)) / (omega + x) - delta * y * (1 + lambda_val * z) + Lambda2 * u
    dzdt = (y + y0) * (u_nonneg + zeta2 * (u_nonneg ** beta)) + xi * y - eta * z
    dudt = -(y + y0) * (u_nonneg + zeta2 * (u_nonneg ** beta)) + Gamma * f_u
    
    return [dxdt, dydt, dzdt, dudt]

# 3) Baseline parameters (edit if needed)
params_base = {
    'zeta1': 0.1,   'beta': 0.5,   'Lambda1': 0.05, 'kappa': 0.2,
    'y0': 0.1,      'omega': 1.0,  'delta': 0.3,    'lambda_val': 0.1,
    'Lambda2': 0.04,'zeta2': 0.1,  'xi': 0.2,       'eta': 0.1,
    'Gamma': 0.05
}

# 4) Simulation settings
initial_state = [0, 0, 0, 1.0]        # x, y, z, u
t_span = (0, 50)
t_eval = np.linspace(*t_span, 300)

# 5) Function to compute peak y for a parameter set
def peak_y_for_params(pdict):
    sol = solve_ivp(lambda t, s: EIM_ODEs(t, s, pdict),
                    t_span, initial_state, t_eval=t_eval)
    return sol.y[1].max()

baseline_peak = peak_y_for_params(params_base)

# 6) Parameters to scan (add/remove as needed)
param_names = ['Lambda1', 'delta', 'kappa', 'Lambda2',
               'beta', 'zeta1', 'zeta2', 'lambda_val',
               'xi', 'eta', 'Gamma']

print("\nParameter   |  -10% change |  +10% change | Effect on peak y (%)")
print("------------|--------------|--------------|----------------------")
for name in param_names:
    minus_params = params_base.copy()
    plus_params  = params_base.copy()
    minus_params[name] *= 0.9
    plus_params[name]  *= 1.1
    
    peak_minus = peak_y_for_params(minus_params)
    peak_plus  = peak_y_for_params(plus_params)
    
    pct_minus = 100 * (peak_minus - baseline_peak) / baseline_peak
    pct_plus  = 100 * (peak_plus  - baseline_peak) / baseline_peak
    
    print(f"{name:<11} | {pct_minus:>+8.2f}%   | {pct_plus:>+8.2f}%   |")

# 7) Optional: show baseline curve for reference
sol_base = solve_ivp(lambda t, s: EIM_ODEs(t, s, params_base),
                     t_span, initial_state, t_eval=t_eval)
plt.figure(figsize=(8, 4))
plt.plot(sol_base.t, sol_base.y[1], label='Baseline y(t)')
plt.xlabel('Time'); plt.ylabel('Immune Response (y)')
plt.title('Baseline Immune‑Response Curve')
plt.legend(); plt.show()
# =====================  END OF SCRIPT  =====================
