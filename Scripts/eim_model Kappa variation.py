# =====================  COMPLETE κ‑SWEEP SCRIPT  =====================
# 1) Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 2) Engram‑Immune ODEs
def EIM_ODEs(t, state, params):
    x, y, z, u = state
    
    # Unpack parameters
    zeta1  = params['zeta1']
    beta   = params['beta']
    Lambda1 = params['Lambda1']
    kappa  = params['kappa']           # <-- parameter we will vary
    y0     = params['y0']
    omega  = params['omega']
    delta  = params['delta']
    lambda_val = params['lambda_val']
    Lambda2 = params['Lambda2']
    zeta2  = params['zeta2']
    xi     = params['xi']
    eta    = params['eta']
    Gamma  = params['Gamma']
    
    # Ensure non‑negative before fractional powers
    x_nonneg = max(x, 0)
    u_nonneg = max(u, 0)
    f_u = u                            # simple choice f(u) = u
    
    dxdt = (1 - z) * (x_nonneg + zeta1 * (x_nonneg ** beta)) + Lambda1 * u
    dydt = x - (kappa * x * (y + y0)) / (omega + x) - delta * y * (1 + lambda_val * z) + Lambda2 * u
    dzdt = (y + y0) * (u_nonneg + zeta2 * (u_nonneg ** beta)) + xi * y - eta * z
    dudt = -(y + y0) * (u_nonneg + zeta2 * (u_nonneg ** beta)) + Gamma * f_u

    return [dxdt, dydt, dzdt, dudt]

# 3) Baseline parameters
params_base = {
    'zeta1': 0.1,  'beta': 0.5,  'Lambda1': 0.05, 'kappa': 0.2,
    'y0': 0.1,     'omega': 1.0, 'delta': 0.3,    'lambda_val': 0.1,
    'Lambda2': 0.04,'zeta2': 0.1,'xi': 0.2,       'eta': 0.1,
    'Gamma': 0.05
}

# 4) Simulation settings
initial_state = [0, 0, 0, 1.0]      # x, y, z, u
t_span = (0, 50)
t_eval = np.linspace(*t_span, 300)

# 5) κ values to test
kappa_values = [0.05, 0.1, 0.2, 0.3]

# 6) Plotting loop and peak collection
plt.figure(figsize=(10, 6))
peak_list = []

for k in kappa_values:
    params = params_base.copy()
    params['kappa'] = k
    sol = solve_ivp(lambda t, s: EIM_ODEs(t, s, params),
                    t_span, initial_state, t_eval=t_eval)
    
    plt.plot(sol.t, sol.y[1], label=f'κ = {k}')
    peak_list.append(sol.y[1].max())

# 7) Print numeric summary
print("κ value  |  Peak immune response")
for k, peak in zip(kappa_values, peak_list):
    print(f"{k:<8} -> {peak:.3f}")

# 8) Finalise plot
plt.xlabel('Time')
plt.ylabel('Immune Response (y)')
plt.title('Sensitivity Analysis: Variation in kappa')
plt.legend()
plt.show()
# =====================  END OF SCRIPT  =====================
