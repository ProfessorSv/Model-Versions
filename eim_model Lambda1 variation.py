import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ----------  Engram‑Immune ODE function (unchanged) ----------
def EIM_ODEs(t, state, params):
    x, y, z, u = state
    zeta1  = params['zeta1']
    beta   = params['beta']
    Lambda1 = params['Lambda1']        # <— we will vary this
    kappa  = params['kappa']
    y0     = params['y0']
    omega  = params['omega']
    delta  = params['delta']
    lambda_val = params['lambda_val']
    Lambda2 = params['Lambda2']
    zeta2  = params['zeta2']
    xi     = params['xi']
    eta    = params['eta']
    Gamma  = params['Gamma']

    x_nonneg = max(x, 0)
    u_nonneg = max(u, 0)
    f_u = u                                # simple choice

    dxdt = (1 - z) * (x_nonneg + zeta1 * (x_nonneg ** beta)) + Lambda1 * u
    dydt = x - (kappa * x * (y + y0))/(omega + x) - delta * y * (1 + lambda_val * z) + Lambda2 * u
    dzdt = (y + y0) * (u_nonneg + zeta2 * (u_nonneg ** beta)) + xi * y - eta * z
    dudt = -(y + y0) * (u_nonneg + zeta2 * (u_nonneg ** beta)) + Gamma * f_u

    return [dxdt, dydt, dzdt, dudt]

# ----------  Base parameters (keep as before) ----------
params_base = {
    'zeta1': 0.1,  'beta': 0.5,  'Lambda1': 0.05,  'kappa': 0.2,
    'y0': 0.1,     'omega': 1.0, 'delta': 0.3,      'lambda_val': 0.1,
    'Lambda2': 0.04,'zeta2': 0.1,'xi': 0.2,         'eta': 0.1,
    'Gamma': 0.05
}

initial_state = [0, 0, 0, 1.0]      # x, y, z, u
t_span = (0, 50)
t_eval = np.linspace(*t_span, 300)

# ----------  Values of Lambda1 to test ----------
lambda1_values = [0.01, 0.03, 0.05, 0.07]

plt.figure(figsize=(10, 6))
for lam in lambda1_values:
    params = params_base.copy()
    params['Lambda1'] = lam               # vary Lambda1
    sol = solve_ivp(lambda t, s: EIM_ODEs(t, s, params),
                    t_span, initial_state, t_eval=t_eval)
    plt.plot(sol.t, sol.y[1], label=f'Lambda1 = {lam}')

# ---- NEW: record and print peak values ----
peak_list = []
for lam in lambda1_values:
    params = params_base.copy()
    params['Lambda1'] = lam
    sol = solve_ivp(lambda t, s: EIM_ODEs(t, s, params),
                    t_span, initial_state, t_eval=t_eval)
    peak_list.append(sol.y[1].max())

print("Λ1 value  |  Peak immune response")
for lam, peak in zip(lambda1_values, peak_list):
    print(f"{lam:<9} -> {peak:.3f}")

# ---- labels and show ----
plt.xlabel('Time')
plt.ylabel('Immune Response (y)')
plt.title('Sensitivity Analysis: Variation in Lambda1')
plt.legend()
plt.show()
