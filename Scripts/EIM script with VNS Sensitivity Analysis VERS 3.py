# =============================================================================
#
#  Full Integrated Script (Feedback Sensitivity Analysis):
#  - Part 1 calibrates the model to IL-4 data.
#  - Part 2 runs a sensitivity analysis on the feedback strength (xi)
#    while VNS is active, to test the "runaway engram" hypothesis.
#
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator
from numpy import trapz

# =============================================================================
#  PART 1: CALIBRATE MODEL TO IL-4 DATA
# =============================================================================

print("--- Starting Part 1: Calibrating Model to IL-4 Data ---")

# 1) Load IL-4 data and normalise
try:
    data = pd.read_csv(r'C:\Users\celal\Desktop\Model-Versions\data\il4_hdm.csv')
except FileNotFoundError:
    print("\nError: 'il4_hdm.csv' not found. Using dummy data.")
    t_data = np.array([0, 2, 4, 8, 24])
    y_data = np.array([3.5, 50, 90, 55, 20])
else:
    t_data = data['time'].values
    y_data = data['il4'].values

y_data_norm = y_data / np.max(y_data)

# 2) Core EIM ODEs
def EIM_core(t, state, p):
    x, y, z, u = state
    x_ = max(x, 0); u_ = max(u, 0)
    dxdt = (1 - z) * (x_ + p['zeta1'] * x_**p['beta']) + p['Lambda1'] * u
    dydt = x - (p['kappa'] * x * (y + p['y0'])) / (p['omega'] + x) - p['delta'] * y * (1 + p['lambda_val'] * z) + p['Lambda2'] * u
    dzdt = (y + p['y0']) * (u_ + p['zeta2'] * u_**p['beta']) + p['xi'] * y - p['eta'] * z
    dudt = -(y + p['y0']) * (u_ + p['zeta2'] * u_**p['beta']) + p['Gamma'] * u
    return [dxdt, dydt, dzdt, dudt]

# 3) Baseline parameters
params_base = {
    'zeta1': 0.1, 'beta': 0.5, 'Lambda1': 0.05, 'kappa': 0.2,
    'y0': 0.1, 'omega': 1.0, 'delta': 0.3, 'lambda_val': 0.1,
    'Lambda2': 0.04, 'zeta2': 0.1, 'xi': 0.2, 'eta': 0.1,
    'Gamma': 0.05
}

# 4) Simulator & Residuals for fitting
def simulate_for_fitting(p, u0):
    sol = solve_ivp(EIM_core, (t_data.min(), t_data.max()), [0, 0, 0, u0], args=(p,), t_eval=t_data)
    return sol.y[1]

def residuals_shape(x):
    δ_val, Λ1_val, Λ2_val, u0_val = x
    p = params_base.copy()
    p.update({'delta': δ_val, 'Lambda1': Λ1_val, 'Lambda2': Λ2_val})
    y_pred = simulate_for_fitting(p, u0_val)
    if np.max(y_pred) < 1e-9: return np.inf
    y_pred_norm = y_pred / np.max(y_pred)
    return y_pred_norm - y_data_norm

# 5) Fit parameters
initial_guess = [params_base['delta'], params_base['Lambda1'], params_base['Lambda2'], 1.0]
bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])
res_shape = least_squares(residuals_shape, initial_guess, bounds=bounds)
δ_fit, Λ1_fit, Λ2_fit, u0_fit = res_shape.x
params_fitted = params_base.copy()
params_fitted.update({'delta': δ_fit, 'Lambda1': Λ1_fit, 'Lambda2': Λ2_fit})
print(f"Fitted Parameters → δ = {δ_fit:.4f}, Λ₁ = {Λ1_fit:.4f}, Λ₂ = {Λ2_fit:.4f}, u₀ = {u0_fit:.2f}")

# =============================================================================
#  PART 2: SENSITIVITY ANALYSIS OF FEEDBACK STRENGTH (xi)
# =============================================================================

print("\n--- Starting Part 2: Sensitivity Analysis for Feedback Strength (ξ) ---")

# 1) Build the v(t) temporal template using sustained suppression shape
days = np.array([0, 42, 84])
suppression_norm = np.array([0.0, 1.0, 1.0])
hours = days * 24
hours_rescaled = np.interp(hours, [hours.min(), hours.max()], [2, 100])
v_interp = PchipInterpolator(hours_rescaled, suppression_norm, extrapolate=False)

# 2) Define the VNS-enabled EIM ODE system
def EIM_vagal(t, state, p, mu):
    x, y, z, u = state
    x_ = max(x, 0); y_ = max(y, 0); u_ = max(u, 0); z_ = z
    v_t = np.nan_to_num(v_interp(t))
    
    dxdt = (1 - z_) * (x_ + p['zeta1'] * x_**p['beta']) + p['Lambda1'] * u_
    dydt = x_ - (p['kappa']*x_*(y_ + p['y0']))/(p['omega']+x_) - p['delta']*y_*(1+p['lambda_val']*z_) + p['Lambda2']*u_ - mu*v_t*y_
    dzdt = (y_ + p['y0']) * (u_ + p['zeta2'] * u_**p['beta']) + p['xi'] * y_ - p['eta'] * z_
    dudt = -(y_ + p['y0']) * (u_ + p['zeta2'] * u_**p['beta']) + p['Gamma'] * u_
    return [dxdt, dydt, dzdt, dudt]

# 3) Set up and run simulations in a loop for different xi values
t_start, t_end = 0, 100
t_eval = np.linspace(t_start, t_end, 500)
initial_state_vns = [0, 0, 0, u0_fit]
mu_fixed = 0.9 # Keep VNS strength high and constant

# Test xi from its baseline of 0.2 up to 1.0
xi_values = np.linspace(0.2, 1.0, 9)
results = []
plt.figure(figsize=(10, 6))

# First, run the baseline simulation (no VNS, baseline xi)
params_baseline = params_fitted.copy()
sol_baseline = solve_ivp(EIM_vagal, (t_start, t_end), initial_state_vns, args=(params_baseline, 0.0), t_eval=t_eval)
auc_base = trapz(sol_baseline.y[1], sol_baseline.t)
plt.plot(sol_baseline.t, sol_baseline.y[1], label=f'Baseline (μ=0, ξ={params_baseline["xi"]:.1f})', lw=3, color='black')

# Loop through different xi values with VNS ON
for xi in xi_values:
    print(f"Running simulation for ξ = {xi:.2f} (with μ = {mu_fixed})...")
    params_current = params_fitted.copy()
    params_current['xi'] = xi # Update the xi parameter
    
    sol_vns = solve_ivp(EIM_vagal, (t_start, t_end), initial_state_vns, args=(params_current, mu_fixed), t_eval=t_eval)
    
    auc_vns = trapz(sol_vns.y[1], sol_vns.t)
    auc_suppression = 100 * (1 - auc_vns / auc_base) if auc_base > 0 else 0
    
    results.append({'xi': xi, 'auc_supp': auc_suppression})
    plt.plot(sol_vns.t, sol_vns.y[1], label=f'VNS, ξ={xi:.2f}')

# 4) Plot the simulation trajectories
plt.title('Effect of Increasing Feedback Strength (ξ) During VNS')
plt.xlabel('Time (hours)')
plt.ylabel('Immune Response y(t)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.show()

# 5) Print summary table to terminal
print("\n" + "="*50)
print(" " * 8 + "FEEDBACK SENSITIVITY ANALYSIS SUMMARY")
print("="*50)
print(f"{'Feedback Strength (ξ)':<25} | {'AUC Suppression (%)':<25}")
print("-" * 50)
for res in results:
    print(f"{res['xi']:<25.2f} | {res['auc_supp']:<25.1f}")
print("="*50)

