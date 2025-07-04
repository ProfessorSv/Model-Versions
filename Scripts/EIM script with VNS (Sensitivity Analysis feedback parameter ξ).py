# =============================================================================
#
#  Full Integrated Script (Sensitivity Analysis Version):
#  - Part 1 calibrates the model to IL-4 data.
#  - Part 2 runs a sensitivity analysis by simulating VNS at different
#    suppression strengths (mu) to see how it affects the outcome.
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

# 1) Load IL-4 data and normalise to peak = 1
try:
    data = pd.read_csv(r'C:\Users\celal\Desktop\Model-Versions\data\il4_hdm.csv')
except FileNotFoundError:
    print("\nError: 'il4_hdm.csv' not found.")
    print("Please make sure the CSV file is in the same directory as this script, or provide the full path.\n")
    print("Creating dummy data to allow the script to run for demonstration purposes.")
    t_data = np.array([0, 2, 4, 8, 24])
    y_data = np.array([3.5, 50, 90, 55, 20])
else:
    t_data = data['time'].values
    y_data = data['il4'].values

y_data_norm = y_data / np.max(y_data)

# 2) Core EIM ODEs (x, y, z, u)
def EIM_core(t, state, p):
    x, y, z, u = state
    ζ1, β = p['zeta1'], p['beta']
    Λ1, κ = p['Lambda1'], p['kappa']
    y0, ω = p['y0'], p['omega']
    δ, λv = p['delta'], p['lambda_val']
    Λ2, ζ2 = p['Lambda2'], p['zeta2']
    ξ, η, Γ = p['xi'], p['eta'], p['Gamma']
    x_ = max(x, 0); u_ = max(u, 0)
    dxdt = (1 - z) * (x_ + ζ1 * x_**β) + Λ1 * u
    dydt = x - (κ * x * (y + y0)) / (ω + x) - δ * y * (1 + λv * z) + Λ2 * u
    dzdt = (y + y0) * (u_ + ζ2 * u_**β) + ξ * y - η * z
    dudt = -(y + y0) * (u_ + ζ2 * u_**β) + Γ * u
    return [dxdt, dydt, dzdt, dudt]

# 3) Baseline core parameters
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
#  PART 2: SENSITIVITY ANALYSIS OF VNS EFFECT
# =============================================================================

print("\n--- Starting Part 2: Sensitivity Analysis for VNS Strength (μ) ---")

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

# 3) Set up and run simulations in a loop for different mu values
t_start, t_end = 0, 100
t_eval = np.linspace(t_start, t_end, 500)
initial_state_vns = [0, 0, 0, u0_fit]

mu_values = np.linspace(0, 1.5, 11) # Test μ from 0 to 1.5
results = []

# First, get baseline metrics
sol_baseline = solve_ivp(EIM_vagal, (t_start, t_end), initial_state_vns, args=(params_fitted, 0.0), t_eval=t_eval)
peak_base = np.max(sol_baseline.y[1])
auc_base = trapz(sol_baseline.y[1], sol_baseline.t)

# Loop through different mu values for VNS
for mu in mu_values:
    print(f"Running simulation for μ = {mu:.2f}...")
    sol_vns = solve_ivp(EIM_vagal, (t_start, t_end), initial_state_vns, args=(params_fitted, mu), t_eval=t_eval)
    
    peak_vns = np.max(sol_vns.y[1])
    auc_vns = trapz(sol_vns.y[1], sol_vns.t)
    
    peak_suppression = 100 * (1 - peak_vns / peak_base) if peak_base > 0 else 0
    auc_suppression = 100 * (1 - auc_vns / auc_base) if auc_base > 0 else 0
    
    results.append({'mu': mu, 'peak_supp': peak_suppression, 'auc_supp': auc_suppression})

# 4) Plot the sensitivity analysis results
results_df = pd.DataFrame(results)
plt.figure(figsize=(8, 6))
plt.plot(results_df['mu'], results_df['peak_supp'], 'o-', label='Peak Response Suppression')
plt.plot(results_df['mu'], results_df['auc_supp'], 's--', label='Total Response (AUC) Suppression')
plt.title('VNS Sensitivity Analysis')
plt.xlabel('VNS Strength (μ)')
plt.ylabel('Suppression (%)')
plt.legend()
plt.grid(True, linestyle=':')
plt.show()

# 5) Print summary table to terminal
print("\n" + "="*60)
print(" " * 12 + "VNS SENSITIVITY ANALYSIS SUMMARY TABLE")
print("="*60)
print(f"{'VNS Strength (μ)':<20} | {'Peak Suppression (%)':<20} | {'AUC Suppression (%)':<20}")
print("-" * 60)
for res in results:
    print(f"{res['mu']:<20.2f} | {res['peak_supp']:<20.1f} | {res['auc_supp']:<20.1f}")
print("="*60)

