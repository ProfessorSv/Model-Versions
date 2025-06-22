# =============================================================================
#
#  Full Integrated Script (Corrected Version 3.1):
#  - Implements proportional VNS suppression.
#  - Uses a MODIFIED Koopman et al. template to model SUSTAINED suppression.
#  - Simulation time is set to 100 hours.
#
#  PART 1: Calibrate Engram-Immune Model (EIM) to IL-4 data
#  PART 2: Simulate Vagal Nerve Stimulation (VNS) using calibrated parameters
#  PART 3: Output results to terminal
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

    x_ = max(x, 0)
    u_ = max(u, 0)

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

# 4) Simulator for fitting y(t)
def simulate_for_fitting(p, u0):
    sol = solve_ivp(
        fun=lambda t, s: EIM_core(t, s, p),
        t_span=(t_data.min(), t_data.max()),
        y0=[0, 0, 0, u0],
        t_eval=t_data
    )
    return sol.y[1]

# 5) Residuals function for fitting
def residuals_shape(x):
    δ_val, Λ1_val, Λ2_val, u0_val = x
    p = params_base.copy()
    p.update({'delta': δ_val, 'Lambda1': Λ1_val, 'Lambda2': Λ2_val})
    y_pred = simulate_for_fitting(p, u0_val)
    if np.max(y_pred) < 1e-9: return np.inf
    y_pred_norm = y_pred / np.max(y_pred)
    return y_pred_norm - y_data_norm

# 6) Fit parameters
initial_guess = [params_base['delta'], params_base['Lambda1'], params_base['Lambda2'], 1.0]
bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])
res_shape = least_squares(residuals_shape, initial_guess, bounds=bounds)
δ_fit, Λ1_fit, Λ2_fit, u0_fit = res_shape.x

params_fitted = params_base.copy()
params_fitted.update({'delta': δ_fit, 'Lambda1': Λ1_fit, 'Lambda2': Λ2_fit})

# =============================================================================
#  PART 2: SIMULATE VNS EFFECT (USING CORRECTED SUSTAINED TEMPLATE)
# =============================================================================

print("\n--- Starting Part 2: Simulating VNS Effect (with Sustained Suppression) ---")

# 1) Build the v(t) temporal template using a sustained suppression shape
days = np.array([0, 42, 84]) # Using 3 points to define the plateau
# *** CORRECTED: Assume suppression remains at its peak to model a sustained effect ***
suppression_norm = np.array([0.0, 1.0, 1.0])
hours = days * 24

# Rescale the time axis to fit the 0-100 hour simulation window
# Start the effect after a small delay (e.g., 2 hours)
hours_rescaled = np.interp(hours, [hours.min(), hours.max()], [2, 100])

# Use PCHIP interpolator for a shape-preserving curve
v_interp = PchipInterpolator(hours_rescaled, suppression_norm, extrapolate=False)

# 2) Define the VNS-enabled EIM ODE system
def EIM_vagal(t, state, p, mu):
    x, y, z, u = state
    x_ = max(x, 0); y_ = max(y, 0); u_ = max(u, 0); z_ = z
    v_t = np.nan_to_num(v_interp(t))
    
    dxdt = (1 - z_) * (x_ + p['zeta1'] * x_**p['beta']) + p['Lambda1'] * u_
    dydt = (
        x_
        - (p['kappa'] * x_ * (y_ + p['y0'])) / (p['omega'] + x_)
        - p['delta'] * y_ * (1 + p['lambda_val'] * z_)
        + p['Lambda2'] * u_
        - mu * v_t * y_ # Proportional suppression
    )
    dzdt = (y_ + p['y0']) * (u_ + p['zeta2'] * u_**p['beta']) + p['xi'] * y_ - p['eta'] * z_
    dudt = -(y_ + p['y0']) * (u_ + p['zeta2'] * u_**p['beta']) + p['Gamma'] * u_
    return [dxdt, dydt, dzdt, dudt]

# 3) Set up and run simulations
t_start, t_end = 0, 100
t_eval = np.linspace(t_start, t_end, 1000)
initial_state_vns = [0, 0, 0, u0_fit]

sol_baseline = solve_ivp(EIM_vagal, (t_start, t_end), initial_state_vns, args=(params_fitted, 0.0), t_eval=t_eval)
sol_vns = solve_ivp(EIM_vagal, (t_start, t_end), initial_state_vns, args=(params_fitted, 0.9), t_eval=t_eval)

# 4) Show graphical plot (removed u(t) plot)
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('EIM Simulation: Effect of Sustained VNS on a Calibrated Model (0-100 hours)', fontsize=16)

# y(t) - Immune Response
axs[0, 0].plot(sol_baseline.t, sol_baseline.y[1], label='Baseline (μ=0)', lw=2)
axs[0, 0].plot(sol_vns.t, sol_vns.y[1], label='VNS (μ=0.9)', lw=2, linestyle='--')
axs[0, 0].set_title('Immune Response'); axs[0, 0].set_ylabel('y(t)'); axs[0, 0].legend(); axs[0, 0].grid(True, linestyle=':')

# x(t) - Engram Signal
axs[0, 1].plot(sol_baseline.t, sol_baseline.y[0], label='Baseline', lw=2); axs[0, 1].plot(sol_vns.t, sol_vns.y[0], label='VNS', lw=2, linestyle='--')
axs[0, 1].set_title('Engram Signal'); axs[0, 1].set_ylabel('x(t)'); axs[0, 1].legend(); axs[0, 1].grid(True, linestyle=':')

# z(t) - Feedback Signal
axs[1, 0].plot(sol_baseline.t, sol_baseline.y[2], label='Baseline', lw=2); axs[1, 0].plot(sol_vns.t, sol_vns.y[2], label='VNS', lw=2, linestyle='--')
axs[1, 0].set_title('Feedback Signal'); axs[1, 0].set_xlabel('Time (hours)'); axs[1, 0].set_ylabel('z(t)'); axs[1, 0].legend(); axs[1, 0].grid(True, linestyle=':')

# Turn off the unused subplot
axs[1, 1].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# =============================================================================
#  PART 3: TERMINAL OUTPUT
# =============================================================================

def generate_ascii_plot(series, height=15, width=60, label=""):
    series = np.nan_to_num(series)
    min_val, max_val = np.min(series), np.max(series)
    if max_val - min_val < 1e-9: max_val += 1e-9
    
    indices = np.linspace(0, len(series) - 1, width, dtype=int)
    sampled_series = series[indices]

    plot = [[' ' for _ in range(width)] for _ in range(height)]
    for w in range(width):
        h = int(((sampled_series[w] - min_val) / (max_val - min_val)) * (height - 1))
        if h >= 0: plot[height - 1 - h][w] = '*'

    print(f"\n--- ASCII Plot: {label} (0-{int(t_end)} hrs) ---")
    for row in plot: print(''.join(row))
    print("-" * width)
    print(f"Min: {min_val:.2f}{' ' * (width - 16 - len(f'{max_val:.2f}'))}Max: {max_val:.2f}")

def print_metrics_summary(sol, label):
    t, y, x = sol.t, sol.y[1], sol.y[0]
    peak_y, peak_t = (np.max(y), t[np.argmax(y)]) if len(y) > 0 else (0, 0)
    auc_y, final_y, final_x = trapz(y, t), y[-1], x[-1]

    print(f"\n--- Metrics for: {label} ---")
    print(f"  Peak Immune Response (y_peak): {peak_y:>8.3f}")
    print(f"  Time to Peak (t_peak):       {peak_t:>8.1f} hours")
    print(f"  Total Response (AUC 0-{int(t_end)}h): {auc_y:>8.1f}")
    print(f"  Final Immune Level (y_final):  {final_y:>8.3f}")
    print(f"  Final Engram Level (x_final):  {final_x:>8.3f}")
    return {'auc': auc_y, 'peak': peak_y}

print("\n" + "="*60)
print(" " * 15 + "TERMINAL OUTPUT OF SIMULATION")
print("="*60)

generate_ascii_plot(sol_baseline.y[1], label="Immune Response y(t) - BASELINE")
generate_ascii_plot(sol_vns.y[1], label="Immune Response y(t) - VNS")

baseline_metrics = print_metrics_summary(sol_baseline, "Baseline (μ=0.0)")
vns_metrics = print_metrics_summary(sol_vns, "VNS (μ=0.9)")

supp_auc_pct = 100 * (1 - vns_metrics['auc'] / baseline_metrics['auc']) if baseline_metrics['auc'] != 0 else 0
supp_peak_pct = 100 * (1 - vns_metrics['peak'] / baseline_metrics['peak']) if baseline_metrics['peak'] != 0 else 0

print("\n--- Overall VNS Suppression Summary (0-100 hrs) ---")
print(f"  Reduction in Total Response (AUC):   {supp_auc_pct:.1f} %")
print(f"  Reduction in Peak Response (y_peak): {supp_peak_pct:.1f} %")
print("="*60)
