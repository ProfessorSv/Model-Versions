# =============================================================================
#
#  Publication-Quality EIM Fit for IL-4 Data (Revised Script)
#
#  This script loads IL-4 data, fits the Engram-Immune Model, and
#  generates a professional two-panel figure showing the model fit
#  and the corresponding residuals, with metrics included.
#
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# =============================================================================
#  1. HELPER FUNCTIONS: Data Loading, Model, and Metrics
# =============================================================================

def load_and_process_il4_data(file_path):
    """
    Loads IL-4 data from a CSV, handles baseline subtraction,
    and normalizes the data for shape-fitting.
    
    Args:
        file_path (str): The path to the 'il4_hdm.csv' file.

    Returns:
        tuple: (time_points, normalized_data) or (None, None) if error.
    """
    try:
        data = pd.read_csv(r'C:\Users\celal\Desktop\Model-Versions\data\il4_hdm.csv')
        t_data = data['time'].values
        y_data = data['il4'].values
        
        # Subtract baseline (assumes t=0 is the first entry)
        baseline = y_data[0]
        y_adj = y_data - baseline
        y_adj[y_adj < 0] = 0.0 # Ensure no negative concentrations
        
        # Normalize for shape fitting
        max_val = np.max(y_adj)
        if max_val < 1e-9:
            print("Warning: Max cytokine value is near zero. Normalization may be unstable.")
            return t_data, y_adj
        
        y_norm = y_adj / max_val
        return t_data, y_norm
        
    except FileNotFoundError:
        print(f"Error: Data file not found at '{file_path}'")
        return None, None
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return None, None

def EIM_core(t, state, p):
    """Defines the four core ODEs of the Engram-Immune Model."""
    x, y, z, u = state
    x_ = max(x, 0); u_ = max(u, 0)
    dxdt = (1 - z) * (x_ + p['zeta1'] * x_**p['beta']) + p['Lambda1'] * u
    dydt = x - (p['kappa'] * x * (y + p['y0'])) / (p['omega'] + x) - p['delta'] * y * (1 + p['lambda_val'] * z) + p['Lambda2'] * u
    dzdt = (y + p['y0']) * (u_ + p['zeta2'] * u_**p['beta']) + p['xi'] * y - p['eta'] * z
    dudt = -(y + p['y0']) * (u_ + p['zeta2'] * u_**p['beta']) + p['Gamma'] * u
    return [dxdt, dydt, dzdt, dudt]

def compute_metrics(y_fit_norm, y_norm):
    """Computes RMSE and R² for a given fit."""
    resid = y_fit_norm - y_norm
    rmse = np.sqrt(np.mean(resid**2))
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y_norm - np.mean(y_norm))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return rmse, r2

# =============================================================================
#  2. MAIN FITTING AND SIMULATION WORKFLOW
# =============================================================================

# --- Configuration ---
IL4_FILE_PATH = 'il4_hdm.csv' # Assumes file is in the same directory

# --- Load and Process Data ---
t_data, y_data_norm = load_and_process_il4_data(IL4_FILE_PATH)

if t_data is None:
    # Exit if data loading failed
    exit()

# --- Define Parameters and Fitting Function ---
params_base = {
    'zeta1': 0.1, 'beta': 0.5, 'Lambda1': 0.05, 'kappa': 0.2, 'y0': 0.1,
    'omega': 1.0, 'delta': 0.3, 'lambda_val': 0.1, 'Lambda2': 0.04,
    'zeta2': 0.1, 'xi': 0.2, 'eta': 0.1, 'Gamma': 0.05
}

def simulate_core(p, u0, t_eval):
    """Runs the ODE solver for a given parameter set."""
    sol = solve_ivp(
        fun=lambda t, s: EIM_core(t, s, p),
        t_span=(t_eval.min(), t_eval.max()),
        y0=[0, 0, 0, u0],
        t_eval=t_eval
    )
    return sol.y[1]

def residuals_shape(params_to_fit, t_points, y_target):
    """Calculates residuals for the least_squares optimizer."""
    δ, Λ1, Λ2, κ_val, u0 = params_to_fit
    p = params_base.copy()
    p.update(delta=δ, Lambda1=Λ1, Lambda2=Λ2, kappa=κ_val)
    
    y_pred = simulate_core(p, u0, t_points)
    
    max_pred = np.max(y_pred)
    if max_pred < 1e-9: return np.inf
    
    y_pred_norm = y_pred / max_pred
    return y_pred_norm - y_target

# --- Run Optimization ---
initial_guess = [params_base['delta'], params_base['Lambda1'], params_base['Lambda2'], params_base['kappa'], 1.0]
bounds = ([0]*5, [np.inf]*5)
fit_result = least_squares(residuals_shape, initial_guess, bounds=bounds, args=(t_data, y_data_norm))
δ_fit, Λ1_fit, Λ2_fit, κ_fit, u0_fit = fit_result.x

print("--- Fitting Complete ---")
print(f"Fitted Parameters → δ={δ_fit:.3f}, Λ₁={Λ1_fit:.3f}, Λ₂={Λ2_fit:.3f}, κ={κ_fit:.3f}, u₀={u0_fit:.2f}")

# --- Generate Final Model Curve and Metrics ---
params_final = params_base.copy()
params_final.update(delta=δ_fit, Lambda1=Λ1_fit, Lambda2=Λ2_fit, kappa=κ_fit)
y_fit = simulate_core(params_final, u0_fit, t_data)
y_fit_norm = y_fit / np.max(y_fit)
rmse_fit, r2_fit = compute_metrics(y_fit_norm, y_data_norm)

print(f"Final Fit Metrics → Normalized RMSE = {rmse_fit:.3f}, Normalized R² = {r2_fit:.3f}")

# =============================================================================
#  3. GENERATE PUBLICATION-QUALITY FIGURE
# =============================================================================

# --- Setup Plot Style ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(
    2, 1, 
    figsize=(8, 8), 
    sharex=True, 
    gridspec_kw={'height_ratios': [3, 1]} # Main plot is 3x taller than residuals
)
fig.suptitle('Model Calibration to Experimental IL-4 Data', fontsize=18, fontweight='bold')

# --- Panel 1: Main Fit Plot ---
# Plot experimental data as points
ax1.plot(t_data, y_data_norm, 'o', color='#006BA4', markersize=8, label='Experimental IL-4 Data')
# Plot experimental data as a line
ax1.plot(t_data, y_data_norm, '-', color='#006BA4', linewidth=2, alpha=0.7, label='_nolegend_')
# Plot model fit
ax1.plot(t_data, y_fit_norm, '--', color='#FF800E', linewidth=2.5, label=f'EIM Fit (R²={r2_fit:.3f})')
ax1.set_ylabel('Normalized IL-4 Level', fontsize=12)
ax1.tick_params(axis='y', labelsize=10)
ax1.legend(loc='upper right', fontsize=10)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# --- Panel 2: Residuals Plot ---
residuals = y_fit_norm - y_data_norm
markerline, stemlines, baseline = ax2.stem(
    t_data, residuals,
    basefmt="k--",
    linefmt='-',
    markerfmt='D'
)
plt.setp(markerline, color='#D62728')
plt.setp(stemlines, color='#D62728')
plt.setp(baseline, color='gray')

ax2.set_xlabel('Time (hours post-challenge)', fontsize=12)
ax2.set_ylabel('Residuals', fontsize=12)
ax2.tick_params(axis='both', which='major', labelsize=10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Add RMSE to the residuals plot
ax2.text(0.95, 0.85, f'RMSE = {rmse_fit:.3f}', 
         transform=ax2.transAxes, ha='right', va='top', 
         fontsize=11, bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

# --- Final Touches ---
plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
plt.savefig("il4_model_fit.png", dpi=300, bbox_inches='tight')
plt.show()

# --- End of Script ---