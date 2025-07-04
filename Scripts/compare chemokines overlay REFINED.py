import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from typing import Dict, List, Any, Tuple

# =============================================================================
# --- Part 1: Configuration & Styling ---
# =============================================================================

# Define file paths and chemokine-specific information in a single, easily editable dictionary.
CHEMOKINE_CONFIG = {
    "CXCL1":    {"file": r'C:\Users\celal\Desktop\Model-Versions\data\CXCL1_hdm.csv', "column": "CXCL1"},
    "Eotaxin-1": {"file": r'C:\Users\celal\Desktop\Model-Versions\data\Eotaxin-1_hdm.csv', "column": "Eotaxin-1"},
    "TARC":     {"file": r'C:\Users\celal\Desktop\Model-Versions\data\TARC_hdm.csv', "column": "TARC"}
}

# Define a professional, colorblind-friendly color palette.
COLOR_PALETTE = {
    "CXCL1":    '#0072B2',  # Blue
    "Eotaxin-1": '#009E73',  # Green
    "TARC":     '#D55E00'   # Vermillion
}

# Set global Matplotlib parameters for a professional, publication-quality aesthetic.
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.autolayout': True,
})

# Baseline model parameters (fixed across all fits).
PARAMS_BASE = {
    'zeta1': 0.1, 'beta': 0.5, 'Lambda1': 0.05, 'kappa': 0.2,
    'y0': 0.1, 'omega': 1.0, 'delta': 0.3, 'lambda_val': 0.1,
    'Lambda2': 0.04, 'zeta2': 0.1, 'xi': 0.2, 'eta': 0.1, 'Gamma': 0.05
}

# =============================================================================
# --- Part 2: Core Modeling Functions ---
# =============================================================================

def EIM_core(t: float, state: List[float], p: Dict[str, float]) -> List[float]:
    """
    Defines the system of four coupled Ordinary Differential Equations (ODEs)
    for the Engram-Immune Model (EIM).
    """
    x, y, z, u = state
    x_, u_ = max(x, 0), max(u, 0)
    
    # Unpack parameters for clarity
    ζ1, β, Λ1, κ = p['zeta1'], p['beta'], p['Lambda1'], p['kappa']
    y0, ω, δ, λv = p['y0'], p['omega'], p['delta'], p['lambda_val']
    Λ2, ζ2, ξ, η, Γ = p['Lambda2'], p['zeta2'], p['xi'], p['eta'], p['Gamma']

    # ODE system
    dxdt = (1 - z) * (x_ + ζ1 * x_**β) + Λ1 * u
    dydt = x - (κ * x * (y + y0)) / (ω + x) - δ * y * (1 + λv * z) + Λ2 * u
    dzdt = (y + y0) * (u_ + ζ2 * u_**β) + ξ * y - η * z
    dudt = -(y + y0) * (u_ + ζ2 * u_**β) + Γ * u
    
    return [dxdt, dydt, dzdt, dudt]

def simulate_model(p: Dict[str, float], u0: float, time_points: np.ndarray) -> np.ndarray:
    """
    Solves the EIM ODEs for a given set of parameters and initial conditions.
    """
    solution = solve_ivp(
        fun=lambda t, s: EIM_core(t, s, p),
        t_span=(time_points.min(), time_points.max()),
        y0=[0, 0, 0, u0],  # Initial state: [x, y, z, u]
        t_eval=time_points,
        method='RK45'
    )
    return solution.y[1]  # Return only the immune response component, y(t)

def residuals_for_fitting(fit_params: List[float], y_data_norm: np.ndarray, t_data: np.ndarray, is_cxcl1: bool) -> np.ndarray:
    """
    Calculates residuals for fitting. Handles the special case for CXCL1 where Λ1 is zero.
    """
    p = PARAMS_BASE.copy()
    
    if is_cxcl1:
        δ_val, Λ2_val, κ_val, u0_val = fit_params
        p.update(delta=δ_val, Lambda1=0, Lambda2=Λ2_val, kappa=κ_val)
    else:
        δ_val, Λ1_val, Λ2_val, κ_val, u0_val = fit_params
        p.update(delta=δ_val, Lambda1=Λ1_val, Lambda2=Λ2_val, kappa=κ_val)
        
    y_predicted = simulate_model(p, u0_val, t_data)
    
    max_pred = np.max(y_predicted)
    if max_pred > 1e-9:
        y_pred_norm = y_predicted / max_pred
    else:
        y_pred_norm = y_predicted
        
    return y_pred_norm - y_data_norm

# =============================================================================
# --- Part 3: Data Processing & Fitting Workflow ---
# =============================================================================

def load_and_normalize_data(filepath: str, chemokine_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads chemokine data from a CSV, subtracts baseline, and normalizes to a peak of 1.
    """
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file was not found at {filepath}")
        return None, None

    time_hours = data['time'].values
    y_raw = data[chemokine_col].values
    
    baseline = y_raw[0]
    y_adjusted = np.maximum(y_raw - baseline, 0)
    
    max_adjusted_val = np.max(y_adjusted)
    if max_adjusted_val < 1e-9:
        print(f"Warning: Max value for {chemokine_col} is near zero.")
        y_normalized = y_adjusted
    else:
        y_normalized = y_adjusted / max_adjusted_val
        
    return time_hours, y_normalized

def fit_model_to_chemokine_data(t_data: np.ndarray, y_norm: np.ndarray, name: str) -> Dict[str, Any]:
    """
    Runs the full fitting pipeline for a single chemokine, handling the CXCL1 special case.
    """
    is_cxcl1 = (name == 'CXCL1')
    
    if is_cxcl1:
        init_guess = [0.3, 0.04, 0.2, 1.0]  # [δ, Λ2, κ, u0]
        bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])
    else:
        init_guess = [0.3, 0.05, 0.04, 0.2, 1.0]  # [δ, Λ1, Λ2, κ, u0]
        bounds = ([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf])

    fit_result = least_squares(
        residuals_for_fitting, init_guess, bounds=bounds, args=(y_norm, t_data, is_cxcl1)
    )
    
    # Unpack parameters and create final parameter dictionary
    p_fit = PARAMS_BASE.copy()
    if is_cxcl1:
        δ, Λ2, κ, u0 = fit_result.x
        Λ1 = 0.0
        p_fit.update(delta=δ, Lambda1=Λ1, Lambda2=Λ2, kappa=κ)
    else:
        δ, Λ1, Λ2, κ, u0 = fit_result.x
        p_fit.update(delta=δ, Lambda1=Λ1, Lambda2=Λ2, kappa=κ)
        
    # Simulate final curve and calculate metrics
    y_fit_raw = simulate_model(p_fit, u0, t_data)
    y_fit_norm = y_fit_raw / np.max(y_fit_raw)
    
    residuals_val = y_fit_norm - y_norm
    rmse = np.sqrt(np.mean(residuals_val**2))
    ss_res = np.sum(residuals_val**2)
    ss_tot = np.sum((y_norm - np.mean(y_norm))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        "params": [δ, Λ1, Λ2, κ, u0],
        "metrics": [rmse, r_squared],
        "fit_curve_norm": y_fit_norm
    }

# =============================================================================
# --- Part 4: Visualization & Reporting ---
# =============================================================================

def create_summary_plot(results: Dict[str, Dict]):
    """
    Generates a publication-quality plot for all chemokine data and model fits.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))

    for name, data in results.items():
        color = COLOR_PALETTE.get(name, 'gray')
        ax.plot(data['time_hours'], data['y_norm'], 'o-', color=color, label=f'{name} Data', markersize=8, alpha=0.8)
        ax.plot(data['time_hours'], data['fit_curve_norm'], '--', color=color, label=f'{name} Model Fit', linewidth=2.5)

    ax.set_title('Normalized Chemokine Dynamics Following HDM Challenge', fontweight='bold', fontsize=18)
    ax.set_xlabel('Time (hours post-challenge)', fontweight='bold')
    ax.set_ylabel('Normalized Chemokine Level', fontweight='bold')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

    metrics_text = [r"$\bf{Goodness-of-Fit}$"]
    for name, data in results.items():
        rmse, r_squared = data['metrics']
        metrics_text.append(f"{name}: $R^2$={r_squared:.3f}, RMSE={rmse:.3f}")
    
    ax.text(0.95, 0.05, "\n".join(metrics_text), transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.show()

def create_summary_table(results: Dict[str, Dict]):
    """
    Creates and prints a formatted pandas DataFrame summarizing fit results.
    """
    param_names = ['δ', 'Λ₁', 'Λ₂', 'κ', 'u₀', 'RMSE', 'R²']
    summary_data = {name: res['params'] + res['metrics'] for name, res in results.items()}
    df = pd.DataFrame(summary_data, index=param_names)
    
    print("\n" + "="*50)
    print("        CHEMOKINE MODEL FITTING SUMMARY")
    print("="*50)
    print(df.round(4))
    print("="*50 + "\n")

# =============================================================================
# --- Part 5: Main Execution Block ---
# =============================================================================

def main():
    """
    Orchestrates the entire workflow: load, fit, plot, and report.
    """
    all_results = {}
    print("Starting the fitting process for all chemokines...")

    for name, config in CHEMOKINE_CONFIG.items():
        print(f"\n--- Processing: {name} ---")
        
        time_hours, y_norm = load_and_normalize_data(config['file'], config['column'])
        
        if time_hours is not None:
            fit_results = fit_model_to_chemokine_data(time_hours, y_norm, name)
            all_results[name] = {"time_hours": time_hours, "y_norm": y_norm, **fit_results}
            print(f"Fit complete for {name}. Norm. R² = {fit_results['metrics'][1]:.4f}")

    if all_results:
        create_summary_plot(all_results)
        create_summary_table(all_results)
    else:
        print("\nNo data was successfully processed. Exiting.")

if __name__ == "__main__":
    main()