import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from typing import Dict, List, Tuple, Any

# =============================================================================
# --- Part 1: Configuration & Styling ---
# =============================================================================

# Define file paths and cytokine-specific information in a single, easily editable dictionary.
# This makes the script scalable and easy to maintain.
CYTOKINE_CONFIG = {
    "IL-4":  {"file": r'C:\Users\celal\Desktop\Model-Versions\data\il4_hdm.csv', "column": "il4"},
    "IL-5":  {"file": r'C:\Users\celal\Desktop\Model-Versions\data\il5_hdm.csv', "column": "il5"},
    "IL-13": {"file": r'C:\Users\celal\Desktop\Model-Versions\data\il13_hdm.csv', "column": "il13"},
    "IFN-γ": {"file": r'C:\Users\celal\Desktop\Model-Versions\data\IFN-y_hdm.csv', "column": "IFN-γ"},
}

# Define a professional, colorblind-friendly color palette.
# Using a dictionary ensures each cytokine is always assigned the same color.
COLOR_PALETTE = {
    "IL-4":  '#0072B2',  # Blue
    "IL-5":  '#D55E00',  # Vermillion
    "IL-13": '#009E73',  # Green
    "IFN-γ": '#CC79A7',  # Reddish Purple
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
    'Lambda2': 0.04, 'zeta2': 0.1, 'xi': 0.2, 'eta': 0.1,
    'Gamma': 0.05, 'tau': 1.0
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
    
    # Ensure variables are non-negative before applying fractional powers
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
    tau = p['tau']
    # Scale real-time hours into the model's intrinsic time units
    model_time_points = time_points / tau
    
    solution = solve_ivp(
        fun=lambda t, s: EIM_core(t, s, p),
        t_span=(model_time_points.min(), model_time_points.max()),
        y0=[0, 0, 0, u0],  # Initial state: [x, y, z, u]
        t_eval=model_time_points,
        method='RK45'
    )
    return solution.y[1]  # Return only the immune response component, y(t)

def residuals_for_fitting(fit_params: List[float], y_data_norm: np.ndarray, t_data: np.ndarray) -> np.ndarray:
    """
    Calculates the residuals between the normalized model output and normalized data.
    This function is minimized by `least_squares`.
    """
    δ_val, Λ1_val, Λ2_val, κ_val, u0_val, tau_val = fit_params
    
    # Create a new parameter dictionary for this simulation run
    p = PARAMS_BASE.copy()
    p.update(delta=δ_val, Lambda1=Λ1_val, Lambda2=Λ2_val, kappa=κ_val, tau=tau_val)
    
    # Simulate the model
    y_predicted = simulate_model(p, u0_val, t_data)
    
    # Normalize the model output for shape-fitting (peak = 1)
    max_pred = np.max(y_predicted)
    if max_pred > 1e-9:  # Avoid division by zero
        y_pred_norm = y_predicted / max_pred
    else:
        y_pred_norm = y_predicted
        
    return y_pred_norm - y_data_norm

# =============================================================================
# --- Part 3: Data Processing & Fitting Workflow ---
# =============================================================================

def load_and_normalize_data(filepath: str, cytokine_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads cytokine data from a CSV, subtracts baseline, and normalizes to a peak of 1.
    Includes robust error handling for file loading and normalization.
    """
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file was not found at {filepath}")
        return None, None

    time_hours = data['time'].values
    y_raw = data[cytokine_col].values
    
    # Subtract baseline (value at t=0)
    baseline = y_raw[0]
    y_adjusted = y_raw - baseline
    y_adjusted[y_adjusted < 0] = 0  # Clamp negative values to zero
    
    # Normalize so that the peak value is 1 for shape-fitting
    max_adjusted_val = np.max(y_adjusted)
    if max_adjusted_val < 1e-9:  # Check to prevent division by zero
        print(f"Warning: Max value for {cytokine_col} is near zero. Normalization may be unstable.")
        y_normalized = y_adjusted
    else:
        y_normalized = y_adjusted / max_adjusted_val
        
    return time_hours, y_normalized

def fit_model_to_cytokine_data(t_data: np.ndarray, y_norm: np.ndarray) -> Dict[str, Any]:
    """
    Runs the full fitting pipeline for a single cytokine dataset.
    Now matches the 'Improved IL-4 EIM fit Publication Quality' approach:
    - Fits only [δ, Λ1, Λ2, κ, u0] (tau fixed at 1.0)
    - Returns metrics and normalized fit
    """
    # Parameters: [δ, Λ1, Λ2, κ, u0]
    init_guess = [0.3, 0.05, 0.04, 0.2, 1.0]
    bounds = ([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf])

    def residuals_shape(params_to_fit, t_points, y_target):
        δ, Λ1, Λ2, κ_val, u0 = params_to_fit
        p = PARAMS_BASE.copy()
        p.update(delta=δ, Lambda1=Λ1, Lambda2=Λ2, kappa=κ_val, tau=1.0)  # tau fixed
        y_pred = simulate_model(p, u0, t_points)
        max_pred = np.max(y_pred)
        if max_pred < 1e-9:
            y_pred_norm = y_pred
        else:
            y_pred_norm = y_pred / max_pred
        return y_pred_norm - y_target

    # Perform non-linear least squares fitting
    fit_result = least_squares(
        residuals_shape,
        init_guess,
        bounds=bounds,
        args=(t_data, y_norm)
    )

    δ_fit, Λ1_fit, Λ2_fit, κ_fit, u0_fit = fit_result.x

    # Simulate the model one last time with the best-fit parameters
    p_fit = {**PARAMS_BASE, 'delta': δ_fit, 'Lambda1': Λ1_fit, 'Lambda2': Λ2_fit, 'kappa': κ_fit, 'tau': 1.0}
    y_fit_raw = simulate_model(p_fit, u0_fit, t_data)
    y_fit_norm = y_fit_raw / np.max(y_fit_raw)

    # Calculate goodness-of-fit metrics
    residuals = y_fit_norm - y_norm
    rmse = np.sqrt(np.mean(residuals**2))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_norm - np.mean(y_norm))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Return all results in a structured dictionary
    return {
        "params": [δ_fit, Λ1_fit, Λ2_fit, κ_fit, u0_fit],
        "metrics": [rmse, r_squared],
        "fit_curve_norm": y_fit_norm
    }

# =============================================================================
# --- Part 4: Visualization & Reporting ---
# =============================================================================

def create_summary_plot(results: Dict[str, Dict]):
    """
    Generates a publication-quality plot comparing all cytokine data and model fits,
    including an annotation box with R-squared and RMSE metrics.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))

    for name, data in results.items():
        color = COLOR_PALETTE.get(name, 'gray')
        # Plot experimental data points
        ax.plot(
            data['time_hours'], data['y_norm'],
            marker='o', markersize=8, linestyle='-',
            color=color, label=f'{name} Data', alpha=0.8
        )
        # Plot model fit as a dashed line
        ax.plot(
            data['time_hours'], data['fit_curve_norm'],
            linestyle='--', linewidth=2.5,
            color=color, label=f'{name} Model Fit'
        )

    # --- Aesthetics and Annotations ---
    ax.set_title('Normalized Cytokine Dynamics Following HDM Challenge', fontweight='bold', fontsize=18)
    ax.set_xlabel('Time (hours post-challenge)', fontweight='bold')
    ax.set_ylabel('Normalized Cytokine Level', fontweight='bold')
    
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

    # --- Create and Add Metrics Annotation Box ---
    metrics_text_lines = [r"$\bf{Goodness-of-Fit}$"] # Bold title using LaTeX
    for name, data in results.items():
        rmse, r_squared = data['metrics']
        # Use LaTeX for R-squared for a professional look
        line = f"{name}: $R^2$={r_squared:.3f}, RMSE={rmse:.3f}"
        metrics_text_lines.append(line)
    
    full_metrics_text = "\n".join(metrics_text_lines)
    
    # Place text in the bottom right corner of the plot
    ax.text(
        0.95, 0.05, full_metrics_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
    )
    
    plt.tight_layout(rect=[0, 0, 0.82, 1]) # Adjust layout to make space for legend
    plt.show()

def create_summary_table(results: Dict[str, Dict]):
    """
    Creates and prints a formatted pandas DataFrame summarizing the fit results.
    """
    param_names = ['δ (Decay)', 'Λ₁ (Engram Drive)', 'Λ₂ (Allergen Drive)', 'κ (Feedback)', 'u₀ (Initial Lesion)', 'Norm. RMSE', 'R²']

    summary_data = {
        name: data['params'] + data['metrics'] for name, data in results.items()
    }

    df = pd.DataFrame(summary_data, index=param_names)

    print("\n" + "="*55)
    print("           MODEL FITTING & METRICS SUMMARY")
    print("="*55)
    print(df.round(4))
    print("="*55 + "\n")

# =============================================================================
# --- Part 5: Main Execution Block ---
# =============================================================================

def main():
    """
    Main function to orchestrate the entire workflow:
    1. Load and process data for all cytokines.
    2. Fit the model to each dataset.
    3. Generate a summary plot and table.
    """
    all_results = {}

    print("Starting the fitting process for all cytokines...")

    for name, config in CYTOKINE_CONFIG.items():
        print(f"\n--- Processing: {name.encode('ascii', errors='replace').decode()} ---")
        
        # 1. Load and normalize data
        time_hours, y_norm = load_and_normalize_data(config['file'], config['column'])
        
        # Proceed only if data was loaded successfully
        if time_hours is not None:
            # 2. Fit model
            fit_results = fit_model_to_cytokine_data(time_hours, y_norm)
            
            # 3. Store all results for this cytokine
            all_results[name] = {
                "time_hours": time_hours,
                "y_norm": y_norm,
                **fit_results
            }
            print(f"Fit complete for {name}. Norm. R² = {fit_results['metrics'][1]:.4f}")

    # 4. Generate final outputs if any results were produced
    if all_results:
        create_summary_plot(all_results)
        create_summary_table(all_results)
    else:
        print("\nNo data was successfully processed. Exiting.")

if __name__ == "__main__":
    main()