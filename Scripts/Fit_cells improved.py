# =============================================================================
# --- Part 1: Setup and Configuration ---
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# Define file paths for the input data. Using a dictionary makes it easy
# to manage paths and keeps the main script clean.
FILE_PATHS = {
    "il5": r'C:\Users\celal\Desktop\Model-Versions\data\il5_hdm.csv',
    "Eotaxin-1": r'C:\Users\celal\Desktop\Model-Versions\data\Eotaxin-1_hdm.csv',
    "CXCL1": r'C:\Users\celal\Desktop\Model-Versions\data\CXCL1_hdm.csv',
    "Eos": r'C:\Users\celal\Desktop\Model-Versions\data\eosinophils_hdm.csv',
    "Neut": r'C:\Users\celal\Desktop\Model-Versions\data\neutrophils_hdm.csv',
}

# A professional color palette for consistency and visual appeal.
# Using specific hex codes ensures the plots look the same everywhere.
PLOT_COLORS = {
    "data": "#333333",  # Dark grey for data points
    "eos_model": "#0072B2",  # A professional blue for the eosinophil model
    "neut_model": "#D55E00", # A professional orange for the neutrophil model
}


# =============================================================================
# --- Part 2: Data Loading and Preprocessing ---
# =============================================================================

def load_and_normalize_data(filepath: str, value_col: str, time_col: str = "time") -> tuple:
    """
    Loads time-course data from a CSV, subtracts the baseline value (at t=0),
    and normalizes the result so the peak value is 1.0.
    This allows us to compare the 'shape' of different biological responses.
    """
    try:
        data = pd.read_csv(filepath)
        time = data[time_col].values
        values = data[value_col].values
    except FileNotFoundError:
        print(f"Error: Could not find the file at {filepath}")
        return None, None
    except KeyError:
        print(f"Error: Column '{value_col}' not found in {filepath}.")
        return None, None

    # Ensure baseline subtraction is safe even if the file has no t=0 point.
    baseline = values[0] if time[0] == 0 else 0
    adjusted_values = np.maximum(0, values - baseline)

    peak = np.max(adjusted_values)
    if peak < 1e-9: # Avoid division by zero
        return time, adjusted_values
    
    return time, adjusted_values / peak


# --- Load all necessary data using the helper function ---

# Load mediator curves that will drive the cell responses.
t_il5, y_il5 = load_and_normalize_data(FILE_PATHS["il5"], "il5")
t_eotaxin, y_eotaxin = load_and_normalize_data(FILE_PATHS["Eotaxin-1"], "Eotaxin-1")
t_cxcl1, y_cxcl1 = load_and_normalize_data(FILE_PATHS["CXCL1"], "CXCL1")

# Load the cell count data that we want to fit our model to.
t_eos, y_eos_norm = load_and_normalize_data(FILE_PATHS["Eos"], "Eos")
t_neuts, y_neuts_norm = load_and_normalize_data(FILE_PATHS["Neut"], "Neut")
# --- Create interpolation functions ---
# These functions allow us to get the value of a mediator at ANY time point,
# not just the ones measured. This is crucial for the ODE solver.
f_il5 = lambda t: np.interp(t, t_il5, y_il5)
f_eotaxin = lambda t: np.interp(t, t_eotaxin, y_eotaxin)
f_cxcl1 = lambda t: np.interp(t, t_cxcl1, y_cxcl1)


# =============================================================================
# --- Part 3: Model Definition and Fitting ---
# =============================================================================

def simulate_cell_dynamics(params: list, t_eval: np.ndarray, drivers: list) -> np.ndarray:
    """
    A general simulator for a cell population driven by one or more mediators.
    Solves a simple ODE: d(Cell)/dt = (alpha * Mediator - delta * Cell) / tau
    """
    *alpha_params, delta, tau = params
    
    def cell_ode(t, cell_pop):
        # Calculate the total drive from all mediators.
        recruitment_drive = sum(alpha * driver_func(t) for alpha, driver_func in zip(alpha_params, drivers))
        # The ODE equation.
        return (recruitment_drive - delta * cell_pop) / tau

    solution = solve_ivp(cell_ode, (t_eval.min(), t_eval.max()), [0], t_eval=t_eval)
    return solution.y[0]

def calculate_residuals(fit_params: list, t_data: np.ndarray, y_data_norm: np.ndarray, drivers: list) -> np.ndarray:
    """
    Calculates the difference between the model simulation and the actual data.
    The goal of the fitting process is to make this difference as small as possible.
    """
    # Simulate the cell dynamics with the current trial parameters.
    y_predicted = simulate_cell_dynamics(fit_params, t_data, drivers)
    
    # Normalize the predicted curve so its peak is 1.0 for a fair comparison.
    peak_predicted = np.max(y_predicted)
    if peak_predicted < 1e-9:
        y_predicted_norm = y_predicted
    else:
        y_predicted_norm = y_predicted / peak_predicted

    return y_predicted_norm - y_data_norm

def fit_model_to_data(t_data: np.ndarray, y_norm: np.ndarray, drivers: list, num_alphas: int) -> dict:
    """

    Performs the least-squares fitting to find the best model parameters.
    """
    # [alpha1, alpha2, ..., delta, tau]
    initial_guesses = [1.0] * num_alphas + [0.5, 1.0] 
    bounds = ([0] * len(initial_guesses), [np.inf] * len(initial_guesses))

    result = least_squares(
        calculate_residuals,
        initial_guesses,
        bounds=bounds,
        args=(t_data, y_norm, drivers)
    )
    
    # --- Calculate final results and goodness-of-fit ---
    fitted_params = result.x
    y_fit = simulate_cell_dynamics(fitted_params, t_data, drivers)
    y_fit_norm = y_fit / np.max(y_fit)
    
    residuals = y_fit_norm - y_norm
    rmse = np.sqrt(np.mean(residuals**2))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_norm - np.mean(y_norm))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        "params": fitted_params,
        "y_fit_norm": y_fit_norm,
        "metrics": {"rmse": rmse, "r_squared": r_squared}
    }

# =============================================================================
# --- Part 4: Main Execution and Plotting ---
# =============================================================================

def main_workflow():
    """
    Orchestrates the fitting for both cell types and generates the plots.
    """
    # --- Fit Eosinophils (driven by IL-5 and Eotaxin-1) ---
    print("--- Fitting Eosinophil Dynamics ---")
    eos_drivers = [f_il5, f_eotaxin]
    eos_results = fit_model_to_data(t_eos, y_eos_norm, eos_drivers, num_alphas=2)
    
    print(f"Eosinophil Fit Parameters (α_IL5, α_Eotaxin, δ, τ): {np.round(eos_results['params'], 4)}")
    print(f"Eosinophil Fit Metrics: RMSE = {eos_results['metrics']['rmse']:.4f}, R² = {eos_results['metrics']['r_squared']:.4f}\n")

    # --- Fit Neutrophils (driven by CXCL1) ---
    print("--- Fitting Neutrophil Dynamics ---")
    neut_drivers = [f_cxcl1]
    neut_results = fit_model_to_data(t_neuts, y_neuts_norm, neut_drivers, num_alphas=1)

    print(f"Neutrophil Fit Parameters (α_CXCL1, δ, τ): {np.round(neut_results['params'], 4)}")
    print(f"Neutrophil Fit Metrics: RMSE = {neut_results['metrics']['rmse']:.4f}, R² = {neut_results['metrics']['r_squared']:.4f}\n")
    
    # --- Create the plots ---
    plot_results(t_eos, y_eos_norm, eos_results['y_fit_norm'], 'Eosinophils', 'Eosinophil_Dynamics_Fit.png')
    plot_results(t_neuts, y_neuts_norm, neut_results['y_fit_norm'], 'Neutrophils', 'Neutrophil_Dynamics_Fit.png')


def plot_results(t_data, y_data, y_fit, cell_type, filename):
    """
    Creates a single, publication-quality plot for a given cell type.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot the experimental data.
    ax.plot(t_data, y_data, marker='o', linestyle='-', color=PLOT_COLORS['data'],
            label=f'Experimental {cell_type} Data', markersize=8)

    # Plot the model fit.
    model_color = PLOT_COLORS['eos_model'] if 'Eosinophil' in cell_type else PLOT_COLORS['neut_model']
    ax.plot(t_data, y_fit, marker='^', linestyle='--', color=model_color,
            label=f'Model Fit for {cell_type}', markersize=7)

    # --- Formatting for Publication Quality ---
    ax.set_title(f'Model Fit to Normalized {cell_type} Counts', fontsize=16, fontweight='bold', pad=12)
    ax.set_xlabel('Time (hours post-HDM challenge)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Cell Count', fontsize=12, fontweight='bold')
    
    # Use a clean and non-intrusive grid.
    ax.grid(True, linestyle=':', alpha=0.5)

    # Remove the top and right spines for a modern look.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=10)

    # Position the legend neatly.
    ax.legend(fontsize=11)
    
    # Ensure all plot elements fit without overlapping.
    plt.tight_layout()

    # Save the figure with high resolution for publication.
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# --- This ensures the main function runs only when the script is executed directly ---
if __name__ == "__main__":
    main_workflow()
