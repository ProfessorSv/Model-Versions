import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# === 1) Load & normalize both IL-4 and IL-5 data ===

#We load both CSVs into separate DataFrames (data_il4, data_il5).
#Subtract each baseline (PBS at time 0) from the raw cytokine values.
#Clamp any negative results to zero (in case baseline > small later value).
#Divide by the maximum of the post‐baseline data so each dataset’s peak is 1.

# IL-4
data_il4     = pd.read_csv(r'C:\Users\celal\Desktop\Model-Versions\il4_hdm.csv')
t_data_il4   = data_il4['time'].values
y_raw_il4    = data_il4['il4'].values
baseline_il4 = y_raw_il4[0]
y_adj_il4    = y_raw_il4 - baseline_il4
y_adj_il4[y_adj_il4 < 0] = 0
y_norm_il4   = y_adj_il4 / np.max(y_adj_il4)



# IL-5
data_il5     = pd.read_csv(r'C:\Users\celal\Desktop\Model-Versions\il5_hdm.csv')
t_data_il5   = data_il5['time'].values
y_raw_il5    = data_il5['il5'].values
baseline_il5 = y_raw_il5[0]
y_adj_il5    = y_raw_il5 - baseline_il5
y_adj_il5[y_adj_il5 < 0] = 0
y_norm_il5   = y_adj_il5 / np.max(y_adj_il5)

# === 2) Define the core Engram-Immune ODEs (same as before) ===

#   Exactly the same 4 ODEs we have been using. No changes needed here.
def EIM_core(t, state, p):
    x, y, z, u = state
    ζ1, β     = p['zeta1'],    p['beta']
    Λ1, κ     = p['Lambda1'],  p['kappa']
    y0, ω     = p['y0'],       p['omega']
    δ, λv     = p['delta'],    p['lambda_val']
    Λ2, ζ2    = p['Lambda2'],  p['zeta2']
    ξ, η, Γ   = p['xi'],       p['eta'], p['Gamma']


    x_, u_    = max(x, 0), max(u, 0)
    dxdt = (1 - z)*(x_ + ζ1*x_**β) + Λ1*u
    dydt = x - (κ*x*(y + y0))/(ω + x) - δ*y*(1 + λv*z) + Λ2*u
    dzdt = (y + y0)*(u_ + ζ2*u_**β) + ξ*y - η*z
    dudt = -(y + y0)*(u_ + ζ2*u_**β) + Γ*u
    return [dxdt, dydt, dzdt, dudt]


# === 3) Baseline parameters (unchanged) ===

#   These are fixed initial guesses for all parameters except 
#   those we will fit: δ, Λ₁, Λ₂, κ, u₀, τ.
params_base = {
    'zeta1': 0.1,   'beta': 0.5,
    'Lambda1': 0.05, 'kappa': 0.2,
    'y0': 0.1,       'omega': 1.0,
    'delta': 0.3,    'lambda_val': 0.1,
    'Lambda2': 0.04, 'zeta2': 0.1,
    'xi': 0.2,       'eta': 0.1,
    'Gamma': 0.05,
    'tau': 1.0       # will be updated by fitting
}

# === 4) Simulator function ===

#Takes a parameter dictionary p and an initial lesion u0 and integrates from t = 0 to maximum time.
#Applies time scaling: real hours t_data / τ = model time.
#Returns y(t) (the immune response) at each data time.

def simulate(p, u0, t_data):
    tau    = p['tau']
    t_mod  = t_data / tau
    sol    = solve_ivp(
                 fun=lambda t, s: EIM_core(t, s, p),
                 t_span=(t_mod.min(), t_mod.max()),
                 y0=[0, 0, 0, u0],
                 t_eval=t_mod
             )
    return sol.y[1]  # return y(t)

# === 5) Residuals function for shape-fitting (δ, Λ1, Λ2, κ, u0, τ) ===

#Given a vector of six parameters [δ, Λ₁, Λ₂, κ, u₀, τ], update p and simulate y(t).
#Normalize the simulated y(t) so peak = 1. Subtract the normalized data (y_norm) to get residuals.
#least_squares will minimize the sum of squares of these residuals.

def residuals(params_vec, y_norm, t_data):
    δ_val, Λ1_val, Λ2_val, κ_val, u0_val, tau_val = params_vec
    p = params_base.copy()
    p.update(delta=δ_val, Lambda1=Λ1_val, Lambda2=Λ2_val, kappa=κ_val, tau=tau_val)
    y_pred = simulate(p, u0_val, t_data)
    y_pred_norm = y_pred / np.max(y_pred)
    return y_pred_norm - y_norm

# === 6) Fit for IL-4 ===

#Use the same initial guess and bounds as before.
#res_il4.x returns the best‐fit six parameters for IL-4.
#We then simulate again with those fits to get y_fit_il4 and y_fit_norm_il4.#

init_guess = [0.3, 0.05, 0.04, 0.2, 1.0, 1.0]       # [δ, Λ1, Λ2, κ, u0, τ]
bounds = ([0,0,0,0,0,0.1], [np.inf]*6)
res_il4 = least_squares(residuals, init_guess, bounds=bounds,
                        args=(y_norm_il4, t_data_il4))
δ_il4, Λ1_il4, Λ2_il4, κ_il4, u0_il4, τ_il4 = res_il4.x

# Simulate IL-4 with fitted parameters
params_il4 = params_base.copy()
params_il4.update(delta=δ_il4, Lambda1=Λ1_il4, Lambda2=Λ2_il4,
                  kappa=κ_il4, tau=τ_il4)
y_fit_il4 = simulate(params_il4, u0_il4, t_data_il4)
y_fit_norm_il4 = y_fit_il4 / np.max(y_fit_il4)

# === 7) Fit for IL-5 ===

#Identical to the IL-4 fit, but uses (y_norm_il5, t_data_il5) instead.
#Outputs best‐fit parameters for IL-5, then simulates y_fit_il5 and y_fit_norm_il5.

res_il5 = least_squares(residuals, init_guess, bounds=bounds,
                        args=(y_norm_il5, t_data_il5))
δ_il5, Λ1_il5, Λ2_il5, κ_il5, u0_il5, τ_il5 = res_il5.x

# Simulate IL-5 with fitted parameters
params_il5 = params_base.copy()
params_il5.update(delta=δ_il5, Lambda1=Λ1_il5, Lambda2=Λ2_il5,
                  kappa=κ_il5, tau=τ_il5)
y_fit_il5 = simulate(params_il5, u0_il5, t_data_il5)
y_fit_norm_il5 = y_fit_il5 / np.max(y_fit_il5)

# === 8) Compute normalized RMSE & R² for both IL-4 and IL-5 ===

#A small helper function computes normalized RMSE and R² given y_fit_norm and y_norm.
#We compute these for both IL-4 and IL-5.

def compute_metrics(y_fit_norm, y_norm):
    resid = y_fit_norm - y_norm
    rmse  = np.sqrt(np.mean(resid**2))
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y_norm - np.mean(y_norm))**2)
    r2    = 1 - ss_res/ss_tot
    return rmse, r2

rmse_il4, r2_il4 = compute_metrics(y_fit_norm_il4, y_norm_il4)
rmse_il5, r2_il5 = compute_metrics(y_fit_norm_il5, y_norm_il5)

# === 9) Plot both normalized curves on the same axes ===

#Plot normalized data & model for IL-4 in blue, and IL-5 in orange on the same axes.
#This makes visual comparison immediate: see if IL-5 peaks earlier/later or is broader/narrower than IL-4.

plt.figure(figsize=(6,4))
plt.plot(t_data_il4, y_norm_il4, 'o-', color='tab:blue', label='IL-4 Data')
plt.plot(t_data_il4, y_fit_norm_il4, 's--', color='tab:blue', label='IL-4 Model')
plt.plot(t_data_il5, y_norm_il5, 'o-', color='tab:orange', label='IL-5 Data')
plt.plot(t_data_il5, y_fit_norm_il5, 's--', color='tab:orange', label='IL-5 Model')
plt.xlabel('Time (h post-challenge)')
plt.ylabel('Normalized Cytokine')
plt.title('Normalized Shape-Fit: IL-4 vs IL-5')
plt.legend()
plt.tight_layout()
plt.show()

# === 10) Create a summary table of fitted parameters and metrics ===
summary = pd.DataFrame({
    'Param': ['δ', 'Λ₁', 'Λ₂', 'κ', 'u₀', 'τ', 'RMSE', 'R²'],
    'IL-4': [δ_il4, Λ1_il4, Λ2_il4, κ_il4, u0_il4, τ_il4, rmse_il4, r2_il4],
    'IL-5': [δ_il5, Λ1_il5, Λ2_il5, κ_il5, u0_il5, τ_il5, rmse_il5, r2_il5]
})
print("\n=== Parameter & Metric Comparison ===")
print(summary.round(3))
