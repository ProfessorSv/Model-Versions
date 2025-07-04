# neut_with_baseline.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# 1) LOAD & NORMALISE CXCL1 (KC) CURVE
def load_norm_csv(path, col):
    df = pd.read_csv(path)
    t  = df['time'].values
    y  = df[col].values
    y_adj = np.maximum(y - y[0], 0)
    return t, y_adj / np.max(y_adj)

# Update this path to where your CXCL1 CSV actually is
t_cxcl1, y_cxcl1 = load_norm_csv(
    r'C:\Users\celal\Desktop\Model-Versions\data\CXCL1_hdm.csv',
    'cxcl1'
)

def f_cxcl1(t):
    return np.interp(t, t_cxcl1, y_cxcl1)

# 2) LOAD & NORMALISE NEUTROPHIL COUNT DATA
# Update this path to where your neutrophils CSV actually is
data_neut   = pd.read_csv(
    r'C:\Users\celal\Desktop\Model-Versions\data\neutrophils_hdm.csv'
)
t_neut      = data_neut['time'].values
y_neut_raw  = data_neut['Neut'].values
y_neut_adj  = np.maximum(y_neut_raw - y_neut_raw[0], 0)
y_neut_norm = y_neut_adj / np.max(y_neut_adj)

# 3) DEFINE RESIDUAL FUNCTION FOR MODEL WITH τ AND BASELINE OFFSET b
def resid_neut_baseline(x):
    alphaN, deltaN, tauN, b = x
    def ode(t, N):
        return (alphaN * f_cxcl1(t) - deltaN * N) / tauN
    sol = solve_ivp(ode, (t_neut.min(), t_neut.max()), [0], t_eval=t_neut)
    N = sol.y[0]
    N_norm = N / np.max(N)
    return (N_norm + b) - y_neut_norm

# 4) FIT PARAMETERS [αN, δN, τN, b]
init_guess = [1.0, 0.5, 1.0, 0.0]                  # initial guesses
bounds     = ([0, 0, 0, -np.inf], [np.inf]*4)      # allow b to be negative if needed
res        = least_squares(resid_neut_baseline, init_guess, bounds=bounds)
alphaN_fit, deltaN_fit, tauN_fit, b_fit = res.x

# 5) SIMULATE WITH THE FITTED PARAMETERS
soln = solve_ivp(
    lambda t, N: (alphaN_fit * f_cxcl1(t) - deltaN_fit * N) / tauN_fit,
    (t_neut.min(), t_neut.max()),
    [0],
    t_eval=t_neut
)
N_fit      = soln.y[0]
N_fit_norm = N_fit / np.max(N_fit) + b_fit

# 6) COMPUTE FIT METRICS RMSE & R²
def compute_metrics(y_pred, y_obs):
    rmse   = np.sqrt(np.mean((y_pred - y_obs)**2))
    ss_res = np.sum((y_obs - y_pred)**2)
    ss_tot = np.sum((y_obs - np.mean(y_obs))**2)
    r2     = 1 - ss_res/ss_tot
    return rmse, r2

rmse_neut, r2_neut = compute_metrics(N_fit_norm, y_neut_norm)

# 7) PRINT RESULTS
print(f"Neutrophil fit with τ & baseline -> alphaN={alphaN_fit:.3f}, deltaN={deltaN_fit:.3f}, tauN={tauN_fit:.3f}, b={b_fit:.3f}")
print(f"  RMSE={rmse_neut:.3f}, R²={r2_neut:.3f}")

# 8) PLOT DATA VS MODEL
plt.figure(figsize=(6,4))
plt.plot(t_neut, y_neut_norm, 'o-', label='Neutrophil data')
plt.plot(t_neut, N_fit_norm, 's--', label='Model (τ + baseline)')
plt.xlabel('Time (h post-challenge)')
plt.ylabel('Normalised neutrophils')
plt.title('Neutrophil Dynamics with Time-Scaling & Baseline Offset')
plt.legend()
plt.tight_layout()
plt.show()
