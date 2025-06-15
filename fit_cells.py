# fit_cells.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# ─── 1) LOAD & NORMALISE MEDIATOR CURVES ──────────────────────────────────
def load_norm_csv(path, col):
    df = pd.read_csv(path)
    t  = df['time'].values
    y  = df[col].values
    y_adj = np.maximum(y - y[0], 0)       # subtract PBS, floor at 0
    return t, y_adj / np.max(y_adj)      # normalise peak→1


# Paths to your mediator CSVs
t_il5,      y_il5      = load_norm_csv(r'C:\Users\celal\Desktop\Model-Versions\data\il5_hdm.csv',       'il5')
t_eotaxin,  y_eotaxin  = load_norm_csv(r'C:\Users\celal\Desktop\Model-Versions\data\Eotaxin-1_hdm.csv',  'eotaxin-1')
t_cxcl1,    y_cxcl1    = load_norm_csv(r'C:\Users\celal\Desktop\Model-Versions\data\CXCL1_hdm.csv',     'cxcl1')

# Interpolation functions to get mediator levels at any t
def f_il5(t):     return np.interp(t, t_il5,     y_il5)
def f_eotaxin(t): return np.interp(t, t_eotaxin, y_eotaxin)
def f_cxcl1(t):   return np.interp(t, t_cxcl1,   y_cxcl1)

# ─── 2) LOAD & NORMALISE CELL DATA ───────────────────────────────────────
data_eos   = pd.read_csv(r'C:\Users\celal\Desktop\Model-Versions\data\eosinophils_hdm.csv')
t_eos      = data_eos['time'].values
y_eos_raw  = data_eos['Eos'].values
y_eos_norm = y_eos_raw / np.max(y_eos_raw)
y_eos_adj  = np.maximum(y_eos_raw - y_eos_raw[0], 0)
y_eos_norm = y_eos_adj / np.max(y_eos_adj)


data_neut  = pd.read_csv(r'C:\Users\celal\Desktop\Model-Versions\data\neutrophils_hdm.csv')
t_neut     = data_neut['time'].values
y_neut_raw = data_neut['Neut'].values
y_neut_norm= y_neut_raw / np.max(y_neut_raw)
y_neut_adj  = np.maximum(y_neut_raw - y_neut_raw[0], 0)
y_neut_norm = y_neut_adj / np.max(y_neut_adj)



# 3) SIMULATORS
def simulate_eos(params, t_eval):
    α1, α2, δE = params
    def ode(t, E):
        return α1*f_il5(t) + α2*f_eotaxin(t) - δE*E
    sol = solve_ivp(ode, (t_eval.min(), t_eval.max()), [0], t_eval=t_eval)
    return sol.y[0]

def simulate_neut(params, t_eval):
    αN, δN = params
    def ode(t, N):
        return αN*f_cxcl1(t) - δN*N
    sol = solve_ivp(ode, (t_eval.min(), t_eval.max()), [0], t_eval=t_eval)
    return sol.y[0]

# 4) RESIDUALS
def resid_eos(x):
    E_pred = simulate_eos(x, t_eos)
    return (E_pred/np.max(E_pred)) - y_eos_norm

def resid_neut(x):
    N_pred = simulate_neut(x, t_neut)
    return (N_pred/np.max(N_pred)) - y_neut_norm

# 5) FIT PARAMETERS
res_eos  = least_squares(resid_eos,  [1.0, 1.0, 0.5], bounds=([0,0,0],[np.inf]*3))
α1_fit, α2_fit, δE_fit = res_eos.x

res_neut = least_squares(resid_neut, [1.0, 0.5],   bounds=([0,0],[np.inf]*2))
αN_fit, δN_fit = res_neut.x

# 6) SIMULATE WITH FITTED PARAMS
E_fit      = simulate_eos([α1_fit, α2_fit, δE_fit], t_eos)
E_fit_norm = E_fit / np.max(E_fit)

N_fit      = simulate_neut([αN_fit, δN_fit], t_neut)
N_fit_norm = N_fit / np.max(N_fit)

# 7) METRIC FUNCTIONS
def compute_metrics(y_pred, y_obs):
    rmse = np.sqrt(np.mean((y_pred - y_obs)**2))
    ss_res = np.sum((y_obs - y_pred)**2)
    ss_tot = np.sum((y_obs - np.mean(y_obs))**2)
    r2 = 1 - ss_res/ss_tot
    return rmse, r2

rmse_eos, r2_eos = compute_metrics(E_fit_norm, y_eos_norm)
rmse_neut, r2_neut = compute_metrics(N_fit_norm, y_neut_norm)

# 8) PRINT RESULTS
print(f"Eosinophil fit → α1={α1_fit:.3f}, α2={α2_fit:.3f}, δE={δE_fit:.3f}")
print(f"  RMSE={rmse_eos:.3f}, R²={r2_eos:.3f}\n")
print(f"Neutrophil fit → αN={αN_fit:.3f}, δN={δN_fit:.3f}")
print(f"  RMSE={rmse_neut:.3f}, R²={r2_neut:.3f}")

# 9) PLOTTING
plt.figure()
plt.plot(t_eos, y_eos_norm, 'o-', label='Eosinophil data')
plt.plot(t_eos, E_fit_norm, 's--', label='Eosinophil model')
plt.xlabel('Time (h)')
plt.ylabel('Normalised eosinophils')
plt.title('Eosinophil Dynamics')
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(t_neut, y_neut_norm, 'o-', label='Neutrophil data')
plt.plot(t_neut, N_fit_norm, 's--', label='Neutrophil model')
plt.xlabel('Time (h)')
plt.ylabel('Normalised neutrophils')
plt.title('Neutrophil Dynamics')
plt.legend()
plt.tight_layout()

plt.show()