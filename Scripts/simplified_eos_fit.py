# simplified_eos_fit.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# 1) LOAD & NORMALISE Eotaxin-1 (CCL11) CURVE
def load_norm_csv(path, col):
    df = pd.read_csv(path)
    t = df['time'].values
    y = df[col].values
    y0 = y[0]
    y_adj = y - y0
    y_adj[y_adj < 0] = 0
    y_norm = y_adj / np.max(y_adj)
    return t, y_norm

t_eotaxin, y_eotaxin = load_norm_csv(
    r'C:\Users\celal\Desktop\Model-Versions\data\Eotaxin-1_hdm.csv',
    'Eotaxin-1'
)


def f_eotaxin(t):
    return np.interp(t, t_eotaxin, y_eotaxin)

# 2) LOAD & NORMALISE EOSINOPHIL DATA
data_eos    = pd.read_csv(
    r'C:\Users\celal\Desktop\Model-Versions\data\eosinophils_hdm.csv'
)
t_eos       = data_eos['time'].values
y_eos_raw   = data_eos['Eos'].values
y_eos_norm  = np.maximum(y_eos_raw - y_eos_raw[0], 0) / np.max(np.maximum(y_eos_raw - y_eos_raw[0], 0))

# 3) DEFINE RESIDUAL FOR CCL11-ONLY MODEL
def resid_eos_simple(x):
    alpha2, deltaE = x
    def ode(t, E):
        return alpha2 * f_eotaxin(t) - deltaE * E
    sol = solve_ivp(ode, (t_eos.min(), t_eos.max()), [0], t_eval=t_eos)
    E = sol.y[0]
    E_norm = E / np.max(E)
    return E_norm - y_eos_norm

# 4) FIT alpha2 & deltaE
init_guess = [1.0, 0.5]
bounds     = ([0, 0], [np.inf, np.inf])
res        = least_squares(resid_eos_simple, init_guess, bounds=bounds)
alpha2_fit, deltaE_fit = res.x

# 5) SIMULATE WITH THE FITTED PARAMETERS
sol2    = solve_ivp(
    lambda t, E: alpha2_fit * f_eotaxin(t) - deltaE_fit * E,
    (t_eos.min(), t_eos.max()),
    [0],
    t_eval=t_eos
)
E2      = sol2.y[0]
E2_norm = E2 / np.max(E2)

# 6) COMPUTE RMSE & R²
def compute_metrics(y_pred, y_obs):
    rmse   = np.sqrt(np.mean((y_pred - y_obs)**2))
    ss_res = np.sum((y_obs - y_pred)**2)
    ss_tot = np.sum((y_obs - np.mean(y_obs))**2)
    r2     = 1 - ss_res/ss_tot
    return rmse, r2

rmse2, r22 = compute_metrics(E2_norm, y_eos_norm)

# 7) PRINT FIT RESULTS
print(f"Simplified Eosinophil fit -> alpha2={alpha2_fit:.3f}, deltaE={deltaE_fit:.3f}")
print(f"  RMSE={rmse2:.3f}, R²={r22:.3f}")

# 8) PLOT DATA VS MODEL
plt.figure(figsize=(6,4))
plt.plot(t_eos, y_eos_norm, 'o-', label='Eosinophil data')
plt.plot(t_eos, E2_norm,    's--', label='CCL11-only model')
plt.xlabel('Time (h post-challenge)')
plt.ylabel('Normalised eosinophils')
plt.title('Eosinophil Dynamics (Eotaxin-1 only)')
plt.legend()
plt.tight_layout()
plt.show()
