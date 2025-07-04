# eos_two_phase.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# 1) LOAD & NORMALISE Eotaxin-1 and IL-5 time-courses
def load_norm_csv(path, col):
    df = pd.read_csv(path)
    t  = df['time'].values
    y  = df[col].values
    y_adj = np.maximum(y - y[0], 0)
    return t, y_adj/np.max(y_adj)

# Update these paths & column names if your CSV differs
t_eot, y_eot = load_norm_csv(
    r'C:\Users\celal\Desktop\Model-Versions\data\Eotaxin-1_hdm.csv',
    'Eotaxin-1'
)
t_il5, y_il5 = load_norm_csv(
    r'C:\Users\celal\Desktop\Model-Versions\data\il5_hdm.csv',
    'il5'
)

f_eot = lambda t: np.interp(t, t_eot, y_eot)
f_il5 = lambda t: np.interp(t, t_il5, y_il5)

# 2) LOAD & NORMALISE Eosinophil count data
df_e = pd.read_csv(
    r'C:\Users\celal\Desktop\Model-Versions\data\eosinophils_hdm.csv'
)
t_e = df_e['time'].values
y_e = df_e['Eos'].values           # column name 'eos'
y_e = np.maximum(y_e - y_e[0], 0)
y_e_norm = y_e / np.max(y_e)

# 3) DEFINE RESIDUAL FUNCTION FOR TWO-PHASE MODEL
def resid_two_phase(x):
    alpha2, deltaE1, tau1, alpha1, deltaE2, tau2 = x
    t_c = 4.0  # switch at 4 h
    def ode(t, E):
        if t < t_c:
            return (alpha2 * f_eot(t) - deltaE1 * E) / tau1
        else:
            return (alpha1 * f_il5(t) - deltaE2 * E) / tau2
    sol = solve_ivp(ode, (t_e.min(), t_e.max()), [0], t_eval=t_e)
    E = sol.y[0]
    E_norm = E / np.max(E)
    return E_norm - y_e_norm

# 4) FIT the six parameters: α₂, δE₁, τ₁, α₁, δE₂, τ₂
init  = [1, 1, 1, 1, 1, 1]
bnds  = ([0]*6, [np.inf]*6)
res   = least_squares(resid_two_phase, init, bounds=bnds)
alpha2_fit, deltaE1_fit, tau1_fit, alpha1_fit, deltaE2_fit, tau2_fit = res.x

# 5) SIMULATE the fitted two-phase model
sol = solve_ivp(
    lambda t, E: (alpha2_fit * f_eot(t) - deltaE1_fit * E) / tau1_fit
                 if t < 4.0
                 else (alpha1_fit * f_il5(t) - deltaE2_fit * E) / tau2_fit,
    (t_e.min(), t_e.max()), [0], t_eval=t_e
)
E_fit = sol.y[0]
E_fit_norm = E_fit / np.max(E_fit)

# 6) COMPUTE RMSE and R²
rmse   = np.sqrt(np.mean((E_fit_norm - y_e_norm)**2))
ss_res = np.sum((y_e_norm - E_fit_norm)**2)
ss_tot = np.sum((y_e_norm - np.mean(y_e_norm))**2)
r2     = 1 - ss_res/ss_tot

# 7) PRINT FIT RESULTS
print(f"Two-phase fit → α₂={alpha2_fit:.3f}, δE₁={deltaE1_fit:.3f}, τ₁={tau1_fit:.3f}, "
      f"α₁={alpha1_fit:.3f}, δE₂={deltaE2_fit:.3f}, τ₂={tau2_fit:.3f}")
print(f"  RMSE={rmse:.3f}, R²={r2:.3f}")

# 8) PLOT data vs. model
plt.figure(figsize=(6,4))
plt.plot(t_e, y_e_norm, 'o-', label='Eosinophil data')
plt.plot(t_e, E_fit_norm, 's--', label='Two-phase model')
plt.xlabel('Time (h post-challenge)')
plt.ylabel('Normalised eosinophils')
plt.title('Two-Phase Eosinophil Dynamics')
plt.legend()
plt.tight_layout()
plt.show()
