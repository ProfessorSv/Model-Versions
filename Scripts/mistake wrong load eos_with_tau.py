# eos_with_tau.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# 1) Load & normalise Eotaxin-1 and IL-5
def load_norm(path, col):
    df = pd.read_csv(path)
    t  = df['time'].values
    y  = df[col].values
    y_adj = np.maximum(y - y[0], 0)
    return t, y_adj/np.max(y_adj)

t_eot, y_eot = load_norm(r'C:\Users\celal\Desktop\Model-Versions\data\Eotaxin-1_hdm.csv',
    'Eotaxin-1')
t_il5, y_il5 = load_norm(r'C:\Users\celal\Desktop\Model-Versions\data\il5_hdm.csv',     'il5')

# Interpolators
f_eot  = lambda t: np.interp(t, t_eot,  y_eot)
f_il5  = lambda t: np.interp(t, t_il5,  y_il5)

# 2) Load & normalise eosinophil counts
df_e = pd.read_csv(r'C:\Users\celal\Desktop\Model-Versions\data\Eotaxin-1_hdm.csv')
t_e = df_e['time'].values
y_e = df_e['Eotaxin-1'].values
y_e_norm = np.maximum(y_e - y_e[0], 0)/np.max(np.maximum(y_e - y_e[0],0))

# 3) Residuals with tau
def resid_tau(x):
    α1, α2, δE, τ = x
    def ode(t, E):
        return (α1*f_il5(t) + α2*f_eot(t) - δE*E)/τ
    sol = solve_ivp(ode, (t_e.min(), t_e.max()), [0], t_eval=t_e)
    E = sol.y[0]
    E_norm = E/np.max(E)
    return E_norm - y_e_norm

# 4) Fit α1, α2, δE, τ
init = [0.5, 1.0, 0.5, 1.0]
bnds = ([0,0,0,0],[np.inf]*4)
res  = least_squares(resid_tau, init, bounds=bnds)
α1_fit, α2_fit, δE_fit, τ_fit = res.x

# 5) Simulate & normalise
sol2 = solve_ivp(
    lambda t,E: (α1_fit*f_il5(t)+α2_fit*f_eot(t)-δE_fit*E)/τ_fit,
    (t_e.min(), t_e.max()), [0], t_eval=t_e
)
E2 = sol2.y[0]; E2_norm = E2/np.max(E2)

# 6) Metrics
rmse = np.sqrt(np.mean((E2_norm - y_e_norm)**2))
ss_res = np.sum((y_e_norm-E2_norm)**2)
ss_tot = np.sum((y_e_norm-np.mean(y_e_norm))**2)
r2 = 1 - ss_res/ss_tot

print(f"Fit with τ → α1={α1_fit:.3f}, α2={α2_fit:.3f}, δE={δE_fit:.3f}, τ={τ_fit:.3f}")
print(f"  RMSE={rmse:.3f}, R²={r2:.3f}")

# 7) Plot
plt.figure(figsize=(6,4))
plt.plot(t_e, y_e_norm, 'o-', label='Data')
plt.plot(t_e, E2_norm,'s--', label='Model w/ τ')
plt.xlabel('Time (h)')
plt.ylabel('Normalised eosinophils')
plt.title('Eosinophil Dynamics with Time‐Scaling')
plt.legend()
plt.tight_layout()
plt.show()
