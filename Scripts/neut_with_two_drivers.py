import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate  import solve_ivp
from scipy.optimize  import least_squares

# — Load & normalise helper —
def load_norm_csv(path, col):
    df = pd.read_csv(path)
    t  = df['time'].values
    y  = df[col].values
    y_adj = np.maximum(y - y[0], 0)
    return t, y_adj/np.max(y_adj)

# 1) Mediator curves
t_ck,  y_ck  = load_norm_csv(r'data/CXCL1_hdm.csv',  'CXCL1')
t_lm,  y_lm  = load_norm_csv(r'data/ltb4_hdm.csv',    'ltb4')
f_ck  = lambda t: np.interp(t, t_ck, y_ck)
f_lm  = lambda t: np.interp(t, t_lm, y_lm)

# 2) Neutrophil data
df = pd.read_csv(r'data/neutrophils_hdm.csv')
tN = df['time'].values
yN = df['Neut'].values
yN = np.maximum(yN - yN[0], 0)
yN_norm = yN/np.max(yN)

# 3) Residuals for two-driver model
def resid(x):
    αK, αL, δN, τN = x
    def ode(t, N):
        return (αK*f_ck(t) + αL*f_lm(t) - δN*N)/τN
    sol = solve_ivp(ode, (tN.min(), tN.max()), [0], t_eval=tN)
    N = sol.y[0]
    return (N/N.max()) - yN_norm

# 4) Fit all four params
init = [1, 1, 0.5, 1]
bnds = ([0,0,0,0], [np.inf]*4)
res  = least_squares(resid, init, bounds=bnds)
αK_fit, αL_fit, δN_fit, τN_fit = res.x

# 5) Simulate & normalise
soln = solve_ivp(lambda t, N: (αK_fit*f_ck(t)+αL_fit*f_lm(t)-δN_fit*N)/τN_fit,
                 (tN.min(), tN.max()), [0], t_eval=tN)
N_fit = soln.y[0]; Nn = N_fit/N_fit.max()

# 6) Metrics
rmse = np.sqrt(np.mean((Nn - yN_norm)**2))
ssr  = np.sum((yN_norm - Nn)**2)
sst  = np.sum((yN_norm - np.mean(yN_norm))**2)
r2   = 1 - ssr/sst

print(f"Two-driver fit → αK={αK_fit:.3f}, αL={αL_fit:.3f}, δN={δN_fit:.3f}, τN={τN_fit:.3f}")
print(f"  RMSE={rmse:.3f}, R²={r2:.3f}")

# 7) Plot
plt.figure()
plt.plot(tN, yN_norm, 'o-', label='Data')
plt.plot(tN, Nn,    's--', label='Model: CXCL1+LTB₄')
plt.xlabel('Time (h)')
plt.ylabel('Normalised neutrophils')
plt.legend(); plt.tight_layout(); plt.show()
