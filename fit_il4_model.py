# ================  fit_il4_model.py  =================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# 1) Load the IL‑4 data
data    = pd.read_csv(r'C:\Users\celal\Desktop\Python\Model Versions\il4_hdm.csv')
t_data  = data['time'].values    # [2, 4, 8, 24]
y_data  = data['il4'].values     # [50, 90, 55, 20]

# 2) Define the Engram‑Immune ODEs (same as before)
def EIM_ODEs(t, state, params):
    x, y, z, u = state
    zeta1    = params['zeta1']
    beta     = params['beta']
    Lambda1  = params['Lambda1']
    kappa    = params['kappa']
    y0       = params['y0']
    omega    = params['omega']
    delta    = params['delta']       # This is what we will fit
    lambda_v = params['lambda_val']
    Lambda2  = params['Lambda2']
    zeta2    = params['zeta2']
    xi       = params['xi']
    eta      = params['eta']
    Gamma    = params['Gamma']
    
    x_nn = max(x, 0)
    u_nn = max(u, 0)
    
    dxdt = (1-z)*(x_nn + zeta1*x_nn**beta) + Lambda1*u
    dydt = x - (kappa*x*(y+y0))/(omega+x) - delta*y*(1+lambda_v*z) + Lambda2*u
    dzdt = (y+y0)*(u_nn + zeta2*u_nn**beta) + xi*y - eta*z
    dudt = -(y+y0)*(u_nn + zeta2*u_nn**beta) + Gamma*u
    
    return [dxdt, dydt, dzdt, dudt]

# 3) Baseline parameters (keep all but delta at their previous values)
params_base = {
    'zeta1':0.1, 'beta':0.5, 'Lambda1':0.05, 'kappa':0.2,
    'y0':0.1,   'omega':1.0, 'delta':0.3,     'lambda_val':0.1,
    'Lambda2':0.04,'zeta2':0.1,'xi':0.2,      'eta':0.1,
    'Gamma':0.05
}

# 4) Simulation helper at exactly the data time‑points
initial_state = [0,0,0,1.0]
def simulate(params):
    sol = solve_ivp(
        fun=lambda t, s: EIM_ODEs(t, s, params),
        t_span=(t_data.min(), t_data.max()),
        y0=initial_state,
        t_eval=t_data
    )
    return sol.y[1]   # immune-response 'y'

# 5) Residuals function for least‑squares (fits δ only)
def residuals(p):
    params = params_base.copy()
    params['delta'] = p[0]
    y_sim = simulate(params)
    return y_sim - y_data

# 6) Run the fit
initial_guess = [params_base['delta']]
res = least_squares(residuals, initial_guess, bounds=(0, np.inf))
delta_fit = res.x[0]
print(f"Fitted δ = {delta_fit:.3f}")

# 7) Plot data vs fitted model
y_fit = simulate({**params_base, 'delta': delta_fit})

plt.figure(figsize=(6,4))
plt.scatter(t_data, y_data, color='black', label='Observed IL-4')
plt.plot   (t_data, y_fit,  color='C1',   label=f'Model (δ={delta_fit:.3f})')
plt.xlabel('Time (h post‑challenge)')
plt.ylabel('IL‑4 (pg/ml)')
plt.title ('Data vs Fitted Model')
plt.legend()
plt.tight_layout()
plt.show()
# ================  END SCRIPT  =================
