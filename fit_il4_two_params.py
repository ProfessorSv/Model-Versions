import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# 1) Load the IL-4 time-course data
data   = pd.read_csv(r'C:\Users\celal\Desktop\Python\Model Versions\il4_hdm.csv')
t_data = data['time'].values    # [ 2,  4,  8, 24]
y_data = data['il4'].values     # [50, 90, 55, 20]

# 2) Define the ODE system
def EIM_ODEs(t, state, params):
    x, y, z, u = state
    ζ1    = params['zeta1']
    β     = params['beta']
    Λ1    = params['Lambda1']
    κ     = params['kappa']
    y0    = params['y0']
    ω     = params['omega']
    δ     = params['delta']
    λv    = params['lambda_val']
    Λ2    = params['Lambda2']
    ζ2    = params['zeta2']
    ξ     = params['xi']
    η     = params['eta']
    Γ     = params['Gamma']
    
    x_ = max(x, 0)
    u_ = max(u, 0)

    dxdt = (1 - z)*(x_ + ζ1*x_**β) + Λ1*u
    dydt = x - (κ*x*(y+y0))/(ω + x) - δ*y*(1 + λv*z) + Λ2*u
    dzdt = (y+y0)*(u_ + ζ2*u_**β) + ξ*y - η*z
    dudt = - (y+y0)*(u_ + ζ2*u_**β) + Γ*u

    return [dxdt, dydt, dzdt, dudt]

# 3) Baseline parameters
params_base = {
    'zeta1':0.1, 'beta':0.5,   'Lambda1':0.05, 'kappa':0.2,
    'y0':0.1,   'omega':1.0,   'delta':0.3,    'lambda_val':0.1,
    'Lambda2':0.04,'zeta2':0.1,'xi':0.2,       'eta':0.1,
    'Gamma':0.05
}

# 4) Simulator at the data time-points
initial_state = [0,0,0,1.0]
def simulate(params):
    sol = solve_ivp(
        lambda t, s: EIM_ODEs(t, s, params),
        (t_data.min(), t_data.max()),
        initial_state,
        t_eval=t_data
    )
    return sol.y[1]  # return y

# 5) Residuals: fits δ and amplitude S
def residuals(p):
    δ, S = p
    ps = params_base.copy()
    ps['delta'] = δ
    y_mod = simulate(ps)
    return S*y_mod - y_data

# 6) Perform the fit
guess = [params_base['delta'], max(y_data)/max(simulate(params_base))]
res = least_squares(residuals, guess, bounds=([0,0],[np.inf,np.inf]))
δ_fit, S_fit = res.x
print(f"Fitted δ = {δ_fit:.4f},  Scaling S = {S_fit:.1f}")

# 7) Plot observed vs fitted
y_fit = S_fit * simulate({**params_base, 'delta':δ_fit})

plt.figure(figsize=(6,4))
plt.scatter(t_data, y_data, color='black', label='Observed IL-4')
plt.plot  (t_data, y_fit,    color='C1',   label=f'Model (δ={δ_fit:.4f}, S={S_fit:.1f})')
plt.xlabel('Time (h post-challenge)')
plt.ylabel('IL-4 (pg/ml)')
plt.title ('Observed vs Fitted IL-4 Time Course')
plt.legend()
plt.tight_layout()
plt.show()
