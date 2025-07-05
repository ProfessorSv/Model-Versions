import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# ——— 1) Load IL-4 data ——————————————————————————————
data   = pd.read_csv(r'C:\Users\celal\Desktop\Model-Versions\data\il4_hdm.csv')
t_data = data['time'].values
y_data = data['il4'].values

# ——— 2) Engram-Immune ODEs ——————————————————————————————
def EIM_ODEs(t, s, p):
    x, y, z, u = s
    ζ1    = p['zeta1'];   β = p['beta']
    Λ1    = p['Lambda1'];  κ = p['kappa']
    y0    = p['y0'];      ω = p['omega']
    δ     = p['delta'];   λv = p['lambda_val']
    Λ2    = p['Lambda2']; ζ2 = p['zeta2']
    ξ     = p['xi'];      η = p['eta']
    Γ     = p['Gamma']
    x_ = max(x,0);  u_ = max(u,0)
    
    dxdt = (1-z)*(x_ + ζ1*x_**β) + Λ1*u
    dydt = x - (κ*x*(y+y0))/(ω+x) - δ*y*(1 + λv*z) + Λ2*u
    dzdt = (y+y0)*(u_ + ζ2*u_**β) + ξ*y - η*z
    dudt = -(y+y0)*(u_ + ζ2*u_**β) + Γ*u
    
    return [dxdt, dydt, dzdt, dudt]

# ——— 3) Baseline parameters ——————————————————————————————
params_base = {
    'zeta1':0.1, 'beta':0.5, 'Lambda1':0.05, 'kappa':0.2,
    'y0':0.1,   'omega':1.0, 'delta':0.3,    'lambda_val':0.1,
    'Lambda2':0.04,'zeta2':0.1,'xi':0.2,     'eta':0.1,
    'Gamma':0.05
}

# ——— 4) Simulator at data times —————————————————————————
initial_state = [0,0,0,1.0]
def simulate(p):
    sol = solve_ivp(lambda t,s: EIM_ODEs(t,s,p),
                    (t_data.min(), t_data.max()),
                    initial_state, t_eval=t_data)
    return sol.y[1]

# ——— 5) Residuals for δ and Λ₂ —————————————————————————
def residuals(x):
    δ_guess, Λ2_guess = x
    p = params_base.copy()
    p['delta']   = δ_guess
    p['Lambda2'] = Λ2_guess
    return simulate(p) - y_data

# ——— 6) Fit both parameters ——————————————————————————
init = [params_base['delta'], params_base['Lambda2']]
res = least_squares(residuals, init, bounds=([0,0],[np.inf,np.inf]))
δ_fit, Λ2_fit = res.x
print(f"\nFitted δ = {δ_fit:.4f},  Λ₂ = {Λ2_fit:.4f}\n")

# ——— 7) Generate fitted curve & metrics ———————————————————
p_fit = {**params_base, 'delta':δ_fit, 'Lambda2':Λ2_fit}
y_fit = simulate(p_fit)

rmse = np.sqrt(np.mean((y_fit - y_data)**2))
ss_res = np.sum((y_data - y_fit)**2)
ss_tot = np.sum((y_data - np.mean(y_data))**2)
r2     = 1 - ss_res/ss_tot

# ——— 8) Plot data vs fit with annotations —————————————————
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(t_data, y_data, color='black', label='Observed IL-4')
ax.plot(t_data, y_fit, color='C1', label='Model fit')
ax.set_xlabel('Time (h post-challenge)')
ax.set_ylabel('IL-4 (pg/ml)')
ax.set_title('Fit of EIM to IL-4 Time Course')
txt = f'δ = {δ_fit:.4f}, Λ₂ = {Λ2_fit:.4f}\nRMSE = {rmse:.1f}\n$R^2$ = {r2:.2f}'
ax.text(0.05,0.95, txt, transform=ax.transAxes,
        va='top', bbox=dict(facecolor='white', alpha=0.7))
ax.legend()
plt.tight_layout()
plt.show()

# ——— 9) Residuals vs time ——————————————————————————
resid = y_fit - y_data
fig2, ax2 = plt.subplots(figsize=(6,4))
ax2.plot(t_data, resid, 'o-', label='Residuals')
ax2.axhline(0, linestyle='--', color='k')
ax2.set_xlabel('Time (h)')
ax2.set_ylabel('Residual (model−data)')
ax2.set_title('Fit Residuals')
ax2.legend()
plt.tight_layout()
plt.show()
