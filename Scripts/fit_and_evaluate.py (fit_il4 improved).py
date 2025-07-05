import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# ——— 1) Load IL-4 data from updated folder ——————————————
data   = pd.read_csv(r'C:\Users\celal\Desktop\Model-Versions\data\il4_hdm.csv')
t_data = data['time'].values
y_data = data['il4'].values

# ——— 2) ODE system (unchanged) —————————————————————————
def EIM_ODEs(t, state, p):
    x, y, z, u = state
    ζ1, β, Λ1, κ = p['zeta1'], p['beta'], p['Lambda1'], p['kappa']
    y0, ω, δ, λv = p['y0'], p['omega'], p['delta'], p['lambda_val']
    Λ2, ζ2, ξ, η = p['Lambda2'], p['zeta2'], p['xi'], p['eta']
    Γ = p['Gamma']
    x_ = max(x,0);    u_ = max(u,0)
    dxdt = (1 - z)*(x_ + ζ1*x_**β) + Λ1*u
    dydt = x - (κ*x*(y+y0))/(ω + x) - δ*y*(1 + λv*z) + Λ2*u
    dzdt = (y+y0)*(u_ + ζ2*u_**β) + ξ*y - η*z
    dudt = -(y+y0)*(u_ + ζ2*u_**β) + Γ*u
    return [dxdt, dydt, dzdt, dudt]

# ——— 3) Baseline params ——————————————————————————————
params_base = {
  'zeta1':0.1,'beta':0.5,'Lambda1':0.05,'kappa':0.2,
  'y0':0.1,'omega':1.0,'delta':0.3,'lambda_val':0.1,
  'Lambda2':0.04,'zeta2':0.1,'xi':0.2,'eta':0.1,'Gamma':0.05
}

# ——— 4) Simulation helper ————————————————————————————
initial_state = [0,0,0,1.0]
def simulate(p):
    sol = solve_ivp(lambda t,s: EIM_ODEs(t,s,p),
                    (t_data.min(),t_data.max()),
                    initial_state, t_eval=t_data)
    return sol.y[1]

# ——— 5) Fit δ, Λ1, and S (scaling) ——————————————————————————————
def residuals(params):
    δ, Λ1, S = params
    ps = params_base.copy()
    ps['delta']   = δ
    ps['Lambda1'] = Λ1
    return S * simulate(ps) - y_data

initial_guess = [params_base['delta'], params_base['Lambda1'], 1.0]
res = least_squares(
    residuals,
    initial_guess,
    bounds=([0, 0, 0], [np.inf, np.inf, np.inf])
)
δ_fit, Λ1_fit, S_fit = res.x
print(f"Fitted δ = {δ_fit:.4f},  Λ₁ = {Λ1_fit:.4f},  S = {S_fit:.4f}")

# ── Recompute fitted curve using all fitted parameters ──
p_fitted = {**params_base, 'delta': δ_fit, 'Lambda1': Λ1_fit}
y_fit = S_fit * simulate(p_fitted)

# ——— 6) Compute fitted curve & metrics ————————————————————
y_fit = S_fit * simulate({**params_base,'delta':δ_fit})
rmse = np.sqrt(np.mean((y_fit-y_data)**2))
ss_res = np.sum((y_data-y_fit)**2)
ss_tot = np.sum((y_data-np.mean(y_data))**2)
r2     = 1 - ss_res/ss_tot

# ——— 7) Plot data vs fit with annotations ——————————————
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(t_data, y_data, color='black', label='Observed IL-4')
ax.plot(t_data, y_fit,   color='C1',   label=f'Model fit')
ax.set_xlabel('Time (h post-challenge)')
ax.set_ylabel('IL-4 (pg/ml)')
ax.set_title('Fit of Engram-Immune Model to IL-4 Data')
# annotate RMSE and R² inside the plot
textstr = f'RMSE = {rmse:.1f} pg/ml\n$R^2$ = {r2:.2f}\nδ = {δ_fit:.3f}, S = {S_fit:.1f}'
ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
ax.legend()
plt.tight_layout()
plt.show()

# ——— 8) Residuals plot ————————————————————————————
resid = y_fit - y_data
fig2, ax2 = plt.subplots(figsize=(6,4))
ax2.plot(t_data, resid, 'o-', label='Residuals')
ax2.axhline(0, color='k', linestyle='--')
ax2.set_xlabel('Time (h)')
ax2.set_ylabel('Residual (model − data)')
ax2.set_title('Residuals vs Time')
ax2.legend()
plt.tight_layout()
plt.show()
