import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def EIM_ODEs(t, state, params):
    """
    Computes the derivatives for the Engram-Immune Model.
    
    Parameters:
    - t: Time (required by the ODE solver)
    - state: A list or array containing [x, y, z, u]
        x: Engram-promoting signal
        y: Immune response
        z: Engram-inhibiting signal
        u: External immune trigger
    - params: Dictionary of parameters.
    
    Returns:
    - A list containing the derivatives [dx/dt, dy/dt, dz/dt, du/dt].
    """
    x, y, z, u = state
    
    # Unpack parameters
    zeta1 = params['zeta1']
    beta = params['beta']
    Lambda1 = params['Lambda1']
    kappa = params['kappa']
    y0 = params['y0']
    omega = params['omega']
    delta = params['delta']
    lambda_val = params['lambda_val']
    Lambda2 = params['Lambda2']
    zeta2 = params['zeta2']
    xi = params['xi']
    eta = params['eta']
    Gamma = params['Gamma']

    # Ensure x and u are non-negative before taking fractional powers
    x_nonneg = max(x, 0)
    u_nonneg = max(u, 0)

    
    # Define f(u); here we assume f(u)=u
    f_u = u
    
    # Calculate derivatives using the non-negative versions of x and u for fractional power operations.
    # These are basically the 4 ODE's in the model.
    dxdt = (1 - z) * (x_nonneg + zeta1 * (x_nonneg ** beta)) + Lambda1 * u
    dydt = x - (kappa * x * (y + y0))/(omega + x) - delta * y * (1 + lambda_val * z) + Lambda2 * u
    dzdt = (y + y0) * (u_nonneg + zeta2 * (u_nonneg ** beta)) + xi * y - eta * z
    dudt = - (y + y0) * (u_nonneg + zeta2 * (u_nonneg ** beta)) + Gamma * f_u
    
    return [dxdt, dydt, dzdt, dudt]

# Define a parameters dictionary (adjust values as needed)
params_base = {
    'zeta1': 0.1,
    'beta': 0.5,
    'Lambda1': 0.05,
    'kappa': 0.2,
    'y0': 0.1,
    'omega': 1.0,
    'delta': 0.3,  # Base value for delta
    'lambda_val': 0.1,
    'Lambda2': 0.04,
    'zeta2': 0.1,
    'xi': 0.2,
    'eta': 0.1,
    'Gamma': 0.05
}


# Initial conditions: [x, y, z, u]
initial_state = [0, 0, 0, 1.0]

# Define the time span for the simulation
t_span = (0, 50)  # from time 0 to 50 (arbitrary units)
t_eval = np.linspace(t_span[0], t_span[1], 300)

# Define a list of delta values to test
delta_values = [0.2, 0.3, 0.4, 0.5]

# Create a plot to show the immune response (y) for each value of delta
plt.figure(figsize=(10, 6))
for d in delta_values:
    # Copy the base parameters and update 'delta'
    params = params_base.copy()
    params['delta'] = d
    
    # Solve the ODE system
    solution = solve_ivp(lambda t, state: EIM_ODEs(t, state, params),
                         t_span, initial_state, t_eval=t_eval)
    
    # Plot the immune response (y is the second variable: index 1)
    plt.plot(solution.t, solution.y[1], label=f'delta = {d}')

plt.xlabel('Time')
plt.ylabel('Immune Response (y)')
plt.title('Sensitivity Analysis: Variation in delta')
plt.legend()
plt.show()
