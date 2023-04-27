import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the system of differential equations
def f(t, y, a, b, c, d):
    x, y, z, w = y
    dxdt = y
    dydt = a * (1 - x**2) * y - x + c * z
    dzdt = w
    dwdt = b * (1 - z**2) * w - z + d * x
    return [dxdt, dydt, dzdt, dwdt]

# Set the parameter values
a = 1.0
b = 3.0
c = 1.0
d = 5.0

# Set the initial conditions
x0 = 0.1
y0 = 0.2
z0 = 0.3
w0 = 0.4

y0_vec = [x0, y0, z0, w0]

# Set the time range to solve over
t_span = [0, 100]

# Set the solver options
opts = {
    'rtol': 1e-6,    # Relative tolerance
    'atol': 1e-6,    # Absolute tolerance
    'max_step': 0.1  # Maximum time step
}

# Solve the differential equation
sol = solve_ivp(lambda t, y: f(t, y, a, b, c, d), t_span, y0_vec, dense_output=True, **opts)

# Extract the solution components
t = sol.t
x, y, z, w = sol.y

# Plot the solution
fig, ax = plt.subplots()
ax.plot(x, y, label='x-y')
ax.plot(z, w, label='z-w')
ax.legend()
ax.set_xlabel('x, z')
ax.set_ylabel('y, w')
ax.set_title('Phase space')
plt.show()