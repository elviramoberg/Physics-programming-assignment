import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Funktion som beräknar den differentialekvation som vi vill lösa
def motion_equations(t, y, k, m, g):
    x1, v1, x2, v2 = y
    dx1_dt = v1
    dv1_dt = -k/m*x1 - g
    dx2_dt = v2
    dv2_dt = -k/m*x2 - g
    return [dx1_dt, dv1_dt, dx2_dt, dv2_dt]

# Definiera de olika variablerna
k = 1
m = 1
g = 9.81
initial_conditions = [1, 0, -1, 0]
t_span = [0, 10]

# Anropa solve_ivp-funktionen för att lösa differentialekvationen
solution = solve_ivp(motion_equations, t_span, initial_conditions, args=(k, m, g), dense_output=True, rtol=1e-6)

# Plotta resultaten
t_eval = np.linspace(t_span[0], t_span[1], 1000)
y_eval = solution.sol(t_eval)

plt.plot(y_eval[0], y_eval[1], label='Massa 1')
plt.plot(y_eval[2], y_eval[3], label='Massa 2')
plt.xlabel('x (m)')
plt.ylabel('v (m/s)')
plt.title('Rörelse hos två massor som är kopplade med en fjäder')
plt.legend()
plt.show()