import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Starting conditions
k = 1
m = 1
x0, y0, v_x0 = 1, 0, 0
v_y0_values = [0.5, 1, np.sqrt(2), 2]

t_int = 0.01     # time interval
t_f = 30        # final time


def energy(x, y, v_x, v_y, m, k):
    K = m/2*((v_x)**2+(v_y)**2)
    r = np.sqrt(x**2+y**2)
    U = -k/r
    E = K + U
    return K, U, E


def angular_momentum(x, y, v_x, v_y, m):
    L = m*(x*v_y - y*v_x)
    return L


# Define the equations of motion
def equations_of_motion(t, z):
    x, y, v_x, v_y = z
    r = np.sqrt(x**2+y**2)
    a_x = -k*x/r**3
    a_y = -k*y/r**3
    return [v_x, v_y, a_x, a_y]


def calculate_solution(x0, y0, v_x0, v_y0, t_f, t_int):
    t_span = (0, t_f)
    z0 = np.array([x0, y0, v_x0, v_y0])
    diff_sol = solve_ivp(equations_of_motion, t_span, z0, t_eval=np.linspace(0, t_f, int(t_f/t_int)+1), max_step=0.01, atol=1e-6)
    return diff_sol


def calculate_trajectory(x0, y0, v_x0, v_y0, t_f, t_int):
    diff_sol = calculate_solution(x0, y0, v_x0, v_y0, t_f, t_int)
    return diff_sol.y[0], diff_sol.y[1]


def plot_trajectories(x0, y0, v_x0, v_y0_values, t_f, t_int):
    for v_y0 in v_y0_values:
        x_sol, y_sol = calculate_trajectory(x0, y0, v_x0, v_y0, t_f, t_int)
        plt.plot(x_sol, y_sol, label=f'vy0 = {round(float(v_y0),1)}')

        plt.title("Object Trajectory")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axhline(y=0, color='gray', linewidth=0.5)
        plt.axvline(x=0, color='gray', linewidth=0.5)
        plt.legend()
        plt.show()


def calculate_energy_components(x0, y0, vx0, vy0, t_f, t_int, m, k):
    diff_sol = calculate_solution(x0, y0, vx0, vy0, t_f, t_int)
    K, U, E = energy(diff_sol.y[0], diff_sol.y[1], diff_sol.y[2], diff_sol.y[3], m, k)
    return diff_sol.t, K, U, E


def plot_energy_components(x0, y0, vx0, vy0_values, t_f, t_int):
    for vy0 in vy0_values:
        t, K, U, E = calculate_energy_components(x0, y0, vx0, vy0, t_f, t_int, m, k)
        plt.plot(t, E, label=f'E(t) for vy0 = {round(float(vy0),1)}')
        plt.plot(t, U, label=f'U(t) for vy0 = {round(float(vy0),1)}')
        plt.plot(t, K, label=f'K(t) for vy0 = {round(float(vy0),1)}')

        plt.title("Energy development")
        plt.xlabel('t')
        plt.ylabel('Energies')
        plt.axhline(y=0, color='gray', linewidth=0.5)
        plt.axvline(x=0, color='gray', linewidth=0.5)
        plt.legend()
        plt.show()


def calculate_angular_momentum(x0, y0, vx0, vy0, t_f, t_int):
    diff_sol = calculate_solution(x0, y0, vx0, vy0, t_f, t_int)
    L = angular_momentum(diff_sol.y[0], diff_sol.y[1], diff_sol.y[2], diff_sol.y[3], m)
    return diff_sol.t, L


def plot_angular_momentum(x0, y0, vx0, vy0_values, t_f, t_int):
    for vy0 in vy0_values:
        t, L = calculate_angular_momentum(x0, y0, vx0, vy0, t_f, t_int)
        plt.plot(t, L, label=f'L(t) for vy0 = {round(float(vy0),1)}')

    plt.title("Angular momentum development")
    plt.xlabel('t')
    plt.ylabel('Angular momentum')
    plt.axhline(y=0, color='gray', linewidth=0.5)
    plt.axvline(x=0, color='gray', linewidth=0.5)
    plt.legend()
    plt.show()


# Plotting trajectories
plot_trajectories(x0, y0, v_x0, v_y0_values, t_f, t_int)

# Plotting time evolution of energy
plot_energy_components(x0, y0, v_x0, v_y0_values, t_f, t_int)

# Plotting time evolution of angular momentum
plot_angular_momentum(x0, y0, v_x0, v_y0_values, t_f, t_int)
