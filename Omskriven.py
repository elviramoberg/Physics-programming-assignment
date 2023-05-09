import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the equations of motion
def calculate_acceleration(x, y):
    r = np.sqrt(x**2 + y**2)
    ax = -x / r**3
    ay = -y / r**3
    return ax, ay

def calculate_trajectory(x0, y0, vx0, vy0, tf, dt):
    t_span = (0, tf)
    z0 = np.array([x0, y0, vx0, vy0])
    sol = solve_ivp(lambda t, z: np.array([z[2], z[3], *calculate_acceleration(*z[:2])]), t_span, z0, t_eval=np.arange(0, tf, dt))
    return sol.y[0], sol.y[1]

def plot_trajectories(x0, y0, vx0, vy0_values, tf, dt):
    plt.figure(figsize=(8, 6))
    for vy0 in vy0_values:
        x, y = calculate_trajectory(x0, y0, vx0, vy0, tf, dt)
        plt.plot(x, y, label=f'vy0 = {vy0:.1f}')

    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Trajectory of the Object")
    plt.grid(True)
    plt.legend()
    plt.show()

def calculate_energy_components(x0, y0, vx0, vy0, tf, dt):
    t_span = (0, tf)
    z0 = np.array([x0, y0, vx0, vy0])
    sol = solve_ivp(lambda t, z: np.array([z[2], z[3], *calculate_acceleration(*z[:2])]), t_span, z0, t_eval=np.arange(0, tf, dt))
    K = 1/2 * (sol.y[2]**2 + sol.y[3]**2)
    U = -1/np.sqrt(sol.y[0]**2 + sol.y[1]**2)
    E = K + U
    return sol.t, K, U, E

def plot_energy_components(x0, y0, vx0, vy0_values, tf, dt):
    plt.figure(figsize=(8, 6))
    for vy0 in vy0_values:
        t, K, U, E = calculate_energy_components(x0, y0, vx0, vy0, tf, dt)
        plt.plot(t, E, label=f'E for vy0 = {vy0:.1f}')
        plt.plot(t, U, label=f'U for vy0 = {vy0:.1f}')
        plt.plot(t, K, label=f'K for vy0 = {vy0:.1f}')

    plt.xlabel('time')
    plt.ylabel('Energies')
    plt.title("Graphs of energies")
    plt.legend()
    plt.show()

def plot_energy(vy0_values):
    x0, y0, vx0 = 1, 0, 0
    dt = 0.1
    tf = 30
    for vy0 in vy0_values:
        z0 = np.array([x0, y0, vx0, vy0])
        sol = solve_ivp(Object, (0, tf), z0, t_eval=np.arange(0, tf, dt))
        K = 0.5 * (sol.y[2] ** 2 + sol.y[3] ** 2)
        U = -1 / np.sqrt(sol.y[0] ** 2 + sol.y[1] ** 2)
        E = K + U

        plt.plot(sol.t, E, label=f'E')
        plt.plot(sol.t, U, label=f'U')
        plt.plot(sol.t, K, label=f'K')

    plt.xlabel('time')
    plt.ylabel('Energies')
    plt.title("Graphs of energies for different initial velocities")
    plt.legend()
    plt.show()


# Starting conditions
x0, y0, vx0 = 1, 0, 0
vy0_values = [0.5, 1, 1.2, np.sqrt(2), 2]

# Setting the time interval and final time
dt = 0.1
tf = 30

# Plotting trajectories
plot_trajectories(x0, y0, vx0, vy0_values, tf, dt)

# Plotting time evolution of energy components
vy0_values = [0.5, 1, 1.2, np.sqrt(2), 2]
plot_energy(vy0_values)





