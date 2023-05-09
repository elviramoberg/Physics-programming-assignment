
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from math import sqrt

# Constants
m = 1.0
k = 1.0
t_0 = 0.0
t_f = 30.0
t_span = [0.0, 100.0]

v_y0 = 1

# Initial conditions
z_t0 = [1, 0, 0, v_y0]


def movement_function(t, z):
    x, y, v_x, v_y = z
    r = np.sqrt(x**2+y**2)
    a_x = -k*x/r**3
    a_y = -k*y/r**3
    return [v_x, v_y, a_x, a_y]


def angular_momentum(x, y, v_y, v_x, m):
    L = m*(x*v_y - y*v_x)
    return L


def kinetic_energy(v_y, v_x, m):
    K = m/2*((v_x)**2+(v_y)**2)
    return K


def potential_energy(x, y, k):
    r = np.sqrt(x**2+y**2)
    U = -k/r
    return U


def total_energy(K, U):
    E_tot = K + U
    return E_tot


def find_trajectory():

    # Solve the ODE
    sol = solve_ivp(movement_function, t_span, z_t0, t_eval=np.linspace(t_0, t_f, 10000))

    # Extract the solution
    x = sol.y[0]
    y = sol.y[1]
    v_x = sol.y[2]
    v_y = sol.y[3]
    t = sol.t

    # Compute the energy
    K = kinetic_energy(v_y, v_x, m)
    U = potential_energy(x, y, k)
    E_tot = total_energy(K, U)

    # Compute the angular momentum
    L = angular_momentum(x, y, v_y, v_x, m)

    # Plot the trajectory
    plt.plot(x, y)
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectory of a particle under inverse square force')

    # Plot the energy and angular momentum
    fig, axs = plt.subplots(2)
    fig.suptitle('Energy and angular momentum of a particle under inverse square force')
    axs[0].plot(t, K, label='Kinetic energy')
    axs[0].plot(t, U, label='Potential energy')
    axs[0].plot(t, E_tot, label='Total energy')
    axs[0].legend()
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Energy')
    axs[1].plot(t, L)
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Angular momentum')
    plt.show()


find_trajectory()
