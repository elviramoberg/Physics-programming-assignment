import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from math import sqrt

v_y0 = 0
z_t0 = {1, 0, 0, v_y0}


def angular_momentum(x, y, v_y, v_x, m):
    L = m*(x*v_y - y*v_x)
    return L


def kinetic_energy(v_y, v_x, m):
    K = m/2*((v_x)**2+(v_y)**2)
    return K


def potential_energy(x, y, k):
    U = -k/sqrt(x**2+y**2)
    return U


def total_energy(K, U):
    E_tot = K - U
    return E_tot

