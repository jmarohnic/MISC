import numpy as np
import matplotlib.pyplot as plt

#Define system variables.
steps = 100
N = 100
dt = 0.01
dx = 1. / N
v = 1
gamma = 1.4
#C = abs(v) * dt / dx

#each element of state is an array of spatial coordinates for the corresponding quantity

def p(state):
    p = (gamma - 1) * state[0] * epsilon(state) 
    return p
    
def epsilon(state):
    eps = (state[2] / state[0]) - (0.5 * v ** 2)
    return eps
    
#Define F(u).
def F(state):
    F_state = [np.zeros(N), np.zeros(N), np.zeros(N)]
    F_state[0] = state[1]
    F_state[1] = state[1] * v + p(state)
    F_state[2] = (state[2] + p(state)) * v
    return F_state

#Define Lax scheme.
def lax(state):
    new_state = [np.zeros(N), np.zeros(N), np.zeros(N)]
    for i in range(3):
        new_state[i][0] = 1.5 #state[i][0]
        new_state[i][N-1] = 1 #state[i][N-1]
        for j in range(1,N-1): 
            new_state[i][j] = 0.5 * ((state)[i][j-1] + (state)[i][j+1]) - (dt / (dx * 2)) * (F(state)[i][j+1] - F(state)[i][j-1])
    return new_state

rho1 = np.array([1] * (N / 2) + [0.125] * (N / 2))
e1 = np.array([1. / ((gamma - 1) * 1)] * (N / 2) + [0.1 / ((gamma - 1) * 0.125)] * (N / 2))
system = np.array([rho1, rho1 * v, e1])
grid = np.linspace(0, 1, N)
initial = system

for i in range(steps):
    plt.scatter(grid, system[0])
    plt.savefig("frame%03i.png" % (i))
    plt.close()
    system = lax(system)
    
