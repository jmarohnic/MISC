import numpy as np
import matplotlib.pyplot as plt

#Define system variables.
steps = 100
N = 50
dt = 0.0025
dx = 1. / N
v1 = 0
gamma = 1.4
#C = abs(v) * dt / dx

#each element of state is an array of spatial coordinates for the corresponding quantity

def p(state):
    p = (gamma - 1) * state[0] * epsilon(state) 
    return p
    
def epsilon(state):
    eps = (state[2] / state[0]) - (0.5 * v(state) ** 2)
    return eps
    
def v(state):
	v = state[1] / state[0]
	return v
    
#Define F(u).
def F(state):
    F_state = [np.zeros(N), np.zeros(N), np.zeros(N)]
    F_state[0] = state[1]
    F_state[1] = state[1] * v(state) + p(state)
    F_state[2] = (state[2] + p(state)) * v(state)
    return F_state

#Define Lax scheme.
def lax(state):
    new_state = [np.zeros(N), np.zeros(N), np.zeros(N)]
    for i in range(3):
        new_state[i][0] = state[i][0]
        new_state[i][N-1] = state[i][N-1]
        for j in range(1,N-1): 
            new_state[i][j] = 0.5 * (state[i][j-1] + state[i][j+1]) - (dt / (dx * 2)) * (F(state)[i][j+1] - F(state)[i][j-1])
    return new_state

rho1 = np.array([1] * (N / 2) + [0.125] * (N / 2))
e1 = np.array([1. / ((gamma - 1) * 1)] * (N / 2) + [0.1 / ((gamma - 1) * 0.125)] * (N / 2))
system = np.array([rho1, rho1 * v1, e1])
grid = np.linspace(0, 1, N)
initial = system

for i in range(steps):
#     plt.scatter(grid, v(system))
#     plt.ylim((-2, 4))
#     plt.savefig("frame%03i.png" % (i))
#     plt.close()
    system = lax(system)
    
plt.subplot(2, 2, 1)
plt.title("density")
plt.scatter(grid, system[0])

plt.subplot(2, 2, 2)
plt.title("pressure")
plt.scatter(grid, p(system))

plt.subplot(2, 2, 3)
plt.title("velocity")
plt.scatter(grid, v(system))

plt.subplot(2, 2, 4)
plt.title("specific internal energy")
plt.scatter(grid, epsilon(system))

plt.show()