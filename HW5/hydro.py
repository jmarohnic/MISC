import numpy as np

#Define system variables.
v = 1
gamma = 1.4

def p(state):
    return p = (gamma - 1) * state[0] * state[2]

#Define F(u).
def F(state):
    new_state = []
    new_state[1] = state[1]
    new_state[2] = state[1] * v + p(state)
    new_state[3] = (state[2] + p(state)) * v
    return new_state

#Define Lax scheme.
def lax(state):
    new_state = []
    new_state = 0.5 * (state
