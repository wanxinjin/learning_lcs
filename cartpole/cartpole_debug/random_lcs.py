import time

from lcs import lcs_learning
import numpy as np
from casadi import *
import lcs.optim as opt
from lcs import lcs_control
import matplotlib.pyplot as plt

n_state = 2
n_control = 1
n_lam = 2

A = np.random.randn(n_state, n_state)
B = np.random.randn(n_state, n_control)
C = np.random.randn(n_state, n_lam)
dyn_offset = np.random.randn(n_state)

D = np.random.randn(n_lam, n_state)
E = np.random.randn(n_lam, n_control)
G = np.random.randn(n_lam, n_lam)
H = np.random.randn(n_lam, n_lam)
F = G @ G.T + H - H.T
lcp_offset = np.random.randn(n_lam)

# form the lcs system
min_sig = min(np.linalg.eigvals(F + F.T))
print(min_sig)

lsc_theta = veccat(vec(A), vec(B),
                   vec(C),
                   dyn_offset,
                   vec(D), vec(E),
                   vec(F),
                   lcp_offset,
                   ).full().flatten()
n_theta = lsc_theta.size

lcs_mats = {
    'n_state': n_state,
    'n_control': n_control,
    'n_lam': n_lam,
    'A': A,
    'B': B,
    'C': C,
    'C_r': sqrt(C),
    'D': D,
    'dyn_offset': dyn_offset,
    'E': E,
    'H': H,
    'G': G,
    'F': F,
    'lcp_offset': lcp_offset,
    'theta': lsc_theta,
    'n_theta': n_theta,
    'min_sig': min_sig}

np.save('random_lcs', lcs_mats)
