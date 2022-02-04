import time

from lcs import lcs_learning
import numpy as np
from casadi import *
import lcs.optim as opt
from lcs import lcs_control
import matplotlib.pyplot as plt

# --------
g = 9.81
mp = 0.411
mc = 0.978
len_p = 0.6
len_com = 0.4267
d1 = 0.35
d2 = -0.35
ks = 100
Ts = 0.01

n_state = 4
n_control = 1
n_lam = 2

A = [[0, 0, 1, 0], [0, 0, 0, 1], [0, g * mp / mc, 0, 0], [0, g * (mc + mp) / (len_com * mc), 0, 0]]
A = np.asarray(A)
B = [[0], [0], [1 / mc], [1 / (len_com * mc)]]
B = np.asarray(B)
C = [[0, 0], [0, 0], [(-1 / mc) + (len_p / (mc * len_com)), (1 / mc) - (len_p / (mc * len_com))],
     [(-1 / (mc * len_com)) + (len_p * (mc + mp)) / (mc * mp * len_com * len_com),
      -((-1 / (mc * len_com)) + (len_p * (mc + mp)) / (mc * mp * len_com * len_com))]]
C = np.asarray(C)
D = [[-1, len_p, 0, 0], [1, -len_p, 0, 0]]
D = np.asarray(D)
F = 1 / ks * np.eye(2)
F = np.asarray(F)
c = [[d1], [-d2]]
c = np.asarray(c)
d = np.zeros((4, 1))
E = np.zeros((2, 1))

A = np.eye(n_state) + Ts * A
B = Ts * B
C = Ts * C
dyn_offset = Ts * d

D = D
E = E
F = F
G_para = [sqrt(F[0, 0]), 0.0, sqrt(F[0, 0])]
H = np.zeros((n_lam, n_lam))
lcp_offset = c

# form the lcs system
min_sig = min(np.linalg.eigvals(F + F.T))

lsc_theta = veccat(vec(A), vec(B),
                   vec(C),
                   dyn_offset,
                   vec(D), vec(E),
                   vec(G_para),
                   vec(H),
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
    'D': D,
    'dyn_offset': dyn_offset,
    'E': E,
    'G_para': G_para,
    'G': sqrt(F),
    'H': H,
    'F': F,
    'lcp_offset': lcp_offset,
    'theta': lsc_theta,
    'n_theta': n_theta,
    'min_sig': min_sig}

np.save('cartpole_lcs', lcs_mats)
